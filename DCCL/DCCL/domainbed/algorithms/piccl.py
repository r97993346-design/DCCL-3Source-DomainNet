"""Paired-Intervention Causal Connectivity Learning (PICCL).

PICCL is implemented as an independent algorithm on top of the existing DCCL
infrastructure.  The frozen pre-trained featurizer is used as an intervention
response reference, not as a distillation teacher.
"""

import copy
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.lib import misc

from .algorithms import DCCL


class PairedInterventionResponseEstimator(nn.Module):
    """Estimate fine-tuning-induced intervention sensitivity.

    Inputs are pooled features with shape ``[B, D]``.  Reference responses are
    detached, and callers should detach the returned tensor before using it as
    a subspace target.
    """

    def forward(
        self,
        adapted: torch.Tensor,
        adapted_intervened: torch.Tensor,
        reference: torch.Tensor,
        reference_intervened: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            adapted.shape
            == adapted_intervened.shape
            == reference.shape
            == reference_intervened.shape
        ):
            raise ValueError(
                "PIRE requires adapted/reference original/intervened features "
                "with identical [B, D] shapes."
            )
        return (adapted_intervened - adapted) - (
            reference_intervened - reference
        ).detach()


class ClassDomainResidualBank(nn.Module):
    """EMA bank for class-conditional, domain-conditional adaptation residuals.

    Buffers:
        prototypes: ``[C, K, D]`` residual means.
        initialized: ``[C, K]`` valid-prototype mask.
        counts: ``[C, K]`` number of EMA updates.
    """

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        feature_dim: int,
        momentum: float = 0.99,
        min_valid_domains: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.min_valid_domains = int(min_valid_domains)

        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.num_domains, self.feature_dim),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.bool),
        )
        self.register_buffer(
            "counts",
            torch.zeros(self.num_classes, self.num_domains, dtype=torch.long),
        )

    @torch.no_grad()
    def update(
        self,
        residuals: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
    ) -> None:
        """Update prototypes from detached ``[B, D]`` residuals."""
        if residuals.ndim != 2 or residuals.shape[1] != self.feature_dim:
            raise ValueError("Residuals must have shape [B, feature_dim].")
        if labels.shape[0] != residuals.shape[0] or domains.shape[0] != residuals.shape[0]:
            raise ValueError("Residual, label and domain batch sizes must match.")

        residuals = residuals.detach().float()
        labels = labels.detach().long()
        domains = domains.detach().long()

        for domain_id in domains.unique().tolist():
            if domain_id < 0 or domain_id >= self.num_domains:
                raise ValueError(
                    "Local source-domain id {} is outside [0, {}).".format(
                        domain_id, self.num_domains
                    )
                )
            domain_mask = domains.eq(domain_id)
            for class_id in labels[domain_mask].unique().tolist():
                if class_id < 0 or class_id >= self.num_classes:
                    continue
                mask = domain_mask & labels.eq(class_id)
                if not mask.any():
                    continue
                batch_mean = residuals[mask].mean(dim=0)
                if self.initialized[class_id, domain_id]:
                    self.prototypes[class_id, domain_id].mul_(self.momentum).add_(
                        batch_mean, alpha=1.0 - self.momentum
                    )
                else:
                    self.prototypes[class_id, domain_id].copy_(batch_mean)
                    self.initialized[class_id, domain_id] = True
                self.counts[class_id, domain_id] += 1

    @torch.no_grad()
    def responses_for(
        self,
        labels: torch.Tensor,
        domains: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return valid per-sample domain responses and their sample mask.

        A response is emitted only if the sample class has prototypes in at
        least ``min_valid_domains`` source domains.
        """
        labels = labels.long()
        domains = domains.long()
        valid_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        responses = []

        for index, (class_id, domain_id) in enumerate(zip(labels.tolist(), domains.tolist())):
            valid_domains = self.initialized[class_id]
            if int(valid_domains.sum().item()) < self.min_valid_domains:
                continue
            if not bool(valid_domains[domain_id].item()):
                continue
            class_center = self.prototypes[class_id, valid_domains].mean(dim=0)
            responses.append(self.prototypes[class_id, domain_id] - class_center)
            valid_mask[index] = True

        if responses:
            return torch.stack(responses, dim=0).to(labels.device), valid_mask
        return self.prototypes.new_zeros((0, self.feature_dim)).to(labels.device), valid_mask


class InterventionSensitiveSubspace(nn.Module):
    """Low-rank intervention-sensitive subspace without a dense D x D matrix."""

    def __init__(self, feature_dim: int, rank: int = 16, eps: float = 1e-8) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.rank = min(int(rank), self.feature_dim)
        self.eps = float(eps)
        basis = torch.empty(self.feature_dim, self.rank)
        nn.init.orthogonal_(basis)
        self.basis = nn.Parameter(basis)

    def orthonormal_basis(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # QR is evaluated in fp32 for numerical stability under AMP.
        q = torch.linalg.qr(self.basis.float(), mode="reduced").Q
        if dtype is not None:
            q = q.to(dtype=dtype)
        return q

    def project(
        self,
        vectors: torch.Tensor,
        receive_basis_grad: bool = True,
    ) -> torch.Tensor:
        if vectors.ndim != 2 or vectors.shape[1] != self.feature_dim:
            raise ValueError("Subspace projection expects [N, feature_dim].")
        q = self.orthonormal_basis(dtype=vectors.dtype)
        if not receive_basis_grad:
            q = q.detach()
        return (vectors @ q) @ q.transpose(0, 1)

    def coverage_loss(
        self,
        responses: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_parts = [part for part in responses if part is not None and part.numel() > 0]
        if not valid_parts:
            zero = self.basis.sum() * 0.0
            return zero, zero.detach()

        targets = torch.cat(valid_parts, dim=0).detach()
        norms_sq = targets.pow(2).sum(dim=1)
        nonzero = norms_sq > self.eps
        if not nonzero.any():
            zero = self.basis.sum() * 0.0
            return zero, zero.detach()

        targets = targets[nonzero]
        norms_sq = norms_sq[nonzero]
        projected = self.project(targets, receive_basis_grad=True)
        residual_sq = (targets - projected).pow(2).sum(dim=1)
        loss = (residual_sq / (norms_sq + self.eps)).mean()
        capture = (projected.pow(2).sum(dim=1) / (norms_sq + self.eps)).mean()
        return loss, capture.detach()

    def orthogonality_loss(self) -> torch.Tensor:
        normalized = F.normalize(self.basis, dim=0, eps=self.eps)
        gram = normalized.transpose(0, 1) @ normalized
        identity = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).mean()


class CausalMediatorProjection(nn.Module):
    """Remove a soft amount of the intervention-sensitive component."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        subspace: InterventionSensitiveSubspace,
        alpha: float,
        receive_basis_grad: bool = False,
    ) -> torch.Tensor:
        sensitive = subspace.project(features, receive_basis_grad=receive_basis_grad)
        return self.norm(features - float(alpha) * sensitive)


class PICCLForwardModel(nn.Module):
    """Inference-only wrapper used by code paths expecting a forward model."""

    def __init__(
        self,
        featurizer: nn.Module,
        subspace: InterventionSensitiveSubspace,
        mediator: CausalMediatorProjection,
        classifier: nn.Module,
        alpha: float,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.subspace = subspace
        self.mediator = mediator
        self.classifier = classifier
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.featurizer(x)
        mediator = self.mediator(
            features, self.subspace, self.alpha, receive_basis_grad=False
        )
        return self.classifier(mediator)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class PICCL(DCCL):
    """Paired-Intervention Causal Connectivity Learning.

    The implementation reuses DCCL's featurizers, projection heads, pre-trained
    anchoring loss and training data pipeline, while replacing the trainable
    representation with an intervention-corrected mediator.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        feature_dim = self.featurizer.n_outputs
        self.paired_response = PairedInterventionResponseEstimator()
        self.residual_bank = ClassDomainResidualBank(
            num_classes=num_classes,
            num_domains=num_domains,
            feature_dim=feature_dim,
            momentum=self._hp("piccl_proto_momentum", 0.99),
            min_valid_domains=self._hp("piccl_min_valid_domains", 2),
        )
        self.sensitive_subspace = InterventionSensitiveSubspace(
            feature_dim=feature_dim,
            rank=self._hp("piccl_rank", 16),
            eps=self._hp("piccl_eps", 1e-8),
        )
        self.mediator_projection = CausalMediatorProjection(feature_dim)
        self.register_buffer("piccl_alpha_state", torch.tensor(0.0))

        self.piccl_gt_mode = str(self._hp("piccl_gt_mode", "replace"))
        if self.piccl_gt_mode not in {"replace", "keep"}:
            raise ValueError("piccl_gt_mode must be 'replace' or 'keep'.")
        self.basis_receive_task_grad = bool(
            self._hp("piccl_basis_receive_task_grad", False)
        )

        # The reference encoder is a frozen intervention-response control.
        for parameter in self.pre_featurizer.parameters():
            parameter.requires_grad_(False)
        self.pre_featurizer.eval()

        # Rebuild the optimizer so PICCL parameters are trainable while keeping
        # DCCL's learning-rate ratios for the existing components.
        lower_cls = 0.1
        lower_proj = 10.0
        weight_decay = self.hparams["weight_decay"]
        parameter_groups = [
            {
                "params": self.featurizer.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": weight_decay,
            },
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams["lr"] / lower_cls,
                "weight_decay": weight_decay,
            },
            {
                "params": self.proj_head.parameters(),
                "lr": self.hparams["lr"] / lower_proj,
                "weight_decay": weight_decay,
            },
            {
                "params": self.pre_proj_head.parameters(),
                "lr": self.hparams["lr"] / lower_proj,
                "weight_decay": weight_decay,
            },
            {
                "params": self.sensitive_subspace.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": 0.0,
            },
            {
                "params": self.mediator_projection.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": weight_decay,
            },
        ]
        if self.piccl_gt_mode == "keep":
            parameter_groups.extend(
                [
                    {"params": self.mean_encoders.parameters(), "lr": self.hparams["lr"] * 10},
                    {"params": self.var_encoders.parameters(), "lr": self.hparams["lr"] * 10},
                ]
            )
        self.optimizer = get_optimizer(self.hparams["optimizer"], parameter_groups)

    def _hp(self, name: str, default):
        return self.hparams[name] if name in self.hparams else default

    def train(self, mode: bool = True):
        super().train(mode)
        # Calling algorithm.train() must never enable BN/dropout updates in the
        # frozen reference branch.
        self.pre_featurizer.eval()
        return self

    def _alpha_and_ramp(self, step: int) -> Tuple[float, float]:
        total_steps = max(int(self._hp("piccl_total_steps", 15001)), 1)
        progress = min(max(float(step) / max(total_steps - 1, 1), 0.0), 1.0)
        warmup = float(self._hp("piccl_warmup_ratio", 0.10))
        ramp_length = max(float(self._hp("piccl_ramp_ratio", 0.20)), 1e-12)
        if progress <= warmup:
            ramp = 0.0
        elif progress >= warmup + ramp_length:
            ramp = 1.0
        else:
            ramp = (progress - warmup) / ramp_length
        return float(self._hp("piccl_alpha_max", 0.5)) * ramp, ramp

    @staticmethod
    def _local_domain_ids(x_by_domain: Sequence[torch.Tensor]) -> torch.Tensor:
        domain_parts = [
            torch.full(
                (batch.shape[0],),
                local_domain,
                device=batch.device,
                dtype=torch.long,
            )
            for local_domain, batch in enumerate(x_by_domain)
        ]
        return torch.cat(domain_parts, dim=0)

    @staticmethod
    def _paired_nt_xent(
        first: torch.Tensor,
        second: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("Intervention contrast requires equal feature shapes.")
        if first.shape[0] < 2:
            return first.sum() * 0.0
        logits = first @ second.transpose(0, 1) / float(temperature)
        targets = torch.arange(first.shape[0], device=first.device)
        return 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.transpose(0, 1), targets)
        )

    @staticmethod
    def _cross_domain_supervised_contrast(
        features: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """SupCon restricted to same-class samples from different domains."""
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.sum() * 0.0
        similarity = features @ features.transpose(0, 1) / float(temperature)
        eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        denominator_mask = ~eye
        positive_mask = (
            labels.view(-1, 1).eq(labels.view(1, -1))
            & domains.view(-1, 1).ne(domains.view(1, -1))
            & denominator_mask
        )
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return features.sum() * 0.0

        masked_similarity = similarity.masked_fill(~denominator_mask, float("-inf"))
        log_denominator = torch.logsumexp(masked_similarity, dim=1, keepdim=True)
        log_probability = similarity - log_denominator
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        per_anchor = -(
            log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count
        )
        return per_anchor[valid_anchor].mean()

    def _gt_loss(self, inter_feats, pre_feats) -> torch.Tensor:
        reg_loss = inter_feats[0].new_zeros(())
        for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            mean = mean_enc(inter_f)
            var = var_enc(inter_f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss = reg_loss + vlb.mean() / 2.0
        return reg_loss

    def update(self, x, y, **kwargs):
        if "x_2" not in kwargs:
            raise KeyError("PICCL requires DCCL's strong augmentation key 'x_2'.")

        all_x = torch.cat(x, dim=0)
        all_y = torch.cat(y, dim=0)
        all_x_intervened = torch.cat(kwargs["x_2"], dim=0)
        all_domains = self._local_domain_ids(x)
        step = int(kwargs.get("step", 0))

        adapted, inter_feats = self.featurizer(all_x, ret_feats=True)
        adapted_intervened, _ = self.featurizer(all_x_intervened, ret_feats=True)

        self.pre_featurizer.eval()
        with torch.no_grad():
            reference, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
            reference_intervened = self.pre_featurizer(all_x_intervened)

        if not (
            adapted.shape
            == adapted_intervened.shape
            == reference.shape
            == reference_intervened.shape
        ):
            raise RuntimeError(
                "PICCL requires identical pooled adapted/reference feature dimensions; "
                "got {}, {}, {}, {}.".format(
                    tuple(adapted.shape),
                    tuple(adapted_intervened.shape),
                    tuple(reference.shape),
                    tuple(reference_intervened.shape),
                )
            )

        delta_pair = self.paired_response(
            adapted, adapted_intervened, reference, reference_intervened
        )
        residual = adapted - reference.detach()
        self.residual_bank.update(residual.detach(), all_y, all_domains)
        delta_domain, valid_domain_mask = self.residual_bank.responses_for(
            all_y, all_domains
        )

        loss_isr, capture_ratio = self.sensitive_subspace.coverage_loss(
            [delta_pair.detach(), delta_domain.detach()]
        )
        loss_orth = self.sensitive_subspace.orthogonality_loss()

        alpha, ramp = self._alpha_and_ramp(step)
        self.piccl_alpha_state.fill_(alpha)
        mediator = self.mediator_projection(
            adapted,
            self.sensitive_subspace,
            alpha,
            receive_basis_grad=self.basis_receive_task_grad,
        )
        mediator_intervened = self.mediator_projection(
            adapted_intervened,
            self.sensitive_subspace,
            alpha,
            receive_basis_grad=self.basis_receive_task_grad,
        )

        logits = self.classifier(mediator)
        loss_cls = F.cross_entropy(logits, all_y)

        projected = F.normalize(self.proj_head(mediator), dim=1)
        projected_intervened = F.normalize(
            self.proj_head(mediator_intervened), dim=1
        )
        temperature = float(self.hparams["t"])
        loss_int = self._paired_nt_xent(
            projected, projected_intervened, temperature
        )
        loss_cross = self._cross_domain_supervised_contrast(
            projected, all_y, all_domains, temperature
        )

        mediator_reference_space = F.normalize(
            self.pre_proj_head(mediator), dim=1
        )
        frozen_reference_space = F.normalize(
            self.pre_proj_head(reference.detach()), dim=1
        )
        reference_views = torch.stack(
            [mediator_reference_space, frozen_reference_space], dim=1
        )
        loss_ref = self.supcon_loss_pre(reference_views, all_y)

        current_int_weight = float(self._hp("piccl_int_weight", 0.1)) * ramp
        loss_ccc = (
            loss_cross
            + current_int_weight * loss_int
            + float(self._hp("piccl_ref_weight", 1.0)) * loss_ref
        )
        loss_inv = (
            1.0 - F.cosine_similarity(mediator, mediator_intervened, dim=1)
        ).mean()

        total_loss = (
            loss_cls
            + float(self._hp("piccl_ccc_weight", 1.0)) * loss_ccc
            + float(self._hp("piccl_isr_weight", 0.1)) * loss_isr
            + float(self._hp("piccl_orth_weight", 1e-3)) * loss_orth
            + float(self._hp("piccl_inv_weight", 0.05)) * loss_inv
        )

        loss_gt = total_loss.new_zeros(())
        if self.piccl_gt_mode == "keep":
            loss_gt = self._gt_loss(inter_feats, pre_feats)
            total_loss = total_loss + float(self.l_d) * loss_gt

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pair_norm = delta_pair.norm(dim=1).mean()
            if delta_domain.numel() > 0:
                domain_norm = delta_domain.norm(dim=1).mean()
            else:
                domain_norm = total_loss.new_zeros(())
            raw_cosine = F.cosine_similarity(
                adapted, adapted_intervened, dim=1
            ).mean()
            mediator_cosine = F.cosine_similarity(
                mediator, mediator_intervened, dim=1
            ).mean()

        result = {
            "loss": float(total_loss.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "loss_ccc": float(loss_ccc.detach().item()),
            "loss_cross": float(loss_cross.detach().item()),
            "loss_int": float(loss_int.detach().item()),
            "loss_ref": float(loss_ref.detach().item()),
            "loss_isr": float(loss_isr.detach().item()),
            "loss_orth": float(loss_orth.detach().item()),
            "loss_inv": float(loss_inv.detach().item()),
            "loss_gt": float(loss_gt.detach().item()),
            "piccl_alpha": float(alpha),
            "paired_response_norm": float(pair_norm.item()),
            "domain_response_norm": float(domain_norm.item()),
            "valid_domain_response_count": float(valid_domain_mask.sum().item()),
            "basis_orth_error": float(loss_orth.detach().item()),
            "response_capture_ratio": float(capture_ratio.item()),
            "raw_intervention_cosine": float(raw_cosine.item()),
            "mediator_intervention_cosine": float(mediator_cosine.item()),
            # Compatibility keys for the repository's optional legacy log path.
            "ce_loss": float(loss_cls.detach().item()),
            "sup_cl_loss": float(loss_cross.detach().item()),
            "pre_cl_loss": float(loss_ref.detach().item()),
        }
        return result

    def _inference_mediator(self, features: torch.Tensor) -> torch.Tensor:
        return self.mediator_projection(
            features,
            self.sensitive_subspace,
            float(self.piccl_alpha_state.item()),
            receive_basis_grad=False,
        )

    def predict(self, x):
        features = self.featurizer(x)
        return self.classifier(self._inference_mediator(features))

    def predict_embed(self, x):
        features = self.featurizer(x)
        return self._inference_mediator(features)

    def get_forward_model(self):
        return PICCLForwardModel(
            self.featurizer,
            self.sensitive_subspace,
            self.mediator_projection,
            self.classifier,
            float(self.piccl_alpha_state.item()),
        )
