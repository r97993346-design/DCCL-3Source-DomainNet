"""PICCL: low-disturbance causal feature mediation on top of original DCCL."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import DCCL, rand_bbox
from domainbed.lib import misc
from domainbed.optimizers import get_optimizer

from .piccl_components import (
    CausalMediatorProjection,
    ClassDomainResidualBank,
    InterventionSensitiveSubspace,
    PairedInterventionResponseEstimator,
    causal_pair_reliability,
    parse_bool,
    reliable_positive_weights,
)
from .reliable_supcon import ReliableSupConLoss


class PICCLForwardModel(nn.Module):
    """Inference/SWAD model with the latest Q and beta stored as buffers."""

    def __init__(
        self,
        featurizer,
        basis_q,
        classifier,
        beta,
        use_residual_gate=False,
        gate_logit=None,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.register_buffer("basis_q", basis_q.detach().clone())
        self.register_buffer("piccl_beta", beta.detach().clone())
        self.use_residual_gate = bool(use_residual_gate)
        if self.use_residual_gate:
            if gate_logit is None or gate_logit.numel() != 1:
                raise ValueError("gate_logit must be a scalar when gating is enabled")
            self.gate_logit = nn.Parameter(gate_logit.detach().clone().reshape(()))
        # swa_utils copies only these explicitly opted-in buffers.  DCCL and
        # other algorithms retain their original SWAD buffer behavior.
        self.swa_latest_buffer_names = ("basis_q", "piccl_beta")

    @property
    def network(self):
        return self

    def predict_embed(self, x):
        z = self.featurizer(x)
        beta = self.piccl_beta.to(z.device, z.dtype)
        if float(beta.detach().item()) == 0.0:
            return z
        q = self.basis_q.to(z.device, z.dtype)
        z_causal = z - beta * ((z @ q) @ q.transpose(0, 1))
        if not self.use_residual_gate:
            return z_causal
        return z + torch.sigmoid(self.gate_logit).to(z.dtype) * (z_causal - z)

    def predict(self, x):
        return self.classifier(self.predict_embed(x))

    def forward(self, x):
        return self.predict(x)


class PICCL(DCCL):
    """Original DCCL objective fed by causally mediated pooled features."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Reading the switch does not consume RNG.  Disabled PICCL constructs
        # exactly the original DCCL modules and optimizer, then stops.
        self.use_piccl = parse_bool(hparams.get("use_piccl", False))
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if not self.use_piccl:
            return

        for parameter in self.pre_featurizer.parameters():
            parameter.requires_grad = False
        self.pre_featurizer.eval()

        feature_dim = self.featurizer.n_outputs
        self.pire = PairedInterventionResponseEstimator()
        self.residual_bank = ClassDomainResidualBank(
            num_classes=num_classes,
            num_domains=num_domains,
            feature_dim=feature_dim,
            momentum=hparams.get("piccl_proto_momentum", 0.99),
            min_count=hparams.get("piccl_min_domain_samples", 8),
            min_valid_domains=hparams.get("piccl_min_valid_domains", 2),
        )
        self.sensitive_subspace = InterventionSensitiveSubspace(
            feature_dim,
            rank=hparams.get("piccl_rank", 16),
            eps=hparams.get("piccl_eps", 1e-8),
        )
        self.causal_mediator = CausalMediatorProjection()
        self.use_residual_gate = parse_bool(
            hparams.get("piccl_use_residual_gate", False)
        )
        if self.use_residual_gate:
            self.gate_logit = nn.Parameter(
                torch.tensor(float(hparams.get("piccl_gate_bias", -2.0)))
            )
        self.reliable_supcon_loss = ReliableSupConLoss(hparams["t"])
        self.register_buffer("piccl_beta", torch.tensor(0.0))
        self.optimizer = get_optimizer(
            hparams["optimizer"], self._piccl_optimizer_groups()
        )

    def _dccl_optimizer_groups(self):
        """Byte-for-byte-equivalent parameter groups from DCCL.__init__."""
        return [
            {
                "params": self.featurizer.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams["lr"] / 0.1,
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.proj_head.parameters(),
                "lr": self.hparams["lr"] / 10,
                "weight_decay": self.hparams["weight_decay"],
            },
            {"params": self.mean_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.var_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.pre_proj_head.parameters(), "lr": self.hparams["lr"] / 10},
        ]

    def _piccl_optimizer_groups(self):
        groups = self._dccl_optimizer_groups()
        groups.append(
            {
                "params": self.sensitive_subspace.parameters(),
                "lr": self.hparams["lr"]
                * float(self.hparams.get("piccl_basis_lr_multiplier", 1.0)),
                "weight_decay": self.hparams.get("piccl_basis_weight_decay", 0.0),
            }
        )
        if self.use_residual_gate:
            groups.append(
                {
                    "params": [self.gate_logit],
                    "lr": self.hparams["lr"],
                    "weight_decay": 0.0,
                }
            )
        return groups

    def train(self, mode=True):
        super().train(mode)
        if self.use_piccl:
            self.pre_featurizer.eval()
        return self

    @staticmethod
    def _domain_ids(x):
        return torch.cat(
            [
                torch.full(
                    (minibatch.shape[0],),
                    domain_id,
                    device=minibatch.device,
                    dtype=torch.long,
                )
                for domain_id, minibatch in enumerate(x)
            ]
        )

    def _resolved_schedule_steps(self, step_key, ratio_key, default_ratio):
        explicit = int(self.hparams.get(step_key, 0))
        if explicit > 0:
            return explicit
        total = max(int(self.hparams.get("piccl_total_steps", 1)), 1)
        return int(total * float(self.hparams.get(ratio_key, default_ratio)))

    def _beta(self, step):
        warmup = self._resolved_schedule_steps(
            "piccl_warmup_steps", "piccl_warmup_ratio", 0.10
        )
        ramp = self._resolved_schedule_steps(
            "piccl_ramp_steps", "piccl_ramp_ratio", 0.20
        )
        if step < warmup:
            progress = 0.0
        elif ramp <= 0:
            progress = 1.0
        else:
            progress = min(max((float(step) - warmup) / float(ramp), 0.0), 1.0)
        self.piccl_beta.fill_(float(self.hparams.get("piccl_beta_max", 0.10)) * progress)
        return self.piccl_beta

    def _reliable_contrast_gamma(self, step):
        if not parse_bool(self.hparams.get("piccl_use_reliable_contrast", False)):
            return 0.0
        warmup = self._resolved_schedule_steps(
            "piccl_reliable_contrast_warmup_steps",
            "piccl_reliable_contrast_warmup_ratio",
            0.30,
        )
        ramp = self._resolved_schedule_steps(
            "piccl_reliable_contrast_ramp_steps",
            "piccl_reliable_contrast_ramp_ratio",
            0.10,
        )
        if step < warmup:
            progress = 0.0
        elif ramp <= 0:
            progress = 1.0
        else:
            progress = min(max((float(step) - warmup) / float(ramp), 0.0), 1.0)
        return float(
            self.hparams.get("piccl_reliable_contrast_gamma_max", 0.30)
        ) * progress

    def _project(self, z, beta):
        z_causal = self.causal_mediator(z, self.sensitive_subspace, beta)
        if z_causal is z or not self.use_residual_gate:
            return z_causal
        return z + torch.sigmoid(self.gate_logit).to(z.dtype) * (z_causal - z)

    def _isr_loss(self, delta_aug, delta_dom):
        min_norm = float(self.hparams.get("piccl_min_delta_norm", 1e-4))
        loss_aug = self.sensitive_subspace.coverage_loss(delta_aug, min_norm=min_norm)
        if delta_dom.numel():
            loss_dom = self.sensitive_subspace.coverage_loss(
                delta_dom, min_norm=min_norm
            )
            loss_isr = (
                float(self.hparams.get("piccl_isr_aug_weight", 0.25)) * loss_aug
                + float(self.hparams.get("piccl_isr_dom_weight", 0.75)) * loss_dom
            )
        else:
            loss_dom = loss_aug * 0.0
            loss_isr = loss_aug
        return loss_isr, loss_aug, loss_dom

    def _pair_reliability(self, z, labels, domains):
        return causal_pair_reliability(
            z,
            labels,
            domains,
            self.sensitive_subspace,
            min_delta_norm=self.hparams.get(
                "piccl_reliable_contrast_min_delta_norm", 1e-6
            ),
        )

    def _reliable_positive_pair_weights(
        self, pair_reliability, labels, domains, gamma
    ):
        return reliable_positive_weights(
            pair_reliability,
            labels,
            domains,
            gamma=gamma,
            min_weight=self.hparams.get(
                "piccl_reliable_contrast_min_weight", 0.5
            ),
            num_views=2,
        )

    def update(self, x, y, **kwargs):
        if not self.use_piccl:
            return super().update(x, y, **kwargs)

        # The remainder follows the original DCCL.update statement order.  The
        # only task-path change is z -> m immediately before classifier/proj heads.
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        x_2 = kwargs["x_2"]
        all_x_2 = torch.cat(x_2)
        step = int(kwargs.get("step", 0))
        beta = self._beta(step).to(all_x.device)

        if self.TN:
            all_x_2, sp_loss = self.TN_network(all_x_2)
            feature_x = self.featurizer(all_x)
            feature_x_2 = self.featurizer(all_x_2)
            causal_x = self._project(feature_x, beta)
            causal_x_2 = self._project(feature_x_2, beta)
            embed_2 = self.proj_head(causal_x_2)
            embed_1 = self.proj_head(causal_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            loss_sup_cl = self.supcon_loss(features, all_y)
            loss = -loss_sup_cl - self.lamda * sp_loss
            self.optimizer_TN.zero_grad()
            loss.backward()
            self.optimizer_TN.step()
            with torch.no_grad():
                all_x_2, sp_loss = self.TN_network(all_x_2)

        r = np.random.rand(1)
        if self.aug and r < self.aug:
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(all_x.size()[0]).to(all_x.device)
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (all_x.size()[-1] * all_x.size()[-2])
            )
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            causal_x = self._project(feature_x, beta)
            pred_x = self.classifier(causal_x)
            loss = F.cross_entropy(pred_x, target_a) * lam + F.cross_entropy(
                pred_x, target_b
            ) * (1 - lam)
        else:
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            causal_x = self._project(feature_x, beta)
            pred_x = self.classifier(causal_x)
            loss = F.cross_entropy(pred_x, all_y)
        ce_loss = loss.item()

        feature_x_2, inter_feats_2 = self.featurizer(all_x_2, ret_feats=True)
        causal_x_2 = self._project(feature_x_2, beta)
        if self.two_ce:
            loss = loss / 2 + F.cross_entropy(
                self.classifier(causal_x_2), all_y
            ) / 2

        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
            pre_pred_x_2 = self.pre_featurizer(all_x_2)

        domains = self._domain_ids(x)
        delta_aug = self.pire(
            feature_x, feature_x_2, pre_pred_x, pre_pred_x_2
        ).detach()
        self.residual_bank.update(
            (feature_x - pre_pred_x).detach(),
            all_y,
            domains,
            min_norm=float(self.hparams.get("piccl_min_delta_norm", 1e-4)),
        )
        delta_dom = self.residual_bank.domain_responses().to(
            feature_x.device, feature_x.dtype
        )
        loss_isr, loss_isr_aug, loss_isr_dom = self._isr_loss(
            delta_aug, delta_dom
        )
        loss_orth = self.sensitive_subspace.orthogonality_loss()

        if self.l_d:
            reg_loss = 0.0
            for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(
                inter_feats,
                pre_feats,
                self.mean_encoders,
                self.var_encoders,
            ):
                mean = mean_enc(inter_f)
                var = var_enc(inter_f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.0
            loss += self.l_d * reg_loss
        else:
            reg_loss = feature_x.new_zeros(())

        reliability_metrics = {
            "reliable_contrast_enabled": 0.0,
            "reliable_contrast_gamma": 0.0,
            "cross_domain_positive_pair_count": 0.0,
        }
        if self.l:
            embed_2 = self.proj_head(causal_x_2)
            embed_1 = self.proj_head(causal_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                if self.pos_mask:
                    pos_mask = 1 - neg_mask
                else:
                    pos_mask = None
                loss_sup_cl = self.supcon_loss(
                    features, all_y, neg_mask=neg_mask, pos_mask=pos_mask
                )
            else:
                if self.sample_d:
                    all_x_2_d = torch.cat(kwargs["x_2_d"])
                    feature_x_2_d = self.featurizer(all_x_2_d)
                    causal_x_2_d = self._project(feature_x_2_d, beta)
                    embed_2_d = self.proj_head(causal_x_2_d)
                    view_2_d = nn.functional.normalize(embed_2_d)
                    add_pos = torch.cat([view_2_d, view_2_d], 0)
                    loss_sup_cl = self.supcon_loss(
                        features, all_y, add_pos=add_pos
                    )
                elif parse_bool(
                    self.hparams.get("piccl_use_reliable_contrast", False)
                ):
                    pair_reliability, _, raw_reliability, pair_delta_norm = (
                        self._pair_reliability(feature_x, all_y, domains)
                    )
                    gamma = self._reliable_contrast_gamma(step)
                    weights, cross_mask, self_aug_mask, _ = (
                        self._reliable_positive_pair_weights(
                            pair_reliability, all_y, domains, gamma
                        )
                    )
                    loss_sup_cl = self.reliable_supcon_loss(
                        features, all_y, positive_weights=weights
                    )
                    cross_weights = weights[cross_mask]
                    reliability_metrics = {
                        "reliable_contrast_enabled": 1.0,
                        "reliable_contrast_gamma": float(gamma),
                        "cross_domain_positive_pair_count": float(
                            cross_mask.sum().item()
                        ),
                        "pair_reliability_raw_mean": float(
                            raw_reliability.mean().item()
                        )
                        if raw_reliability.numel()
                        else 1.0,
                        "pair_weight_effective_mean": float(
                            cross_weights.mean().item()
                        )
                        if cross_weights.numel()
                        else 1.0,
                        "mean_pair_delta_norm": float(pair_delta_norm.mean().item())
                        if pair_delta_norm.numel()
                        else 0.0,
                        "self_aug_weights_are_one": float(
                            torch.all(weights[self_aug_mask] == 1).item()
                        ),
                    }
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
            loss += self.l * loss_sup_cl

        pre_cl_loss = 0.0
        if self.l_layer:
            embed_1 = self.pre_proj_head(causal_x)
            embed_2 = self.pre_proj_head(pre_pred_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            all_y_pre = all_y

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                if self.pos_mask:
                    pos_mask = 1 - neg_mask
                else:
                    pos_mask = None
                pre_cl_loss += self.supcon_loss_pre(
                    features,
                    all_y_pre,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                pre_cl_loss += self.supcon_loss_pre(features, all_y_pre)
            loss += self.l_layer * pre_cl_loss

        dccl_total_loss = loss
        weighted_isr = float(
            self.hparams.get("piccl_isr_weight", 0.05)
        ) * loss_isr
        weighted_orth = float(
            self.hparams.get("piccl_orth_weight", 1e-4)
        ) * loss_orth
        loss = loss + weighted_isr + weighted_orth

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            basis_stats = self.sensitive_subspace.diagnostics()
            gated_projection_delta = (causal_x - feature_x).norm(dim=1).mean()
            direct_causal_x = self.causal_mediator(
                feature_x, self.sensitive_subspace, beta
            )
            direct_projection_delta = (
                direct_causal_x - feature_x
            ).norm(dim=1).mean()
            original_norm = feature_x.norm(dim=1).mean().clamp_min(1e-12)
            if delta_aug.numel():
                projected = self.sensitive_subspace.project(
                    delta_aug, detach_basis=True
                )
                projection_ratio = projected.pow(2).sum(dim=1).div(
                    delta_aug.pow(2).sum(dim=1).clamp_min(1e-8)
                ).mean()
            else:
                projection_ratio = feature_x.new_zeros(())

        loss_dict = {
            "loss": float(loss.item()),
            "ce_loss": float(ce_loss),
            "loss_cls": float(ce_loss),
            "dccl_total_loss": float(dccl_total_loss.item()),
            "reg_loss": float(reg_loss.item())
            if torch.is_tensor(reg_loss)
            else float(reg_loss),
            "loss_isr": float(loss_isr.item()),
            "loss_isr_aug": float(loss_isr_aug.item()),
            "loss_isr_dom": float(loss_isr_dom.item()),
            "loss_orth": float(loss_orth.item()),
            "weighted_loss_isr": float(weighted_isr.item()),
            "weighted_loss_orth": float(weighted_orth.item()),
            "piccl_beta": float(beta.item()),
            "feature_delta_norm": float(gated_projection_delta.item()),
            "feature_delta_ratio": float(
                (gated_projection_delta / original_norm).item()
            ),
            "piccl_gate_value": float(
                torch.sigmoid(self.gate_logit).item()
            )
            if self.use_residual_gate
            else 1.0,
            "direct_projection_delta_norm": float(
                direct_projection_delta.item()
            ),
            "gated_projection_delta_norm": float(
                gated_projection_delta.item()
            ),
            "gated_feature_delta_ratio": float(
                (gated_projection_delta / original_norm).item()
            ),
            "projection_ratio": float(projection_ratio.item()),
            "valid_domain_response_count": float(delta_dom.shape[0]),
            "basis_orthogonality_error": float(
                basis_stats["basis_orthogonality_error"].item()
            ),
            "basis_rank": float(basis_stats["basis_rank"]),
            "basis_norm": float(basis_stats["basis_norm"].item()),
            "has_nan_or_inf": float(not torch.isfinite(loss).item()),
            "piccl_has_nan_or_inf": float(not torch.isfinite(loss).item()),
        }
        if self.l:
            loss_dict["sup_cl_loss"] = float(loss_sup_cl.item())
        if self.l_layer:
            loss_dict["pre_cl_loss"] = float(pre_cl_loss.item())
        loss_dict.update(reliability_metrics)
        return loss_dict

    def predict_embed(self, x):
        if not self.use_piccl:
            return super().predict_embed(x)
        return self._project(self.featurizer(x), self.piccl_beta)

    def predict(self, x):
        if not self.use_piccl:
            return super().predict(x)
        return self.classifier(self.predict_embed(x))

    def get_forward_model(self):
        if not self.use_piccl:
            return super().get_forward_model()
        q = self.sensitive_subspace.orthonormal_basis(
            detach=True, dtype=torch.float32
        )
        return PICCLForwardModel(
            self.featurizer,
            q,
            self.classifier,
            self.piccl_beta,
            use_residual_gate=self.use_residual_gate,
            gate_logit=self.gate_logit if self.use_residual_gate else None,
        )

    def clone(self):
        if not self.use_piccl:
            return super().clone()
        clone = copy.deepcopy(self)
        clone.optimizer = get_optimizer(
            clone.hparams["optimizer"], clone._piccl_optimizer_groups()
        )
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone
