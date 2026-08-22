"""CIPT + single-view direct causal supervised contrastive ablation.

Execution modes:
1) cipt_pure=True: standard single-view CIPT with no contrastive objective.
2) cipt_pure=False: standard CIPT on the original image plus supervised
   contrastive learning directly on the original causal representations.

This branch deliberately uses no augmented contrastive view and no projection
head. For each anchor, positives are only other samples in the same minibatch
with the same class label. Anchors with no same-class peer are excluded from the
contrastive average; if a whole minibatch has no valid positive pair, the
contrastive loss is exactly zero and the update reduces to the CIPT objective.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)
from domainbed.optimizers import get_optimizer


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT with direct single-view causal-space supervised contrastive loss."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_single_view_contrastive = bool(
            hparams.get("cipt_single_view_contrastive", True)
        )
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        if not self.cipt_pure and not self.cipt_single_view_contrastive:
            raise ValueError(
                "This branch is the single-view contrastive ablation. Set "
                "cipt_single_view_contrastive=true or use cipt_pure=true."
            )

        # Direct causal contrastive learning should not be hidden behind an MLP
        # projection head. Remove both inherited DCCL projection modules and
        # rebuild the optimizer without their parameters.
        for module_name in ("proj_head", "pre_proj_head"):
            if hasattr(self, module_name):
                delattr(self, module_name)

        # Isolate one DCCL-side mechanism: same-class supervised contrastive
        # clustering directly in the causal representation space.
        self.l_layer = 0.0
        self.l_d = 0.0
        if hasattr(self, "reg_log_variance"):
            self.reg_log_variance.requires_grad_(False)

        self.contrastive_weight = float(
            hparams.get(
                "cipt_causal_contrastive_weight",
                hparams.get("cipt_contrastive_weight", 0.1),
            )
        )
        self.contrastive_warmup_steps = max(
            0, int(hparams.get("cipt_contrastive_warmup_steps", 500))
        )
        self.single_view_temperature = float(
            getattr(self.supcon_loss, "temperature", hparams["t"])
        )
        self.register_buffer(
            "_causal_contrastive_step",
            torch.zeros((), dtype=torch.long),
        )

        if self.cipt_pure:
            self.contrastive_weight = 0.0

        trainable = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            trainable,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable
        )
        self.frozen_parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )

        print(
            "CIPTDCCL single-view-causal-contrastive: pure_cipt={}, "
            "single_view={}, template_mode={}, K={}, tda_heads={}, lr={}, "
            "contrastive_weight={}, contrastive_warmup_steps={}, t={}, "
            "augmentation_view=False, projection_head=False, pre_cl=False, "
            "reg=False".format(
                self.cipt_pure,
                self.cipt_single_view_contrastive,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.contrastive_warmup_steps,
                self.single_view_temperature,
            )
        )
        print(
            "CIPTDCCL single-view-causal-contrastive parameters: "
            "trainable={}, frozen={}".format(
                self.trainable_parameter_count,
                self.frozen_parameter_count,
            )
        )

    def _intervention_features(self, labels=None):
        if self.cipt_template_mode == "b5b":
            return self.text_features.intervention_features(labels=labels)
        return self.text_features.irrelevant_text_features

    def _contrastive_scale(self):
        """Linearly warm the direct causal contrastive coefficient."""
        if self.cipt_pure or self.contrastive_weight <= 0.0:
            return 0.0
        if self.contrastive_warmup_steps <= 0:
            return self.contrastive_weight

        step = int(self._causal_contrastive_step.item())
        ramp = min(1.0, step / float(self.contrastive_warmup_steps))
        return self.contrastive_weight * ramp

    def _single_view_supcon(self, causal, labels):
        """Supervised contrastive loss with one feature per sample.

        Positives for anchor i are only j != i with y_j == y_i. Singleton
        anchors are not included in the contrastive average, but remain in the
        denominator as negatives for valid anchors. If no anchor has a positive
        peer, return a differentiable zero loss.
        """
        features = F.normalize(causal, dim=-1)
        labels = labels.view(-1)
        batch_size = features.shape[0]

        if batch_size <= 1:
            zero = features.sum() * 0.0
            return zero, 0, 0

        logits = torch.matmul(features, features.T) / self.single_view_temperature
        self_mask = torch.eye(
            batch_size, dtype=torch.bool, device=features.device
        )
        contrast_mask = ~self_mask
        positive_mask = labels[:, None].eq(labels[None, :]) & contrast_mask
        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0
        valid_anchor_count = int(valid_anchor.sum().item())
        positive_link_count = int(positive_mask.sum().item())

        if valid_anchor_count == 0:
            zero = features.sum() * 0.0
            return zero, 0, 0

        # Stable log-softmax over every non-self sample. Singleton anchors are
        # excluded only as anchors; they still contribute as negatives.
        masked_logits = logits.masked_fill(~contrast_mask, float("-inf"))
        logits_max = masked_logits.max(dim=1, keepdim=True).values
        stable_logits = logits - logits_max.detach()
        exp_logits = torch.exp(stable_logits) * contrast_mask.float()
        log_prob = stable_logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        mean_log_prob_pos = (
            (positive_mask.float() * log_prob).sum(dim=1)[valid_anchor]
            / positive_count[valid_anchor].float()
        )
        loss = -mean_log_prob_pos.mean()
        return loss, valid_anchor_count, positive_link_count

    def _cipt_terms(self, x, y):
        """Compute the unchanged CIPT objective terms on one original view."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        visual = self._visual(all_x)
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()

        causal_logits = self._logits(
            causal[:, None, :], class_features
        )[:, 0]
        spurious_logits = self._logits(
            spurious[:, None, :], class_features
        )[:, 0]
        loss_de = cipt_decomposition_loss(
            causal_logits, spurious_logits, labels
        )
        loss_ind = cipt_independence_loss(causal, spurious)

        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )
        return (
            labels,
            visual,
            causal,
            spurious,
            class_features,
            interventions,
            logits,
            loss_cls,
            loss_de,
            loss_ind,
            cipt_base_loss,
        )

    def _update_pure(self, x, y):
        """Single-original-image CIPT path with no contrastive objective."""
        (
            labels,
            visual,
            causal,
            spurious,
            class_features,
            interventions,
            logits,
            loss_cls,
            loss_de,
            loss_ind,
            cipt_base_loss,
        ) = self._cipt_terms(x, y)
        total = cipt_base_loss

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        zero = causal.new_zeros(())
        return {
            "total_loss": total.item(),
            "cipt_base_loss": cipt_base_loss.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de.item(),
            "cipt_de_aug_loss": zero.item(),
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": zero.item(),
            "contrastive_weight_eff": 0.0,
            "contrastive_valid_anchors": 0.0,
            "contrastive_valid_anchor_ratio": 0.0,
            "contrastive_positive_links": 0.0,
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def _update_single_view(self, x, y):
        """CIPT + same-class contrastive clustering on original causal features."""
        (
            labels,
            visual,
            causal,
            spurious,
            class_features,
            interventions,
            logits,
            loss_cls,
            loss_de,
            loss_ind,
            cipt_base_loss,
        ) = self._cipt_terms(x, y)

        loss_contrastive, valid_anchor_count, positive_link_count = (
            self._single_view_supcon(causal, labels)
        )

        self._causal_contrastive_step.add_(1)
        contrastive_weight_eff = self._contrastive_scale()
        total = (
            cipt_base_loss
            + contrastive_weight_eff * loss_contrastive
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL single-view causal shapes: mode={} v={} e={} s={} "
                "z_k={} text_features={} logits={} valid_anchors={}/{}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
                    tuple(interventions.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                    valid_anchor_count,
                    labels.numel(),
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        zero = causal.new_zeros(())
        batch_size = max(1, int(labels.numel()))
        return {
            "total_loss": total.item(),
            "cipt_base_loss": cipt_base_loss.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de.item(),
            "cipt_de_aug_loss": zero.item(),
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "contrastive_weight_eff": float(contrastive_weight_eff),
            "contrastive_valid_anchors": float(valid_anchor_count),
            "contrastive_valid_anchor_ratio": valid_anchor_count / batch_size,
            "contrastive_positive_links": float(positive_link_count),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def update(self, x, y, **kwargs):
        if self.cipt_pure:
            return self._update_pure(x, y)
        return self._update_single_view(x, y)

    def predict(self, x):
        if self.cipt_template_mode != "b5b":
            return super().predict(x)

        # At inference labels are unknown. Score every candidate class using its
        # own class-conditioned B5b intervention contexts and average over K.
        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        diverse_features = self.text_features.intervention_features(
            labels=None
        )

        num_classes, num_templates, dim = diverse_features.shape
        batch = causal.shape[0]
        causal_flat = (
            causal[:, None, :]
            .expand(batch, num_classes, dim)
            .reshape(batch * num_classes, dim)
        )
        context_flat = (
            diverse_features[None, :, :, :]
            .expand(batch, num_classes, num_templates, dim)
            .reshape(batch * num_classes, num_templates, dim)
        )
        z = self.tda(causal_flat, context_flat).reshape(
            batch, num_classes, num_templates, dim
        )
        z = F.normalize(z, dim=-1)
        text = F.normalize(class_features, dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        return scale * torch.einsum(
            "bckd,cd->bck", z, text
        ).mean(dim=-1)
