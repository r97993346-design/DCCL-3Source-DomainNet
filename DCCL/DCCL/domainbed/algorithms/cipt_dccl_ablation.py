"""CIPT ablation with optional single-view class-level supervised contrastive learning.

This branch intentionally removes the full DCCL fusion path. Training uses only
one original image view with official CLIP preprocessing. The only auxiliary
objective retained on top of pure CIPT is supervised contrastive learning over
causal features from different samples that share the same class label.

Modes:
1) cipt_pure=True: pure CIPT, no contrastive objective.
2) cipt_pure=False and cipt_class_supcon=True: pure CIPT + single-view class SupCon.

Removed from this branch:
- augmented-image positive pairs / x_2
- causal consistency between original and augmented views
- augmented-view decomposition loss
- pretrained-feature contrastive anchoring (pre-CL)
- representation/Gaussian anchoring regularizer (reg)
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)


class CIPTDCCL(_BaseCIPTDCCL):
    """Pure CIPT with an optional same-class single-view SupCon objective."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_class_supcon = bool(
            hparams.get("cipt_class_supcon", True)
        ) and not self.cipt_pure
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        # This ablation deliberately removes both DCCL anchoring objectives.
        self.l_layer = 0.0
        self.l_d = 0.0
        for parameter in self.pre_proj_head.parameters():
            parameter.requires_grad_(False)
        self.reg_log_variance.requires_grad_(False)

        # In pure-CIPT mode the projection head is unused as well.
        if not self.cipt_class_supcon:
            for parameter in self.proj_head.parameters():
                parameter.requires_grad_(False)

        # The base class builds an optimizer before the ablation flags are known.
        # Rebuild it so removed/frozen DCCL-side parameters are not optimized.
        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        self.optimizer = self.new_optimizer(trainable)
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable
        )
        self.frozen_parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )

        print(
            "CIPT class-SupCon ablation: pure_cipt={}, class_supcon={}, "
            "template_mode={}, K={}, tda_heads={}, lr={}, "
            "contrastive_weight={}, temperature={}, l_layer=0, l_d=0".format(
                self.cipt_pure,
                self.cipt_class_supcon,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight if self.cipt_class_supcon else 0.0,
                hparams["t"],
            )
        )
        print(
            "CIPT class-SupCon parameters: trainable={}, frozen={}".format(
                self.trainable_parameter_count,
                self.frozen_parameter_count,
            )
        )

    def _intervention_features(self, labels=None):
        if self.cipt_template_mode == "b5b":
            return self.text_features.intervention_features(labels=labels)
        return self.text_features.irrelevant_text_features

    def _single_view_class_supcon(self, projected, labels):
        """Supervised contrastive loss for one view per real training sample.

        Same-label samples are positives; every non-self sample remains in the
        contrast set and therefore can act as a negative. Anchors whose class
        occurs only once in the current concatenated source-domain batch are
        skipped instead of dividing by zero. This preserves those singleton
        samples as negatives for the valid anchors.
        """
        features = F.normalize(projected, dim=-1)
        batch_size = features.shape[0]

        if batch_size <= 1:
            return projected.sum() * 0.0, 0, 0

        temperature = float(self.supcon_loss.temperature)
        base_temperature = float(self.supcon_loss.base_temperature)

        logits = torch.matmul(features, features.T) / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(
            batch_size, dtype=torch.bool, device=features.device
        )
        same_class = labels.view(-1, 1).eq(labels.view(1, -1))
        positive_mask = same_class & ~self_mask
        contrast_mask = ~self_mask

        exp_logits = torch.exp(logits) * contrast_mask.to(logits.dtype)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0
        valid_count = int(valid_anchor.sum().item())
        positive_pair_count = int(positive_mask.sum().item())

        if valid_count == 0:
            return projected.sum() * 0.0, 0, positive_pair_count

        mean_log_prob_pos = (
            (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1)[valid_anchor]
            / positive_count[valid_anchor].to(log_prob.dtype)
        )
        loss = -(
            temperature / base_temperature
        ) * mean_log_prob_pos.mean()
        return loss, valid_count, positive_pair_count

    def _update_single_view(self, x, y):
        """Run pure CIPT, optionally adding only class-level causal SupCon."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        # One original/basic CLIP-preprocessed image per sample. No x_2 is used.
        visual = self._visual(all_x)
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()

        # Original CIPT decomposition objective only.
        causal_logits = self._logits(causal[:, None, :], class_features)[:, 0]
        spurious_logits = self._logits(spurious[:, None, :], class_features)[:, 0]
        loss_de = cipt_decomposition_loss(
            causal_logits, spurious_logits, labels
        )
        loss_ind = cipt_independence_loss(causal, spurious)

        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        zero = causal.new_zeros(())
        loss_contrastive = zero
        valid_anchor_count = 0
        positive_pair_count = 0

        if self.cipt_class_supcon:
            projected = self.proj_head(causal)
            (
                loss_contrastive,
                valid_anchor_count,
                positive_pair_count,
            ) = self._single_view_class_supcon(projected, labels)

        total = (
            loss_cls
            + self.beta * loss_de
            + self.gamma * loss_ind
            + (
                self.contrastive_weight * loss_contrastive
                if self.cipt_class_supcon
                else zero
            )
        )

        if self.debug_shapes:
            print(
                "CIPT class-SupCon shapes: mode={} v={} e={} s={} z_k={} "
                "text_features={} logits={} valid_anchors={}/{} positive_pairs={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
                    tuple(interventions.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                    valid_anchor_count,
                    labels.shape[0],
                    positive_pair_count,
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        valid_anchor_ratio = (
            float(valid_anchor_count) / float(labels.shape[0])
            if labels.shape[0] > 0
            else 0.0
        )

        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de.item(),
            "cipt_de_aug_loss": zero.item(),
            "cipt_ind_loss": loss_ind.item(),
            "class_supcon_loss": loss_contrastive.item(),
            # Keep the historical key for comparison with older result files.
            "dccl_contrastive_loss": loss_contrastive.item(),
            "contrastive_valid_anchor_ratio": valid_anchor_ratio,
            "contrastive_positive_pairs": float(positive_pair_count),
            "causal_consistency_loss": zero.item(),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def update(self, x, y, **kwargs):
        # kwargs may contain legacy dataset fields, but this branch intentionally
        # ignores every augmented/pretrained-anchor input and consumes x/y only.
        return self._update_single_view(x, y)

    def predict(self, x):
        if self.cipt_template_mode != "b5b":
            return super().predict(x)

        # At inference labels are unknown. Score every candidate class using its
        # own class-conditioned B5b intervention contexts and average over K.
        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        diverse_features = self.text_features.intervention_features(labels=None)

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
        return scale * torch.einsum("bckd,cd->bck", z, text).mean(dim=-1)
