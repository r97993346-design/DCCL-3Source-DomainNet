"""CIPT + direct causal contrastive ablation.

Execution modes:
1) cipt_pure=True: standard single-view CIPT.
2) cipt_pure=False: standard CIPT on the original image plus one augmented
   image. The augmented image is causally decomposed, receives a lightweight
   decomposition loss, and its causal feature is used as the positive
   contrastive view.

This branch deliberately removes projection heads from the contrastive path.
The direct causal representations e(x) and e(T(x)) are normalized and sent to
SupCon. The augmented view does not participate in TDA classification,
independence, causal-consistency, pre-CL, or representation-regularization
losses; it contributes only augmented-view causal decomposition and causal
contrastive supervision.
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
    """CIPT with direct causal-space supervised contrastive regularization."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        # Direct causal contrastive learning should not be hidden behind an MLP
        # projection head. Remove both DCCL projection modules inherited from
        # the compatibility base class and rebuild the optimizer so their
        # parameters are not optimized at all.
        for module_name in ("proj_head", "pre_proj_head"):
            if hasattr(self, module_name):
                delattr(self, module_name)

        # Keep only the direct causal-space contrastive mechanism from DCCL.
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

        # Relative strength of the augmented-view decomposition objective.
        # The effective coefficient is beta * cipt_aug_decomp_weight, so the
        # default 0.5 gives the augmented view half the decomposition strength
        # of the original view while keeping it an auxiliary branch.
        self.aug_decomp_weight = float(
            hparams.get("cipt_aug_decomp_weight", 0.5)
        )
        if self.aug_decomp_weight < 0.0:
            raise ValueError("cipt_aug_decomp_weight must be non-negative")

        self.register_buffer(
            "_causal_contrastive_step",
            torch.zeros((), dtype=torch.long),
        )

        if self.cipt_pure:
            self.contrastive_weight = 0.0
            self.aug_decomp_weight = 0.0

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
            "CIPTDCCL direct-causal-contrastive: pure_cipt={}, "
            "template_mode={}, K={}, tda_heads={}, lr={}, "
            "contrastive_weight={}, contrastive_warmup_steps={}, "
            "aug_decomp_weight={}, aug_decomp_beta_eff={}, "
            "projection_head=False, pre_cl=False, reg=False".format(
                self.cipt_pure,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.contrastive_warmup_steps,
                self.aug_decomp_weight,
                self.beta * self.aug_decomp_weight,
            )
        )
        print(
            "CIPTDCCL direct-causal-contrastive parameters: "
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

    def _update_pure(self, x, y):
        """Single-original-image CIPT path with no contrastive objective."""
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
            "aug_decomp_weight_eff": 0.0,
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": zero.item(),
            "contrastive_weight_eff": 0.0,
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def _update_fusion(self, x, y, x_2):
        """CIPT main view + augmented decomposition + direct causal SupCon."""
        all_x = torch.cat(x)
        all_x_aug = torch.cat(x_2)
        labels = torch.cat(y)

        # Frozen CLIP encodes both views. The same causal-decomposition module
        # is shared by original and augmented images.
        visual = self._visual(all_x)
        visual_aug = self._visual(all_x_aug)
        causal, spurious = self.causal_decomposition(visual)
        causal_aug, spurious_aug = self.causal_decomposition(visual_aug)

        class_features = self.text_features.class_features()

        # Original-view CIPT decomposition objective.
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

        # Augmented-view causal decomposition supervision. This uses the same
        # CIPT decomposition objective: e_aug must remain class-discriminative,
        # while s_aug is encouraged to be class-uninformative. The augmented
        # branch deliberately does not receive TDA classification or an extra
        # independence loss.
        causal_aug_logits = self._logits(
            causal_aug[:, None, :], class_features
        )[:, 0]
        spurious_aug_logits = self._logits(
            spurious_aug[:, None, :], class_features
        )[:, 0]
        loss_de_aug = cipt_decomposition_loss(
            causal_aug_logits, spurious_aug_logits, labels
        )

        # Only the original causal representation enters the CIPT TDA task
        # branch, so the augmented image remains an auxiliary training view.
        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        # No projection head: contrast directly in causal representation space.
        contrast_features = torch.stack(
            (
                F.normalize(causal, dim=-1),
                F.normalize(causal_aug, dim=-1),
            ),
            dim=1,
        )
        loss_contrastive = self.supcon_loss(
            contrast_features, labels
        )

        # Preserve the original CIPT objective, then add two auxiliary terms:
        # 1) lightweight augmented-view decomposition;
        # 2) direct causal-space supervised contrastive learning.
        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )
        aug_decomp_weight_eff = self.beta * self.aug_decomp_weight

        self._causal_contrastive_step.add_(1)
        contrastive_weight_eff = self._contrastive_scale()
        total = (
            cipt_base_loss
            + aug_decomp_weight_eff * loss_de_aug
            + contrastive_weight_eff * loss_contrastive
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL direct causal shapes: mode={} v={} v_aug={} "
                "e={} e_aug={} s={} s_aug={} z_k={} text_features={} logits={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(visual_aug.shape),
                    tuple(causal.shape),
                    tuple(causal_aug.shape),
                    tuple(spurious.shape),
                    tuple(spurious_aug.shape),
                    tuple(interventions.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                )
            )

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
            "cipt_de_aug_loss": loss_de_aug.item(),
            "aug_decomp_weight_eff": float(aug_decomp_weight_eff),
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "contrastive_weight_eff": float(contrastive_weight_eff),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_e_aug_norm": causal_aug.norm(dim=-1).mean().item(),
            "mean_e_aug_cosine": F.cosine_similarity(
                causal, causal_aug, dim=-1
            ).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_s_aug_norm": spurious_aug.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
            "mean_e_aug_s_aug_cosine": F.cosine_similarity(
                causal_aug, spurious_aug, dim=-1
            ).mean().item(),
        }

    def update(self, x, y, **kwargs):
        if self.cipt_pure:
            return self._update_pure(x, y)

        if "x_2" not in kwargs:
            raise KeyError(
                "CIPTDCCL fusion mode requires x_2: one original image "
                "and one augmented view per sample."
            )
        return self._update_fusion(x, y, kwargs["x_2"])

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
