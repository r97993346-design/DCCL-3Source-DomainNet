"""CIPTDCCL ablation wrapper with optional shallow visual prompt tuning.

The branch keeps CLIP weights frozen and does not add MetaPrompt episodic/meta
learning or AC loss. When visual prompt tuning is enabled, the prompted CLIP
visual feature feeds causal decomposition, while the original unprompted frozen
CLIP feature remains the DCCL pretrained semantic anchor.

Execution modes:
1) cipt_pure=True: single original image; no DCCL contrastive, causal
   consistency, pre-CL, or representation regularizer.
2) cipt_pure=False: original + augmented view for SupCon, causal consistency,
   augmented L_de, and DCCL pretrained anchoring.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)
from domainbed.algorithms.cipt_visual_prompt import (
    VisualPromptLearner,
    encode_image_with_visual_prompt,
)
from domainbed.optimizers import get_optimizer


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT+DCCL with selectable text bank and optional visual prompt tuning."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)
        self.causal_consistency_weight = float(
            hparams.get("cipt_causal_consistency_weight", 1.0)
        )

        self.visual_prompt_enabled = bool(
            hparams.get("cipt_visual_prompt_enabled", True)
        )
        self.visual_prompt_length = int(
            hparams.get("cipt_visual_prompt_length", 4)
        )
        self.visual_prompt = None
        if self.visual_prompt_enabled:
            self.visual_prompt = VisualPromptLearner(
                self.clip_model, prompt_length=self.visual_prompt_length
            )

        # Pure CIPT removes DCCL-side objectives. The optional visual prompt is
        # independent: disable cipt_visual_prompt_enabled for the exact old
        # frozen-visual pure-CIPT reproduction.
        if self.cipt_pure:
            self.contrastive_weight = 0.0
            self.l_layer = 0.0
            self.l_d = 0.0

        # _BaseCIPTDCCL built its optimizer before this subclass added the
        # visual prompt. Rebuild it so prompt tokens are included exactly once.
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            trainable,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.trainable_parameter_count = sum(p.numel() for p in trainable)
        self.frozen_parameter_count = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )

        print(
            "CIPTDCCL ablation: pure_cipt={}, template_mode={}, K={}, "
            "tda_heads={}, lr={}, contrastive_weight={}, l_layer={}, l_d={}, "
            "causal_consistency_weight={}, visual_prompt={}, visual_prompt_length={}".format(
                self.cipt_pure,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.l_layer,
                self.l_d,
                0.0 if self.cipt_pure else self.causal_consistency_weight,
                self.visual_prompt_enabled,
                self.visual_prompt_length if self.visual_prompt_enabled else 0,
            )
        )
        print(
            "CIPTDCCL visual-prompt branch parameters: trainable={}, frozen={}".format(
                self.trainable_parameter_count, self.frozen_parameter_count
            )
        )

    def _intervention_features(self, labels=None):
        if self.cipt_template_mode == "b5b":
            return self.text_features.intervention_features(labels=labels)
        return self.text_features.irrelevant_text_features

    def _visual_anchor(self, images):
        """Original pretrained CLIP visual representation, always frozen."""
        with torch.no_grad():
            return self.clip_model.encode_image(images).float()

    def _visual(self, images):
        """Feature used by causal decomposition and prediction."""
        if not self.visual_prompt_enabled:
            return self._visual_anchor(images)
        return encode_image_with_visual_prompt(
            self.clip_model, images, self.visual_prompt
        )

    def _update_pure(self, x, y):
        """Single-original-image CIPT path with no DCCL/augmentation losses."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        visual = self._visual(all_x)
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()

        causal_logits = self._logits(causal[:, None, :], class_features)[:, 0]
        spurious_logits = self._logits(spurious[:, None, :], class_features)[:, 0]
        loss_de = cipt_decomposition_loss(causal_logits, spurious_logits, labels)
        loss_ind = cipt_independence_loss(causal, spurious)

        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        total = loss_cls + self.beta * loss_de + self.gamma * loss_ind

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        zero = causal.new_zeros(())
        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de.item(),
            "cipt_de_aug_loss": zero.item(),
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": zero.item(),
            "dccl_contrastive_loss": zero.item(),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def _update_fusion(self, x, y, x_2):
        """Original + augmented-view fusion with a shared visual prompt."""
        all_x = torch.cat(x)
        all_x_aug = torch.cat(x_2)
        labels = torch.cat(y)

        # Prompted features adapt the frozen visual encoder to the DG task.
        # The same learnable prompt parameters are shared across both views.
        visual = self._visual(all_x)
        visual_aug = self._visual(all_x_aug)

        # Keep the original unprompted CLIP representation as a fixed semantic
        # anchor for DCCL's pre-CL and representation regularizer. Avoid an
        # extra CLIP pass when visual prompting is disabled.
        visual_anchor = (
            self._visual_anchor(all_x)
            if self.visual_prompt_enabled
            else visual.detach()
        )

        causal, spurious = self.causal_decomposition(visual)
        causal_aug, spurious_aug = self.causal_decomposition(visual_aug)
        class_features = self.text_features.class_features()

        causal_logits = self._logits(causal[:, None, :], class_features)[:, 0]
        spurious_logits = self._logits(spurious[:, None, :], class_features)[:, 0]
        loss_de_orig = cipt_decomposition_loss(
            causal_logits, spurious_logits, labels
        )

        causal_logits_aug = self._logits(
            causal_aug[:, None, :], class_features
        )[:, 0]
        spurious_logits_aug = self._logits(
            spurious_aug[:, None, :], class_features
        )[:, 0]
        loss_de_aug = cipt_decomposition_loss(
            causal_logits_aug, spurious_logits_aug, labels
        )
        loss_de = 0.5 * (loss_de_orig + loss_de_aug)

        # Keep CIPT independence on the original decomposition only. The
        # augmented branch roles stay unchanged from clip-preprocess-ablation.
        loss_ind = cipt_independence_loss(causal, spurious)

        loss_causal_consistency = (
            1.0 - F.cosine_similarity(causal, causal_aug, dim=-1)
        ).mean()

        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        projected = self.proj_head(causal)
        projected_aug = self.proj_head(causal_aug)
        contrast_features = torch.stack(
            (
                F.normalize(projected, dim=-1),
                F.normalize(projected_aug, dim=-1),
            ),
            dim=1,
        )
        loss_contrastive = self.supcon_loss(contrast_features, labels)

        # DCCL anchoring intentionally targets the UNPROMPTED pretrained CLIP
        # feature, so the target itself cannot drift with the visual prompt.
        pre_features = torch.stack(
            (
                F.normalize(self.pre_proj_head(causal), dim=-1),
                F.normalize(self.pre_proj_head(visual_anchor), dim=-1),
            ),
            dim=1,
        )
        pre_cl_loss = (
            self.supcon_loss_pre(pre_features, labels)
            if self.l_layer
            else causal.new_zeros(())
        )

        variance = F.softplus(self.reg_log_variance) + 1e-5
        reg_loss = (
            (((causal - visual_anchor).pow(2) / variance) + variance.log()).mean() / 2
            if self.l_d
            else causal.new_zeros(())
        )

        total = (
            loss_cls
            + self.beta * loss_de
            + self.gamma * loss_ind
            + self.causal_consistency_weight * loss_causal_consistency
            + self.contrastive_weight * loss_contrastive
            + self.l_layer * pre_cl_loss
            + self.l_d * reg_loss
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL shapes: mode={} v={} v_aug={} v_anchor={} e={} e_aug={} "
                "s={} s_aug={} z_k={} text_features={} logits={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(visual_aug.shape),
                    tuple(visual_anchor.shape),
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

        return {
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_de_orig_loss": loss_de_orig.item(),
            "cipt_de_aug_loss": loss_de_aug.item(),
            "cipt_ind_loss": loss_ind.item(),
            "causal_consistency_loss": loss_causal_consistency.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "pre_cl_loss": pre_cl_loss.item(),
            "reg_loss": reg_loss.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

    def update(self, x, y, **kwargs):
        if self.cipt_pure:
            return self._update_pure(x, y)

        if "x_2" not in kwargs:
            raise KeyError(
                "CIPTDCCL fusion mode requires x_2: one original image and one "
                "augmented view per sample."
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
