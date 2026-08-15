"""Ablation wrapper for selectable CIPTDCCL intervention prompt banks.

This file deliberately keeps the high-performance feature/multiprompt training
path unchanged except for:
1) the selected text intervention bank (B5a/B5b/B5c),
2) the official squared-cosine L_ind implemented in cipt_losses.py, and
3) an optional minimal CIPT reproduction switch that only disables contrastive
   losses while leaving all other model/training logic unchanged.
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
    """High-performance CIPTDCCL with B5a/B5b/B5c prompt ablations."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        # Minimal reproduction switch: only disable contrastive objectives.
        # Do not change prompts, TDA heads, adapter initialization, optimizer,
        # regularization, SWAD, or any other existing logic.
        if self.cipt_pure:
            self.contrastive_weight = 0.0
            self.l_layer = 0.0

        print(
            "CIPTDCCL ablation: pure_cipt={}, template_mode={}, K={}, "
            "tda_heads={}, lr={}, contrastive_weight={}, l_layer={}".format(
                self.cipt_pure,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.l_layer,
            )
        )

    def update(self, x, y, **kwargs):
        # B5a and B5c are class-agnostic [K,D] banks, so the original known-good
        # update path is preserved exactly. cipt_losses.py supplies the new L_ind.
        if self.cipt_template_mode != "b5b":
            return super().update(x, y, **kwargs)

        # B5b needs ground-truth labels to choose the class-conditioned ImageNet
        # intervention contexts during training. Everything else mirrors the
        # original high-performance update implementation.
        all_x = torch.cat(x)
        all_x_aug = torch.cat(kwargs["x_2"])
        labels = torch.cat(y)

        visual = self._visual(all_x)
        visual_aug = self._visual(all_x_aug)
        causal, spurious = self.causal_decomposition(visual)
        causal_aug, spurious_aug = self.causal_decomposition(visual_aug)

        class_features = self.text_features.class_features()
        causal_logits = self._logits(causal[:, None, :], class_features)[:, 0]
        spurious_logits = self._logits(spurious[:, None, :], class_features)[:, 0]
        loss_de = cipt_decomposition_loss(causal_logits, spurious_logits, labels)
        loss_ind = 0.5 * (
            cipt_independence_loss(causal, spurious)
            + cipt_independence_loss(causal_aug, spurious_aug)
        )

        diverse_features = self.text_features.intervention_features(labels=labels)
        interventions = self.tda(causal, diverse_features)
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

        pre_features = torch.stack(
            (
                F.normalize(self.pre_proj_head(causal), dim=-1),
                F.normalize(self.pre_proj_head(visual.detach()), dim=-1),
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
            (((causal - visual.detach()).pow(2) / variance) + variance.log()).mean() / 2
            if self.l_d
            else causal.new_zeros(())
        )

        total = (
            loss_cls
            + self.beta * loss_de
            + self.gamma * loss_ind
            + self.contrastive_weight * loss_contrastive
            + self.l_layer * pre_cl_loss
            + self.l_d * reg_loss
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL shapes: mode={} v={} e={} s={} projected_e={} z_k={} text_features={} logits={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
                    tuple(projected.shape),
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
            "cipt_ind_loss": loss_ind.item(),
            "dccl_contrastive_loss": loss_contrastive.item(),
            "pre_cl_loss": pre_cl_loss.item(),
            "reg_loss": reg_loss.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(causal, spurious, dim=-1).mean().item(),
        }

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
