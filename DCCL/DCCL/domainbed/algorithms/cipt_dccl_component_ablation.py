"""Independent component ablations for the no-augmentation CIPT+DCCL branch.

This wrapper keeps the architecture and preprocessing of
``cipt_dccl_ablation.CIPTDCCL`` unchanged, while exposing four independent
training switches through hparams/config:

- ``cipt_use_de``: decomposition loss L_de
- ``cipt_use_ind``: independence loss L_ind
- ``cipt_use_tda``: text diversity augmentation (TDA)
- ``cipt_use_contrastive``: single-view causal supervised contrastive loss

When TDA is disabled, classification falls back to direct causal-feature
classification against the same learned class text features. This keeps a
classification objective in every ablation setting and changes only the TDA
intervention itself.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl_ablation import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPTDCCL with independent L_de, L_ind, TDA, and contrastive switches."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.use_de = bool(hparams.get("cipt_use_de", True))
        self.use_ind = bool(hparams.get("cipt_use_ind", True))
        self.use_tda = bool(hparams.get("cipt_use_tda", True))
        self.use_contrastive = bool(
            hparams.get("cipt_use_contrastive", True)
        )

        # Backward compatibility: the legacy cipt_pure switch has always meant
        # "CIPT without the added causal contrastive objective".
        if self.cipt_pure:
            self.use_contrastive = False

        print(
            "CIPTDCCL component switches: L_de={}, L_ind={}, TDA={}, "
            "Contrastive={} (legacy cipt_pure={})".format(
                self.use_de,
                self.use_ind,
                self.use_tda,
                self.use_contrastive,
                self.cipt_pure,
            )
        )

    def _contrastive_scale(self):
        """Linearly warm the direct causal contrastive coefficient when enabled."""
        if not self.use_contrastive or self.contrastive_weight <= 0.0:
            return 0.0
        if self.contrastive_warmup_steps <= 0:
            return self.contrastive_weight

        step = int(self._causal_contrastive_step.item())
        ramp = min(1.0, step / float(self.contrastive_warmup_steps))
        return self.contrastive_weight * ramp

    def update(self, x, y, **kwargs):
        """Single-view CIPT update with independently switchable components."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        visual = self._visual(all_x)
        causal, spurious = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        zero = causal.new_zeros(())

        if self.use_de:
            causal_logits = self._logits(
                causal[:, None, :], class_features
            )[:, 0]
            spurious_logits = self._logits(
                spurious[:, None, :], class_features
            )[:, 0]
            loss_de = cipt_decomposition_loss(
                causal_logits, spurious_logits, labels
            )
        else:
            loss_de = zero

        if self.use_ind:
            loss_ind = cipt_independence_loss(causal, spurious)
        else:
            loss_ind = zero

        if self.use_tda:
            classification_features = self.tda(
                causal, self._intervention_features(labels=labels)
            )
        else:
            # TDA-off ablation: classify the causal representation directly.
            classification_features = causal[:, None, :]

        logits = self._logits(classification_features, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )

        if (
            not self.use_contrastive
            or self.contrastive_weight <= 0.0
        ):
            loss_contrastive = zero
            valid_anchor_fraction = zero
            contrastive_weight_eff = 0.0
        else:
            loss_contrastive, valid_anchor_fraction = self._single_view_supcon(
                causal, labels
            )
            self._causal_contrastive_step.add_(1)
            contrastive_weight_eff = self._contrastive_scale()

        total = (
            cipt_base_loss
            + contrastive_weight_eff * loss_contrastive
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL component-ablation shapes: mode={} v={} e={} s={} "
                "cls_features={} text_features={} logits={} "
                "switches(de={}, ind={}, tda={}, con={})".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
                    tuple(classification_features.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                    self.use_de,
                    self.use_ind,
                    self.use_tda,
                    self.use_contrastive,
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

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
            "contrastive_valid_anchor_fraction": valid_anchor_fraction.item(),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
            "ablation_de_enabled": float(self.use_de),
            "ablation_ind_enabled": float(self.use_ind),
            "ablation_tda_enabled": float(self.use_tda),
            "ablation_contrastive_enabled": float(self.use_contrastive),
        }

    def predict(self, x):
        if self.use_tda:
            return super().predict(x)

        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self.text_features.class_features()
        return self._logits(
            causal[:, None, :], class_features
        )[:, 0]
