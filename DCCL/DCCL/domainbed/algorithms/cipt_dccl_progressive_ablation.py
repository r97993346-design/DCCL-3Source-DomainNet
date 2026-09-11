"""Progressive module ablations for the no-augmentation CIPT branch.

The full Safe-Diverse prompt-selection implementation remains unchanged in
``cipt_dccl_ablation.CIPTDCCL``.  This wrapper only controls whether the three
paper-level modules are active, matching the progressive ablation table:

1) Base:                 v <-> class text
2) + Causal:             v -> (e, s), e <-> class text
3) + Text Diversity:     v -> (e, s), e -> Safe-Diverse/TDA -> z_k <-> class text
4) + Contrastive:        the same prediction path + WBC-CL on e

The switches are hierarchical by design.  Text Diversity requires Causal, and
Contrastive requires Text Diversity, so each valid configuration differs from
the previous one by exactly one paper-level module.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl_ablation import CIPTDCCL as _FullCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)


class CIPTDCCL(_FullCIPTDCCL):
    """CIPTDCCL with clean progressive Causal/Text/Contrastive ablations."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.use_causal_module = bool(
            hparams.get("cipt_use_causal_module", True)
        )
        self.use_text_diversity = bool(
            hparams.get("cipt_use_text_diversity", True)
        )
        self.use_contrastive_module = bool(
            hparams.get("cipt_use_contrastive_module", True)
        )

        # Legacy compatibility: cipt_pure has always meant no added CCL.
        if self.cipt_pure:
            self.use_contrastive_module = False

        # Keep the main ablation table strictly progressive rather than allowing
        # arbitrary module combinations that do not correspond to the method.
        if self.use_text_diversity and not self.use_causal_module:
            raise ValueError(
                "cipt_use_text_diversity=true requires "
                "cipt_use_causal_module=true."
            )
        if self.use_contrastive_module and not self.use_text_diversity:
            raise ValueError(
                "cipt_use_contrastive_module=true requires "
                "cipt_use_text_diversity=true."
            )

        print(
            "CIPTDCCL progressive ablation: Causal={}, TextDiversity={}, "
            "Contrastive={}, selector_mode={} (Safe-Diverse unchanged)".format(
                self.use_causal_module,
                self.use_text_diversity,
                self.use_contrastive_module,
                self.prompt_selector_mode,
            )
        )

    def _contrastive_enabled(self):
        return (
            self.use_contrastive_module
            and not self.cipt_pure
            and self.contrastive_weight > 0.0
        )

    def update(self, x, y, **kwargs):
        """Train one of the four progressive ablation configurations."""
        all_x = torch.cat(x)
        labels = torch.cat(y)
        domain_ids = torch.cat(
            [
                domain_labels.new_full(
                    domain_labels.shape, domain_index
                )
                for domain_index, domain_labels in enumerate(y)
            ]
        )

        visual = self._visual(all_x)
        class_features = self.text_features.class_features()
        zero = visual.new_zeros(())

        # ------------------------------------------------------------------
        # 1) Base: bypass causal decomposition completely and classify the
        # frozen CLIP visual feature directly against the same learned class
        # text features used by every other row.
        # ------------------------------------------------------------------
        if not self.use_causal_module:
            causal = None
            spurious = None
            loss_de = zero
            loss_ind = zero
            classification_features = visual[:, None, :]
            selector_metrics = self._empty_selector_metrics(
                visual, prompt_count=0, candidate_count=0, active=False
            )

        # ------------------------------------------------------------------
        # 2) + Causal: decompose v -> (e, s), optimize L_de and L_ind, and use
        # e directly for CLIP image-text classification when Text Diversity is
        # disabled.
        # ------------------------------------------------------------------
        else:
            causal, spurious = self.causal_decomposition(visual)

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

            # --------------------------------------------------------------
            # 3) + Text Diversity: use the existing selector/TDA path exactly
            # as implemented in cipt_dccl_ablation.py.  No selector rule is
            # changed here.
            # --------------------------------------------------------------
            if self.use_text_diversity:
                classification_features, selector_metrics = (
                    self._select_interventions(
                        visual, causal, class_features, labels=labels
                    )
                )
            else:
                classification_features = causal[:, None, :]
                selector_metrics = self._empty_selector_metrics(
                    causal, prompt_count=0, candidate_count=0, active=False
                )

        logits = self._logits(classification_features, class_features)
        loss_cls = cipt_classification_loss(logits, labels)
        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )

        # ------------------------------------------------------------------
        # 4) + Contrastive: prediction path is unchanged. WBC-CL groups only e
        # by source domain and class; it never consumes s, z_k, text, or x_2.
        # ------------------------------------------------------------------
        if not self._contrastive_enabled():
            loss_contrastive = zero
            wbc_metrics = self._empty_wbc_metrics(visual)
            contrastive_weight_eff = 0.0
        else:
            loss_contrastive, wbc_metrics = self._wbc_contrastive(
                causal, labels, domain_ids
            )
            self._causal_contrastive_step.add_(1)
            contrastive_weight_eff = self._contrastive_scale()

        total = cipt_base_loss + contrastive_weight_eff * loss_contrastive

        if self.debug_shapes:
            causal_shape = None if causal is None else tuple(causal.shape)
            spurious_shape = None if spurious is None else tuple(spurious.shape)
            print(
                "CIPTDCCL progressive shapes: mode={} v={} e={} s={} "
                "cls_features={} text_features={} logits={} "
                "modules(causal={}, text={}, contrastive={})".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    causal_shape,
                    spurious_shape,
                    tuple(classification_features.shape),
                    tuple(class_features.shape),
                    tuple(logits.shape),
                    self.use_causal_module,
                    self.use_text_diversity,
                    self.use_contrastive_module,
                )
            )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        if causal is None:
            mean_e_norm = zero
            mean_s_norm = zero
            mean_es_cosine = zero
        else:
            mean_e_norm = causal.norm(dim=-1).mean()
            mean_s_norm = spurious.norm(dim=-1).mean()
            mean_es_cosine = F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean()

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
            "wbc_contrastive_loss": loss_contrastive.item(),
            "contrastive_weight_eff": float(contrastive_weight_eff),
            "contrastive_valid_anchor_fraction": wbc_metrics[
                "valid_anchor_fraction"
            ].item(),
            "wbc_domain_coverage_fraction": wbc_metrics[
                "domain_coverage_fraction"
            ].item(),
            "wbc_weakest_positive_similarity": wbc_metrics[
                "weakest_positive_similarity"
            ].item(),
            "wbc_hard_negative_similarity": wbc_metrics[
                "hard_negative_similarity"
            ].item(),
            "wbc_violation_fraction": wbc_metrics[
                "violation_fraction"
            ].item(),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_v_norm": visual.norm(dim=-1).mean().item(),
            "mean_e_norm": mean_e_norm.item(),
            "mean_s_norm": mean_s_norm.item(),
            "mean_es_cosine": mean_es_cosine.item(),
            "ablation_causal_enabled": float(self.use_causal_module),
            "ablation_text_diversity_enabled": float(self.use_text_diversity),
            "ablation_contrastive_enabled": float(
                self.use_contrastive_module
            ),
            **self._selector_metric_items(selector_metrics),
        }

    def predict(self, x):
        """Use the prediction path corresponding exactly to the active row."""
        # Full current Text Diversity path, including the unchanged adaptive
        # Safe-Diverse selector at test time.
        if self.use_text_diversity:
            return super().predict(x)

        visual = self._visual(x)
        class_features = self.text_features.class_features()

        # Base: v <-> t_c.
        if not self.use_causal_module:
            return self._logits(
                visual[:, None, :], class_features
            )[:, 0]

        # + Causal: e <-> t_c, without TDA/text intervention.
        causal, _ = self.causal_decomposition(visual)
        return self._logits(
            causal[:, None, :], class_features
        )[:, 0]
