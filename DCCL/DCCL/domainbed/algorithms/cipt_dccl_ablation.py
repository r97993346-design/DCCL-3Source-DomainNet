"""CIPT + switchable single-view contrastive learning in causal space E.

Execution modes:
1) cipt_pure=True: single-view CIPT only.
2) cipt_pure=False: the same single-view CIPT plus supervised contrastive
   learning directly on original-image causal representations.

This branch deliberately removes projection heads and removes the augmented
image branch entirely. Positive pairs are same-class original causal features
within the merged source-domain batch. No x_2/e_aug path is required.
"""

import math

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
)
from domainbed.algorithms.cipt_neighbor_contrastive import (
    empty_neighbor_stats,
    neighbor_retention_weights,
    validate_neighbor_options,
)
from domainbed.algorithms.cipt_modules import SafeDiversePromptSelector
from domainbed.optimizers import get_optimizer


CLASS_CONDITIONED_TDA_MODES = {"b5b", "bconst"}
SELECTABLE_TDA_MODES = {"b5a", "b5c"}


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT with single-view supervised contrastive learning in causal space."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))

        # Neutral-subject robustness experiment. Only the TDA intervention bank
        # changes; learned class prompts and the visual/classification paths stay
        # untouched. Valid choices are subject/thing/object/entity.
        self.cipt_neutral_subject = str(
            hparams.get("cipt_neutral_subject", "subject")
        ).lower()
        self.text_features.set_neutral_subject(self.cipt_neutral_subject)

        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

        self.prompt_selector_mode = str(
            hparams.get("cipt_selector_mode", "random")
        ).lower()
        if self.prompt_selector_mode not in ("random", "all", "adaptive"):
            raise ValueError(
                "Unknown cipt_selector_mode={!r}; expected random, all, or "
                "adaptive.".format(self.prompt_selector_mode)
            )
        if (self.prompt_selector_mode != "random"
                and self.cipt_template_mode not in SELECTABLE_TDA_MODES):
            raise ValueError(
                "cipt_selector_mode={!r} requires the diverse class-agnostic "
                "b5a or b5c bank, got {!r}.".format(
                    self.prompt_selector_mode, self.cipt_template_mode
                )
            )
        self.prompt_selector = SafeDiversePromptSelector(
            k=hparams["cipt_k"],
            candidate_count=hparams.get("cipt_selector_candidates", 8),
        )

        # Keep contrastive learning directly in the causal representation space.
        for module_name in ("proj_head", "pre_proj_head"):
            if hasattr(self, module_name):
                delattr(self, module_name)

        # Disable inherited DCCL-side auxiliary objectives.
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
        if not math.isfinite(self.contrastive_weight):
            raise ValueError("cipt_causal_contrastive_weight must be finite")
        if self.contrastive_weight < 0.0:
            raise ValueError("cipt_causal_contrastive_weight must be non-negative")
        self.contrastive_warmup_steps = max(
            0, int(hparams.get("cipt_contrastive_warmup_steps", 500))
        )
        # Keep the causal contrastive temperature independently tunable.  The
        # generic ``t`` option remains a backwards-compatible fallback, but a
        # new method run no longer has to inherit the standard DCCL setting.
        self.contrastive_temperature = float(
            hparams.get("cipt_contrastive_temperature", hparams.get("t", 0.1))
        )
        if not math.isfinite(self.contrastive_temperature):
            raise ValueError("cipt_contrastive_temperature must be finite")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("cipt_contrastive_temperature must be positive")
        self.contrastive_type = str(hparams.get("cipt_contrastive_type", "supcon")).lower()
        if self.contrastive_type not in ("supcon", "neighbor_retention"):
            raise ValueError("cipt_contrastive_type must be supcon or neighbor_retention")
        self.neighbor_k = hparams.get("cipt_neighbor_k", 5)
        self.neighbor_alpha = float(hparams.get("cipt_neighbor_alpha", 0.5))
        validate_neighbor_options(self.neighbor_k, self.neighbor_alpha)
        self.neighbor_diagnostics = bool(hparams.get("cipt_neighbor_diagnostics", False))
        self.register_buffer(
            "_causal_contrastive_step",
            torch.zeros((), dtype=torch.long),
        )

        if self.cipt_pure:
            self.contrastive_weight = 0.0

        # Rebuild optimizer after removing projection heads.
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
            "template_mode={}, neutral_subject={}, K={}, tda_heads={}, lr={}, "
            "contrastive_weight={}, contrastive_warmup_steps={}, temp={}, "
            "selector_mode={}, selector_candidates={}, "
            "visual_l2_norm=False, adapter_init=default, augmented_view=False, "
            "projection_head=False, pre_cl=False, reg=False".format(
                self.cipt_pure,
                self.cipt_template_mode,
                self.cipt_neutral_subject,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.contrastive_warmup_steps,
                self.contrastive_temperature,
                self.prompt_selector_mode,
                self.prompt_selector.candidate_count,
            )
        )
        print(
            "CIPTDCCL single-view-causal-contrastive parameters: "
            "trainable={}, frozen={}".format(
                self.trainable_parameter_count,
                self.frozen_parameter_count,
            )
        )
        print(
            "CIPTDCCL contrastive_type={}, neighbor_k={}, neighbor_alpha={}, "
            "neighbor_diagnostics={}, detached_weights=True".format(
                self.contrastive_type, self.neighbor_k, self.neighbor_alpha,
                self.neighbor_diagnostics,
            )
        )

    def _intervention_features(self, labels=None):
        if self.cipt_template_mode in CLASS_CONDITIONED_TDA_MODES:
            return self.text_features.intervention_features(labels=labels)
        return self.text_features.irrelevant_text_features

    @staticmethod
    def _empty_selector_metrics(reference, prompt_count, candidate_count=0):
        zero = reference.new_zeros(())
        return {
            "prompt_selector_active": zero,
            "prompt_count": reference.new_tensor(float(prompt_count)),
            "prompt_selector_candidates": reference.new_tensor(float(candidate_count)),
            "prompt_selector_relevance": zero,
            "prompt_selector_js": zero,
            "prompt_selector_pairwise_cosine": zero,
            "prompt_selector_safe_fraction": zero,
            "prompt_selector_safe_candidates": zero,
            "prompt_selector_fallback_fraction": zero,
            "prompt_selector_unique": zero,
        }

    @staticmethod
    def _selector_metric_items(selector_metrics):
        return {
            name: float(value.detach().item())
            for name, value in selector_metrics.items()
        }

    def _select_interventions(self, visual, causal, class_features, labels=None):
        """Select B5a/B5c contexts with the same policy during train and eval.

        Random mode preserves the paired protocol's random train K and fixed
        eval K. Class-conditioned and constant banks retain their original path.
        Safety uses predictions, never training labels or the spurious feature.
        """
        if (self.cipt_template_mode not in SELECTABLE_TDA_MODES
                or self.prompt_selector_mode == "random"):
            contexts = self._intervention_features(labels=labels)
            interventions = self.tda(causal, contexts)
            return interventions, self._empty_selector_metrics(
                causal, interventions.shape[1]
            )

        prompt_bank = self.text_features.full_intervention_features()
        if self.prompt_selector_mode == "all":
            interventions = self.tda(causal, prompt_bank)
            return interventions, self._empty_selector_metrics(
                causal, interventions.shape[1], prompt_bank.shape[0]
            )

        # CLIP image/text similarity selects candidates in their shared frozen
        # embedding space. E and S are not used as relevance queries.
        candidate_indices, candidate_relevance = self.prompt_selector.shortlist(
            visual, prompt_bank
        )
        candidate_contexts = prompt_bank[candidate_indices]
        candidate_interventions = self.tda(causal, candidate_contexts)

        # Index selection is label-free and detached. The gathered K features
        # keep their gradient path through TDA and the causal adapter.
        with torch.no_grad():
            base_logits = self._logits(
                causal.detach()[:, None, :], class_features.detach()
            )[:, 0]
            candidate_logits = self._logits(
                candidate_interventions.detach(), class_features.detach()
            )
        local_indices, _, metrics = self.prompt_selector.select(
            candidate_indices, candidate_relevance, causal,
            candidate_interventions, base_logits, candidate_logits,
        )
        interventions = self.prompt_selector.batch_gather(
            candidate_interventions, local_indices
        )
        metrics.update({
            "prompt_selector_active": causal.new_ones(()),
            "prompt_count": causal.new_tensor(float(interventions.shape[1])),
            "prompt_selector_candidates": causal.new_tensor(
                float(candidate_interventions.shape[1])
            ),
        })
        return interventions, metrics

    def _contrastive_scale(self):
        """Linearly warm the direct causal contrastive coefficient."""
        if self.cipt_pure or self.contrastive_weight <= 0.0:
            return 0.0
        if self.contrastive_warmup_steps <= 0:
            return self.contrastive_weight

        step = int(self._causal_contrastive_step.item())
        ramp = min(1.0, step / float(self.contrastive_warmup_steps))
        return self.contrastive_weight * ramp

    def _single_view_supcon(self, causal, labels, anchor_weights=None):
        """Supervised contrastive loss over original causal features only.

        Positives are other samples in the merged source-domain batch that share
        the same class label. Self-pairs are excluded. Anchors with no positive
        sample are safely ignored instead of producing NaNs.
        """
        features = F.normalize(causal.float(), dim=-1)
        batch_size = features.shape[0]
        if batch_size <= 1:
            return features.sum() * 0.0, features.new_zeros(())

        temperature = max(self.contrastive_temperature, 1e-8)
        logits = features @ features.t() / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(
            batch_size, device=features.device, dtype=torch.bool
        )
        logits_mask = ~self_mask

        labels_col = labels.contiguous().view(-1, 1)
        positive_mask = labels_col.eq(labels_col.t()) & logits_mask
        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0

        if not valid_anchor.any():
            return features.sum() * 0.0, valid_anchor.float().mean()

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        mean_log_prob_pos = (
            positive_mask.float() * log_prob
        ).sum(dim=1) / positive_count.clamp_min(1).float()

        if anchor_weights is None:
            loss = -mean_log_prob_pos[valid_anchor].mean()
        else:
            if anchor_weights.shape != (batch_size,):
                raise ValueError("anchor_weights must have shape [B]")
            # Preserve the original mean over valid anchors, not sum(weights).
            loss = -(
                mean_log_prob_pos[valid_anchor]
                * anchor_weights.detach()[valid_anchor]
            ).mean()
        valid_fraction = valid_anchor.float().mean()
        return loss, valid_fraction

    def _causal_contrastive_loss(self, visual, causal, labels, label_batches):
        """Shared route for standard, weighted and disabled contrastive modes."""
        enabled = (
            not self.cipt_pure
            and getattr(self, "use_contrastive", True)
            and self.contrastive_weight > 0.0
        )
        apply_weights = (
            enabled and self.contrastive_type == "neighbor_retention"
            and self.neighbor_alpha > 0.0
        )
        stats = empty_neighbor_stats(causal)
        stats["nbr_wmean"] = causal.new_ones(())
        stats["nbr_wmax"] = causal.new_ones(())
        anchor_weights = None
        if apply_weights or self.neighbor_diagnostics:
            # The trainer supplies one label tensor per SOURCE domain, in the
            # same order as torch.cat(x)/torch.cat(y). No target labels enter.
            domains = torch.cat([
                torch.full_like(batch_labels, domain, dtype=torch.long)
                for domain, batch_labels in enumerate(label_batches)
            ])
            candidate_weights, measured = neighbor_retention_weights(
                visual, causal, labels, domains,
                k=self.neighbor_k, alpha=self.neighbor_alpha,
            )
            stats.update(measured)
            if apply_weights:
                anchor_weights = candidate_weights
                valid = labels[:, None].eq(labels[None, :]).sum(dim=1) > 1
                count = valid.float().sum().clamp_min(1)
                stats["nbr_wmean"] = (
                    1.0 + ((candidate_weights - 1.0) * valid).sum() / count
                )
                if candidate_weights.numel():
                    stats["nbr_wmax"] = candidate_weights.max()

        if not enabled:
            zero = causal.new_zeros(())
            return zero, zero, 0.0, stats

        loss, valid_fraction = self._single_view_supcon(causal, labels, anchor_weights)
        self._causal_contrastive_step.add_(1)
        return loss, valid_fraction, self._contrastive_scale(), stats

    def update(self, x, y, **kwargs):
        """Single-view CIPT update; x_2 is intentionally not consumed."""
        all_x = torch.cat(x)
        labels = torch.cat(y)

        # Frozen CLIP image encoder. CausalDecomposition directly applies two
        # default-initialized linear adapters to the frozen visual features.
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

        interventions, selector_metrics = self._select_interventions(
            visual, causal, class_features, labels=labels
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )

        loss_contrastive, valid_anchor_fraction, contrastive_weight_eff, neighbor_stats = (
            self._causal_contrastive_loss(visual, causal, labels, y)
        )

        total = (
            cipt_base_loss
            + contrastive_weight_eff * loss_contrastive
        )

        if self.debug_shapes:
            print(
                "CIPTDCCL single-view causal shapes: mode={} v={} e={} s={} "
                "z_k={} text_features={} logits={}".format(
                    self.cipt_template_mode,
                    tuple(visual.shape),
                    tuple(causal.shape),
                    tuple(spurious.shape),
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
            **{name: value.item() for name, value in neighbor_stats.items()},
            **self._selector_metric_items(selector_metrics),
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
        }

    def predict(self, x):
        if self.cipt_template_mode not in CLASS_CONDITIONED_TDA_MODES | SELECTABLE_TDA_MODES:
            return super().predict(x)

        if self.cipt_template_mode in SELECTABLE_TDA_MODES:
            visual = self._visual(x)
            causal, _ = self.causal_decomposition(visual)
            class_features = self.text_features.class_features()
            interventions, _ = self._select_interventions(
                visual, causal, class_features
            )
            return self._logits(interventions, class_features).mean(dim=1)

        # At inference labels are unknown. For B0/Bconst, score every candidate
        # class using that candidate's own intervention contexts and average K.
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
