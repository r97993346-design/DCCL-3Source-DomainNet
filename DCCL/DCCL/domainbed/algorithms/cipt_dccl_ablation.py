"""CIPT + intervention-reliable contrastive learning on original causal e.

Execution modes:
1) cipt_pure=True: single-view CIPT only.
2) cipt_pure=False: the same single-view CIPT plus reference-weighted
   supervised contrastive learning on original-image causal representations.

This branch deliberately removes projection heads and removes the augmented
image branch entirely. Positive pairs are same-class original causal features
within the merged source-domain batch. Existing TDA predictions determine
detached positive-reference weights; they never enter the similarity matrix.
Uniform and pre-TDA confidence weights are available as matched controls.
No x_2/e_aug path is required.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.algorithms import CIPTDCCL as _BaseCIPTDCCL
from domainbed.algorithms.cipt_losses import (
    classification_loss as cipt_classification_loss,
    decomposition_loss as cipt_decomposition_loss,
    independence_loss as cipt_independence_loss,
    intervention_reference_reliability,
    confidence_reference_reliability,
    reference_weighted_causal_contrastive_loss,
)
from domainbed.optimizers import get_optimizer


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT with reference-weighted contrastive learning only in e space."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.cipt_pure = bool(hparams.get("cipt_pure", False))
        self.cipt_template_mode = str(
            hparams.get("cipt_template_mode", "b5a")
        ).lower()
        self.text_features.set_template_mode(self.cipt_template_mode)

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
        self.contrastive_warmup_steps = max(
            0, int(hparams.get("cipt_contrastive_warmup_steps", 500))
        )
        self.contrastive_temperature = float(hparams.get("t", 0.1))
        self.contrastive_reference = str(
            hparams.get("cipt_contrastive_reference", "intervention")
        ).lower()
        if self.contrastive_reference not in ("intervention", "confidence", "uniform"):
            raise ValueError(
                "cipt_contrastive_reference must be intervention, confidence or uniform"
            )
        self.contrastive_reference_floor = float(
            hparams.get("cipt_contrastive_reference_floor", 0.1)
        )
        if not 0.0 < self.contrastive_reference_floor <= 1.0:
            raise ValueError("cipt_contrastive_reference_floor must be in (0, 1]")
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
            "CIPTDCCL reference-weighted-causal-contrastive: pure_cipt={}, "
            "template_mode={}, K={}, tda_heads={}, lr={}, "
            "contrastive_weight={}, contrastive_warmup_steps={}, temp={}, "
            "reference={}, reference_floor={}, "
            "visual_l2_norm=False, adapter_init=default, augmented_view=False, "
            "projection_head=False, pre_cl=False, reg=False".format(
                self.cipt_pure,
                self.cipt_template_mode,
                hparams["cipt_k"],
                hparams["cipt_tda_heads"],
                hparams["lr"],
                self.contrastive_weight,
                self.contrastive_warmup_steps,
                self.contrastive_temperature,
                self.contrastive_reference,
                self.contrastive_reference_floor,
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
        """Keep the legacy helper available as the uniform-weight control."""
        loss, metrics = reference_weighted_causal_contrastive_loss(
            causal, labels, temperature=self.contrastive_temperature,
            reference_floor=self.contrastive_reference_floor,
        )
        return loss, metrics["valid_anchor_fraction"]

    @staticmethod
    def _empty_contrastive_metrics(reference):
        return {name: reference.new_zeros(()) for name in (
            "valid_anchor_fraction", "irc_reliability_mean", "irc_reliability_std",
            "irc_positive_ess_fraction", "irc_weighted_anchor_fraction",
            "irc_intervention_active", "irc_intervention_count", "irc_tda_fallback",
        )}

    @staticmethod
    def _contrastive_metric_items(metrics):
        # Transfer these small diagnostics together, rather than adding one
        # GPU synchronization per new metric to every training step.
        names = list(metrics)
        values = torch.stack([metrics[name].detach() for name in names]).cpu().tolist()
        return {
            ("contrastive_valid_anchor_fraction" if name == "valid_anchor_fraction" else name): value
            for name, value in zip(names, values)
        }

    def _causal_reference_contrastive(
        self, causal, labels, intervention_logits=None, causal_logits=None,
    ):
        """Reuse existing predictions for scores; only e enters the loss graph."""
        intervention_active = (
            self.contrastive_reference == "intervention"
            and intervention_logits is not None
        )
        tda_fallback = (
            self.contrastive_reference == "intervention"
            and intervention_logits is None
        )
        if intervention_active:
            reliability = intervention_reference_reliability(intervention_logits, labels)
        elif self.contrastive_reference == "confidence":
            if causal_logits is None:
                raise ValueError("confidence reference mode requires pre-TDA causal logits")
            reliability = confidence_reference_reliability(causal_logits, labels)
        else:
            # The independent TDA-off ablation uses ordinary uniform SupCon.
            # No intervention score is inferred from a non-intervened prediction.
            reliability = None

        loss, metrics = reference_weighted_causal_contrastive_loss(
            causal, labels, reference_reliability=reliability,
            temperature=self.contrastive_temperature,
            reference_floor=self.contrastive_reference_floor,
        )
        metrics.update({
            "irc_intervention_active": causal.new_tensor(float(intervention_active)),
            "irc_intervention_count": causal.new_tensor(
                float(intervention_logits.shape[1]) if intervention_active else 0.0
            ),
            "irc_tda_fallback": causal.new_tensor(float(tda_fallback)),
        })
        return loss, metrics

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

        interventions = self.tda(
            causal, self._intervention_features(labels=labels)
        )
        logits = self._logits(interventions, class_features)
        loss_cls = cipt_classification_loss(logits, labels)

        cipt_base_loss = (
            loss_cls + self.beta * loss_de + self.gamma * loss_ind
        )

        if self.cipt_pure or self.contrastive_weight <= 0.0:
            loss_contrastive = causal.new_zeros(())
            contrastive_metrics = self._empty_contrastive_metrics(causal)
            contrastive_weight_eff = 0.0
        else:
            loss_contrastive, contrastive_metrics = self._causal_reference_contrastive(
                causal, labels, intervention_logits=logits, causal_logits=causal_logits,
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
            **self._contrastive_metric_items(contrastive_metrics),
            "pre_cl_loss": zero.item(),
            "reg_loss": zero.item(),
            "mean_e_norm": causal.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(
                causal, spurious, dim=-1
            ).mean().item(),
        }

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
