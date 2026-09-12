"""CRCC: causal-reliability-guided connectivity contrastive learning for CIPT.

This module keeps the official-no-augmentation CIPT training path intact and
only replaces the contrastive objective applied to the causal representation
``e``.  The spurious representation ``s`` is deliberately not used by CRCC:
CIPT already drives its class prediction toward a uniform distribution through
L_de and decorrelates it from ``e`` through L_ind.

CRCC differs from vanilla supervised contrastive learning in two ways:

1. Positive connectivity aggregation.
   Same-class samples form a positive bag.  The loss aggregates their evidence
   instead of forcing every same-class pair to be individually compact.  This
   preserves legitimate cross-domain variation (e.g. PACS Photo vs. Sketch).

2. Adaptive pair importance in causal space.
   Positive samples can be weighted by the true-class confidence of their own
   causal feature, while negatives can be weighted by how strongly the anchor
   confuses their class.  Both weights are detached so the network cannot lower
   the loss by manipulating the weighting mechanism itself.

The existing ``cipt_use_contrastive`` switch and contrastive warm-up/weight are
reused.  ``cipt_contrastive_type=supcon`` restores the previous single-view
SupCon objective for direct ablation.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl_component_ablation import (
    CIPTDCCL as _BaseCIPTDCCL,
)


class CIPTDCCL(_BaseCIPTDCCL):
    """CIPT with CRCC operating only on the invariant causal feature ``e``."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.contrastive_type = str(
            hparams.get("cipt_contrastive_type", "crcc")
        ).lower()
        if self.contrastive_type not in {"crcc", "supcon"}:
            raise ValueError(
                "cipt_contrastive_type must be one of {'crcc', 'supcon'}, "
                f"got {self.contrastive_type!r}"
            )

        self.crcc_use_reliability = bool(
            hparams.get("cipt_crcc_reliability", True)
        )
        self.crcc_use_confusion_negative = bool(
            hparams.get("cipt_crcc_confusion_negative", True)
        )
        self.crcc_eps = float(hparams.get("cipt_crcc_eps", 1e-12))

        # Cached diagnostics from the latest CRCC call.  They are plain tensors
        # (not parameters/buffers) because they are logging-only state.
        self._crcc_last = {
            "positive_reliability": 0.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "valid_anchor_fraction": 0.0,
        }

        print(
            "CIPTDCCL contrastive objective: type={}, reliability={}, "
            "confusion_negative={}, operates_on=e_only, uses_s=False".format(
                self.contrastive_type,
                self.crcc_use_reliability,
                self.crcc_use_confusion_negative,
            )
        )

    def _causal_class_probabilities(self, causal):
        """Return detached class probabilities predicted directly from ``e``.

        These probabilities are used only as weighting signals.  The actual
        causal classification/decomposition losses remain the original CIPT
        objectives in the parent class.
        """
        with torch.no_grad():
            class_features = self.text_features.class_features()
            causal_logits = self._logits(
                causal[:, None, :], class_features
            )[:, 0]
            return F.softmax(causal_logits.float(), dim=-1)

    @staticmethod
    def _weighted_logsumexp(logits, weights, mask, eps):
        """Log of a normalized weighted exponential average.

        Row-wise weight normalization removes dependence on the number of
        positives/negatives in the minibatch.  Masked entries are exactly
        excluded rather than merely assigned a tiny finite weight.
        """
        weights = weights * mask.float()
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(eps)
        weights = weights / normalizer

        neg_inf = torch.full_like(logits, float("-inf"))
        weighted_logits = torch.where(
            mask,
            logits + torch.log(weights.clamp_min(eps)),
            neg_inf,
        )
        return torch.logsumexp(weighted_logits, dim=1)

    def _crcc_loss(self, causal, labels):
        """Causal Reliability-Guided Connectivity Contrastive loss.

        Positive evidence is aggregated over the same-class bag, with optional
        source-positive reliability weights derived from each sample's own
        causal classification confidence.  Negative evidence is aggregated over
        different-class samples, with optional anchor-specific confusion weights.

        Importantly, the spurious feature ``s`` is never consumed here.
        """
        features = F.normalize(causal.float(), dim=-1)
        batch_size = features.shape[0]
        if batch_size <= 1:
            zero = features.sum() * 0.0
            return zero, features.new_zeros(())

        temperature = max(float(self.contrastive_temperature), 1e-8)
        similarity = features @ features.t() / temperature

        eye = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        labels_col = labels.contiguous().view(-1, 1)
        same_class = labels_col.eq(labels_col.t()) & ~eye
        different_class = ~labels_col.eq(labels_col.t())

        valid_anchor = same_class.any(dim=1) & different_class.any(dim=1)
        if not valid_anchor.any():
            zero = features.sum() * 0.0
            self._crcc_last = {
                "positive_reliability": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "valid_anchor_fraction": float(valid_anchor.float().mean().item()),
            }
            return zero, valid_anchor.float().mean()

        need_probabilities = (
            self.crcc_use_reliability
            or self.crcc_use_confusion_negative
        )
        if need_probabilities:
            causal_prob = self._causal_class_probabilities(causal)
        else:
            causal_prob = None

        # Positive reliability is a property of the candidate positive sample:
        # P(y_j | e_j).  Detaching the weighting signal is intentional.
        if self.crcc_use_reliability:
            sample_index = torch.arange(batch_size, device=features.device)
            positive_reliability = causal_prob[
                sample_index, labels
            ].detach().clamp_min(self.crcc_eps)
        else:
            positive_reliability = features.new_ones(batch_size)

        positive_weights = positive_reliability.unsqueeze(0).expand(
            batch_size, batch_size
        )

        # Negative weight for candidate j under anchor i is P(y_j | e_i):
        # classes that the anchor currently confuses with its own class receive
        # more attention.  This is also detached from the optimization graph.
        if self.crcc_use_confusion_negative:
            negative_weights = causal_prob[:, labels].detach().clamp_min(
                self.crcc_eps
            )
        else:
            negative_weights = features.new_ones(
                batch_size, batch_size
            )

        positive_score = self._weighted_logsumexp(
            similarity,
            positive_weights,
            same_class,
            self.crcc_eps,
        )
        negative_score = self._weighted_logsumexp(
            similarity,
            negative_weights,
            different_class,
            self.crcc_eps,
        )

        # Evidence-ranking form: same-class connectivity evidence should exceed
        # confusing different-class evidence.  Unlike pairwise SupCon, distant
        # positives are not each assigned an independent large pull-together loss.
        per_anchor_loss = F.softplus(negative_score - positive_score)
        loss = per_anchor_loss[valid_anchor].mean()

        with torch.no_grad():
            self._crcc_last = {
                "positive_reliability": float(
                    positive_reliability.mean().item()
                ),
                "positive_score": float(
                    positive_score[valid_anchor].mean().item()
                ),
                "negative_score": float(
                    negative_score[valid_anchor].mean().item()
                ),
                "valid_anchor_fraction": float(
                    valid_anchor.float().mean().item()
                ),
            }

        return loss, valid_anchor.float().mean()

    def _single_view_supcon(self, causal, labels):
        """Dispatch to CRCC or the previous single-view SupCon baseline."""
        if self.contrastive_type == "supcon":
            self._crcc_last = {
                "positive_reliability": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "valid_anchor_fraction": 0.0,
            }
            return super()._single_view_supcon(causal, labels)
        return self._crcc_loss(causal, labels)

    def update(self, x, y, **kwargs):
        metrics = super().update(x, y, **kwargs)
        metrics.update(
            {
                "contrastive_is_crcc": float(
                    self.contrastive_type == "crcc"
                ),
                "crcc_positive_reliability": self._crcc_last[
                    "positive_reliability"
                ],
                "crcc_positive_score": self._crcc_last["positive_score"],
                "crcc_negative_score": self._crcc_last["negative_score"],
            }
        )
        return metrics
