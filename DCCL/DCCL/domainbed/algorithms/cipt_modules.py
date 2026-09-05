"""Small causal decomposition and intervention modules used by CIPT-DCCL."""

import torch
from torch import nn
import torch.nn.functional as F


class CausalDecomposition(nn.Module):
    """CIPT's two linear, embedding-preserving adapters."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.causal_adapter = nn.Linear(embedding_dim, embedding_dim)
        self.spurious_adapter = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, visual_features):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        return self.causal_adapter(visual_features), self.spurious_adapter(visual_features)


class TextDiversityAugmentation(nn.Module):
    """The single residual cross-attention layer used for CIPT intervention."""

    def __init__(self, embedding_dim, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, causal_features, irrelevant_text_features):
        """Apply K independent one-token text interventions.

        Args:
            causal_features: [B, D].
            irrelevant_text_features: either shared [K, D] contexts (B5a/B5c)
                or sample-specific [B, K, D] contexts (B5b).
        """
        batch, dim = causal_features.shape
        if irrelevant_text_features.ndim == 2:
            k = irrelevant_text_features.shape[0]
            context = irrelevant_text_features[None, :, :].expand(batch, k, dim)
        elif irrelevant_text_features.ndim == 3:
            if irrelevant_text_features.shape[0] != batch:
                raise ValueError(
                    "Batch mismatch: causal_features has {}, contexts have {}".format(
                        batch, irrelevant_text_features.shape[0]
                    )
                )
            k = irrelevant_text_features.shape[1]
            context = irrelevant_text_features
        else:
            raise ValueError(
                "Expected intervention contexts [K,D] or [B,K,D], got {}".format(
                    tuple(irrelevant_text_features.shape)
                )
            )

        query = causal_features[:, None, :].expand(batch, k, dim).reshape(batch * k, 1, dim)
        context = context.reshape(batch * k, 1, dim)
        attended, _ = self.attention(query, context, context, need_weights=False)
        return self.layer_norm(query + attended).reshape(batch, k, dim)

    def prompt_effects(self, text_features):
        """Return the exact one-token attention residual for each prompt.

        TDA applies every prompt independently as a single key/value token. The
        attention softmax is therefore one, so the residual is determined only
        by the existing value and output projections. Exposing that residual
        lets the selector rank the intervention that TDA will actually apply,
        instead of ranking raw text embeddings in a different representation.
        """
        if text_features.ndim != 2:
            raise ValueError(
                "Expected a shared prompt bank [M,D], got {}".format(
                    tuple(text_features.shape)
                )
            )

        dim = self.attention.embed_dim
        in_proj_weight = self.attention.in_proj_weight
        if in_proj_weight is None:
            raise RuntimeError(
                "Prompt-effect extraction requires MultiheadAttention "
                "with a combined in_proj_weight."
            )
        value_weight = in_proj_weight[2 * dim : 3 * dim]
        in_proj_bias = self.attention.in_proj_bias
        value_bias = None
        if in_proj_bias is not None:
            value_bias = in_proj_bias[2 * dim : 3 * dim]

        value = F.linear(text_features, value_weight, value_bias)
        return self.attention.out_proj(value)

    def apply_prompt_effects(self, causal_features, prompt_effects):
        """Apply precomputed TDA prompt residuals to causal features.

        Args:
            causal_features: [B,D].
            prompt_effects: shared [K,D] or sample-specific [B,K,D].
        """
        batch, dim = causal_features.shape
        if prompt_effects.ndim == 2:
            effects = prompt_effects[None, :, :].expand(batch, -1, -1)
        elif prompt_effects.ndim == 3:
            if prompt_effects.shape[0] != batch:
                raise ValueError(
                    "Batch mismatch: causal_features has {}, effects have {}".format(
                        batch, prompt_effects.shape[0]
                    )
                )
            effects = prompt_effects
        else:
            raise ValueError(
                "Expected prompt effects [K,D] or [B,K,D], got {}".format(
                    tuple(prompt_effects.shape)
                )
            )
        if effects.shape[-1] != dim:
            raise ValueError(
                "Embedding mismatch: causal_features has D={}, effects have D={}".format(
                    dim, effects.shape[-1]
                )
            )
        query = causal_features[:, None, :].expand(batch, effects.shape[1], dim)
        return self.layer_norm(query + effects)


class SafeDiversePromptSelector(nn.Module):
    """Parameter-free, per-sample prompt shortlist and reranking.

    Relevance is derived from the existing spurious/causal decomposition. A
    label-free Jensen-Shannon penalty protects the current class distribution,
    and a greedy MMR penalty avoids selecting redundant interventions.
    """

    def __init__(
        self,
        k,
        candidate_count=8,
        causal_penalty=0.5,
        js_weight=1.0,
        diversity_weight=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.candidate_count = int(candidate_count)
        self.causal_penalty = float(causal_penalty)
        self.js_weight = float(js_weight)
        self.diversity_weight = float(diversity_weight)
        self.eps = float(eps)

        if self.k < 1:
            raise ValueError("Prompt selector k must be positive.")
        if self.candidate_count < 1:
            raise ValueError("Prompt selector candidate_count must be positive.")
        if self.causal_penalty < 0.0:
            raise ValueError("Prompt selector causal_penalty must be non-negative.")
        if self.js_weight < 0.0:
            raise ValueError("Prompt selector js_weight must be non-negative.")
        if self.diversity_weight < 0.0:
            raise ValueError("Prompt selector diversity_weight must be non-negative.")

    @staticmethod
    def batch_gather(features, indices):
        """Gather [B,L,*] features with per-sample [B,K] indices."""
        if features.ndim < 2 or indices.ndim != 2:
            raise ValueError("Expected features [B,L,...] and indices [B,K].")
        if features.shape[0] != indices.shape[0]:
            raise ValueError("Batch mismatch while gathering prompt features.")
        batch_indices = torch.arange(
            features.shape[0], device=features.device
        )[:, None]
        return features[batch_indices, indices]

    def shortlist(self, causal_features, spurious_features, prompt_effects):
        """Select Top-L prompts that affect s while avoiding e."""
        if causal_features.shape != spurious_features.shape:
            raise ValueError("Causal and spurious feature shapes must match.")
        if prompt_effects.ndim != 2:
            raise ValueError("Expected shared prompt effects [M,D].")
        if causal_features.shape[-1] != prompt_effects.shape[-1]:
            raise ValueError("Image and prompt-effect dimensions must match.")

        num_prompts = prompt_effects.shape[0]
        if self.k > num_prompts:
            raise ValueError(
                "Cannot select K={} unique prompts from a bank of {}.".format(
                    self.k, num_prompts
                )
            )
        candidate_count = max(
            self.k, min(self.candidate_count, num_prompts)
        )

        with torch.no_grad():
            causal = F.normalize(causal_features.detach().float(), dim=-1)
            spurious = F.normalize(spurious_features.detach().float(), dim=-1)
            effects = F.normalize(prompt_effects.detach().float(), dim=-1)
            spurious_similarity = spurious @ effects.t()
            causal_similarity = (causal @ effects.t()).abs()
            relevance = (
                spurious_similarity
                - self.causal_penalty * causal_similarity
            )
            candidate_relevance, candidate_indices = relevance.topk(
                candidate_count, dim=-1, largest=True, sorted=True
            )
        return candidate_indices, candidate_relevance

    def js_divergence(self, base_logits, candidate_logits):
        """Compute JS(P(y|e), P(y|z_k)) for every candidate prompt."""
        if base_logits.ndim != 2 or candidate_logits.ndim != 3:
            raise ValueError(
                "Expected base logits [B,C] and candidate logits [B,L,C]."
            )
        if (
            base_logits.shape[0] != candidate_logits.shape[0]
            or base_logits.shape[-1] != candidate_logits.shape[-1]
        ):
            raise ValueError("Base and candidate logit shapes are incompatible.")

        base_prob = F.softmax(base_logits.float(), dim=-1).clamp_min(self.eps)
        candidate_prob = F.softmax(
            candidate_logits.float(), dim=-1
        ).clamp_min(self.eps)
        base_prob = base_prob[:, None, :].expand_as(candidate_prob)
        midpoint = 0.5 * (base_prob + candidate_prob)
        js = 0.5 * (
            (base_prob * (base_prob.log() - midpoint.log())).sum(dim=-1)
            + (
                candidate_prob
                * (candidate_prob.log() - midpoint.log())
            ).sum(dim=-1)
        )
        return js

    def _greedy_mmr(self, quality, candidate_effects):
        """Select K high-quality but non-redundant candidates."""
        batch, candidate_count = quality.shape
        normalized = F.normalize(
            candidate_effects.detach().float(), dim=-1
        )
        pairwise = torch.bmm(normalized, normalized.transpose(1, 2))
        available = torch.ones(
            batch, candidate_count, dtype=torch.bool, device=quality.device
        )
        selected = []

        for _ in range(self.k):
            score = quality
            if selected:
                selected_so_far = torch.stack(selected, dim=1)
                gather_index = selected_so_far[:, None, :].expand(
                    batch, candidate_count, -1
                )
                redundancy = pairwise.gather(2, gather_index).max(dim=-1).values
                # Negative similarity is not treated as an extra reward.
                score = score - self.diversity_weight * redundancy.clamp_min(0.0)
            score = score.masked_fill(~available, float("-inf"))
            next_index = score.argmax(dim=-1)
            selected.append(next_index)
            available.scatter_(1, next_index[:, None], False)

        return torch.stack(selected, dim=1)

    def rerank(
        self,
        candidate_indices,
        candidate_relevance,
        candidate_effects,
        base_logits,
        candidate_logits,
    ):
        """Apply semantic-safety and diversity reranking to a shortlist."""
        with torch.no_grad():
            js = self.js_divergence(
                base_logits.detach(), candidate_logits.detach()
            )
            quality = candidate_relevance - self.js_weight * js
            local_indices = self._greedy_mmr(quality, candidate_effects)
            global_indices = candidate_indices.gather(1, local_indices)
            selected_relevance = candidate_relevance.gather(1, local_indices)
            selected_js = js.gather(1, local_indices)
            selected_effects = self.batch_gather(
                candidate_effects.detach(), local_indices
            )

            if self.k > 1:
                normalized = F.normalize(selected_effects.float(), dim=-1)
                similarity = torch.bmm(
                    normalized, normalized.transpose(1, 2)
                )
                upper = torch.triu_indices(
                    self.k, self.k, offset=1, device=similarity.device
                )
                pairwise_cosine = similarity[:, upper[0], upper[1]].mean()
            else:
                pairwise_cosine = quality.new_zeros(())

            metrics = {
                "prompt_selector_relevance": selected_relevance.mean(),
                "prompt_selector_js": selected_js.mean(),
                "prompt_selector_pairwise_cosine": pairwise_cosine,
                "prompt_selector_unique": quality.new_tensor(
                    float(torch.unique(global_indices).numel())
                ),
            }
        return local_indices, global_indices, metrics
