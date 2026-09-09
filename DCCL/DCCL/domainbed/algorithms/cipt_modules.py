"""Small causal decomposition and intervention modules used by CIPT-DCCL."""

import torch
from torch import nn
import torch.nn.functional as F


class CausalDecomposition(nn.Module):
    """Two linear adapters for causal/spurious decomposition.

    This branch intentionally keeps the original DomainBed-side behavior:
    no pre-decomposition visual L2 normalization and default PyTorch linear
    initialization for both adapters.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.causal_adapter = nn.Linear(embedding_dim, embedding_dim)
        self.spurious_adapter = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, visual_features):
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
    """Parameter-free visual-relevant, safe and diverse prompt selection.

    The frozen CLIP visual feature is used only for image/text relevance because
    it shares CLIP's embedding space with the fixed B5c text bank.  Causal
    predictions before and after TDA provide a label-free semantic-safety test.
    Diversity is measured between the *actual* per-sample TDA intervention
    directions instead of between raw prompt embeddings.

    The spurious representation is deliberately not consumed: its class
    distribution is trained to be uniform and it has no objective aligning it
    with CLIP's text space.
    """

    def __init__(
        self,
        k,
        candidate_count=8,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.candidate_count = int(candidate_count)
        self.eps = float(eps)

        if self.k < 1:
            raise ValueError("Prompt selector k must be positive.")
        if self.candidate_count < 1:
            raise ValueError("Prompt selector candidate_count must be positive.")

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

    def shortlist(self, visual_features, prompt_features):
        """Select the Top-L B5c prompts in the frozen CLIP space.

        Args:
            visual_features: Frozen CLIP image features ``[B,D]``.
            prompt_features: Frozen CLIP B5c text features ``[M,D]``.
        """
        if visual_features.ndim != 2 or prompt_features.ndim != 2:
            raise ValueError(
                "Expected visual features [B,D] and prompt features [M,D]."
            )
        if visual_features.shape[-1] != prompt_features.shape[-1]:
            raise ValueError("Image and prompt dimensions must match.")

        num_prompts = prompt_features.shape[0]
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
            visual = F.normalize(visual_features.detach().float(), dim=-1)
            prompts = F.normalize(prompt_features.detach().float(), dim=-1)
            relevance = visual @ prompts.t()
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

    def _eligible_candidates(self, safe_mask, js):
        """Keep every safe prompt and add minimum-JS fallbacks if required."""
        safe_count = safe_mask.sum(dim=1)
        fallback_needed = (self.k - safe_count).clamp_min(0)

        # Rank only unsafe prompts by JS.  Safe prompts receive +inf and are not
        # accidentally included in the fallback count.
        unsafe_js = js.masked_fill(safe_mask, float("inf"))
        fallback_order = unsafe_js.argsort(dim=1)
        fallback_rank = torch.empty_like(fallback_order)
        positions = torch.arange(
            js.shape[1], device=js.device
        )[None, :].expand_as(fallback_order)
        fallback_rank.scatter_(1, fallback_order, positions)
        fallback_mask = (~safe_mask) & (
            fallback_rank < fallback_needed[:, None]
        )
        return safe_mask | fallback_mask

    def _greedy_farthest(
        self, candidate_relevance, intervention_directions, safe_mask, eligible
    ):
        """Select the most relevant seed, then maximize minimum distance.

        Safe candidates always have priority.  Minimum-JS fallback candidates
        are considered only after all available safe candidates have been used.
        """
        batch, candidate_count = candidate_relevance.shape
        raw_norm = intervention_directions.norm(dim=-1, keepdim=True)
        normalized = intervention_directions / raw_norm.clamp_min(self.eps)
        pairwise_cosine = torch.bmm(
            normalized, normalized.transpose(1, 2)
        ).clamp(-1.0, 1.0)
        pairwise_distance = 1.0 - pairwise_cosine

        # A zero intervention is not evidence of diversity.
        nonzero = raw_norm.squeeze(-1) > self.eps
        valid_pairs = nonzero[:, :, None] & nonzero[:, None, :]
        pairwise_distance = pairwise_distance.masked_fill(~valid_pairs, 0.0)

        available = eligible.clone()
        safe_available = available & safe_mask
        first_pool = torch.where(
            safe_available.any(dim=1, keepdim=True), safe_available, available
        )
        first = candidate_relevance.masked_fill(
            ~first_pool, float("-inf")
        ).argmax(dim=1)
        selected = [first]
        available.scatter_(1, first[:, None], False)

        gather_first = first[:, None, None].expand(batch, candidate_count, 1)
        min_distance = pairwise_distance.gather(
            2, gather_first
        ).squeeze(-1)

        for _ in range(1, self.k):
            safe_available = available & safe_mask
            step_pool = torch.where(
                safe_available.any(dim=1, keepdim=True),
                safe_available,
                available,
            )
            next_index = min_distance.masked_fill(
                ~step_pool, float("-inf")
            ).argmax(dim=1)
            selected.append(next_index)
            available.scatter_(1, next_index[:, None], False)

            gather_next = next_index[:, None, None].expand(
                batch, candidate_count, 1
            )
            distance_to_next = pairwise_distance.gather(
                2, gather_next
            ).squeeze(-1)
            min_distance = torch.minimum(min_distance, distance_to_next)

        return torch.stack(selected, dim=1), pairwise_cosine

    def select(
        self,
        candidate_indices,
        candidate_relevance,
        causal_features,
        candidate_interventions,
        base_logits,
        candidate_logits,
    ):
        """Apply hard class safety and farthest-direction selection."""
        if candidate_indices.ndim != 2 or candidate_relevance.ndim != 2:
            raise ValueError("Expected candidate indices/relevance [B,L].")
        if candidate_indices.shape != candidate_relevance.shape:
            raise ValueError("Candidate indices and relevance shapes must match.")
        if causal_features.ndim != 2 or candidate_interventions.ndim != 3:
            raise ValueError(
                "Expected causal features [B,D] and interventions [B,L,D]."
            )
        if (
            causal_features.shape[0] != candidate_indices.shape[0]
            or candidate_interventions.shape[:2] != candidate_indices.shape
            or candidate_interventions.shape[-1] != causal_features.shape[-1]
        ):
            raise ValueError("Candidate intervention shapes are incompatible.")
        if self.k > candidate_indices.shape[1]:
            raise ValueError("Cannot select more prompts than the shortlist.")

        with torch.no_grad():
            js = self.js_divergence(
                base_logits.detach(), candidate_logits.detach()
            )
            base_prediction = base_logits.detach().argmax(dim=-1)
            candidate_prediction = candidate_logits.detach().argmax(dim=-1)
            safe_mask = candidate_prediction.eq(base_prediction[:, None])
            eligible = self._eligible_candidates(safe_mask, js)

            intervention_directions = (
                candidate_interventions.detach().float()
                - causal_features.detach().float()[:, None, :]
            )
            local_indices, pairwise = self._greedy_farthest(
                candidate_relevance.detach().float(),
                intervention_directions,
                safe_mask,
                eligible,
            )
            global_indices = candidate_indices.gather(1, local_indices)
            selected_relevance = candidate_relevance.gather(1, local_indices)
            selected_js = js.gather(1, local_indices)
            selected_safe = safe_mask.gather(1, local_indices)

            if self.k > 1:
                selected_pairwise = self.batch_gather(
                    pairwise, local_indices
                )
                selected_pairwise = selected_pairwise.gather(
                    2, local_indices[:, None, :].expand(-1, self.k, -1)
                )
                upper = torch.triu_indices(
                    self.k, self.k, offset=1, device=pairwise.device
                )
                pairwise_cosine = selected_pairwise[
                    :, upper[0], upper[1]
                ].mean()
            else:
                pairwise_cosine = candidate_relevance.new_zeros(())

            metrics = {
                "prompt_selector_relevance": selected_relevance.mean(),
                "prompt_selector_js": selected_js.mean(),
                "prompt_selector_pairwise_cosine": pairwise_cosine,
                "prompt_selector_safe_fraction": selected_safe.float().mean(),
                "prompt_selector_safe_candidates": safe_mask.float().sum(
                    dim=1
                ).mean(),
                "prompt_selector_fallback_fraction": (
                    ~selected_safe
                ).float().mean(),
                "prompt_selector_unique": candidate_relevance.new_tensor(
                    float(torch.unique(global_indices).numel())
                ),
            }
        return local_indices, global_indices, metrics
