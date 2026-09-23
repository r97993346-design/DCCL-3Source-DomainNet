"""Small causal decomposition and intervention modules used by CIPT-DCCL."""

import torch
from torch import nn
import torch.nn.functional as F


class CausalDecomposition(nn.Module):
    """Switchable CIPT decomposition: dual linear adapters or causal mask.

    dual_linear exactly preserves the official-CIPT-style decomposition:
    two independently trainable identity-initialized linear adapters.

    causal_mask uses a sample-specific complementary binary-concrete mask:
        E = M(V) * V
        S = (1 - M(V)) * V
    so the two branches partition the same frozen CLIP representation instead
    of learning two unconstrained feature transforms.
    """

    SUPPORTED_MODES = {"dual_linear", "causal_mask"}

    def __init__(
        self,
        embedding_dim,
        mode="dual_linear",
        mask_hidden_dim=128,
        mask_temperature=1.0,
        mask_hard=False,
        eps=1e-6,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.mode = str(mode).lower()
        self.mask_temperature = float(mask_temperature)
        self.mask_hard = bool(mask_hard)
        self.eps = float(eps)
        self._last_mask = None

        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                "Unknown decomposition mode {!r}; expected one of {}.".format(
                    self.mode, sorted(self.SUPPORTED_MODES)
                )
            )
        if self.mask_temperature <= 0.0:
            raise ValueError("mask_temperature must be positive")

        if self.mode == "dual_linear":
            self.causal_adapter = nn.Linear(self.embedding_dim, self.embedding_dim)
            self.spurious_adapter = nn.Linear(self.embedding_dim, self.embedding_dim)

            # Official CIPT initialization: both E/S start from the normalized
            # frozen CLIP visual feature and are then independently optimized.
            nn.init.eye_(self.causal_adapter.weight)
            nn.init.zeros_(self.causal_adapter.bias)
            nn.init.eye_(self.spurious_adapter.weight)
            nn.init.zeros_(self.spurious_adapter.bias)
        else:
            hidden = int(mask_hidden_dim)
            if hidden <= 0:
                raise ValueError("mask_hidden_dim must be positive")
            self.mask_generator = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embedding_dim),
            )
            # Start close to a balanced soft split while keeping non-zero
            # gradients through both layers from the first update.
            nn.init.xavier_uniform_(self.mask_generator[0].weight)
            nn.init.zeros_(self.mask_generator[0].bias)
            nn.init.normal_(self.mask_generator[2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.mask_generator[2].bias)

    def _gumbel_sigmoid(self, logits):
        """Binary-Concrete/Gumbel-Sigmoid relaxation with deterministic eval."""
        if self.training:
            uniform = torch.rand_like(logits).clamp(
                min=self.eps, max=1.0 - self.eps
            )
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft = torch.sigmoid(
                (logits + logistic_noise) / self.mask_temperature
            )
        else:
            soft = torch.sigmoid(logits / self.mask_temperature)

        if not self.mask_hard:
            return soft

        hard = (soft >= 0.5).to(dtype=soft.dtype)
        if self.training:
            # Straight-through estimator: hard forward, soft backward.
            return hard.detach() - soft.detach() + soft
        return hard

    def forward(self, visual_features):
        if self.mode == "dual_linear":
            self._last_mask = None
            return (
                self.causal_adapter(visual_features),
                self.spurious_adapter(visual_features),
            )

        mask_logits = self.mask_generator(visual_features)
        mask = self._gumbel_sigmoid(mask_logits)
        self._last_mask = mask
        causal = mask * visual_features
        spurious = (1.0 - mask) * visual_features
        return causal, spurious

    @property
    def last_mask(self):
        """Mask from the latest forward pass, or None in dual-linear mode."""
        return self._last_mask

    def mask_sparsity_loss(self):
        """Mean mask activation used to prevent the trivial M -> 1 solution."""
        if self.mode != "causal_mask" or self._last_mask is None:
            return None
        return self._last_mask.mean()


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

class SafeDiversePromptSelector(nn.Module):
    """Parameter-free visual-relevant, safe and diverse prompt selection.

    The frozen CLIP visual feature is used only for image/text relevance because
    it shares CLIP's embedding space with the fixed class-agnostic text bank.  Causal
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
        """Select the Top-L class-agnostic prompts in the frozen CLIP space.

        Args:
            visual_features: Frozen CLIP image features ``[B,D]``.
            prompt_features: Frozen CLIP text features ``[M,D]``.
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
