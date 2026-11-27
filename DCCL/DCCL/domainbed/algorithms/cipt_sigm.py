"""Text-IF-inspired semantic interaction guidance for class-agnostic prompts.

The class-agnostic text vector controls channel-wise scale and shift of the
learnable causal E feature. No labels or spurious S features enter this module.
Supports [K,D] shared and [B,K,D] per-image text conditions.

The output head is initialized small, rather than identically zero: an exactly
identity mapping for every prompt yields zero pairwise MCC and hence cannot
train a prompt-only consistency branch from scratch.
"""

import math

import torch
from torch import nn


class SemanticInteractionGuidance(nn.Module):
    """SIGM: E'_k = (1 + scale * gamma(t_k)) * E + scale * beta(t_k)."""

    def __init__(self, embedding_dim, hidden_dim=None, scale=1.0, init_std=0.02):
        super().__init__()
        embedding_dim = int(embedding_dim)
        hidden_dim = max(1, embedding_dim // 2) if hidden_dim is None else int(hidden_dim)
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("SIGM embedding and hidden dimensions must be positive")
        for key, value in (("scale", scale), ("init_std", init_std)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("SIGM " + key + " must be positive and finite")
        self.scale = float(scale)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim * 2),
        )
        nn.init.normal_(self.projection[-1].weight, std=float(init_std))
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, causal_features, text_features):
        if causal_features.ndim != 2:
            raise ValueError("SIGM expects E features [B,D]")
        batch, dim = causal_features.shape
        if text_features.ndim == 2:
            if text_features.shape[-1] != dim:
                raise ValueError("SIGM text and E dimensions must agree")
            conditions = text_features.unsqueeze(0).expand(batch, -1, -1)
        elif text_features.ndim == 3:
            if text_features.shape[0] != batch or text_features.shape[-1] != dim:
                raise ValueError("SIGM text conditions must be [B,K,D]")
            conditions = text_features
        else:
            raise ValueError("SIGM expects text conditions [K,D] or [B,K,D]")
        gamma, beta = self.projection(conditions).chunk(2, dim=-1)
        reference = causal_features[:, None, :]
        return (1.0 + self.scale * gamma) * reference + self.scale * beta
