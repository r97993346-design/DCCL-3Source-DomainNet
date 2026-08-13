"""Official-aligned causal decomposition and TDA modules for CIPT-DCCL."""

from torch import nn


class CausalDecomposition(nn.Module):
    """Two single-linear adapters f_e and f_s used by official CIPT."""

    def __init__(self, embedding_dim, identity_init=True):
        super().__init__()
        self.causal_adapter = nn.Linear(embedding_dim, embedding_dim)
        self.spurious_adapter = nn.Linear(embedding_dim, embedding_dim)

        if identity_init:
            nn.init.eye_(self.causal_adapter.weight)
            nn.init.zeros_(self.causal_adapter.bias)
            nn.init.eye_(self.spurious_adapter.weight)
            nn.init.zeros_(self.spurious_adapter.bias)

    def forward(self, visual_features):
        return self.causal_adapter(visual_features), self.spurious_adapter(visual_features)


class TextDiversityAugmentation(nn.Module):
    """CIPT text-based diversity augmentation, Eq. (17)-(19)."""

    def __init__(self, embedding_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim={} must be divisible by num_heads={}".format(
                    embedding_dim, num_heads
                )
            )
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, causal_features, irrelevant_text_features):
        """Fuse causal features with K text interventions.

        Args:
            causal_features: [B, D]
            irrelevant_text_features: [K, D] shared across the batch or
                [B, K, D] sample-specific class-conditioned contexts.

        Returns:
            [B, K, D] intervention features.
        """
        if irrelevant_text_features.ndim == 2:
            irrelevant_text_features = irrelevant_text_features.unsqueeze(0).expand(
                causal_features.shape[0], -1, -1
            )
        if irrelevant_text_features.ndim != 3:
            raise ValueError(
                "irrelevant_text_features must be [K, D] or [B, K, D], got {}".format(
                    tuple(irrelevant_text_features.shape)
                )
            )
        if irrelevant_text_features.shape[0] != causal_features.shape[0]:
            raise ValueError("Batch size mismatch between causal and text features.")

        batch, num_templates, dim = irrelevant_text_features.shape
        query = causal_features[:, None, :].expand(-1, num_templates, -1).reshape(
            batch * num_templates, 1, dim
        )
        key_value = irrelevant_text_features.reshape(batch * num_templates, 1, dim)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        z = self.layer_norm(query + self.dropout(attended))
        return z.squeeze(1).reshape(batch, num_templates, dim)
