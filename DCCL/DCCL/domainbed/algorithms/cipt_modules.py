"""Small causal decomposition and intervention modules used by CIPT-DCCL."""

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
