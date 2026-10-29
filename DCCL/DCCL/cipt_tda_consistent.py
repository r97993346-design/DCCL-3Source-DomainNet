"""Training/inference-consistent candidate-class TDA for original CIPT.

Only the class-conditioned classification-score construction is changed.
All actual CLIP, prompt learner, E/S adapters, TDA attention, and loss
implementations are inherited from the *pinned upstream* CIPT submodule.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cipt.model import CIPTModel, CIPTOutput, load_openai_clip


class ConsistentCIPTModel(CIPTModel):
    """Compute *every* candidate's TDA score in BOTH train and inference.

    Unlike upstream CIPT.forward(), gold labels are NEVER used to select the
    TDA text context. Training labels are passed only to cipt_loss outside
    the model. The original K-averaged CE loss, E/S and IND losses remain.
    """

    def __init__(self, *args, class_chunk_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        if class_chunk_size < 1:
            raise ValueError("class_chunk_size must be >= 1")
        self.class_chunk_size = class_chunk_size
        if not self.class_conditioned_templates:
            raise ValueError("ConsistentCIPTModel expects class-conditioned ImageNet templates")

    def classwise_tda_logits(
        self,
        causal_features: Tensor,
        text_features: Tensor,
        template_indices: Tensor | None = None,
    ) -> Tensor:
        """Return [B, K, C] logits: each score c uses its own text context c.

        Training and inference call this SAME method. Training retains
        upstream random K templates; inference retains upstream fixed first K.
        Candidate-class chunks avoid OOM on datasets such as OfficeHome.
        """
        bank = self._select_diverse_features(labels=None, indices=template_indices)
        if bank.ndim != 3:
            raise ValueError("Expected [C, K, D] class-conditioned template bank")

        batch, dim = causal_features.shape
        classes, k, text_dim = bank.shape
        if text_features.shape != (classes, dim) or text_dim != dim:
            raise ValueError(
                f"Feature mismatch: E={causal_features.shape}, "
                f"text={text_features.shape}, bank={bank.shape}"
            )

        class_text = F.normalize(text_features.float(), dim=-1)
        chunks = []
        for start in range(0, classes, self.class_chunk_size):
            end = min(start + self.class_chunk_size, classes)
            count = end - start

            # [B*C', D]: same image representation for all class candidates.
            e = causal_features[:, None, :].expand(batch, count, dim)
            e = e.reshape(batch * count, dim)

            # [B*C', K, D]: each class has its own K text contexts.
            p = bank[start:end][None, :, :, :].expand(batch, count, k, dim)
            p = p.reshape(batch * count, k, dim)

            z = self.diversity_augmentation(e, p.float())
            z = F.normalize(z.reshape(batch, count, k, dim).float(), dim=-1)
            scores = self.logit_scale * torch.einsum(
                "bckd,cd->bck", z, class_text[start:end]
            )
            chunks.append(scores)

        return torch.cat(chunks, dim=1).permute(0, 2, 1).contiguous()

    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        template_indices: Tensor | None = None,
    ) -> CIPTOutput:
        # Deliberately do not access labels. They ONLY enter cipt_loss.
        image_features = self.encode_image_features(images)
        text_features = self.encode_prompt_features()
        causal_features = self.causal_adapter(image_features)
        spurious_features = self.spurious_adapter(image_features)
        return CIPTOutput(
            interventional_logits=self.classwise_tda_logits(
                causal_features, text_features, template_indices=template_indices
            ),
            causal_logits=self._logits(causal_features, text_features),
            spurious_logits=self._logits(spurious_features, text_features),
            image_features=image_features,
            causal_features=causal_features,
            spurious_features=spurious_features,
            text_features=text_features,
            augmented_features=None,  # avoid storing [B,C,K,D]
        )

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        use_tda: bool = True,
        template_indices: Tensor | None = None,
    ) -> Tensor:
        old_mode = self.training
        self.eval()
        try:
            e = self.causal_adapter(self.encode_image_features(images))
            text = self.encode_prompt_features()
            if not use_tda:
                return self._logits(e, text)
            return self.classwise_tda_logits(
                e, text, template_indices=template_indices
            ).mean(dim=1)
        finally:
            self.train(old_mode)


def build_consistent_cipt(
    classnames,
    backbone="ViT-B/16",
    device=None,
    jit=False,
    download_root=None,
    class_chunk_size=8,
    **kwargs,
):
    """Upstream build_cipt() equivalent, constructing the subclass instead."""
    clip_model, preprocess, tokenize = load_openai_clip(
        backbone=backbone, device=device, jit=jit, download_root=download_root
    )
    model = ConsistentCIPTModel(
        clip_model,
        classnames=classnames,
        tokenize=tokenize,
        class_chunk_size=class_chunk_size,
        **kwargs,
    )
    model.to(next(clip_model.parameters()).device)
    return model, preprocess
