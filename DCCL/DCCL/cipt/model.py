from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .templates import DEFAULT_CONTEXT_INIT, IMAGENET_TEMPLATES

TokenizeFn = Callable[[str | Sequence[str]], Tensor]


@dataclass
class CIPTOutput:
    """Forward output used by the training loss."""

    interventional_logits: Tensor | None
    causal_logits: Tensor
    spurious_logits: Tensor
    image_features: Tensor
    causal_features: Tensor
    spurious_features: Tensor
    text_features: Tensor
    augmented_features: Tensor | None


def _clip_dtype(clip_model: nn.Module) -> torch.dtype:
    return getattr(clip_model, "dtype", next(clip_model.parameters()).dtype)


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _format_template(template: str, class_name: str) -> str:
    clean_name = class_name.replace("_", " ")
    if "{class}" in template:
        return template.format(**{"class": clean_name})
    if "{}" in template:
        return template.format(clean_name)
    return template


def _has_class_placeholder(templates: Sequence[str]) -> bool:
    return any(("{}" in template) or ("{class}" in template) for template in templates)


class OpenAITextEncoder(nn.Module):
    """Text encoder wrapper that accepts prompt embeddings.

    OpenAI CLIP exposes ``encode_text`` for token ids, but prompt tuning needs
    to feed learnable token embeddings into the frozen text transformer.
    """

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = _clip_dtype(clip_model)

    def forward(self, prompts: Tensor, tokenized_prompts: Tensor) -> Tensor:
        x = prompts.to(dtype=self.dtype)
        pos = self.positional_embedding[: x.shape[1]].to(device=x.device, dtype=self.dtype)
        x = x + pos
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=self.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
        return x @ self.text_projection


class PromptLearner(nn.Module):
    """CoOp-style learnable prompt tokens ``{t1,...,tM,[CLASS]}``."""

    def __init__(
        self,
        classnames: Sequence[str],
        clip_model: nn.Module,
        tokenize: TokenizeFn,
        n_ctx: int = 16,
        ctx_init: str | None = DEFAULT_CONTEXT_INIT,
    ) -> None:
        super().__init__()
        if n_ctx < 1:
            raise ValueError("n_ctx must be positive.")

        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(self.classnames)
        self.n_ctx = n_ctx
        self.tokenize = tokenize
        dtype = _clip_dtype(clip_model)
        device = _module_device(clip_model)
        ctx_dim = clip_model.token_embedding.weight.shape[1]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        if ctx_init:
            tokenized = tokenize(ctx_init).to(device)
            with torch.no_grad():
                init_embedding = clip_model.token_embedding(tokenized).float()[0]
            eot_idx = int(tokenized[0].argmax().item())
            init_len = max(0, min(n_ctx, eot_idx - 1))
            if init_len > 0:
                ctx_vectors[:init_len].copy_(init_embedding[1 : 1 + init_len])

        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in self.classnames]
        tokenized_prompts = tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype=dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

    def forward(self) -> Tensor:
        ctx = self.ctx.to(dtype=self.token_prefix.dtype)
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class FeatureAdapter(nn.Module):
    """Single-linear feature adapter used for ``f_e`` and ``f_s``."""

    def __init__(self, dim: int, identity_init: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        if identity_init:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DiversityAugmentation(nn.Module):
    """Text-based diversity augmentation, Eq. (17)-(19)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, causal_features: Tensor, text_features: Tensor) -> Tensor:
        """Fuse each causal feature with K diverse text embeddings.

        Args:
            causal_features: ``[B, D]`` tensor.
            text_features: either ``[K, D]`` shared by the batch or
                ``[B, K, D]`` sample-specific text features.

        Returns:
            Augmented features ``z`` with shape ``[B, K, D]``.
        """

        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(0).expand(causal_features.shape[0], -1, -1)
        if text_features.ndim != 3:
            raise ValueError(f"text_features must be [K, D] or [B, K, D], got {text_features.shape}.")
        if text_features.shape[0] != causal_features.shape[0]:
            raise ValueError("Batch size of causal_features and text_features does not match.")

        batch, num_templates, dim = text_features.shape
        query = causal_features[:, None, :].expand(-1, num_templates, -1).reshape(batch * num_templates, 1, dim)
        key_value = text_features.reshape(batch * num_templates, 1, dim)
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        z = self.norm(query + self.dropout(attn_out))
        return z.squeeze(1).reshape(batch, num_templates, dim)


class CIPTModel(nn.Module):
    """Core CIPT network built on OpenAI CLIP."""

    def __init__(
        self,
        clip_model: nn.Module,
        classnames: Sequence[str],
        tokenize: TokenizeFn,
        n_ctx: int = 16,
        ctx_init: str | None = DEFAULT_CONTEXT_INIT,
        templates: Sequence[str] = IMAGENET_TEMPLATES,
        num_diverse_templates: int = 4,
        num_heads: int = 8,
        sample_templates: bool = True,
    ) -> None:
        super().__init__()
        if len(classnames) == 0:
            raise ValueError("classnames cannot be empty.")
        if num_diverse_templates < 1:
            raise ValueError("num_diverse_templates must be positive.")

        self.clip_model = clip_model
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.templates = list(templates)
        self.num_diverse_templates = num_diverse_templates
        self.sample_templates = sample_templates
        self.class_conditioned_templates = _has_class_placeholder(self.templates)
        self.tokenize = tokenize

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        dim = clip_model.visual.output_dim
        self.text_encoder = OpenAITextEncoder(clip_model)
        self.prompt_learner = PromptLearner(self.classnames, clip_model, tokenize, n_ctx=n_ctx, ctx_init=ctx_init)
        self.causal_adapter = FeatureAdapter(dim)
        self.spurious_adapter = FeatureAdapter(dim)
        self.diversity_augmentation = DiversityAugmentation(dim, num_heads=num_heads)

        diverse_features = self._build_diverse_text_features()
        self.register_buffer("diverse_text_features", diverse_features, persistent=False)

    @property
    def logit_scale(self) -> Tensor:
        return self.clip_model.logit_scale.exp().float()

    def train(self, mode: bool = True) -> "CIPTModel":
        super().train(mode)
        self.clip_model.eval()
        self.text_encoder.eval()
        return self

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    @torch.no_grad()
    def _build_diverse_text_features(self, batch_size: int = 256) -> Tensor:
        device = _module_device(self.clip_model)
        if self.class_conditioned_templates:
            texts = [
                _format_template(template, class_name)
                for class_name in self.classnames
                for template in self.templates
            ]
            out_shape = (len(self.classnames), len(self.templates), -1)
        else:
            texts = list(self.templates)
            out_shape = (len(self.templates), -1)

        features = []
        for start in range(0, len(texts), batch_size):
            tokenized = self.tokenize(texts[start : start + batch_size]).to(device)
            encoded = self.clip_model.encode_text(tokenized).float()
            features.append(F.normalize(encoded, dim=-1))
        return torch.cat(features, dim=0).reshape(out_shape)

    def _select_template_indices(self, num_available: int, device: torch.device, indices: Tensor | None = None) -> Tensor:
        k = min(self.num_diverse_templates, num_available)
        if indices is not None:
            indices = indices.to(device=device, dtype=torch.long)
            if indices.numel() > k:
                indices = indices[:k]
            return indices
        if self.training and self.sample_templates:
            return torch.randperm(num_available, device=device)[:k]
        return torch.arange(k, device=device)

    def _select_diverse_features(self, labels: Tensor | None = None, indices: Tensor | None = None) -> Tensor:
        bank = self.diverse_text_features
        device = bank.device
        if self.class_conditioned_templates:
            idx = self._select_template_indices(bank.shape[1], device, indices)
            selected = bank.index_select(1, idx)
            if labels is None:
                return selected
            return selected[labels.to(device=device, dtype=torch.long)]

        idx = self._select_template_indices(bank.shape[0], device, indices)
        return bank.index_select(0, idx)

    def encode_image_features(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            image_features = self.clip_model.encode_image(
                images.to(device=_module_device(self.clip_model), dtype=_clip_dtype(self.clip_model))
            )
        return F.normalize(image_features.float(), dim=-1)

    def encode_prompt_features(self) -> Tensor:
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return F.normalize(text_features.float(), dim=-1)

    def _logits(self, features: Tensor, text_features: Tensor) -> Tensor:
        features = F.normalize(features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        if features.ndim == 2:
            return self.logit_scale * features @ text_features.t()
        if features.ndim == 3:
            return self.logit_scale * torch.einsum("bkd,cd->bkc", features, text_features)
        raise ValueError(f"Expected features with shape [B, D] or [B, K, D], got {features.shape}.")

    def forward(self, images: Tensor, labels: Tensor | None = None, template_indices: Tensor | None = None) -> CIPTOutput:
        image_features = self.encode_image_features(images)
        text_features = self.encode_prompt_features()
        causal_features = self.causal_adapter(image_features)
        spurious_features = self.spurious_adapter(image_features)
        causal_logits = self._logits(causal_features, text_features)
        spurious_logits = self._logits(spurious_features, text_features)

        augmented_features = None
        interventional_logits = None
        if self.class_conditioned_templates and labels is None:
            return CIPTOutput(
                interventional_logits=interventional_logits,
                causal_logits=causal_logits,
                spurious_logits=spurious_logits,
                image_features=image_features,
                causal_features=causal_features,
                spurious_features=spurious_features,
                text_features=text_features,
                augmented_features=augmented_features,
            )

        diverse_features = self._select_diverse_features(labels=labels, indices=template_indices)
        augmented_features = self.diversity_augmentation(causal_features, diverse_features.float())
        interventional_logits = self._logits(augmented_features, text_features)
        return CIPTOutput(
            interventional_logits=interventional_logits,
            causal_logits=causal_logits,
            spurious_logits=spurious_logits,
            image_features=image_features,
            causal_features=causal_features,
            spurious_features=spurious_features,
            text_features=text_features,
            augmented_features=augmented_features,
        )

    @torch.no_grad()
    def predict(self, images: Tensor, use_tda: bool = True, template_indices: Tensor | None = None) -> Tensor:
        """Return classification logits.

        If the diversity templates contain class placeholders, inference scores
        each candidate class with its own template embeddings and averages over
        K templates. If templates are class-agnostic, it averages the TDA logits
        over K templates.
        """

        was_training = self.training
        self.eval()
        image_features = self.encode_image_features(images)
        text_features = self.encode_prompt_features()
        causal_features = self.causal_adapter(image_features)

        if not use_tda:
            logits = self._logits(causal_features, text_features)
            self.train(was_training)
            return logits

        diverse_features = self._select_diverse_features(labels=None, indices=template_indices)
        if self.class_conditioned_templates:
            num_classes, num_templates, dim = diverse_features.shape
            batch = causal_features.shape[0]
            causal_flat = causal_features[:, None, :].expand(batch, num_classes, dim).reshape(batch * num_classes, dim)
            diverse_flat = (
                diverse_features[None, :, :, :]
                .expand(batch, num_classes, num_templates, dim)
                .reshape(batch * num_classes, num_templates, dim)
            )
            z = self.diversity_augmentation(causal_flat, diverse_flat.float()).reshape(
                batch, num_classes, num_templates, dim
            )
            z = F.normalize(z.float(), dim=-1)
            text = F.normalize(text_features.float(), dim=-1)
            logits = self.logit_scale * torch.einsum("bckd,cd->bck", z, text).mean(dim=-1)
        else:
            z = self.diversity_augmentation(causal_features, diverse_features.float())
            logits = self._logits(z, text_features).mean(dim=1)

        self.train(was_training)
        return logits


def load_openai_clip(
    backbone: str = "ViT-B/16",
    device: str | torch.device | None = None,
    jit: bool = False,
    download_root: str | None = None,
):
    """Load OpenAI CLIP and return ``(model, preprocess, tokenize)``."""

    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is required. Install it with "
            "`pip install git+https://github.com/openai/CLIP.git`."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(backbone, device=device, jit=jit, download_root=download_root)
    return model, preprocess, clip.tokenize


def build_cipt(
    classnames: Sequence[str],
    backbone: str = "ViT-B/16",
    device: str | torch.device | None = None,
    jit: bool = False,
    download_root: str | None = None,
    **kwargs,
) -> tuple[CIPTModel, object]:
    """Convenience constructor using OpenAI CLIP pretrained weights."""

    clip_model, preprocess, tokenize = load_openai_clip(
        backbone=backbone,
        device=device,
        jit=jit,
        download_root=download_root,
    )
    model = CIPTModel(clip_model, classnames=classnames, tokenize=tokenize, **kwargs)
    model.to(_module_device(clip_model))
    return model, preprocess
