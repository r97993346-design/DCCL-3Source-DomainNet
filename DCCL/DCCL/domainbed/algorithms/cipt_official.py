from __future__ import absolute_import

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CONTEXT_INIT = "a photo of a"

# Official CIPT ImageNet diversity templates.
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def _import_local_clip():
    """Import OpenAI CLIP, preferring the repository-local copy."""
    try:
        import clip
        return clip
    except ImportError:
        repo_root = Path(__file__).resolve().parents[4]
        clip_root = repo_root / "CLIP"
        if not clip_root.exists():
            raise ImportError(
                "CIPT requires OpenAI CLIP. The repository-local CLIP directory "
                "was not found and the clip package is not installed."
            )
        clip_root = str(clip_root)
        if clip_root not in sys.path:
            sys.path.insert(0, clip_root)
        import clip
        return clip


def load_frozen_clip(model_name="ViT-B/16", model_path="", download_root=""):
    """Load the frozen OpenAI CLIP model used by official CIPT text modules."""
    clip = _import_local_clip()
    source = model_path if model_path else model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(
        source,
        device=device,
        jit=False,
        download_root=download_root if download_root else None,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, clip.tokenize


def _clip_dtype(clip_model):
    try:
        return clip_model.dtype
    except (AttributeError, RuntimeError):
        return next(clip_model.parameters()).dtype


def _module_device(module):
    return next(module.parameters()).device


def _format_template(template, class_name):
    clean_name = class_name.replace("_", " ")
    if "{class}" in template:
        return template.format(**{"class": clean_name})
    if "{}" in template:
        return template.format(clean_name)
    return template


def _has_class_placeholder(templates):
    return any(("{}" in t) or ("{class}" in t) for t in templates)


class OpenAITextEncoder(nn.Module):
    """Official CIPT wrapper for learnable prompt embeddings."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = _clip_dtype(clip_model)

    def forward(self, prompts, tokenized_prompts):
        x = prompts.to(dtype=self.dtype)
        pos = self.positional_embedding[: x.shape[1]].to(
            device=x.device, dtype=self.dtype
        )
        x = x + pos
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=self.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            eot_indices,
        ]
        return x @ self.text_projection


class PromptLearner(nn.Module):
    """Official CoOp-style CIPT prompt learner."""

    def __init__(
        self,
        classnames,
        clip_model,
        tokenize,
        n_ctx=16,
        ctx_init=DEFAULT_CONTEXT_INIT,
    ):
        super().__init__()
        if n_ctx < 1:
            raise ValueError("n_ctx must be positive")

        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(self.classnames)
        self.n_ctx = int(n_ctx)
        self.tokenize = tokenize

        dtype = _clip_dtype(clip_model)
        device = _module_device(clip_model)
        ctx_dim = clip_model.token_embedding.weight.shape[1]

        ctx_vectors = torch.empty(
            self.n_ctx, ctx_dim, dtype=torch.float32, device=device
        )
        nn.init.normal_(ctx_vectors, std=0.02)

        if ctx_init:
            tokenized = tokenize(ctx_init).to(device)
            with torch.no_grad():
                init_embedding = clip_model.token_embedding(tokenized).float()[0]
            eot_idx = int(tokenized[0].argmax().item())
            init_len = max(0, min(self.n_ctx, eot_idx - 1))
            if init_len > 0:
                ctx_vectors[:init_len].copy_(
                    init_embedding[1 : 1 + init_len]
                )

        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompts = [
            "{} {}.".format(prompt_prefix, name)
            for name in self.classnames
        ]
        tokenized_prompts = tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(
                dtype=dtype
            )

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )

    def forward(self):
        ctx = self.ctx.to(dtype=self.token_prefix.dtype)
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat(
            [self.token_prefix, ctx, self.token_suffix], dim=1
        )


class FeatureAdapter(nn.Module):
    """Official CIPT single-linear adapter for causal/spurious features."""

    def __init__(self, dim, identity_init=True):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        if identity_init:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DiversityAugmentation(nn.Module):
    """Official CIPT TDA, with a torch-1.7 compatible attention call."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "dim={} must be divisible by num_heads={}".format(
                    dim, num_heads
                )
            )

        # Official CIPT uses batch_first=True. torch==1.7 (the DCCL
        # requirement) does not expose that argument, so the fallback below
        # transposes [N,L,E] <-> [L,N,E] while preserving identical semantics.
        try:
            self.attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self._batch_first = True
        except TypeError:
            self.attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
            )
            self._batch_first = False

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, causal_features, text_features):
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0).expand(
                causal_features.shape[0], -1, -1
            )
        if text_features.dim() != 3:
            raise ValueError(
                "text_features must be [K,D] or [B,K,D], got {}".format(
                    text_features.shape
                )
            )
        if text_features.shape[0] != causal_features.shape[0]:
            raise ValueError(
                "Batch size of causal_features and text_features does not match"
            )

        batch, num_templates, dim = text_features.shape
        query = (
            causal_features[:, None, :]
            .expand(-1, num_templates, -1)
            .reshape(batch * num_templates, 1, dim)
        )
        key_value = text_features.reshape(
            batch * num_templates, 1, dim
        )

        if self._batch_first:
            attn_out, _ = self.attn(
                query, key_value, key_value, need_weights=False
            )
        else:
            query_t = query.transpose(0, 1)
            key_value_t = key_value.transpose(0, 1)
            attn_out, _ = self.attn(
                query_t,
                key_value_t,
                key_value_t,
                need_weights=False,
            )
            attn_out = attn_out.transpose(0, 1)

        z = self.norm(query + self.dropout(attn_out))
        return z.squeeze(1).reshape(batch, num_templates, dim)


def tda_classification_loss(logits, labels):
    """Official CIPT Eq. (21): CE averaged over K interventions."""
    if logits.dim() != 3:
        raise ValueError(
            "Expected logits [B,K,C], got {}".format(logits.shape)
        )
    batch, num_templates, num_classes = logits.shape
    repeated_labels = labels[:, None].expand(
        batch, num_templates
    ).reshape(-1)
    return F.cross_entropy(
        logits.reshape(batch * num_templates, num_classes),
        repeated_labels,
    )


def decomposition_loss(causal_logits, spurious_logits, labels):
    """Official CIPT Eq. (11): causal CE + KL(uniform || spurious)."""
    if causal_logits.shape != spurious_logits.shape:
        raise ValueError(
            "Causal and spurious logits must have the same shape"
        )
    num_classes = causal_logits.shape[-1]
    causal_ce = F.cross_entropy(causal_logits, labels)
    log_spurious = F.log_softmax(spurious_logits, dim=-1)
    uniform = torch.full_like(
        log_spurious, fill_value=1.0 / float(num_classes)
    )
    spurious_kl = F.kl_div(
        log_spurious, uniform, reduction="batchmean"
    )
    return causal_ce + spurious_kl, causal_ce, spurious_kl


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """Official CIPT Eq. (14)-(15): squared cosine correlation."""
    cov = F.cosine_similarity(
        causal_features, spurious_features, dim=-1, eps=eps
    )
    return 0.5 * cov.pow(2).mean()


def cipt_loss(
    interventional_logits,
    causal_logits,
    spurious_logits,
    causal_features,
    spurious_features,
    labels,
    beta=2.0,
    gamma=5.0,
):
    """Official CIPT Eq. (22): L_c + beta*L_de + gamma*L_ind."""
    classification = tda_classification_loss(
        interventional_logits, labels
    )
    decomposition, causal_ce, spurious_kl = decomposition_loss(
        causal_logits, spurious_logits, labels
    )
    independence = independence_loss(
        causal_features, spurious_features
    )
    total = classification + beta * decomposition + gamma * independence
    return {
        "loss": total,
        "classification": classification,
        "decomposition": decomposition,
        "independence": independence,
        "causal_ce": causal_ce,
        "spurious_kl": spurious_kl,
    }


class OfficialCIPTAuxiliary(nn.Module):
    """Official CIPT graph with image encoding replaced by an input feature.

    Everything after CIPT's image-feature extraction is kept faithful to the
    official implementation: prompt learner, causal/spurious adapters, cosine
    classifier, class-conditioned template bank, TDA, and losses. The caller is
    responsible only for supplying an image representation in CLIP's output
    dimension; in DCCL this is produced by the fusion bridge.
    """

    def __init__(
        self,
        clip_model,
        classnames,
        tokenize,
        n_ctx=16,
        ctx_init=DEFAULT_CONTEXT_INIT,
        templates=IMAGENET_TEMPLATES,
        num_diverse_templates=4,
        num_heads=8,
        dropout=0.0,
        sample_templates=True,
    ):
        super().__init__()
        if len(classnames) == 0:
            raise ValueError("classnames cannot be empty")
        if num_diverse_templates < 1:
            raise ValueError("num_diverse_templates must be positive")

        self.clip_model = clip_model
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.templates = list(templates)
        self.num_diverse_templates = int(num_diverse_templates)
        self.sample_templates = bool(sample_templates)
        self.class_conditioned_templates = _has_class_placeholder(
            self.templates
        )
        self.tokenize = tokenize

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.dim = int(clip_model.visual.output_dim)
        self.text_encoder = OpenAITextEncoder(clip_model)
        self.prompt_learner = PromptLearner(
            self.classnames,
            clip_model,
            tokenize,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
        )
        self.causal_adapter = FeatureAdapter(self.dim)
        self.spurious_adapter = FeatureAdapter(self.dim)
        self.diversity_augmentation = DiversityAugmentation(
            self.dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        diverse_features = self._build_diverse_text_features()
        self.register_buffer(
            "diverse_text_features",
            diverse_features,
            persistent=False,
        )

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale.exp().float()

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.text_encoder.eval()
        return self

    def trainable_parameters(self):
        return (
            param
            for param in self.parameters()
            if param.requires_grad
        )

    @torch.no_grad()
    def _build_diverse_text_features(self, batch_size=256):
        device = _module_device(self.clip_model)

        if self.class_conditioned_templates:
            texts = [
                _format_template(template, class_name)
                for class_name in self.classnames
                for template in self.templates
            ]
            out_shape = (
                len(self.classnames),
                len(self.templates),
                -1,
            )
        else:
            texts = list(self.templates)
            out_shape = (len(self.templates), -1)

        features = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenize(
                texts[start : start + batch_size]
            ).to(device)
            encoded = self.clip_model.encode_text(tokens).float()
            features.append(F.normalize(encoded, dim=-1))

        return torch.cat(features, dim=0).reshape(out_shape)

    def _select_template_indices(self, num_available, device, indices=None):
        k = min(self.num_diverse_templates, num_available)
        if indices is not None:
            indices = indices.to(device=device, dtype=torch.long)
            if indices.numel() > k:
                indices = indices[:k]
            return indices
        if self.training and self.sample_templates:
            return torch.randperm(num_available, device=device)[:k]
        return torch.arange(k, device=device)

    def _select_diverse_features(self, labels=None, indices=None):
        bank = self.diverse_text_features
        device = bank.device

        if self.class_conditioned_templates:
            idx = self._select_template_indices(
                bank.shape[1], device, indices
            )
            selected = bank.index_select(1, idx)
            if labels is None:
                return selected
            return selected[
                labels.to(device=device, dtype=torch.long)
            ]

        idx = self._select_template_indices(
            bank.shape[0], device, indices
        )
        return bank.index_select(0, idx)

    def encode_prompt_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(
            prompts, tokenized_prompts
        )
        return F.normalize(text_features.float(), dim=-1)

    def _logits(self, features, text_features):
        features = F.normalize(features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        if features.dim() == 2:
            return self.logit_scale * features @ text_features.t()
        if features.dim() == 3:
            return self.logit_scale * torch.einsum(
                "bkd,cd->bkc", features, text_features
            )
        raise ValueError(
            "Expected features [B,D] or [B,K,D], got {}".format(
                features.shape
            )
        )

    def forward(self, image_features, labels, template_indices=None):
        # Official CIPT normalize-after-image-encoder behavior. In the DCCL
        # integration, image_features are supplied by the DCCL->CIPT bridge.
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = self.encode_prompt_features()

        causal_features = self.causal_adapter(image_features)
        spurious_features = self.spurious_adapter(image_features)
        causal_logits = self._logits(
            causal_features, text_features
        )
        spurious_logits = self._logits(
            spurious_features, text_features
        )

        diverse_features = self._select_diverse_features(
            labels=labels,
            indices=template_indices,
        )
        augmented_features = self.diversity_augmentation(
            causal_features, diverse_features.float()
        )
        interventional_logits = self._logits(
            augmented_features, text_features
        )

        return {
            "interventional_logits": interventional_logits,
            "causal_logits": causal_logits,
            "spurious_logits": spurious_logits,
            "image_features": image_features,
            "causal_features": causal_features,
            "spurious_features": spurious_features,
            "text_features": text_features,
            "augmented_features": augmented_features,
        }
