"""Standalone TPAMI CIPT baseline adapted to the DomainBed training interface.

This module intentionally contains no DCCL components: no second image view,
no SupCon/projection head, no PMA/GT-style losses, and no SWAD-specific model.
The core implementation follows the public CIPT code path:
  frozen OpenAI CLIP -> causal/spurious linear adapters -> prompt learner ->
  class-conditioned ImageNet intervention templates -> TDA ->
  L_cls + beta * L_de + gamma * L_ind.
"""

import torch
from torch import nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import Algorithm
from domainbed.algorithms.cipt_losses import (
    classification_loss,
    decomposition_loss,
    independence_loss,
)
from domainbed.algorithms.cipt_prompt import (
    B5B_IMAGENET_TEMPLATES,
    CLIPTextEncoder,
    PromptLearner,
    load_frozen_clip,
)


class FeatureAdapter(nn.Module):
    """Official CIPT single-linear adapter with identity initialization."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DiversityAugmentation(nn.Module):
    """CIPT text diversity augmentation: LN(e + Attention(e, p_k, p_k))."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim={} must be divisible by num_heads={}".format(dim, num_heads))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, causal_features, text_features):
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(0).expand(causal_features.shape[0], -1, -1)
        if text_features.ndim != 3:
            raise ValueError("text_features must be [K,D] or [B,K,D], got {}".format(tuple(text_features.shape)))
        if text_features.shape[0] != causal_features.shape[0]:
            raise ValueError("causal/text batch sizes do not match")

        batch, k, dim = text_features.shape
        query = causal_features[:, None, :].expand(-1, k, -1).reshape(batch * k, 1, dim)
        key_value = text_features.reshape(batch * k, 1, dim)
        attended, _ = self.attn(query, key_value, key_value, need_weights=False)
        z = self.norm(query + self.dropout(attended))
        return z.squeeze(1).reshape(batch, k, dim)


class CIPT(Algorithm):
    """Official CIPT baseline selectable with ``--algorithm CIPT``."""

    use_official_clip_preprocess = True
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        class_names = list(hparams.get("cipt_class_names", ["class {}".format(i) for i in range(num_classes)]))
        if len(class_names) != num_classes:
            raise ValueError("CIPT class-name count {} != num_classes {}".format(len(class_names), num_classes))

        self.beta = float(hparams.get("cipt_beta", 4.0))
        self.gamma = float(hparams.get("cipt_gamma", 5.0))
        self.k = int(hparams.get("cipt_k", 4))
        self.debug_shapes = bool(hparams.get("cipt_debug_shapes", False))

        self.clip_model, self.tokenize = load_frozen_clip(
            hparams.get("cipt_clip_backbone", "ViT-B/16"),
            hparams.get("cipt_clip_path", ""),
        )
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        dim = int(self.clip_model.visual.output_dim)
        self.text_encoder = CLIPTextEncoder(self.clip_model)
        self.prompt_learner = PromptLearner(
            class_names,
            self.clip_model,
            self.tokenize,
            int(hparams.get("cipt_prompt_length", 16)),
            hparams.get("cipt_prompt_init", "a photo of a"),
        )
        self.causal_adapter = FeatureAdapter(dim)
        self.spurious_adapter = FeatureAdapter(dim)
        self.tda = DiversityAugmentation(dim, num_heads=int(hparams.get("cipt_tda_heads", 8)))

        self.class_names = [name.replace("_", " ") for name in class_names]
        self.register_buffer("diverse_text_features", self._build_diverse_text_features(), persistent=False)

        trainable = [p for p in self.parameters() if p.requires_grad]
        lr = float(hparams.get("cipt_lr", 2.5e-3))
        weight_decay = float(hparams.get("cipt_weight_decay", 0.0))
        self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

        total_steps = max(1, int(hparams.get("cipt_total_steps", 1)))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=0.0)

        print(
            "Official CIPT: backbone={}, beta={}, gamma={}, K={}, heads={}, lr={}, wd={}, schedule_steps={}".format(
                hparams.get("cipt_clip_backbone", "ViT-B/16"), self.beta, self.gamma, self.k,
                int(hparams.get("cipt_tda_heads", 8)), lr, weight_decay, total_steps
            )
        )

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def _build_diverse_text_features(self, batch_size=256):
        device = next(self.clip_model.parameters()).device
        texts = [
            template.format(class_name)
            for class_name in self.class_names
            for template in B5B_IMAGENET_TEMPLATES
        ]
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenize(texts[start : start + batch_size]).to(device)
            encoded = self.clip_model.encode_text(tokens).float()
            chunks.append(F.normalize(encoded, dim=-1))
        bank = torch.cat(chunks, dim=0)
        return bank.reshape(len(self.class_names), len(B5B_IMAGENET_TEMPLATES), -1)

    def _select_template_indices(self):
        num_available = self.diverse_text_features.shape[1]
        k = min(self.k, num_available)
        device = self.diverse_text_features.device
        if self.training:
            return torch.randperm(num_available, device=device)[:k]
        return torch.arange(k, device=device)

    def _select_diverse_features(self, labels=None):
        idx = self._select_template_indices()
        selected = self.diverse_text_features.index_select(1, idx)
        if labels is None:
            return selected
        labels = labels.to(device=selected.device, dtype=torch.long)
        return selected[labels]

    def _encode_image(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images.to(dtype=self.clip_model.dtype)).float()
        return F.normalize(features, dim=-1)

    def _class_features(self):
        features = self.text_encoder(self.prompt_learner(), self.prompt_learner.tokenized_prompts)
        return F.normalize(features.float(), dim=-1)

    def _logits(self, features, text_features):
        features = F.normalize(features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        if features.ndim == 2:
            return scale * features @ text_features.t()
        if features.ndim == 3:
            return scale * torch.einsum("bkd,cd->bkc", features, text_features)
        raise ValueError("Expected [B,D] or [B,K,D], got {}".format(tuple(features.shape)))

    def update(self, x, y, **kwargs):
        images = torch.cat(x)
        labels = torch.cat(y)

        image_features = self._encode_image(images)
        text_features = self._class_features()
        causal_features = self.causal_adapter(image_features)
        spurious_features = self.spurious_adapter(image_features)

        causal_logits = self._logits(causal_features, text_features)
        spurious_logits = self._logits(spurious_features, text_features)
        loss_de = decomposition_loss(causal_logits, spurious_logits, labels)
        loss_ind = independence_loss(causal_features, spurious_features)

        diverse_features = self._select_diverse_features(labels)
        augmented_features = self.tda(causal_features, diverse_features.float())
        interventional_logits = self._logits(augmented_features, text_features)
        loss_cls = classification_loss(interventional_logits, labels)

        total = loss_cls + self.beta * loss_de + self.gamma * loss_ind

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.debug_shapes:
            print(
                "CIPT shapes: image={} e={} s={} p={} z={} logits={}".format(
                    tuple(image_features.shape), tuple(causal_features.shape), tuple(spurious_features.shape),
                    tuple(diverse_features.shape), tuple(augmented_features.shape), tuple(interventional_logits.shape)
                )
            )

        return {
            "loss": total.item(),
            "total_loss": total.item(),
            "cipt_cls_loss": loss_cls.item(),
            "cipt_de_loss": loss_de.item(),
            "cipt_ind_loss": loss_ind.item(),
            "cipt_lr": self.optimizer.param_groups[0]["lr"],
            "mean_e_norm": causal_features.norm(dim=-1).mean().item(),
            "mean_s_norm": spurious_features.norm(dim=-1).mean().item(),
            "mean_es_cosine": F.cosine_similarity(causal_features, spurious_features, dim=-1).mean().item(),
            "ce_loss": loss_cls.item(),
            "sup_cl_loss": 0.0,
            "pre_cl_loss": 0.0,
        }

    @torch.no_grad()
    def predict(self, x):
        was_training = self.training
        self.eval()

        image_features = self._encode_image(x)
        causal_features = self.causal_adapter(image_features)
        text_features = self._class_features()
        diverse_features = self._select_diverse_features(labels=None)

        num_classes, num_templates, dim = diverse_features.shape
        batch = causal_features.shape[0]
        causal_flat = causal_features[:, None, :].expand(batch, num_classes, dim).reshape(batch * num_classes, dim)
        context_flat = (
            diverse_features[None, :, :, :]
            .expand(batch, num_classes, num_templates, dim)
            .reshape(batch * num_classes, num_templates, dim)
        )
        z = self.tda(causal_flat, context_flat.float()).reshape(batch, num_classes, num_templates, dim)
        z = F.normalize(z.float(), dim=-1)
        text = F.normalize(text_features.float(), dim=-1)
        scale = self.clip_model.logit_scale.exp().detach().float()
        logits = scale * torch.einsum("bckd,cd->bck", z, text).mean(dim=-1)

        self.train(was_training)
        return logits

    @torch.no_grad()
    def predict_embed(self, x):
        image_features = self._encode_image(x)
        return self.causal_adapter(image_features)
