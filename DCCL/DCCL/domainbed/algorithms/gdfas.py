"""Faithful multi-class DomainBed port of GD-FAS (ICCV 2025).

Jung et al., Group-wise Scaling and Orthogonal Decomposition for
Domain-Invariant Feature Extraction in Face Anti-Spoofing.
https://github.com/SeungjinJung/GD-FAS

Unchanged training components: CLIP ViT-B/16 trainable image/text encoders,
six prompts per class ensembled *before* text normalization, cosine IT CE +
group-wise scaling, Gram-Schmidt FOD on raw image embeddings, same-domain
supervised contrast on the raw orthogonal residual, 512-4096-4096-256 BN MLP,
paired-view instance contrast, and auxiliary CE. The paper's inference uses
raw CLIP image-text cosine logits (NOT projected E-only logits).

Necessary application changes: 2 -> C classes, data from DomainBed minibatches,
class names/prompts adapted to natural object recognition, missing-group
handling, device-independent masks, and official two-view augmentation exposed
by DomainBed's dataset adapter.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import Algorithm
from domainbed.algorithms.cipt_prompt import clip
from domainbed.algorithms.gdfas_losses import (
    fod_decomposition, fod_domain_loss, groupwise_scaling_loss,
    paired_image_instance_loss,
)


CLASS_PROMPTS = (
    "This is an example of a {}",
    "This is a {}",
    "This is a photo of a {}",
    "This is how a {} looks like",
    "a photo of a {}",
    "a picture showing a {}",
)


class GDFAS(Algorithm):
    """Full GD-FAS objective adapted to N-class domain generalization."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError("GDFAS requires RGB image input")
        if num_domains < 2:
            raise ValueError("GDFAS's groupwise scaling requires >=2 source domains")
        name = str(hparams.get("gdfas_clip_path", "")).strip() or str(
            hparams.get("gdfas_clip_backbone", "ViT-B/16")
        )
        model, _ = clip.load(name, device="cpu", jit=False)
        # OpenAI CLIP loaded on CPU is fp32, keeping optimizer gradients fp32.
        self.clip_model = model.float()
        d = int(self.clip_model.visual.output_dim)
        if d != 512:
            raise ValueError("Original GD-FAS auxiliary head expects 512-dim ViT-B/16 features")
        class_names = hparams.get("gdfas_class_names")
        if not class_names or len(class_names) != num_classes:
            raise ValueError("GDFAS requires dataset class names (gdfas_class_names)")
        prompts = [template.format(c.replace("_", " ")) for c in class_names
                   for template in CLASS_PROMPTS]
        tokens = clip.tokenize(prompts)
        self.register_buffer("class_tokens", tokens, persistent=True)
        self.prompts_per_class = len(CLASS_PROMPTS)
        self.image_mlp = nn.Sequential(
            nn.Linear(d, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256, num_classes, bias=True)
        self.fod_weight = float(hparams.get("gdfas_fod_weight", 0.8))
        self.ii_weight = float(hparams.get("gdfas_ii_weight", 0.1))
        self.aux_weight = float(hparams.get("gdfas_aux_weight", 1.0))
        self.use_gs = bool(hparams.get("gdfas_gs", True))
        self.temperature = float(hparams.get("gdfas_temperature", 0.1))
        self.beta = float(hparams.get("gdfas_beta", 1.5))
        self.gs_alpha = math.log(num_domains) / 2.0
        if self.temperature <= 0:
            raise ValueError("gdfas_temperature must be positive")
        # Official run defaults: Adam(lr=3e-6, wd=1e-6), entire CLIP trainable.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(hparams.get("lr", 3e-6)),
            weight_decay=float(hparams.get("weight_decay", 1e-6))
        )
        self.lr_step_size = int(hparams.get("gdfas_lr_step_size", 10))
        self.lr_gamma = float(hparams.get("gdfas_lr_gamma", 0.1))
        self.lr_steps_per_epoch = int(hparams.get("gdfas_lr_steps_per_epoch", 10))
        self._update_count = 0
        print("GDFAS: C={}, source domains={}, prompts/class={}, "
              "CLIP fine-tuned=True, FOD={}, GS={}, II={}, auxiliary={}.".format(
                  num_classes, num_domains, self.prompts_per_class,
                  self.fod_weight, self.use_gs, self.ii_weight, self.aux_weight))

    def _text_embeddings(self):
        text = self.clip_model.encode_text(self.class_tokens)
        return text.view(self.num_classes, self.prompts_per_class, -1).mean(dim=1)

    def _similarity_logits(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return self.clip_model.logit_scale.exp() * (image_features @ text_features.t())

    def update(self, x, y, **kwargs):
        # Every source domain provides two independently augmented views of the
        # same image; order is [all first views, all second views] (official).
        if "x_2" not in kwargs:
            raise RuntimeError("GDFAS requires independent second views x_2")
        first = torch.cat(x, dim=0)
        second = torch.cat(kwargs["x_2"], dim=0)
        labels = torch.cat(y, dim=0).long()
        domains = torch.cat([
            torch.full((len(yi),), i, device=yi.device, dtype=torch.long)
            for i, yi in enumerate(y)
        ], dim=0)
        images = torch.cat([first, second], dim=0)
        labels = labels.repeat(2)
        domains = domains.repeat(2)

        image_features = self.clip_model.encode_image(images).float()
        text_features = self._text_embeddings().float()
        logits = self._similarity_logits(image_features, text_features)

        if self.use_gs:
            loss_it = groupwise_scaling_loss(
                logits, labels, domains, beta=self.beta, alpha=self.gs_alpha
            )
        else:
            loss_it = F.cross_entropy(logits, labels)

        # Both Gram-Schmidt and the detached projection reproduce the released
        # FOD code; domain loss is on S, not the invariant projection E.
        invariant, specific, basis = fod_decomposition(
            image_features, text_features, detach_projection=True
        )
        loss_fod = fod_domain_loss(
            specific, domains, temperature=self.temperature,
            official_reduction=True
        )
        z = self.image_mlp(image_features)
        loss_ii = paired_image_instance_loss(z, temperature=0.8)
        loss_aux = F.cross_entropy(self.classifier(z), labels)

        total = (
            loss_it + self.fod_weight * loss_fod
            + self.ii_weight * loss_ii + self.aux_weight * loss_aux
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self._update_count += 1
        # Official outer script performs StepLR every 10 iterations. The
        # schedule is optional here because DomainBed often uses >5000 steps.
        if self.lr_step_size > 0 and self.lr_steps_per_epoch > 0:
            interval = self.lr_step_size * self.lr_steps_per_epoch
            if self._update_count % interval == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] *= self.lr_gamma
        with torch.no_grad():
            cosine = F.cosine_similarity(invariant, specific, dim=-1).abs().mean()
        return {
            "loss": float(total.detach()),
            "it_loss": float(loss_it.detach()),
            "fod_loss": float(loss_fod.detach()),
            "ii_loss": float(loss_ii.detach()),
            "aux_loss": float(loss_aux.detach()),
            "basis_rank": float(basis.shape[0]),
            "orth_cos": float(cosine.detach()),
        }

    def predict(self, x):
        image_features = self.clip_model.encode_image(x).float()
        text_features = self._text_embeddings().float()
        # Original paper: use the IT classifier, NOT the auxiliary MLP.
        return self._similarity_logits(image_features, text_features)

    def predict_embed(self, x):
        return self.clip_model.encode_image(x).float()
