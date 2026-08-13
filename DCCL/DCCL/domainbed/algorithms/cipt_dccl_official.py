"""Strict official-alignment wrapper for the feature/multiprompt CIPTDCCL path."""

import copy

import torch
from torch import nn
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl import CIPTDCCL as _CIPTDCCLBase


class CIPTDCCLForwardModel(nn.Module):
    """Inference-only CIPTDCCL view used by SWAD.

    Only trainable parameters that affect prediction are registered:
    causal decomposition, prompt context and TDA. Frozen CLIP/text encoders are
    kept as non-registered references so SWAD does not iterate over ~150M frozen
    parameters at every training step. AveragedModel deep-copies this wrapper
    once at construction, so the frozen encoders are still available for final
    SWAD evaluation without copying the full training optimizer/state each step.
    """

    def __init__(self, model):
        super().__init__()
        self.causal_decomposition = model.causal_decomposition
        self.prompt_learner = model.text_features.prompt_learner
        self.tda = model.tda
        self.k = int(model.text_features.k)

        self.register_buffer(
            "diverse_text_features",
            model.text_features.diverse_text_features.detach(),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_mean", model._imagenet_mean.detach(), persistent=False
        )
        self.register_buffer(
            "_imagenet_std", model._imagenet_std.detach(), persistent=False
        )
        self.register_buffer(
            "_clip_mean", model._clip_mean.detach(), persistent=False
        )
        self.register_buffer(
            "_clip_std", model._clip_std.detach(), persistent=False
        )

        # Bypass nn.Module registration for frozen encoders. This is deliberate:
        # they must be present for inference but must not participate in SWAD's
        # parameter-averaging loop.
        object.__setattr__(self, "_clip_model_ref", model.clip_model)
        object.__setattr__(
            self, "_text_encoder_ref", model.text_features.text_encoder
        )

    @property
    def network(self):
        # Compatibility with generic DomainBed/SWAD utilities.
        return self

    def train(self, mode=True):
        super().train(mode)
        self._clip_model_ref.eval()
        self._text_encoder_ref.eval()
        return self

    def _visual(self, images):
        image_pixels = images * self._imagenet_std + self._imagenet_mean
        clip_images = (image_pixels - self._clip_mean) / self._clip_std
        with torch.no_grad():
            visual = self._clip_model_ref.encode_image(clip_images).float()
        return F.normalize(visual, dim=-1)

    def _class_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        features = self._text_encoder_ref(prompts, tokenized_prompts)
        return F.normalize(features.float(), dim=-1)

    def _diverse_features(self):
        k = min(self.k, self.diverse_text_features.shape[1])
        return self.diverse_text_features[:, :k, :]

    @torch.no_grad()
    def predict(self, x):
        was_training = self.training
        self.eval()

        visual = self._visual(x)
        causal, _ = self.causal_decomposition(visual)
        class_features = self._class_features()
        diverse_features = self._diverse_features()

        num_classes, num_templates, dim = diverse_features.shape
        batch = causal.shape[0]
        causal_flat = causal[:, None, :].expand(batch, num_classes, dim).reshape(
            batch * num_classes, dim
        )
        diverse_flat = (
            diverse_features[None, :, :, :]
            .expand(batch, num_classes, num_templates, dim)
            .reshape(batch * num_classes, num_templates, dim)
        )
        z = self.tda(causal_flat, diverse_flat.float()).reshape(
            batch, num_classes, num_templates, dim
        )
        z = F.normalize(z.float(), dim=-1)
        text = F.normalize(class_features.float(), dim=-1)
        scale = self._clip_model_ref.logit_scale.exp().detach().float()
        logits = scale * torch.einsum("bckd,cd->bck", z, text).mean(dim=-1)

        self.train(was_training)
        return logits

    def forward(self, x):
        return self.predict(x)

    def predict_embed(self, x):
        causal, _ = self.causal_decomposition(self._visual(x))
        return causal


class CIPTDCCL(_CIPTDCCLBase):
    """Use official CIPT defaults while preserving the DCCL integration."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        effective_hparams = copy.deepcopy(dict(hparams))
        effective_hparams.setdefault("cipt_tda_heads", 8)
        effective_hparams.setdefault("cipt_lr", 2.5e-3)
        effective_hparams.setdefault("cipt_weight_decay", 0.0)

        super().__init__(
            input_shape, num_classes, num_domains, effective_hparams
        )

        # DomainBed/DCCL transforms normalize with ImageNet statistics. OpenAI
        # CLIP expects different normalization. Re-normalize inside the model so
        # DCCL's augmentation/data pipeline remains unchanged while the frozen
        # CLIP encoder sees its official input distribution.
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        print(
            "CIPTDCCL official alignment: tda_heads={}, cipt_lr={}, "
            "cipt_weight_decay={}, cosine_steps={}, CLIP input renormalization=on".format(
                effective_hparams["cipt_tda_heads"],
                effective_hparams["cipt_lr"],
                effective_hparams["cipt_weight_decay"],
                self.cipt_schedule_steps,
            )
        )

    def _visual(self, images):
        # Undo DomainBed ImageNet normalization and apply OpenAI CLIP stats.
        image_pixels = images * self._imagenet_std + self._imagenet_mean
        clip_images = (image_pixels - self._clip_mean) / self._clip_std
        with torch.no_grad():
            visual = self.clip_model.encode_image(clip_images).float()
        return F.normalize(visual, dim=-1)

    def update(self, x, y, **kwargs):
        metrics = super().update(x, y, **kwargs)
        # Compatibility aliases for the optional legacy loss logger in trainer.py.
        metrics["ce_loss"] = metrics["cipt_cls_loss"]
        metrics["sup_cl_loss"] = metrics["dccl_contrastive_loss"]
        return metrics

    def get_forward_model(self):
        # Do not deepcopy the full training object/optimizer every SWAD step.
        return CIPTDCCLForwardModel(self)
