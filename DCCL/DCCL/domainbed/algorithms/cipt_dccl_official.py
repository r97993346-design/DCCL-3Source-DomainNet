"""Strict official-alignment wrapper for the feature/multiprompt CIPTDCCL path."""

import copy

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl import CIPTDCCL as _CIPTDCCLBase


class CIPTDCCL(_CIPTDCCLBase):
    """Use official CIPT defaults while preserving the DCCL integration."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        effective_hparams = copy.deepcopy(dict(hparams))

        # train_all.py historically defaulted this option to 1. Official CIPT's
        # public implementation defaults to 8 heads; upgrade the legacy default
        # in official-alignment mode. Values other than 1 are preserved.
        if int(effective_hparams.get("cipt_tda_heads", 8)) == 1:
            effective_hparams["cipt_tda_heads"] = 8

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
            "cipt_weight_decay={}, CLIP input renormalization=on".format(
                effective_hparams["cipt_tda_heads"],
                effective_hparams["cipt_lr"],
                effective_hparams["cipt_weight_decay"],
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
