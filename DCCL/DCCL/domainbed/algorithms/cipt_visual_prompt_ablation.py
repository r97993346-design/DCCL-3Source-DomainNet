"""Minimal visual-prompt wrapper for CIPT causal learning.

This wrapper adds learnable visual prompt tokens P_v on top of the existing
feature/cipt-causal-contrastive-no-proj implementation without introducing any
prompt-specific loss. With cipt_pure=True, P_v is optimized only through the
original CIPT objective: L_cls + beta * L_de + gamma * L_ind.

If cipt_pure=False is explicitly enabled, the inherited direct causal-space
SupCon term is also active and can additionally backpropagate through P_v.
"""

from domainbed.algorithms.cipt_dccl_ablation import (
    CIPTDCCL as _CausalContrastiveCIPTDCCL,
)
from domainbed.algorithms.cipt_visual_prompt import (
    VisualPromptLearner,
    encode_image_with_visual_prompt,
)
from domainbed.optimizers import get_optimizer


class CIPTDCCL(_CausalContrastiveCIPTDCCL):
    """CIPT with a shallow learnable visual prompt and no new loss terms."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.visual_prompt_enabled = bool(
            hparams.get("cipt_visual_prompt_enabled", True)
        )
        self.visual_prompt_length = int(
            hparams.get("cipt_visual_prompt_length", 4)
        )

        # CLIP remains completely frozen. The only new visual-side trainable
        # parameters are the prompt tokens below.
        self.clip_model.requires_grad_(False)
        self.visual_prompt = None
        if self.visual_prompt_enabled:
            self.visual_prompt = VisualPromptLearner(
                self.clip_model,
                prompt_length=self.visual_prompt_length,
            )

        # Parent optimizer was built before P_v existed, so rebuild it after
        # adding the prompt tokens. No optimizer group or special LR is added.
        trainable = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            trainable,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable
        )
        self.frozen_parameter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )

        print(
            "CIPTDCCL visual-prompt causal-loss-only: enabled={}, length={}, "
            "pure_cipt={}, prompt_specific_loss=False, trainable={}, frozen={}".format(
                self.visual_prompt_enabled,
                self.visual_prompt_length if self.visual_prompt_enabled else 0,
                self.cipt_pure,
                self.trainable_parameter_count,
                self.frozen_parameter_count,
            )
        )
        if self.cipt_pure:
            print(
                "Visual prompt gradients: L_cls + beta*L_de + gamma*L_ind only"
            )
        else:
            print(
                "Visual prompt gradients: CIPT base loss + inherited direct causal SupCon"
            )

    def _visual_anchor(self, images):
        """Unprompted frozen CLIP visual feature for diagnostics/ablation."""
        with __import__("torch").no_grad():
            return self.clip_model.encode_image(images).float()

    def _visual(self, images):
        """Prompted visual feature; downstream losses learn P_v automatically."""
        if not self.visual_prompt_enabled:
            return self._visual_anchor(images)
        return encode_image_with_visual_prompt(
            self.clip_model,
            images,
            self.visual_prompt,
        )
