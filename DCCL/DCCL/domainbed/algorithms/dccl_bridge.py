"""Official Bridge CBB integration for DCCL.

The CBB internals come from the official Bridge implementation. This module
only adapts the feature interface: for torchvision ResNet-18/50, the CBB is
inserted after ``layer4`` and before global average pooling. DCCL's losses,
positive/negative construction, pretrained anchor, and SWAD interface remain
inherited unchanged from :class:`DCCL`.
"""

import torch
import torch.nn as nn

from domainbed import networks
from domainbed.optimizers import get_optimizer
from domainbed.models.bridge_cbb_official import MultiScaleBasisBlock

from .algorithms import DCCL


OFFICIAL_BRIDGE_COMMIT = "88946a9793e61016f65f4f99ee30e326ae992c54"


def _get_hparam(hparams, name, default):
    try:
        return hparams[name]
    except (KeyError, TypeError):
        return default


class OfficialBridgePreResNet(networks.PreResNet):
    """PreResNet with the official Bridge CBB before global pooling."""

    def __init__(self, input_shape, hparams, freeze=0):
        model_name = hparams["model"]
        if model_name not in ("resnet18", "resnet50"):
            raise ValueError(
                "DCCLBridgeOfficial currently supports torchvision resnet18/resnet50 "
                f"only, got model={model_name!r}."
            )

        super().__init__(input_shape, hparams, freeze=freeze)

        self.bridge_block = MultiScaleBasisBlock(
            in_channels=self.n_outputs,
            basis_reduction=_get_hparam(
                hparams, "bridge_basis_reduction", 2
            ),
            basis_reduction_mode=_get_hparam(
                hparams, "bridge_basis_reduction_mode", "div"
            ),
            with_ssp=_get_hparam(hparams, "bridge_with_ssp", True),
            with_query=_get_hparam(hparams, "bridge_with_query", True),
            with_input_subspace=_get_hparam(
                hparams, "bridge_with_input_subspace", False
            ),
            with_dropout=_get_hparam(
                hparams, "bridge_with_dropout", False
            ),
            basis_normalize=_get_hparam(
                hparams, "bridge_basis_normalize", True
            ),
            conv_kernel_size=_get_hparam(
                hparams, "bridge_conv_kernel_size", 3
            ),
        )

    def forward(self, x, ret_feats=False):
        """Run ResNet, apply CBB to layer4 map, then preserve DCCL output API."""
        self.clear_features()
        network = self.network

        x = network.conv1(x)
        x = network.bn1(x)
        x = network.relu(x)
        x = network.maxpool(x)

        x = network.layer1(x)
        x = network.layer2(x)
        x = network.layer3(x)
        x = network.layer4(x)

        # The only DCCL-specific interface adaptation: official CBB receives
        # the final 4-D feature map before ResNet's global average pooling.
        x = self.bridge_block(x)

        x = network.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.dropout(x)

        if ret_feats:
            # Hooks registered by PreResNet still expose the same stem/layer
            # tensors used by DCCL's original layer-wise regularization.
            return out, self._features
        return out


class DCCLBridgeOfficial(DCCL):
    """DCCL with the official Bridge MultiScaleBasisBlock feature module."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Build the untouched DCCL components first, including the frozen
        # pretrained anchor and all original loss modules.
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # Replace only the trainable feature path. The frozen pre_featurizer is
        # intentionally left unchanged and never passes through Bridge.
        self.featurizer = OfficialBridgePreResNet(
            input_shape, self.hparams, freeze=0
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.proj = nn.Sequential(self.featurizer, self.proj_head)

        # Rebuild the optimizer with exactly the same DCCL parameter-group
        # learning-rate policy, now including official CBB parameters through
        # self.featurizer.parameters().
        lower_cls = 0.1
        lower_proj = 10
        optimized_list = [
            {
                "params": self.featurizer.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams["lr"] / lower_cls,
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.proj_head.parameters(),
                "lr": self.hparams["lr"] / lower_proj,
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.mean_encoders.parameters(),
                "lr": self.hparams["lr"] * 10,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": self.hparams["lr"] * 10,
            },
            {
                "params": self.pre_proj_head.parameters(),
                "lr": self.hparams["lr"] / lower_proj,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            optimized_list,
        )
