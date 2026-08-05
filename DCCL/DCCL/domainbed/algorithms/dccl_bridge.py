"""Official Bridge CBB integration for DCCL.

The CBB internals come from the official Bridge implementation. This module
adapts the feature interface for torchvision ResNet-18/50. The high-dimensional
``layer4`` map is reduced to the channel width used by the official module,
processed by CBB, expanded again, and fused through an identity-initialized
residual gate. DCCL's losses, positive/negative construction, pretrained
anchor, and SWAD interface remain inherited from :class:`DCCL`.
"""

import torch
import torch.nn as nn

from domainbed import networks
from domainbed.optimizers import get_optimizer
from domainbed.models.bridge_cbb_official import ResidualBridgeBlock

from .algorithms import DCCL


OFFICIAL_BRIDGE_COMMIT = "88946a9793e61016f65f4f99ee30e326ae992c54"


def _get_hparam(hparams, name, default):
    try:
        return hparams[name]
    except (KeyError, TypeError):
        return default


class OfficialBridgePreResNet(nn.Module):
    """Wrap DCCL's existing PreResNet and insert CBB before global pooling."""

    def __init__(self, base_featurizer, hparams):
        super().__init__()
        model_name = hparams["model"]
        if model_name not in ("resnet18", "resnet50"):
            raise ValueError(
                "DCCLBridgeOfficial currently supports torchvision resnet18/resnet50 "
                f"only, got model={model_name!r}."
            )
        if not isinstance(base_featurizer, networks.PreResNet):
            raise TypeError(
                "DCCLBridgeOfficial requires DCCL's PreResNet featurizer, got "
                f"{type(base_featurizer).__name__}."
            )

        # Reuse the exact DCCL-initialized backbone instead of constructing a
        # second ResNet. This preserves its initialization and feature hooks.
        self.base_featurizer = base_featurizer
        self.n_outputs = base_featurizer.n_outputs

        self.bridge_adapter = ResidualBridgeBlock(
            in_channels=self.n_outputs,
            bridge_channels=int(_get_hparam(hparams, "bridge_channels", 256)),
            gate_init=float(_get_hparam(hparams, "bridge_gate_init", 0.0)),
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

    def backbone_parameters(self):
        return self.base_featurizer.parameters()

    def bridge_parameters(self):
        return self.bridge_adapter.parameters()

    def forward(self, x, ret_feats=False):
        """Run ResNet, apply CBB to layer4 map, then preserve DCCL output API."""
        base = self.base_featurizer
        base.clear_features()
        network = base.network

        x = network.conv1(x)
        x = network.bn1(x)
        x = network.relu(x)
        x = network.maxpool(x)

        x = network.layer1(x)
        x = network.layer2(x)
        x = network.layer3(x)
        x = network.layer4(x)

        # CBB sees a compact representation and starts as an exact identity
        # residual, preserving the pretrained DCCL feature distribution.
        x = self.bridge_adapter(x)

        # The final generative-alignment target must describe the feature map
        # consumed by the classifier, rather than the pre-Bridge layer4 map.
        if base._features:
            base._features[-1] = x

        x = network.avgpool(x)
        x = torch.flatten(x, 1)
        out = base.dropout(x)

        if ret_feats:
            # Hooks registered by PreResNet still expose the same stem/layer
            # tensors used by DCCL's original layer-wise regularization.
            return out, base._features
        return out


class DCCLBridgeOfficial(DCCL):
    """DCCL with the official Bridge MultiScaleBasisBlock feature module."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Build the untouched DCCL components first, including the frozen
        # pretrained anchor and all original loss modules.
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # Wrap only the existing trainable feature path. The frozen
        # pre_featurizer is intentionally left unchanged and never passes
        # through Bridge.
        self.featurizer = OfficialBridgePreResNet(
            self.featurizer, self.hparams
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.proj = nn.Sequential(self.featurizer, self.proj_head)

        # Keep the pretrained backbone on its fine-tuning LR while allowing the
        # randomly initialized adapter to learn at a separately controlled LR.
        lower_cls = 0.1
        lower_proj = 10
        bridge_lr = self.hparams["lr"] * float(
            _get_hparam(self.hparams, "bridge_lr_multiplier", 10.0)
        )
        optimized_list = [
            {
                "params": self.featurizer.backbone_parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.featurizer.bridge_parameters(),
                "lr": bridge_lr,
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
