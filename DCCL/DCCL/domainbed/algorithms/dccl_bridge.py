"""Official Bridge CBB integration for DCCL.

The CBB internals come from the official Bridge implementation. This module
adapts the feature interface for torchvision ResNet-18/50. The high-dimensional
``layer4`` map is reduced to the channel width used by the official module,
processed by CBB, expanded again, and fused through a fixed-scale residual
adapter.

The important optimization split is intentional:

* CE and DCCL supervised contrastive loss use the post-Bridge representation.
* DCCL generative alignment (reg_loss) and pretrained contrastive alignment
  (pre_cl) use the pre-Bridge backbone representation.

This prevents the pretrained-anchor objectives from directly suppressing the
feature correction that CBB is supposed to learn.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import networks
from domainbed.lib import misc
from domainbed.optimizers import get_optimizer
from domainbed.models.bridge_cbb_official import ResidualBridgeBlock

from .algorithms import DCCL, rand_bbox


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
        # second ResNet. Its own hparams keep backbone BatchNorm frozen.
        self.base_featurizer = base_featurizer
        self.n_outputs = base_featurizer.n_outputs
        self.last_pre_bridge_output = None

        self.bridge_adapter = ResidualBridgeBlock(
            in_channels=self.n_outputs,
            bridge_channels=int(_get_hparam(hparams, "bridge_channels", 256)),
            residual_scale=float(
                _get_hparam(hparams, "bridge_residual_scale", 0.1)
            ),
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
        """Return post-Bridge output while preserving pre-Bridge DCCL anchors."""
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

        # Hooks have now captured the original DCCL stem/layer1..layer4 maps.
        # Keep them unchanged so reg_loss aligns the backbone, not the CBB
        # correction, to the frozen pretrained anchor.
        pre_bridge_pooled = network.avgpool(x)
        pre_bridge_pooled = torch.flatten(pre_bridge_pooled, 1)
        self.last_pre_bridge_output = base.dropout(pre_bridge_pooled)

        # CBB is used only on the task representation consumed by CE/SupCon.
        x = self.bridge_adapter(x)
        x = network.avgpool(x)
        x = torch.flatten(x, 1)
        out = base.dropout(x)

        if ret_feats:
            # Return a snapshot rather than the mutable hook list itself. A
            # second forward must not overwrite the first forward's reg targets.
            return out, list(base._features)
        return out


class DCCLBridgeOfficial(DCCL):
    """DCCL with the official Bridge CBB feature module."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # ``freeze_bn=False`` is exposed to the trainer only so final SWAD BN
        # refresh is triggered. DCCL's ResNet backbones themselves must remain
        # frozen exactly as in the baseline, therefore build the inherited DCCL
        # components from a private copy with freeze_bn=True.
        dccl_hparams = copy.deepcopy(hparams)
        dccl_hparams["freeze_bn"] = True
        super().__init__(input_shape, num_classes, num_domains, dccl_hparams)

        # Keep externally visible hparams (including freeze_bn=False for the
        # SWAD refresh trigger) while the already-created backbone modules retain
        # their private freeze_bn=True configuration.
        self.hparams = hparams

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

    def update(self, x, y, **kwargs):
        """DCCL update with pretrained-anchor losses before Bridge.

        The original DCCL task path is preserved for CE/SupCon. Only the two
        pretrained-anchor objectives are redirected to the pre-Bridge backbone
        representation so they do not penalize the CBB residual itself.
        """
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        x_2 = kwargs["x_2"]
        all_x_2 = torch.cat(x_2)

        if self.TN:
            all_x_2, sp_loss = self.TN_network(all_x_2)
            feature_x = self.featurizer(all_x)
            feature_x_2 = self.featurizer(all_x_2)
            embed_2 = self.proj_head(feature_x_2)
            embed_1 = self.proj_head(feature_x)
            view_1 = F.normalize(embed_1)
            view_2 = F.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            loss_sup_cl = self.supcon_loss(features, all_y)
            loss = -loss_sup_cl - self.lamda * sp_loss
            self.optimizer_TN.zero_grad()
            loss.backward()
            self.optimizer_TN.step()
            with torch.no_grad():
                all_x_2, sp_loss = self.TN_network(all_x_2)

        r = np.random.rand(1)
        if self.aug and r < self.aug:
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(all_x.size()[0], device=all_x.device)
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / (all_x.size()[-1] * all_x.size()[-2])
            )
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            anchor_feature_x = self.featurizer.last_pre_bridge_output
            pred_x = self.classifier(feature_x)
            loss = (
                F.cross_entropy(pred_x, target_a) * lam
                + F.cross_entropy(pred_x, target_b) * (1 - lam)
            )
        else:
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            anchor_feature_x = self.featurizer.last_pre_bridge_output
            pred_x = self.classifier(feature_x)
            loss = F.cross_entropy(pred_x, all_y)

        if anchor_feature_x is None:
            raise RuntimeError("Missing pre-Bridge feature for DCCL anchor losses")

        ce_loss = loss.item()

        # This second forward cannot overwrite ``inter_feats`` because the
        # wrapper returned a list snapshot above.
        feature_x_2, _inter_feats_2 = self.featurizer(all_x_2, ret_feats=True)
        if self.two_ce:
            loss = loss / 2 + F.cross_entropy(
                self.classifier(feature_x_2), all_y
            ) / 2

        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = None
        if self.l_d:
            reg_loss = 0.0
            for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(
                inter_feats, pre_feats, self.mean_encoders, self.var_encoders
            ):
                mean = mean_enc(inter_f)
                var = var_enc(inter_f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.0
            loss += self.l_d * reg_loss

        loss_sup_cl = None
        if self.l:
            embed_2 = self.proj_head(feature_x_2)
            embed_1 = self.proj_head(feature_x)
            view_1 = F.normalize(embed_1)
            view_2 = F.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                pos_mask = 1 - neg_mask if self.pos_mask else None
                loss_sup_cl = self.supcon_loss(
                    features,
                    all_y,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            elif self.sample_d:
                all_x_2_d = torch.cat(kwargs["x_2_d"])
                feature_x_2_d = self.featurizer(all_x_2_d)
                embed_2_d = self.proj_head(feature_x_2_d)
                view_2_d = F.normalize(embed_2_d)
                add_pos = torch.cat([view_2_d, view_2_d], 0)
                loss_sup_cl = self.supcon_loss(
                    features, all_y, add_pos=add_pos
                )
            else:
                loss_sup_cl = self.supcon_loss(features, all_y)
            loss += self.l * loss_sup_cl

        pre_cl_loss = 0.0
        if self.l_layer:
            # Crucial separation: pre_cl regularizes the backbone representation
            # before CBB. CE/SupCon above still optimize the post-CBB feature.
            embed_1 = self.pre_proj_head(anchor_feature_x)
            embed_2 = self.pre_proj_head(pre_pred_x)
            view_1 = F.normalize(embed_1)
            view_2 = F.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            all_y_pre = all_y

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                pos_mask = 1 - neg_mask if self.pos_mask else None
                pre_cl_loss += self.supcon_loss_pre(
                    features,
                    all_y_pre,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                pre_cl_loss += self.supcon_loss_pre(features, all_y_pre)
            loss += self.l_layer * pre_cl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss}
        if self.l:
            loss_dict["sup_cl_loss"] = loss_sup_cl.item()
        if self.l_layer:
            loss_dict["pre_cl_loss"] = pre_cl_loss.item()
        if reg_loss is not None:
            loss_dict["reg_loss"] = reg_loss.item()
        return loss_dict
