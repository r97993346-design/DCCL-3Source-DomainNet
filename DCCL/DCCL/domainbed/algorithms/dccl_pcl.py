import numpy as np
import torch
import torch.nn.functional as F

from domainbed.lib import misc
from domainbed.lib.pcl import PCLLoss
from .algorithms import DCCL as BaseDCCL
from .algorithms import rand_bbox


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


class DCCL(BaseDCCL):
    """DCCL with an optional PCL-style auxiliary alignment loss.

    The original DCCL SupCon positive definition is left unchanged. PCL is
    computed only on the original-view projection features, grouped by ground
    truth class and source-domain pair. Its partial-OT plan provides an
    additional local alignment signal; unmatched pairs are never converted to
    negatives and are never removed from DCCL's supervised positive set.

    When ``use_pcl`` is false, ``update`` delegates directly to BaseDCCL so the
    baseline forward/loss/RNG path remains unchanged.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.use_pcl = _as_bool(hparams.get("use_pcl", False))
        self.pcl_weight = float(hparams.get("pcl_weight", 0.1))
        self.pcl_align_weight = float(hparams.get("pcl_align_weight", 1.0))
        self.pcl_uniform_weight = float(hparams.get("pcl_uniform_weight", 0.1))

        self.pcl_criterion = PCLLoss(
            transport_mass=float(hparams.get("pcl_transport_mass", 0.8)),
            sinkhorn_epsilon=float(hparams.get("pcl_sinkhorn_epsilon", 0.05)),
            sinkhorn_iters=int(hparams.get("pcl_sinkhorn_iters", 50)),
            uniform_temperature=float(hparams.get("pcl_uniform_temperature", 2.0)),
            match_threshold=float(hparams.get("pcl_match_threshold", 0.5)),
        )

    @staticmethod
    def _domain_ids_from_minibatches(y):
        """Create stable source-domain IDs from the environment minibatch list."""
        ids = []
        for domain_id, labels in enumerate(y):
            ids.append(
                torch.full(
                    (labels.shape[0],),
                    domain_id,
                    device=labels.device,
                    dtype=torch.long,
                )
            )
        return torch.cat(ids, dim=0)

    def update(self, x, y, **kwargs):
        # Critical regression guarantee: no extra PCL computation, RNG draw, or
        # changed loss path when the feature is disabled.
        if not self.use_pcl:
            return super().update(x, y, **kwargs)

        all_x = torch.cat(x)
        all_y = torch.cat(y)
        pcl_domains = self._domain_ids_from_minibatches(y)

        # PCL must use real/original samples only. Preserve an untouched tensor
        # because DCCL's optional CutMix mutates all_x in-place.
        pcl_original_x = all_x.clone()

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
        cutmix_applied = False
        if self.aug and r < self.aug:
            cutmix_applied = True
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
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
            pred_x = self.classifier(feature_x)
            loss = (
                F.cross_entropy(pred_x, target_a) * lam
                + F.cross_entropy(pred_x, target_b) * (1 - lam)
            )
        else:
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            pred_x = self.classifier(feature_x)
            loss = F.cross_entropy(pred_x, all_y)
        ce_loss = loss.item()

        feature_x_2, inter_feats_2 = self.featurizer(all_x_2, ret_feats=True)
        if self.two_ce:
            loss = loss / 2 + F.cross_entropy(self.classifier(feature_x_2), all_y) / 2

        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        if self.l_d:
            reg_loss = 0.0
            for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(
                inter_feats,
                pre_feats,
                self.mean_encoders,
                self.var_encoders,
            ):
                mean = mean_enc(inter_f)
                var = var_enc(inter_f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.0
            loss += self.l_d * reg_loss

        # Keep the original DCCL supervised contrastive path unchanged.
        pcl_view = None
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

                if self.pos_mask:
                    pos_mask = 1 - neg_mask
                else:
                    pos_mask = None
                loss_sup_cl = self.supcon_loss(
                    features,
                    all_y,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                if self.sample_d:
                    all_x_2_d = torch.cat(kwargs["x_2_d"])
                    feature_x_2_d = self.featurizer(all_x_2_d)
                    embed_2_d = self.proj_head(feature_x_2_d)
                    view_2_d = F.normalize(embed_2_d)
                    add_pos = torch.cat([view_2_d, view_2_d], 0)
                    loss_sup_cl = self.supcon_loss(
                        features,
                        all_y,
                        add_pos=add_pos,
                    )
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
            loss += self.l * loss_sup_cl

            # If the original view was not modified by CutMix, reuse exactly the
            # same normalized projection produced above.
            if not cutmix_applied:
                pcl_view = view_1

        pre_cl_loss = 0.0
        if self.l_layer:
            embed_1 = self.pre_proj_head(feature_x)
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

                if self.pos_mask:
                    pos_mask = 1 - neg_mask
                else:
                    pos_mask = None
                pre_cl_loss += self.supcon_loss_pre(
                    features,
                    all_y_pre,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                pre_cl_loss += self.supcon_loss_pre(features, all_y_pre)
            loss += self.l_layer * pre_cl_loss

        # PCL operates only on original-view projection features. When CutMix
        # changed DCCL's feature_x (or contrastive loss is disabled), obtain an
        # independent original-sample representation here.
        if pcl_view is None:
            pcl_feature = self.featurizer(pcl_original_x)
            pcl_embed = self.proj_head(pcl_feature)
            pcl_view = F.normalize(pcl_embed)

        pcl_align_loss, pcl_uniform_loss, pcl_stats = self.pcl_criterion(
            pcl_view,
            all_y,
            pcl_domains,
        )
        pcl_loss = (
            self.pcl_align_weight * pcl_align_loss
            + self.pcl_uniform_weight * pcl_uniform_loss
        )
        loss += self.pcl_weight * pcl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            "loss": loss.item(),
            "ce_loss": ce_loss,
            "pcl_loss": pcl_loss.item(),
            "pcl_align_loss": pcl_align_loss.item(),
            "pcl_uniform_loss": pcl_uniform_loss.item(),
            "pcl_valid_domain_pairs": pcl_stats["valid_domain_pairs"],
            "pcl_valid_class_pairs": pcl_stats["valid_class_pairs"],
            "pcl_transport_mass": pcl_stats["transport_mass"],
            "pcl_matching_ratio": pcl_stats["matching_ratio"],
        }
        if self.l:
            loss_dict["sup_cl_loss"] = loss_sup_cl.item()
        if self.l_layer:
            loss_dict["pre_cl_loss"] = pre_cl_loss.item()
        return loss_dict
