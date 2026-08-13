from __future__ import absolute_import

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.lib import misc

from .algorithms import DCCL, rand_bbox
from .cipt_official import (
    DEFAULT_CONTEXT_INIT,
    OfficialCIPTAuxiliary,
    cipt_loss,
    load_frozen_clip,
)

__all__ = ["DCCLCIPT"]


_DATASET_SUBDIRS = {
    "PACS": "PACS",
    "VLCS": "VLCS",
    "OfficeHome": "office_home",
    "TerraIncognita": "terra_incognita_fixed",
    "DomainNet": "domain_net",
}

_KNOWN_CLASSNAMES = {
    "PACS": [
        "dog",
        "elephant",
        "giraffe",
        "guitar",
        "horse",
        "house",
        "person",
    ],
}


def _parse_explicit_classnames(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_classnames(hparams, num_classes):
    """Resolve ImageFolder class names in the exact label-index order."""
    explicit = _parse_explicit_classnames(
        hparams.get("cipt_classnames", "")
    )
    if explicit:
        if len(explicit) != num_classes:
            raise ValueError(
                "cipt_classnames has {} names but dataset has {} classes".format(
                    len(explicit), num_classes
                )
            )
        return explicit

    dataset_name = str(hparams.get("dataset", ""))
    data_dir = str(hparams.get("data_dir", ""))
    subdir = _DATASET_SUBDIRS.get(dataset_name)

    if data_dir and subdir:
        dataset_root = Path(data_dir) / subdir
        if dataset_root.exists():
            env_dirs = sorted(
                path for path in dataset_root.iterdir() if path.is_dir()
            )
            for env_dir in env_dirs:
                classnames = sorted(
                    path.name for path in env_dir.iterdir() if path.is_dir()
                )
                if len(classnames) == num_classes:
                    return classnames

    known = _KNOWN_CLASSNAMES.get(dataset_name, [])
    if len(known) == num_classes:
        return list(known)

    raise RuntimeError(
        "Unable to resolve CIPT class names for dataset={!r}. "
        "Pass --cipt_classnames name1,name2,... in ImageFolder label order, "
        "or make sure --data_dir points to the dataset root.".format(
            dataset_name
        )
    )


class DCCLToCIPTBridge(nn.Module):
    """The only feature-interface layer introduced by the DCCL/CIPT fusion."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class DCCLCIPT(DCCL):
    """DCCL trained with official CIPT as a training-only auxiliary objective.

    DCCL's prediction path, classifier, SupCon views, pre-trained alignment and
    SWAD forward model are unchanged. CIPT receives a side branch from DCCL's
    final feature through one bridge layer. After that bridge, the official CIPT
    prompt learner, causal/spurious adapters, cosine logits, TDA and Eq. (22)
    loss are preserved.

    Training objective:
        L_total = L_DCCL + lambda_cipt(step) * L_CIPT

    Inference objective/path:
        x -> DCCL featurizer -> DCCL classifier
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DCCLCIPT, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.cipt_weight = float(hparams.get("cipt_weight", 0.05))
        self.cipt_warmup_steps = int(
            hparams.get("cipt_warmup_steps", 500)
        )
        self.cipt_beta = float(hparams.get("cipt_beta", 4.0))
        self.cipt_gamma = float(hparams.get("cipt_gamma", 5.0))
        self.cipt_lr = float(hparams.get("cipt_lr", 2.5e-3))
        self.cipt_weight_decay = float(
            hparams.get("cipt_weight_decay", 0.0)
        )

        classnames = _resolve_classnames(hparams, num_classes)
        clip_model, tokenize = load_frozen_clip(
            model_name=hparams.get("cipt_clip_model", "ViT-B/16"),
            model_path=hparams.get("cipt_clip_path", ""),
            download_root=hparams.get("cipt_clip_download_root", ""),
        )
        clip_dim = int(clip_model.visual.output_dim)

        self.cipt_bridge = DCCLToCIPTBridge(
            self.featurizer.n_outputs,
            clip_dim,
        )
        self.cipt_aux = OfficialCIPTAuxiliary(
            clip_model=clip_model,
            classnames=classnames,
            tokenize=tokenize,
            n_ctx=int(hparams.get("cipt_n_ctx", 16)),
            ctx_init=hparams.get(
                "cipt_ctx_init", DEFAULT_CONTEXT_INIT
            ),
            num_diverse_templates=int(
                hparams.get("cipt_num_text_views", 4)
            ),
            num_heads=int(hparams.get("cipt_tda_heads", 8)),
            dropout=float(hparams.get("cipt_tda_dropout", 0.0)),
            sample_templates=bool(
                hparams.get("cipt_sample_templates", True)
            ),
        )

        # Keep DCCL's optimizer exactly as constructed by DCCL. Official CIPT
        # uses Adam(lr=2.5e-3, wd=0) for its trainable prompt/adapters/TDA; the
        # bridge is optimized with the same auxiliary optimizer. CIPT gradients
        # still reach the shared DCCL backbone through the total loss, and that
        # backbone is stepped only by DCCL's original optimizer.
        cipt_parameters = list(self.cipt_bridge.parameters()) + list(
            self.cipt_aux.trainable_parameters()
        )
        self.cipt_optimizer = torch.optim.Adam(
            cipt_parameters,
            lr=self.cipt_lr,
            weight_decay=self.cipt_weight_decay,
        )

    def train(self, mode=True):
        super(DCCLCIPT, self).train(mode)
        # Official CIPT freezes CLIP and keeps its text encoder in eval mode.
        self.cipt_aux.clip_model.eval()
        self.cipt_aux.text_encoder.eval()
        return self

    def _current_cipt_weight(self, step):
        if self.cipt_weight <= 0.0:
            return 0.0
        if self.cipt_warmup_steps <= 0:
            return self.cipt_weight
        progress = min(
            max(float(step) / float(self.cipt_warmup_steps), 0.0),
            1.0,
        )
        return self.cipt_weight * progress

    def _compute_cipt_auxiliary(self, feature_x, labels, step):
        bridge_features = self.cipt_bridge(feature_x)
        output = self.cipt_aux(bridge_features, labels)
        losses = cipt_loss(
            output["interventional_logits"],
            output["causal_logits"],
            output["spurious_logits"],
            output["causal_features"],
            output["spurious_features"],
            labels,
            beta=self.cipt_beta,
            gamma=self.cipt_gamma,
        )
        weight = self._current_cipt_weight(step)

        with torch.no_grad():
            causal_acc = (
                output["causal_logits"].argmax(dim=-1) == labels
            ).float().mean()
            tda_logits = output["interventional_logits"].mean(dim=1)
            tda_acc = (
                tda_logits.argmax(dim=-1) == labels
            ).float().mean()
            spurious_prob = F.softmax(
                output["spurious_logits"], dim=-1
            )
            spurious_entropy = -(
                spurious_prob
                * torch.log(spurious_prob.clamp_min(1e-8))
            ).sum(dim=-1).mean()

        metrics = {
            "cipt_loss": losses["loss"].item(),
            "cipt_cls_loss": losses["classification"].item(),
            "cipt_de_loss": losses["decomposition"].item(),
            "cipt_ind_loss": losses["independence"].item(),
            "cipt_causal_ce": losses["causal_ce"].item(),
            "cipt_spurious_kl": losses["spurious_kl"].item(),
            "cipt_weight": float(weight),
            "cipt_weighted_loss": float(weight * losses["loss"].item()),
            "cipt_causal_acc": causal_acc.item(),
            "cipt_tda_acc": tda_acc.item(),
            "cipt_spurious_entropy": spurious_entropy.item(),
        }
        return losses["loss"] * weight, metrics

    def update(self, x, y, **kwargs):
        # This is the original DCCL update path. The only algorithmic addition
        # is the CIPT auxiliary loss immediately before backward().
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
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            loss_sup_cl = self.supcon_loss(features, all_y)
            loss = -loss_sup_cl - self.lamda * sp_loss
            self.optimizer_TN.zero_grad()
            loss.backward()
            self.optimizer_TN.step()
            with torch.no_grad():
                all_x_2, sp_loss = self.TN_network(all_x_2)

        cutmix_active = False
        r = np.random.rand(1)
        if self.aug and r < self.aug:
            cutmix_active = True
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
            feature_x, inter_feats = self.featurizer(
                all_x, ret_feats=True
            )
            pred_x = self.classifier(feature_x)
            loss = (
                F.cross_entropy(pred_x, target_a) * lam
                + F.cross_entropy(pred_x, target_b) * (1 - lam)
            )
        else:
            feature_x, inter_feats = self.featurizer(
                all_x, ret_feats=True
            )
            pred_x = self.classifier(feature_x)
            loss = F.cross_entropy(pred_x, all_y)

        ce_loss = loss.item()

        feature_x_2, inter_feats_2 = self.featurizer(
            all_x_2, ret_feats=True
        )
        if self.two_ce:
            loss = (
                loss / 2
                + F.cross_entropy(
                    self.classifier(feature_x_2), all_y
                ) / 2
            )

        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(
                all_x, ret_feats=True
            )

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

        if self.l:
            embed_2 = self.proj_head(feature_x_2)
            embed_1 = self.proj_head(feature_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(
                    torch.cat([all_d, all_d_2]), 1
                ).float()
                neg_mask = torch.eq(d, d.T).float()
                pos_mask = 1 - neg_mask if self.pos_mask else None
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
                    view_2_d = nn.functional.normalize(embed_2_d)
                    add_pos = torch.cat([view_2_d, view_2_d], 0)
                    loss_sup_cl = self.supcon_loss(
                        features, all_y, add_pos=add_pos
                    )
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
            loss += self.l * loss_sup_cl

        pre_cl_loss = 0.0
        if self.l_layer:
            embed_1 = self.pre_proj_head(feature_x)
            embed_2 = self.pre_proj_head(pre_pred_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            all_y_pre = all_y

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(
                    torch.cat([all_d, all_d_2]), 1
                ).float()
                neg_mask = torch.eq(d, d.T).float()
                pos_mask = 1 - neg_mask if self.pos_mask else None
                pre_cl_loss += self.supcon_loss_pre(
                    features,
                    all_y_pre,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                pre_cl_loss += self.supcon_loss_pre(
                    features, all_y_pre
                )
            loss += self.l_layer * pre_cl_loss

        cipt_metrics = {
            "cipt_loss": 0.0,
            "cipt_cls_loss": 0.0,
            "cipt_de_loss": 0.0,
            "cipt_ind_loss": 0.0,
            "cipt_causal_ce": 0.0,
            "cipt_spurious_kl": 0.0,
            "cipt_weight": self._current_cipt_weight(
                kwargs.get("step", 0)
            ),
            "cipt_weighted_loss": 0.0,
            "cipt_causal_acc": 0.0,
            "cipt_tda_acc": 0.0,
            "cipt_spurious_entropy": 0.0,
        }

        # A CutMix image has two labels, while official CIPT assumes one class
        # label per image for class-conditioned prompts/templates. DCCL's
        # published/default experiments use aug=0; if CutMix is enabled we keep
        # DCCL's behavior and simply omit CIPT on that mixed-label step.
        current_weight = self._current_cipt_weight(kwargs.get("step", 0))
        if (not cutmix_active) and current_weight > 0.0:
            cipt_term, cipt_metrics = self._compute_cipt_auxiliary(
                feature_x,
                all_y,
                kwargs.get("step", 0),
            )
            loss = loss + cipt_term

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "DCCLCIPT produced a non-finite total loss"
            )

        self.optimizer.zero_grad()
        self.cipt_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cipt_optimizer.step()

        loss_dict = {
            "loss": loss.item(),
            "ce_loss": ce_loss,
        }
        if self.l:
            loss_dict["sup_cl_loss"] = loss_sup_cl.item()
        if self.l_layer:
            loss_dict["pre_cl_loss"] = pre_cl_loss.item()
        if self.l_d:
            loss_dict["reg_loss"] = reg_loss.item()
        loss_dict.update(cipt_metrics)
        return loss_dict

    # predict(), predict_embed() and get_forward_model() are intentionally
    # inherited from DCCL. Therefore CIPT is training-only and cannot alter
    # DCCL/SWAD inference behavior.
