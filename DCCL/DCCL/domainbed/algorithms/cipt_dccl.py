from __future__ import absolute_import

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed import networks
from domainbed.lib import misc
from domainbed.optimizers import get_optimizer

from .algorithms import DCCL, rand_bbox

__all__ = ["DCCLCIPT"]


# Fixed, class-agnostic visual-style contexts adapted from the CIPT
# ImageNet template bank. They provide perturbation directions only; there is
# deliberately no PromptLearner or learnable text token branch.
CIPT_STYLE_TEMPLATES = [
    "a bad photo.",
    "a low resolution photo.",
    "a rendering.",
    "graffiti.",
    "a cropped photo.",
    "a bright photo.",
    "a dark photo.",
    "a drawing.",
    "a black and white photo.",
    "a painting.",
    "a pixelated photo.",
    "a sketch.",
    "a cartoon.",
]


def _import_local_clip():
    """Import the repository-local OpenAI CLIP package lazily."""
    try:
        import clip
        return clip
    except ImportError:
        repo_root = Path(__file__).resolve().parents[4]
        clip_root = repo_root / "CLIP"
        if not clip_root.exists():
            raise ImportError(
                "CIPT text perturbation requires the repository-local CLIP "
                "directory or an installed OpenAI CLIP package."
            )
        clip_root = str(clip_root)
        if clip_root not in sys.path:
            sys.path.insert(0, clip_root)
        import clip
        return clip


class FeatureAdapter(nn.Module):
    """CIPT-style single-linear feature adapter with identity initialization."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class FixedTextFeatureBank(nn.Module):
    """Frozen CLIP text embeddings for class-agnostic style perturbations."""

    def __init__(
        self,
        model_name="ViT-B/16",
        model_path="",
        download_root="",
        encode_batch_size=64,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.download_root = download_root
        self.encode_batch_size = int(encode_batch_size)
        # Non-persistent: this bank is deterministic from the fixed templates
        # and CLIP checkpoint and does not need to inflate checkpoints.
        self.register_buffer("bank", torch.empty(0), persistent=False)

    def _build(self, device):
        clip = _import_local_clip()
        source = self.model_path if self.model_path else self.model_name
        download_root = self.download_root if self.download_root else None
        clip_model, _ = clip.load(
            source,
            device=device,
            jit=False,
            download_root=download_root,
        )
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        features = []
        with torch.no_grad():
            for start in range(0, len(CIPT_STYLE_TEMPLATES), self.encode_batch_size):
                texts = CIPT_STYLE_TEMPLATES[start : start + self.encode_batch_size]
                tokens = clip.tokenize(texts).to(device)
                encoded = clip_model.encode_text(tokens).float()
                features.append(F.normalize(encoded, dim=-1))
        bank = torch.cat(features, dim=0)

        # Only fixed text embeddings are retained; the CLIP visual encoder and
        # text transformer never participate in DCCL training.
        del clip_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self.bank = bank.detach()

    def get(self, num_views, device, training=True):
        if self.bank.numel() == 0:
            self._build(device)
        elif self.bank.device != device:
            self.bank = self.bank.to(device)

        k = min(int(num_views), self.bank.shape[0])
        if k < 1:
            raise ValueError("cipt_num_text_views must be >= 1")
        if training:
            indices = torch.randperm(self.bank.shape[0], device=device)[:k]
        else:
            indices = torch.arange(k, device=device)
        return self.bank.index_select(0, indices)


class DiversityAugmentation(nn.Module):
    """CIPT text-diversity attention adapted to the DCCL feature dimension.

    Input text embeddings are first projected from CLIP's 512-d text space to
    the native DCCL feature space. The DCCL representation itself is never
    compressed or replaced by a CLIP visual feature.
    """

    def __init__(self, visual_dim, text_dim=512, num_heads=8, dropout=0.0):
        super().__init__()
        if visual_dim % num_heads != 0:
            raise ValueError(
                "visual_dim={} must be divisible by num_heads={}".format(
                    visual_dim, num_heads
                )
            )
        self.text_projection = nn.Linear(text_dim, visual_dim, bias=False)
        nn.init.xavier_uniform_(self.text_projection.weight)
        # batch_first is intentionally not used for compatibility with the
        # project's older PyTorch versions.
        self.attn = nn.MultiheadAttention(
            visual_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, causal_features, text_features):
        """
        Args:
            causal_features: [B, D]
            text_features: [K, T] fixed CLIP text embeddings
        Returns:
            z: [B, K, D]
        """
        if causal_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError(
                "Expected causal_features [B,D] and text_features [K,T], got "
                "{} and {}".format(causal_features.shape, text_features.shape)
            )

        batch, dim = causal_features.shape
        text_projected = self.text_projection(text_features.float())
        text_projected = F.normalize(text_projected, dim=-1)
        num_views = text_projected.shape[0]

        # MultiheadAttention in torch 1.7 uses [L, N, E].
        query = (
            causal_features[:, None, :]
            .expand(-1, num_views, -1)
            .reshape(batch * num_views, 1, dim)
            .transpose(0, 1)
        )
        key_value = (
            text_projected[None, :, :]
            .expand(batch, -1, -1)
            .reshape(batch * num_views, 1, dim)
            .transpose(0, 1)
        )
        attn_out, _ = self.attn(
            query, key_value, key_value, need_weights=False
        )
        z = self.norm(query + self.dropout(attn_out))
        z = z.transpose(0, 1).squeeze(1)
        return z.reshape(batch, num_views, dim)


def spurious_uniform_kl(spurious_logits):
    """KL(p_uniform || p_spurious), matching CIPT Eq. (11)."""
    num_classes = spurious_logits.shape[-1]
    log_spurious = F.log_softmax(spurious_logits, dim=-1)
    uniform = torch.full_like(log_spurious, 1.0 / float(num_classes))
    return F.kl_div(log_spurious, uniform, reduction="batchmean")


def independence_loss(causal_features, spurious_features, eps=1e-6):
    """CIPT Eq. (14)-(15): 0.5 * squared cosine correlation."""
    corr = F.cosine_similarity(
        causal_features, spurious_features, dim=-1, eps=eps
    )
    return 0.5 * corr.pow(2).mean()


class DCCLCIPT(DCCL):
    """DCCL with CIPT-inspired causal separation and text perturbation.

    Main path:
        DCCL backbone f
          -> causal adapter e -> shared DCCL classifier / SupCon
          -> spurious adapter s -> KL-to-uniform through a detached shared head
        e -> fixed CLIP text perturbation -> z_k
          -> shared classifier L_c
          -> additional SupCon views

    No CLIP visual backbone and no learnable prompt branch are used.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DCCLCIPT, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        dim = self.featurizer.n_outputs
        self.causal_adapter = FeatureAdapter(dim)
        self.spurious_adapter = FeatureAdapter(dim)

        self.cipt_text_enable = bool(hparams.get("cipt_text_enable", True))
        self.cipt_num_text_views = int(hparams.get("cipt_num_text_views", 4))
        self.cipt_kl_weight = float(hparams.get("cipt_kl_weight", 0.1))
        self.cipt_ind_weight = float(hparams.get("cipt_ind_weight", 0.01))
        self.cipt_lc_weight = float(hparams.get("cipt_lc_weight", 0.1))

        self.text_bank = FixedTextFeatureBank(
            model_name=hparams.get("cipt_clip_model", "ViT-B/16"),
            model_path=hparams.get("cipt_clip_path", ""),
            download_root=hparams.get("cipt_clip_download_root", ""),
            encode_batch_size=hparams.get("cipt_text_batch_size", 64),
        )
        self.diversity_augmentation = DiversityAugmentation(
            visual_dim=dim,
            text_dim=512,
            num_heads=int(hparams.get("cipt_tda_heads", 8)),
            dropout=float(hparams.get("cipt_tda_dropout", 0.0)),
        )

        # The classifier is shared by e, s and z. KL uses a detached copy of
        # this same decision boundary so it cannot be minimized by collapsing a
        # separate spurious classification head.
        self.network = nn.Sequential(
            self.featurizer, self.causal_adapter, self.classifier
        )
        self._rebuild_optimizer()

    def _rebuild_optimizer(self):
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
            {
                "params": self.causal_adapter.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": self.spurious_adapter.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            },
        ]
        if self.cipt_text_enable:
            optimized_list.append(
                {
                    "params": self.diversity_augmentation.parameters(),
                    "lr": self.hparams["lr"],
                    "weight_decay": self.hparams["weight_decay"],
                }
            )

        self.optimizer = get_optimizer(
            self.hparams["optimizer"], optimized_list
        )

    def _detached_classifier(self, features):
        """Apply the shared classifier while freezing the head for KL."""
        # DCCL defines classifier as Sequential(Linear(...)).
        linear = self.classifier[0]
        return F.linear(
            features,
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
        )

    def _make_text_views(self, causal_features):
        text_features = self.text_bank.get(
            self.cipt_num_text_views,
            causal_features.device,
            training=self.training,
        )
        return self.diversity_augmentation(
            causal_features, text_features
        )

    def _text_classification_loss(self, z, labels):
        batch, num_views, dim = z.shape
        logits = self.classifier(z.reshape(batch * num_views, dim))
        repeated_labels = (
            labels[:, None].expand(batch, num_views).reshape(-1)
        )
        return F.cross_entropy(logits, repeated_labels), logits

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        x_2 = kwargs["x_2"]
        all_x_2 = torch.cat(x_2)

        if self.TN:
            # Keep DCCL's optional TN behavior, but perform its contrast in the
            # causal feature space.
            all_x_2, sp_loss = self.TN_network(all_x_2)
            feature_tn_1 = self.causal_adapter(self.featurizer(all_x))
            feature_tn_2 = self.causal_adapter(self.featurizer(all_x_2))
            embed_1 = self.proj_head(feature_tn_1)
            embed_2 = self.proj_head(feature_tn_2)
            view_1 = F.normalize(embed_1, dim=-1)
            view_2 = F.normalize(embed_2, dim=-1)
            features = torch.stack([view_1, view_2], dim=1)
            loss_tn_sup = self.supcon_loss(features, all_y)
            loss_tn = -loss_tn_sup - self.lamda * sp_loss
            self.optimizer_TN.zero_grad()
            loss_tn.backward()
            self.optimizer_TN.step()
            with torch.no_grad():
                all_x_2, _ = self.TN_network(all_x_2)

        cutmix_active = False
        r = np.random.rand(1)
        if self.aug and r < self.aug:
            cutmix_active = True
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(all_x.size(0), device=all_x.device)
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                / float(all_x.size(-1) * all_x.size(-2))
            )

        feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
        causal_x = self.causal_adapter(feature_x)
        spurious_x = self.spurious_adapter(feature_x)
        pred_x = self.classifier(causal_x)

        if cutmix_active:
            loss_ce = (
                F.cross_entropy(pred_x, target_a) * lam
                + F.cross_entropy(pred_x, target_b) * (1 - lam)
            )
        else:
            loss_ce = F.cross_entropy(pred_x, all_y)

        feature_x_2, inter_feats_2 = self.featurizer(
            all_x_2, ret_feats=True
        )
        causal_x_2 = self.causal_adapter(feature_x_2)

        if self.two_ce:
            ce_2 = F.cross_entropy(
                self.classifier(causal_x_2), all_y
            )
            loss_ce = loss_ce / 2.0 + ce_2 / 2.0

        loss = loss_ce

        # Split former L_de explicitly:
        #   L_ce is exactly DCCL's classification CE on e (above).
        #   L_kl only suppresses category information in s.
        spurious_logits = self._detached_classifier(spurious_x)
        loss_kl = spurious_uniform_kl(spurious_logits)
        loss_ind = independence_loss(causal_x, spurious_x)
        loss = (
            loss
            + self.cipt_kl_weight * loss_kl
            + self.cipt_ind_weight * loss_ind
        )

        z = None
        loss_c = loss.new_tensor(0.0)
        aug_logits = None
        # CutMix has a mixed target and therefore no single class-preserving
        # text intervention target. The default DCCL experiments use aug=0.
        if self.cipt_text_enable and not cutmix_active:
            z = self._make_text_views(causal_x)
            loss_c, aug_logits = self._text_classification_loss(z, all_y)
            loss = loss + self.cipt_lc_weight * loss_c

        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(
                all_x, ret_feats=True
            )

        reg_loss = loss.new_tensor(0.0)
        if self.l_d:
            for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(
                inter_feats,
                pre_feats,
                self.mean_encoders,
                self.var_encoders,
            ):
                mean = mean_enc(inter_f)
                var = var_enc(inter_f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss = reg_loss + vlb.mean() / 2.0
            loss = loss + self.l_d * reg_loss

        loss_sup_cl = loss.new_tensor(0.0)
        if self.l:
            embed_1 = self.proj_head(causal_x)
            embed_2 = self.proj_head(causal_x_2)
            view_1 = F.normalize(embed_1, dim=-1)
            view_2 = F.normalize(embed_2, dim=-1)
            view_list = [view_1, view_2]

            if z is not None:
                batch, num_text_views, dim = z.shape
                z_embed = self.proj_head(
                    z.reshape(batch * num_text_views, dim)
                ).reshape(batch, num_text_views, -1)
                z_embed = F.normalize(z_embed, dim=-1)
                for k in range(num_text_views):
                    view_list.append(z_embed[:, k, :])

            features = torch.stack(view_list, dim=1)

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                domain_views = [all_d, all_d_2]
                if z is not None:
                    domain_views.extend(
                        [all_d for _ in range(z.shape[1])]
                    )
                d = torch.unsqueeze(
                    torch.cat(domain_views), 1
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
                    feature_x_2_d = self.causal_adapter(
                        self.featurizer(all_x_2_d)
                    )
                    embed_2_d = self.proj_head(feature_x_2_d)
                    view_2_d = F.normalize(embed_2_d, dim=-1)
                    add_pos = torch.cat(
                        [view_2_d for _ in range(features.shape[1])],
                        dim=0,
                    )
                    loss_sup_cl = self.supcon_loss(
                        features, all_y, add_pos=add_pos
                    )
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
            loss = loss + self.l * loss_sup_cl

        pre_cl_loss = loss.new_tensor(0.0)
        if self.l_layer:
            # Preserve original DCCL pre_cl semantics on the raw backbone
            # feature, rather than forcing the frozen teacher into the adapter
            # space.
            embed_1_pre = self.pre_proj_head(feature_x)
            embed_2_pre = self.pre_proj_head(pre_pred_x)
            view_1_pre = F.normalize(embed_1_pre, dim=-1)
            view_2_pre = F.normalize(embed_2_pre, dim=-1)
            features_pre = torch.stack(
                [view_1_pre, view_2_pre], dim=1
            )

            if self.re_w:
                all_d = torch.cat(kwargs["d"])
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(
                    torch.cat([all_d, all_d_2]), 1
                ).float()
                neg_mask = torch.eq(d, d.T).float()
                pos_mask = 1 - neg_mask if self.pos_mask else None
                pre_cl_loss = self.supcon_loss_pre(
                    features_pre,
                    all_y,
                    neg_mask=neg_mask,
                    pos_mask=pos_mask,
                )
            else:
                pre_cl_loss = self.supcon_loss_pre(
                    features_pre, all_y
                )
            loss = loss + self.l_layer * pre_cl_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "DCCLCIPT produced a non-finite total loss."
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            causal_acc = (pred_x.argmax(dim=1) == all_y).float().mean()
            spurious_prob = F.softmax(spurious_logits, dim=-1)
            spurious_entropy = -(
                spurious_prob
                * torch.log(spurious_prob.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            if aug_logits is not None:
                aug_pred = aug_logits.argmax(dim=-1)
                repeated = (
                    all_y[:, None]
                    .expand(all_y.shape[0], z.shape[1])
                    .reshape(-1)
                )
                augmented_acc = (
                    aug_pred.reshape(-1) == repeated
                ).float().mean()
            else:
                augmented_acc = loss.new_tensor(0.0)

        loss_dict = {
            "loss": loss.item(),
            "ce_loss": loss_ce.item(),
            "kl_loss": loss_kl.item(),
            "ind_loss": loss_ind.item(),
            "c_loss": loss_c.item(),
            "causal_acc": causal_acc.item(),
            "spurious_entropy": spurious_entropy.item(),
            "augmented_acc": augmented_acc.item(),
        }
        if self.l:
            loss_dict["sup_cl_loss"] = loss_sup_cl.item()
        if self.l_layer:
            loss_dict["pre_cl_loss"] = pre_cl_loss.item()
        if self.l_d:
            loss_dict["reg_loss"] = reg_loss.item()
        return loss_dict

    def predict(self, x):
        feature = self.featurizer(x)
        causal = self.causal_adapter(feature)
        return self.classifier(causal)

    def predict_embed(self, x):
        return self.causal_adapter(self.featurizer(x))

    def get_forward_model(self):
        # SWAD averages exactly the inference-time path, including the new
        # causal adapter, while excluding training-only spurious/text modules.
        return nn.Sequential(
            self.featurizer, self.causal_adapter, self.classifier
        )
