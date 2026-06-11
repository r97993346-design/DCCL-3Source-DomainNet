# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from asyncio import ALL_COMPLETED
import copy
import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from domainbed.lib import misc
#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer
from domainbed import rise

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams, freeze=0, pre=True)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model
    
    def predict_embed(self, x):
        return self.network[0](x)


def get_shapes(model, input_shape):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]

    return shapes

class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x

class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps






class FeatureMasker(nn.Module):
    """Small MLP that predicts feature-wise mask logits for CIRL."""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z_orig, z_inter):
        return self.net(torch.cat([z_orig, z_inter], dim=1))

class ForwardModel(nn.Module):
    """Forward model is used to reduce gpu memory usage of SWAD.
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)

    # def predict_embed(self, x):
    #     return self.network[0](x)

class TN(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        down_layer1 = self.create_down_layer(in_channels, 32, kernel_size=(3, 3))
        down_layer2 = self.create_down_layer(32, 16, kernel_size=(3, 3))
        down_layer3 = self.create_down_layer(16, 8, kernel_size=(3, 3))

        up_layer1 = self.create_up_layer(8, 8, kernel_size=(3, 3))
        up_layer2 = self.create_up_layer(8, 16, kernel_size=(3, 3))
        up_layer3 = self.create_up_layer(16, 32, kernel_size=(3, 3))
        self.network = nn.Sequential(down_layer1, down_layer2, down_layer3, up_layer1, up_layer2, up_layer3)



    def create_down_layer(self, in_channels, out_channels, kernel_size, ):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                        nn.ReLU(),
                        nn.BatchNorm2d(out_channels),
                        nn.MaxPool2d(2)
                        )

    def create_up_layer(self, in_channels, out_channels, kernel_size, ):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                        nn.ReLU(),
                        nn.BatchNorm2d(out_channels),
                        nn.Upsample(scale_factor=2, mode='nearest')
                        )
    def forward(self, x):
        out = self.network(x)
        out = F.sigmoid(torch.max(out, dim=1, keepdim=True)[0])
        loss = torch.mean(out)
        return x*out, loss


class DCCL(Algorithm):
    """
    DCCL: Domain-Connecting Contrastive Learning
    
    The algorithm combines multiple loss components:
    1. Cross-entropy loss for classification
    2. Contrastive loss between augmented views (controlled by --l)
    3. Generative alignment regularization loss (controlled by --l_d) 
    4. Layer-wise contrastive loss with pre-trained features (controlled by --l_layer)
    5. Optional: We haven’t applied this parameter in our experiments. Transform Network for adversarial augmentation. (controlled by --TN)
    6. Optional: We haven’t applied this parameter in our experiments. CutMix data augmentation. Don't use in the exps. (controlled by --aug)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DCCL, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams, freeze=0, pre=True)
        self.pre_featurizer = networks.Featurizer(input_shape, self.hparams, freeze="all", pre=True)
        classifier_input_dim = self.featurizer.n_outputs
        hidden_num_1, hidden_num_2 = classifier_input_dim//4, classifier_input_dim//4
        self.num_classes = num_classes
        self.aug = hparams["aug"]
        # CIRL options. All branches are inert unless their flags are enabled.
        self.use_fourier_intervention = hparams.get("use_fourier_intervention", False)
        self.use_intervention_reliability = hparams.get("use_intervention_reliability", False)
        self.fourier_mix_alpha = hparams.get("fourier_mix_alpha", 1.0)
        self.fourier_mix_min = hparams.get("fourier_mix_min", 0.0)
        self.fourier_mix_max = hparams.get("fourier_mix_max", 1.0)
        self.intervention_mu = hparams.get("intervention_mu", 0.5)
        self.intervention_temperature = hparams.get("intervention_temperature", 0.1)
        self.reliability_min_weight = hparams.get("reliability_min_weight", 0.1)
        self.reliability_loss_weight = hparams.get("reliability_loss_weight", 1.0)
        self.use_factorization_loss = hparams.get("use_factorization_loss", False)
        self.factorization_loss_weight = hparams.get("factorization_loss_weight", 0.1)
        self.factorization_offdiag_weight = hparams.get("factorization_offdiag_weight", 0.005)
        self.factorization_eps = hparams.get("factorization_eps", 1e-4)
        self.factorization_feature_source = hparams.get("factorization_feature_source", "projection")
        self.use_adversarial_masker = hparams.get("use_adversarial_masker", False)
        self.masker_aux_loss_weight = hparams.get("masker_aux_loss_weight", 0.1)
        self.masker_adversarial_weight = hparams.get("masker_adversarial_weight", 1.0)
        self.mask_keep_ratio = hparams.get("mask_keep_ratio", 0.5)
        self.gumbel_temperature = hparams.get("gumbel_temperature", 1.0)
        self.gumbel_hard = hparams.get("gumbel_hard", True)
        self.masker_hidden_dim = hparams.get("masker_hidden_dim", 512)
        self.masker_update_interval = hparams.get("masker_update_interval", 1)
        self.update_count = 0
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim, num_classes))
        self.network = nn.Sequential(self.featurizer, self.classifier)
        if hparams["n_layer"]==1:
            self.proj_head = nn.Sequential(nn.Linear(self.featurizer.n_outputs, hidden_num_1))
        else:
            self.proj_head = nn.Sequential(nn.Linear(self.featurizer.n_outputs, hidden_num_1), nn.BatchNorm1d(hidden_num_1), nn.ReLU(), nn.Linear(hidden_num_1, hidden_num_2))
        self.proj = nn.Sequential(self.featurizer, self.proj_head)
        self.proj_output_dim = hidden_num_1 if hparams["n_layer"] == 1 else hidden_num_2
        if self.use_adversarial_masker:
            self.masker = FeatureMasker(self.proj_output_dim, self.masker_hidden_dim)
            self.superior_classifier = nn.Linear(self.proj_output_dim, num_classes)
            self.inferior_classifier = nn.Linear(self.proj_output_dim, num_classes)
        else:
            self.masker = None
            self.superior_classifier = None
            self.inferior_classifier = None

        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])

        # loss trade-offs
        self.l_layer = hparams["l_layer"]
        self.l = hparams["l"]
        self.l_d = hparams["l_d"]
        self.two_ce = hparams["two_ce"]
        self.pos_mask = hparams["pos_mask"]
        self.TN = hparams["TN"]
        self.lamda = hparams["lamda"]
        self.sample_d = hparams["sample_d"]
        self.use_rise = hparams.get("use_rise", False)
        self.use_rise_kd = self.use_rise and hparams.get("use_rise_kd", False)
        self.use_rise_proto = self.use_rise and hparams.get("use_rise_proto", False)
        self.rise_kd_weight = hparams.get("rise_kd_weight", 0.0)
        self.rise_proto_weight = hparams.get("rise_proto_weight", 0.0)
        self.rise_kd_temperature = hparams.get("rise_kd_temperature", 2.0)
        self.re_w = hparams.get("re_w", False)

        if self.use_rise:
            class_names = hparams.get("class_names", None)
            if class_names is None or len(class_names) != num_classes:
                raise ValueError(
                    "RISE-guided DCCL requires class_names aligned with labels. "
                    f"Expected {num_classes} names, got {0 if class_names is None else len(class_names)}."
                )
            self.rise_clip_model, self.rise_clip_module = rise.load_clip_teacher(
                hparams.get("rise_clip_model_name", "ViT-B/32"),
                device="cpu",
                freeze=hparams.get("rise_freeze_clip", True),
                download_root=hparams.get("rise_clip_download_root", None),
            )
            rise_prompt_mode = hparams.get("rise_prompt_mode", "multi")
            text_prototypes = rise.build_text_prototypes(
                self.rise_clip_model,
                self.rise_clip_module,
                class_names,
                rise_prompt_mode,
                device="cpu",
            )
            self.register_buffer("rise_text_prototypes", text_prototypes)
            self.rise_clip_input_normalize = rise.CLIPInputNormalize()
            self.rise_clip_dim = text_prototypes.shape[1]
            rise_prompt_count = rise.prompt_count(rise_prompt_mode)
            if rise_prompt_mode == "rise80" and rise_prompt_count != 80:
                raise ValueError(f"rise80 prompt mode must contain 80 templates, got {rise_prompt_count}.")
            first_class_prompts = rise.build_prompts(class_names[:1], rise_prompt_mode)[0][:5]
            logging.getLogger("SingletonLogger").info(
                "RISE prompt mode=%s, prompt_count=%d, E_proto shape=%s, "
                "first_class=%r, first_5_prompts=%s",
                rise_prompt_mode,
                rise_prompt_count,
                tuple(text_prototypes.shape),
                class_names[0],
                first_class_prompts,
            )
            if self.use_rise_proto:
                projection_dim = hparams.get("rise_projection_dim", self.rise_clip_dim)
                if projection_dim != self.rise_clip_dim:
                    raise ValueError(
                        "--rise_projection_dim must match the CLIP text embedding dimension "
                        f"({self.rise_clip_dim}) for {hparams.get('rise_clip_model_name', 'ViT-B/32')}. "
                        f"Got {projection_dim}."
                    )
                self.rise_projector = nn.Linear(classifier_input_dim, projection_dim)
            else:
                self.rise_projector = None
        else:
            self.rise_clip_model = None
            self.rise_projector = None

        if self.TN:
            self.TN_network = TN()
        self.weight_matrix=None
        
        # losses - simplified to only essential parameters
        self.supcon_loss = SupConLoss(hparams["t"])
        self.supcon_loss_pre = SupConLoss(hparams["t_pre"])
        self.con_loss = ConLoss(hparams["t"])

        # layer-wise contrast
        if hparams["n_layer"]==1:
            self.pre_proj_head = nn.Sequential(nn.Linear(self.featurizer.n_outputs, hidden_num_1))
        else:
            self.pre_proj_head = nn.Sequential(nn.Linear(self.featurizer.n_outputs, hidden_num_1), nn.BatchNorm1d(hidden_num_1), nn.ReLU(), nn.Linear(hidden_num_1, hidden_num_2))
        lower_cls=0.1
        lower_proj=10
        
        optimized_list = [{'params':self.featurizer.parameters(), 'lr':self.hparams["lr"], 'weight_decay':self.hparams["weight_decay"]}, 
            {'params':self.classifier.parameters(), 'lr':self.hparams["lr"]/lower_cls, 'weight_decay':self.hparams["weight_decay"]}, 
            {'params':self.proj_head.parameters(), 'lr':self.hparams["lr"]/lower_proj, 'weight_decay':self.hparams["weight_decay"]},
            {"params": self.mean_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.var_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.pre_proj_head.parameters(), "lr": self.hparams["lr"]/lower_proj}
            ]
        if self.use_adversarial_masker:
            optimized_list.extend([
                {"params": self.superior_classifier.parameters(), "lr": self.hparams["lr"], "weight_decay": self.hparams["weight_decay"]},
                {"params": self.inferior_classifier.parameters(), "lr": self.hparams["lr"], "weight_decay": self.hparams["weight_decay"]},
            ])
        if self.use_rise_proto:
            optimized_list.append({
                "params": self.rise_projector.parameters(),
                "lr": self.hparams["lr"],
                "weight_decay": self.hparams["weight_decay"],
            })
        if self.TN:
            self.optimizer_TN = get_optimizer(
            hparams["optimizer"],
            [{"params": self.TN_network.parameters(), "lr": self.hparams["lr"]}]
            )
        
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            optimized_list
        )
        if self.use_adversarial_masker:
            self.masker_optimizer = get_optimizer(
                hparams["optimizer"],
                [{"params": self.masker.parameters(), "lr": self.hparams["lr"], "weight_decay": self.hparams["weight_decay"]}],
            )

    def _imagenet_denormalize(self, x):
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
        return (x * std + mean).clamp(0.0, 1.0)

    def _imagenet_normalize(self, x):
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
        return (x.clamp(0.0, 1.0) - mean) / std

    def _cross_domain_donor_indices(self, domains):
        batch_size = domains.numel()
        perm = torch.randperm(batch_size, device=domains.device)
        if batch_size <= 1:
            return perm
        donor = perm.clone()
        for i in range(batch_size):
            candidates = torch.nonzero(domains != domains[i], as_tuple=False).flatten()
            if candidates.numel() > 0:
                donor[i] = candidates[torch.randint(candidates.numel(), (1,), device=domains.device)]
        return donor

    def _fourier_intervention(self, x, domains=None):
        if domains is None:
            donor_idx = torch.randperm(x.size(0), device=x.device)
        else:
            donor_idx = self._cross_domain_donor_indices(domains)
        x01 = self._imagenet_denormalize(x)
        donor01 = x01[donor_idx]
        fft_x = torch.fft.fft2(x01, dim=(-2, -1))
        fft_donor = torch.fft.fft2(donor01, dim=(-2, -1))
        amp_x, phase_x = torch.abs(fft_x), torch.angle(fft_x)
        amp_donor = torch.abs(fft_donor)
        alpha = self.fourier_mix_alpha
        if alpha < 0:
            low = min(self.fourier_mix_min, self.fourier_mix_max)
            high = max(self.fourier_mix_min, self.fourier_mix_max)
            alpha = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(low, high)
        else:
            alpha = float(max(self.fourier_mix_min, min(self.fourier_mix_max, alpha)))
        mixed_amp = (1.0 - alpha) * amp_x + alpha * amp_donor
        mixed_fft = torch.polar(mixed_amp, phase_x)
        x_inter = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
        return self._imagenet_normalize(x_inter)

    def _reliable_supcon_loss(self, features, labels, domains, reliability):
        batch_size = labels.shape[0]
        contrast_count = features.shape[1]
        flat_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = torch.matmul(flat_features, flat_features.T) / self.supcon_loss.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        repeated_labels = labels.repeat(contrast_count)
        repeated_domains = domains.repeat(contrast_count)
        label_mask = torch.eq(repeated_labels.view(-1, 1), repeated_labels.view(1, -1)).float()
        logits_mask = torch.ones_like(label_mask)
        logits_mask.scatter_(1, torch.arange(batch_size * contrast_count, device=features.device).view(-1, 1), 0)
        cross_domain = torch.ne(repeated_domains.view(-1, 1), repeated_domains.view(1, -1)).float()
        rel = reliability.repeat(contrast_count)
        rel_pair = torch.clamp(rel.view(-1, 1) * rel.view(1, -1), self.reliability_min_weight, 1.0)
        pos_weight = torch.ones_like(label_mask)
        pos_weight = torch.where((label_mask > 0) & (cross_domain > 0), rel_pair, pos_weight)
        mask = label_mask * logits_mask * pos_weight
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        denom = mask.sum(1).clamp_min(1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom
        valid = (mask.sum(1) > 0).float()
        loss = -mean_log_prob_pos * valid
        return loss.sum() / valid.sum().clamp_min(1.0)

    def _factorization_loss(self, z_orig, z_inter):
        z_orig = (z_orig - z_orig.mean(0)) / (z_orig.std(0, unbiased=False) + self.factorization_eps)
        z_inter = (z_inter - z_inter.mean(0)) / (z_inter.std(0, unbiased=False) + self.factorization_eps)
        corr = torch.matmul(z_orig.T, z_inter) / z_orig.size(0)
        on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
        off_diag = corr.flatten()[:-1].view(corr.size(0) - 1, corr.size(0) + 1)[:, 1:].flatten().pow(2).sum()
        return on_diag + self.factorization_offdiag_weight * off_diag

    def _gumbel_topk_mask(self, logits):
        keep = max(1, min(logits.size(1), int(round(logits.size(1) * self.mask_keep_ratio))))
        if self.training:
            noise = -torch.empty_like(logits).exponential_().log()
            scores = (logits + noise) / max(self.gumbel_temperature, 1e-8)
        else:
            scores = logits
        if self.gumbel_hard:
            idx = scores.topk(keep, dim=1).indices
            hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)
            soft = F.softmax(scores, dim=1) * keep
            return hard + soft - soft.detach()
        return (F.softmax(scores, dim=1) * keep).clamp(0.0, 1.0)

    def _masker_aux_losses(self, z_orig, z_inter, labels, detach_mask_for_aux):
        mask_logits = self.masker(z_orig, z_inter)
        mask = self._gumbel_topk_mask(mask_logits)
        if detach_mask_for_aux:
            mask = mask.detach()
        sup_z = z_orig * mask
        inf_z = z_orig * (1.0 - mask)
        sup_loss = F.cross_entropy(self.superior_classifier(sup_z), labels)
        inf_loss = F.cross_entropy(self.inferior_classifier(inf_z), labels)
        return sup_loss, inf_loss, mask, sup_z, inf_z

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        x_2 = kwargs["x_2"]
        all_x_2 = torch.cat(x_2)
        if self.TN:
            # all_x = self.TN_network(all_x)
            # update generator
            all_x_2, sp_loss = self.TN_network(all_x_2)
            feature_x = self.featurizer(all_x)
            feature_x_2 = self.featurizer(all_x_2)
            embed_2 = self.proj_head(feature_x_2)
            embed_1 = self.proj_head(feature_x)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
            features = torch.stack([view_1, view_2], dim=1)
            loss_sup_cl = self.supcon_loss(features, all_y)
            loss = -loss_sup_cl-self.lamda*sp_loss
            self.optimizer_TN.zero_grad()
            loss.backward()
            self.optimizer_TN.step()
            # update main
            with torch.no_grad():
                all_x_2, sp_loss = self.TN_network(all_x_2)
        r = np.random.rand(1)
        if self.aug and r<self.aug:
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            # cutmix
            bbx1, bby1, bbx2, bby2 = rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2]))
            # mix up
            # all_x = all_x*lam + all_x[rand_index]*(1-lam)
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            pred_x = self.classifier(feature_x)
            loss = F.cross_entropy(pred_x, target_a)*lam+F.cross_entropy(pred_x, target_b)*(1-lam)
        else:
            feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
            pred_x = self.classifier(feature_x)
            loss = F.cross_entropy(pred_x, all_y)
        ce_loss = loss.item()

        all_d = torch.cat(kwargs["d"]) if "d" in kwargs else None
        feature_x_2, inter_feats_2 = self.featurizer(all_x_2, ret_feats=True)
        z_orig = None
        z_inter = None
        reliability = None
        x_inter = None
        if self.use_fourier_intervention:
            x_inter = self._fourier_intervention(all_x, all_d)
            feature_x_inter = self.featurizer(x_inter)
            z_orig = nn.functional.normalize(self.proj_head(feature_x), dim=1)
            z_inter = nn.functional.normalize(self.proj_head(feature_x_inter), dim=1)
        if self.two_ce:
            loss = loss/2+F.cross_entropy(self.classifier(feature_x_2), all_y)/2
        with torch.no_grad():
            pre_pred_x, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
        if self.l_d:
            
            reg_loss = 0.
            for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(inter_feats, pre_feats, self.mean_encoders, self.var_encoders):
                mean = mean_enc(inter_f)
                var = var_enc(inter_f)
                vlb = (mean - pre_f).pow(2).div(var) + var.log()
                reg_loss += vlb.mean() / 2.

            loss += self.l_d*reg_loss
        if self.l:
            embed_2 = self.proj_head(feature_x_2)
            embed_1 = self.proj_head(feature_x)

            view_1 = nn.functional.normalize(embed_1, dim=1)
            view_2 = nn.functional.normalize(embed_2, dim=1)
            if z_orig is None:
                z_orig = view_1
            if z_inter is None:
                z_inter = view_2
            features = torch.stack([view_1, view_2], dim=1)
            if self.use_intervention_reliability:
                if all_d is None:
                    all_d = torch.zeros_like(all_y)
                features = torch.stack([z_orig, z_inter], dim=1)
                sim = F.cosine_similarity(z_orig.detach(), z_inter.detach(), dim=1)
                reliability = torch.sigmoid((sim - self.intervention_mu) / self.intervention_temperature)
                loss_sup_cl = self._reliable_supcon_loss(features, all_y, all_d, reliability)
                loss += self.l * self.reliability_loss_weight * loss_sup_cl
            elif self.re_w:
                all_d_2 = torch.cat(kwargs["d_2"])
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()

                if self.pos_mask:
                    pos_mask = 1-neg_mask
                else:
                    pos_mask = None
                loss_sup_cl = self.supcon_loss(features, all_y, neg_mask=neg_mask, pos_mask=pos_mask)
                loss += self.l*loss_sup_cl
            else:
                if self.sample_d:
                    all_x_2_d = torch.cat(kwargs["x_2_d"])
                    feature_x_2_d = self.featurizer(all_x_2_d)
                    embed_2_d = self.proj_head(feature_x_2_d)
                    view_2_d = nn.functional.normalize(embed_2_d, dim=1)
                    add_pos = torch.cat([view_2_d,view_2_d],0)
                    loss_sup_cl = self.supcon_loss(features, all_y, add_pos = add_pos)
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
                loss += self.l*loss_sup_cl
        pre_cl_loss = 0.
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
                d = torch.unsqueeze(torch.cat([all_d, all_d_2]), 1).float()
                neg_mask = torch.eq(d, d.T).float()

                if self.pos_mask:
                    pos_mask = 1-neg_mask
                else:
                    pos_mask = None
                pre_cl_loss += self.supcon_loss_pre(features, all_y_pre, neg_mask=neg_mask, pos_mask=pos_mask)
            else:
                pre_cl_loss += self.supcon_loss_pre(features, all_y_pre)
            loss += self.l_layer*pre_cl_loss

        factorization_loss = None
        if self.use_factorization_loss:
            if z_orig is None:
                z_orig = nn.functional.normalize(self.proj_head(feature_x), dim=1)
            if z_inter is None:
                z_inter = nn.functional.normalize(self.proj_head(feature_x_2), dim=1)
            factor_z_orig = z_orig
            factor_z_inter = z_inter
            factorization_loss = self._factorization_loss(factor_z_orig, factor_z_inter)
            loss += self.factorization_loss_weight * factorization_loss

        masker_aux_loss = None
        masker_loss = None
        if self.use_adversarial_masker:
            if z_orig is None:
                z_orig = nn.functional.normalize(self.proj_head(feature_x), dim=1)
            if z_inter is None:
                z_inter = nn.functional.normalize(self.proj_head(feature_x_2), dim=1)
            sup_aux, inf_aux, mask, sup_z, inf_z = self._masker_aux_losses(
                z_orig, z_inter, all_y, detach_mask_for_aux=True
            )
            masker_aux_loss = sup_aux + inf_aux
            loss += self.masker_aux_loss_weight * masker_aux_loss
            if self.update_count % max(1, self.masker_update_interval) == 0:
                self.masker_optimizer.zero_grad()
                sup_m, inf_m, _, _, _ = self._masker_aux_losses(
                    z_orig.detach(), z_inter.detach(), all_y, detach_mask_for_aux=False
                )
                masker_loss = sup_m - self.masker_adversarial_weight * inf_m
                masker_loss.backward()
                self.masker_optimizer.step()

        rise_kd_loss = None
        rise_proto_loss = None
        rise_clip_top1_match_label_ratio = None
        rise_clip_confidence_mean = None
        rise_proto_cosine_mean = None

        if self.use_rise and self.use_rise_kd:
            with torch.no_grad():
                clip_x = self.rise_clip_input_normalize(all_x)
                clip_img_feat = self.rise_clip_model.encode_image(clip_x).float()
                clip_img_feat = F.normalize(clip_img_feat, dim=1)
                clip_logits = clip_img_feat @ self.rise_text_prototypes.detach().T
                logit_scale = getattr(self.rise_clip_model, "logit_scale", None)
                if logit_scale is not None:
                    clip_logits = clip_logits * logit_scale.exp().detach()
                clip_logits = clip_logits.detach()
                clip_probs = F.softmax(clip_logits, dim=1)
                rise_clip_confidence_mean = clip_probs.max(dim=1).values.mean()
                rise_clip_top1_match_label_ratio = (clip_probs.argmax(dim=1) == all_y).float().mean()
            rise_kd_loss = rise.clip_kd_loss(pred_x, clip_logits, self.rise_kd_temperature)
            loss += self.rise_kd_weight * rise_kd_loss

        if self.use_rise and self.use_rise_proto:
            projected_feature = self.rise_projector(feature_x)
            rise_proto_loss, rise_proto_cosine_mean = rise.prototype_alignment_loss(
                projected_feature, all_y, self.rise_text_prototypes
            )
            loss += self.rise_proto_weight * rise_proto_loss

        loss_opt = loss
        self.optimizer.zero_grad()
        loss_opt.backward()
        self.optimizer.step()
        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss}
        if self.use_fourier_intervention:
            loss_dict["fourier_intervention"] = 1.0
        if self.use_intervention_reliability:
            loss_dict["reliability_mean"] = reliability.mean().item() if reliability is not None else 0.0
            loss_dict["reliability_loss_weight"] = float(self.reliability_loss_weight)
        if self.use_factorization_loss:
            loss_dict["loss_factorization"] = factorization_loss.item() if factorization_loss is not None else 0.0
        if self.use_adversarial_masker:
            loss_dict["loss_masker_aux"] = masker_aux_loss.item() if masker_aux_loss is not None else 0.0
            loss_dict["loss_masker"] = masker_loss.item() if masker_loss is not None else 0.0
        if self.l:
            loss_dict["sup_cl_loss"] = loss_sup_cl.item()
        if self.l_layer:
            loss_dict["pre_cl_loss"] = pre_cl_loss.item()
        if self.use_rise:
            loss_dict.update({
                "loss_cls": ce_loss,
                "loss_dccl": loss_sup_cl.item() if self.l else 0.0,
                "loss_rise_kd": rise_kd_loss.item() if rise_kd_loss is not None else 0.0,
                "loss_rise_proto": rise_proto_loss.item() if rise_proto_loss is not None else 0.0,
                "rise_kd_weight": float(self.rise_kd_weight),
                "rise_proto_weight": float(self.rise_proto_weight),
                "rise_clip_top1_match_label_ratio": (
                    rise_clip_top1_match_label_ratio.item()
                    if rise_clip_top1_match_label_ratio is not None else 0.0
                ),
                "rise_clip_confidence_mean": (
                    rise_clip_confidence_mean.item()
                    if rise_clip_confidence_mean is not None else 0.0
                ),
                "rise_proto_cosine_mean": (
                    rise_proto_cosine_mean.item()
                    if rise_proto_cosine_mean is not None else 0.0
                ),
            })
        self.update_count += 1
        return loss_dict

    def predict(self, x):
        return self.network(x)

    def predict_embed(self, x):
        return self.featurizer(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.3, mask_out=False, neg_mix=False, not_sup=False, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.neg_mix = neg_mix
        self.base_temperature = temperature
        self.mask_out = mask_out
        self.not_sup = not_sup

    def forward(self, features, labels=None, mask=None, neg_mask=None, pos_mask=None, add_pos=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif self.not_sup:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if self.neg_mix:
            self_dot = torch.sum(contrast_feature*contrast_feature, 1, keepdim=True)/self.temperature
            # lam = np.random.beta(1, 1)
            anchor_dot_contrast_neg = 0.5*anchor_dot_contrast+0.5*self_dot
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_out = 1-mask
        # mask-out self-contrast cases
        # self, dimension->index (value) along which dim, index, src->consistent indexing with index
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1, 
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # diagnal
        mask = mask * logits_mask
        # CUDA_VISIBLE_DEVICES=1 python train_all.py SiamOH0_sup_low_1 --dataset TerraIncognita --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data --algorithm SupContrast --l 1 --t 0.1 --sup --sample_d --re_w --l_d 0
        # compute log_prob
        if self.neg_mix:
            exp_logits = torch.exp(anchor_dot_contrast_neg - logits_max.detach()) * logits_mask
        else:
            exp_logits = torch.exp(logits) * logits_mask
        if neg_mask is not None:
            exp_logits = exp_logits*neg_mask
        if self.mask_out:
            exp_logits = exp_logits*mask_out
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if pos_mask is not None:
            log_prob = log_prob*pos_mask
        if add_pos is not None:
            add_logits = torch.sum(add_pos*contrast_feature, 1, keepdim=True)/self.temperature - logits_max.detach()
            add_logits = torch.squeeze(add_logits)
            mean_log_prob_pos = ((mask * log_prob).sum(1)+add_logits) / (mask.sum(1)+1)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class ConLoss(nn.Module):
    def __init__(self, temperature=0.3, contrast_mode='all'):
        super(ConLoss, self).__init__()
        self.t = temperature

    def forward(self, x1, x2, mask=None, weight=False):
        T = self.t
        batch_size, _ = x1.size()
        
        # batch_size *= 2
        # x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # if weight:
        #     sim_matrix = sim_matrix*mask
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        
        return loss

class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
