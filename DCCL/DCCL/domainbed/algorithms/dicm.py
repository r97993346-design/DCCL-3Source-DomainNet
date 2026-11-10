"""DICM replaces DCCL's GT branch and feeds all student tasks through E.

The original DCCL class remains intact. A1-A4 inherit Algorithm directly and do
not construct GT. Only the explicit A5 control restores the original GT branch.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from domainbed import networks
from domainbed.lib import misc
from domainbed.lib.dicm_hparams import finalize_dicm_hparams
from domainbed.optimizers import get_optimizer

from .algorithms import Algorithm, MeanEncoder, SupConLoss, VarianceEncoder, get_shapes
from .dicm_components import (
    ClassDomainResidualBank,
    PairedInterventionResponseEstimator,
    StableFeatureProjection,
)


class DICMForwardModel(nn.Module):
    """Exactly the inference path; SWAD averages its raw B before fresh QR."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def predict_embed(self, x):
        return self.network[1](self.network[0](x))

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)


class DICM(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        hparams = finalize_dicm_hparams(dict(hparams))
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.mode = hparams["dicm_ablation"]
        self.use_projection = self.mode != "A1"
        self.use_gt = self.mode == "A5"
        self.gt_weight = float(hparams["l_d"]) if self.use_gt else 0.0
        self.l = float(hparams["l"])
        self.l_layer = float(hparams["l_layer"])
        self.isr_weight = float(hparams["dicm_isr_weight"])
        self.aug_weight = float(hparams["dicm_aug_weight"])
        self.dom_weight = float(hparams["dicm_dom_weight"])
        self.min_norm = float(hparams["dicm_min_delta_norm"])
        if not math.isfinite(self.min_norm) or self.min_norm < 0:
            raise ValueError("dicm_min_delta_norm must be finite and nonnegative")
        self.two_ce = bool(hparams.get("two_ce", False))
        self.sample_d = bool(hparams.get("sample_d", False))
        self.re_w = False

        self.featurizer = networks.Featurizer(input_shape, hparams, freeze=0, pre=True)
        self.pre_featurizer = networks.Featurizer(input_shape, hparams, freeze="all", pre=True)
        self.pre_featurizer.requires_grad_(False)
        self.pre_featurizer.eval()
        feature_dim = self.featurizer.n_outputs
        hidden_dim = feature_dim // 4
        if hidden_dim < 1:
            raise ValueError("DCCL heads require at least four pooled dimensions")
        self.classifier = nn.Sequential(nn.Linear(feature_dim, num_classes))
        self.proj_head = self._head(feature_dim, hidden_dim, int(hparams["n_layer"]))

        if self.use_gt:
            shapes = get_shapes(self.pre_featurizer, input_shape)
            if not shapes:
                raise ValueError("A5 GT requires hooked intermediate backbone features")
            self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
            self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
            self.pre_featurizer.clear_features()
        else:
            # Match DCCL's dummy RNG draw without constructing/executing GT.
            # This aligns common initial parameters for default dropout=0.
            torch.rand(1, *input_shape)
        self.pre_proj_head = self._head(feature_dim, hidden_dim, int(hparams["n_layer"]))
        self.supcon_loss = SupConLoss(float(hparams["t"]))
        self.supcon_loss_pre = SupConLoss(float(hparams["t_pre"]))

        if self.use_projection:
            seed = int(hparams["dicm_basis_seed"])
            if seed < 0:
                seed = torch.initial_seed()
            self.projection = StableFeatureProjection(
                feature_dim, int(hparams["dicm_rank"]), float(hparams["dicm_beta"]), seed,
            )
            self.pire = PairedInterventionResponseEstimator()
            minimum_domains = int(hparams["dicm_min_valid_domains"]) or num_domains
            self.residual_bank = ClassDomainResidualBank(
                num_classes, num_domains, feature_dim,
                momentum=float(hparams["dicm_proto_momentum"]),
                min_count=int(hparams["dicm_min_domain_samples"]),
                min_updates=int(hparams["dicm_min_domain_updates"]),
                min_valid_domains=minimum_domains,
            )
        else:
            # A1 constructs no intervention parameters or prototype state.
            self.projection = nn.Identity()
        self.network = nn.Sequential(self.featurizer, self.projection, self.classifier)
        self.optimizer = get_optimizer(hparams["optimizer"], self._optimizer_groups())

    @staticmethod
    def _head(feature_dim, hidden_dim, layers):
        if layers == 1:
            return nn.Sequential(nn.Linear(feature_dim, hidden_dim))
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def _optimizer_groups(self):
        lr, wd = float(self.hparams["lr"]), float(self.hparams["weight_decay"])
        groups = [
            {"params": self.featurizer.parameters(), "lr": lr, "weight_decay": wd},
            {"params": self.classifier.parameters(), "lr": lr / 0.1, "weight_decay": wd},
        ]
        if self.l:
            groups.append({"params": self.proj_head.parameters(), "lr": lr / 10, "weight_decay": wd})
        if self.use_gt:
            # MeanEncoder is currently parameter-free; variance uses original 10x lr.
            gt_parameters = list(self.mean_encoders.parameters()) + list(self.var_encoders.parameters())
            groups.append({"params": gt_parameters, "lr": lr * 10, "weight_decay": 0.0})
        if self.l_layer:
            # Original DCCL does not assign weight decay to this group.
            groups.append({"params": self.pre_proj_head.parameters(), "lr": lr / 10, "weight_decay": 0.0})
        if self.use_projection:
            basis_lr = float(self.hparams["dicm_basis_lr"]) or lr
            if not math.isfinite(basis_lr) or basis_lr <= 0:
                raise ValueError("dicm_basis_lr must be positive or 0 to inherit lr")
            groups.append({"params": self.projection.parameters(), "lr": basis_lr, "weight_decay": 0.0})
        return groups

    def train(self, mode=True):
        super().train(mode)
        self.pre_featurizer.eval()
        return self

    def _domain_ids(self, minibatches):
        if len(minibatches) != self.num_domains:
            raise ValueError("DICM.update expects one minibatch per source domain only")
        return torch.cat([
            torch.full((len(batch),), d, dtype=torch.long, device=batch.device)
            for d, batch in enumerate(minibatches)
        ])

    def _isr_loss(self, delta_aug, delta_dom, q):
        subspace = self.projection.subspace
        aug, n_aug = subspace.coverage_loss(delta_aug, q, self.min_norm)
        dom, n_dom = subspace.coverage_loss(delta_dom, q, self.min_norm)
        aw = self.aug_weight if n_aug else 0.0
        dw = self.dom_weight if n_dom else 0.0
        total_weight = aw + dw
        if total_weight:
            aw, dw = aw / total_weight, dw / total_weight
        # Do not retain a zero-weight graph, including Adam momentum on B.
        loss = aug.new_zeros(())
        if aw:
            loss = loss + aw * aug
        if dw:
            loss = loss + dw * dom
        return loss, aug, dom, n_aug, n_dom, aw, dw

    def _stable(self, features, q):
        return self.projection(features, q=q) if self.use_projection else features

    def _gt_loss(self, student_features, reference_features):
        if not self.use_gt:
            raise RuntimeError("GT exists only in A5")
        if not student_features or not reference_features:
            raise ValueError("A5 GT needs nonempty intermediate feature snapshots")
        loss = student_features[0].new_zeros(())
        for student, reference, mean_encoder, variance_encoder in misc.zip_strict(
                student_features, reference_features, self.mean_encoders, self.var_encoders):
            if student.shape != reference.shape:
                raise ValueError("GT student/reference intermediate shapes must match")
            mean, variance = mean_encoder(student), variance_encoder(student)
            loss = loss + ((mean - reference.detach()).square() / variance + variance.log()).mean() / 2.0
        return loss

    @staticmethod
    def _gradient_norm(parameters):
        gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return float(torch.stack(gradients).norm().item()) if gradients else 0.0

    def update(self, x, y, **kwargs):
        if not self.training:
            raise RuntimeError("DICM.update is training-only; predict never updates EMA")
        domains = self._domain_ids(x)
        x_aug = kwargs["x_2"]
        if len(x_aug) != len(x) or len(y) != len(x):
            raise ValueError("Source labels and paired views must match source minibatches")
        if any(a.shape != b.shape or len(a) != len(label) for a, b, label in zip(x, x_aug, y)):
            raise ValueError("Each source requires matching paired views and labels")
        all_x, all_x_aug, all_y = torch.cat(x), torch.cat(x_aug), torch.cat(y)
        step = int(kwargs.get("step", 0))
        # Keep the default DCCL per-update NumPy RNG consumption (aug=0).
        np.random.rand(1)
        z = self.featurizer(all_x)
        gt_student = gt_reference = None
        if self.use_gt:
            # main's shared hook list makes GT read the SECOND student view.
            # Preserve that actual baseline pairing explicitly, without aliasing.
            z_aug, gt_student = self.featurizer(all_x_aug, ret_feats=True)
            gt_student = tuple(gt_student)
        else:
            z_aug = self.featurizer(all_x_aug)
        with torch.no_grad():
            if self.use_gt:
                z_ref, gt_reference = self.pre_featurizer(all_x, ret_feats=True)
                # Freeze this FIRST-reference-view container before PIRE's next call.
                gt_reference = tuple(gt_reference)
            else:
                z_ref = self.pre_featurizer(all_x)
            z_aug_ref = self.pre_featurizer(all_x_aug) if self.use_projection else None

        zero = z.new_zeros(())
        q = None
        loss_isr = loss_aug = loss_dom = zero
        n_aug = n_dom = 0
        aw = dw = 0.0
        delta_aug = delta_dom = z.detach().new_empty((0, z.shape[1]))
        if self.use_projection:
            q = self.projection.subspace.orthonormal_basis()
            delta_aug = self.pire(z, z_aug, z_ref, z_aug_ref)
            self.residual_bank.update((z - z_ref).detach(), all_y, domains, step, self.min_norm)
            delta_dom = self.residual_bank.domain_responses()
            loss_isr, loss_aug, loss_dom, n_aug, n_dom, aw, dw = self._isr_loss(delta_aug, delta_dom, q)

        e, e_aug = self._stable(z, q), self._stable(z_aug, q)
        logits = self.classifier(e)
        first_ce = F.cross_entropy(logits, all_y)
        ce = first_ce
        if self.two_ce:
            ce = 0.5 * (ce + F.cross_entropy(self.classifier(e_aug), all_y))

        sup_cl = zero
        if self.l:
            # Preserve original head call order and view-major SupCon ordering.
            view_aug = F.normalize(self.proj_head(e_aug), dim=1)
            view = F.normalize(self.proj_head(e), dim=1)
            features = torch.stack([view, view_aug], dim=1)
            if self.sample_d:
                z_extra = self.featurizer(torch.cat(kwargs["x_2_d"]))
                extra = F.normalize(self.proj_head(self._stable(z_extra, q)), dim=1)
                sup_cl = self.supcon_loss(features, all_y, add_pos=torch.cat([extra, extra]))
            else:
                sup_cl = self.supcon_loss(features, all_y)

        pre_cl = zero
        if self.l_layer:
            view = F.normalize(self.pre_proj_head(e), dim=1)
            # z_ref is frozen, but this shared head still trains on both sides.
            reference_view = F.normalize(self.pre_proj_head(z_ref), dim=1)
            pre_cl = self.supcon_loss_pre(torch.stack([view, reference_view], dim=1), all_y)

        task_loss = ce + self.l * sup_cl + self.l_layer * pre_cl
        reg_loss = self._gt_loss(gt_student, gt_reference) if self.use_gt else zero
        weighted_gt = self.gt_weight * reg_loss
        weighted_isr = self.isr_weight * loss_isr
        loss = task_loss
        if self.use_gt:
            loss = loss + weighted_gt
        if self.use_projection and self.isr_weight > 0 and (aw > 0 or dw > 0):
            loss = loss + weighted_isr
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DICM total loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Diagnostics belong to the same (theta_t, B_t) as this forward, before step.
        with torch.no_grad():
            metrics = {
                "loss": float(loss.item()), "ce_loss": float(first_ce.item()),
                "loss_cls": float(ce.item()), "sup_cl_loss": float(sup_cl.item()),
                "pre_cl_loss": float(pre_cl.item()), "task_loss": float(task_loss.item()),
                "reg_loss": float(reg_loss.item()), "gt_enabled": float(self.use_gt),
                "gt_weight": self.gt_weight, "weighted_gt_loss": float(weighted_gt.item()),
                "gt_layer_count": float(len(self.mean_encoders)) if self.use_gt else 0.0,
                "gt_grad_norm": self._gradient_norm(self.var_encoders.parameters()) if self.use_gt else 0.0,
                "loss_isr": float(loss_isr.item()), "loss_isr_aug": float(loss_aug.item()),
                "loss_isr_dom": float(loss_dom.item()), "weighted_loss_isr": float(weighted_isr.item()),
                "isr_aug_weight_effective": aw, "isr_dom_weight_effective": dw,
                "valid_aug_response_count": float(n_aug), "valid_dom_response_count": float(n_dom),
                "dicm_enabled": float(self.use_projection),
                "supcon_enabled": float(bool(self.l)), "precl_enabled": float(bool(self.l_layer)),
                "supcon_weight": self.l, "precl_weight": self.l_layer,
                "original_feature_norm": float(z.norm(dim=1).mean().item()),
                "stable_feature_norm": float(e.norm(dim=1).mean().item()),
                "feature_delta_ratio": float(((e - z).norm(dim=1).mean() / z.norm(dim=1).mean().clamp_min(1e-12)).item()),
                "raw_aug_distance": float((z_aug - z).norm(dim=1).mean().item()),
                "stable_aug_distance": float((e_aug - e).norm(dim=1).mean().item()),
                "batch_accuracy": float((logits.argmax(1) == all_y).float().mean().item()),
                "backbone_grad_norm": self._gradient_norm(self.featurizer.parameters()),
                "classifier_grad_norm": self._gradient_norm(self.classifier.parameters()),
                "proj_grad_norm": self._gradient_norm(self.proj_head.parameters()),
                "pre_proj_grad_norm": self._gradient_norm(self.pre_proj_head.parameters()),
                "has_nan_or_inf": 0.0,
            }
            if self.use_projection:
                metrics.update(self.projection.subspace.diagnostics(q))
                sensitive = self.projection.subspace.project(z.detach(), q=q)
                energy = z.detach().float().square().sum(1).clamp_min(1e-12)
                valid = self.residual_bank.valid_mask()
                initialized = self.residual_bank.initialized
                ages = step - self.residual_bank.last_update_step[initialized]
                counts = torch.bincount(all_y, minlength=self.num_classes).float()
                centers = z.detach().float().new_zeros((self.num_classes, z.shape[1]))
                centers.index_add_(0, all_y, z.detach().float())
                present = counts > 0
                centers = centers[present] / counts[present, None] - z.detach().float().mean(0)
                class_sensitive = self.projection.subspace.project(centers, q=q)
                class_energy = (counts[present] * centers.square().sum(1)).sum().clamp_min(1e-12)
                metrics.update({
                    "dicm_beta": float(self.projection.beta.item()),
                    "sensitive_energy_ratio": float((sensitive.float().square().sum(1) / energy).mean().item()),
                    "isr_aug_capture": 1.0 - float(loss_aug.item()) if n_aug else 0.0,
                    "isr_dom_capture": 1.0 - float(loss_dom.item()) if n_dom else 0.0,
                    "delta_aug_norm": float(delta_aug.norm(dim=1).mean().item()),
                    "delta_dom_norm": float(delta_dom.norm(dim=1).mean().item()) if delta_dom.numel() else 0.0,
                    "valid_prototype_count": float(valid.sum().item()),
                    "valid_prototype_classes": float((valid.sum(1) >= self.residual_bank.min_valid_domains).sum().item()),
                    "prototype_age_max": float(ages.max().item()) if ages.numel() else 0.0,
                    "prototype_sample_count_min": float(self.residual_bank.counts[valid].min().item()) if valid.any() else 0.0,
                    "prototype_update_count_min": float(self.residual_bank.update_counts[valid].min().item()) if valid.any() else 0.0,
                    "class_between_removed_ratio": float(((counts[present] * class_sensitive.square().sum(1)).sum() / class_energy).item()),
                    "reference_aug_distance": float((z_aug_ref - z_ref).norm(dim=1).mean().item()),
                    "basis_grad_norm": self._gradient_norm(self.projection.parameters()),
                })
            for index, group in enumerate(self.optimizer.param_groups):
                metrics["param_group_lr_" + str(index)] = float(group["lr"])
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite DICM diagnostics or gradients; optimizer step skipped")
        self.optimizer.step()
        return metrics

    def predict_embed(self, x):
        return self.projection(self.featurizer(x))

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        return DICMForwardModel(self.network)

    def clone(self):
        # Hook containers may hold non-leaf tensors after a training forward.
        for module in self.modules():
            if hasattr(module, "clear_features"):
                module.clear_features()
        clone = copy.deepcopy(self)
        clone.optimizer = get_optimizer(clone.hparams["optimizer"], clone._optimizer_groups())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone
