"""Low-disturbance Paired-Intervention Causal Connectivity Learning."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import DCCL
from domainbed.optimizers import get_optimizer


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "y", "on"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


class PairedInterventionResponseEstimator(nn.Module):
    def forward(self, z, z_int, z_ref, z_int_ref):
        tensors = {"z": z, "z_int": z_int, "z_ref": z_ref, "z_int_ref": z_int_ref}
        if any(t.dim() != 2 for t in tensors.values()):
            raise ValueError("PIRE inputs must be pooled feature tensors [B,D]")
        if len({tuple(t.shape) for t in tensors.values()}) != 1:
            raise ValueError("PIRE inputs must have identical shapes")
        return (z_int - z) - (z_int_ref - z_ref).detach()


class ClassDomainResidualBank(nn.Module):
    """EMA class/domain residual prototypes, usable only after sufficient data."""
    def __init__(self, num_classes, num_domains, feature_dim, momentum=0.99,
                 min_count=8, min_valid_domains=2):
        super().__init__()
        self.num_classes, self.num_domains, self.feature_dim = num_classes, num_domains, feature_dim
        self.momentum, self.min_count, self.min_valid_domains = momentum, min_count, min_valid_domains
        self.register_buffer("prototypes", torch.zeros(num_classes, num_domains, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, num_domains, dtype=torch.bool))
        self.register_buffer("counts", torch.zeros(num_classes, num_domains, dtype=torch.long))

    @torch.no_grad()
    def update(self, residual, labels, domains):
        for c in labels.detach().long().unique().tolist():
            for d in domains.detach().long().unique().tolist():
                if not (0 <= c < self.num_classes and 0 <= d < self.num_domains):
                    continue
                mask = (labels == c) & (domains == d)
                if not mask.any():
                    continue
                current = residual.detach()[mask].mean(0)
                if self.initialized[c, d]:
                    self.prototypes[c, d].mul_(self.momentum).add_(current, alpha=1 - self.momentum)
                else:
                    self.prototypes[c, d].copy_(current)
                    self.initialized[c, d] = True
                self.counts[c, d].add_(int(mask.sum()))

    def domain_responses(self):
        responses = []
        for c in range(self.num_classes):
            valid = self.initialized[c] & (self.counts[c] >= self.min_count)
            if int(valid.sum()) < self.min_valid_domains:
                continue
            prototype = self.prototypes[c, valid]
            responses.append(prototype - prototype.mean(0, keepdim=True))
        if not responses:
            return self.prototypes.new_zeros((0, self.feature_dim))
        return torch.cat(responses).detach()


class InterventionSensitiveSubspace(nn.Module):
    def __init__(self, feature_dim, rank=16, eps=1e-8):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(feature_dim, min(int(rank), feature_dim)) * .02)
        self.eps = eps

    def orthonormal_basis(self, dtype=None):
        q = torch.linalg.qr(self.basis.float(), mode="reduced").Q
        return q if dtype is None else q.to(dtype=dtype)

    def project(self, v, detach_basis=True):
        if v.numel() == 0:
            return v
        q = self.orthonormal_basis(v.dtype).to(v.device)
        if detach_basis:
            q = q.detach()
        return (v @ q) @ q.T

    def coverage_loss(self, responses):
        responses = responses.detach()
        valid = responses.pow(2).sum(1) > self.eps
        if not valid.any():
            return self.basis.sum() * 0
        v = responses[valid]
        projected = self.project(v, detach_basis=False)
        return ((v - projected).pow(2).sum(1) / (v.pow(2).sum(1) + self.eps)).mean()


class CausalMediatorProjection(nn.Module):
    """Orthogonal low-rank causal projection; beta=0 is bitwise identity."""
    def forward(self, z, subspace, beta):
        if float(beta.detach()) == 0.0:
            return z
        sensitive = subspace.project(z, detach_basis=True)
        return z - beta.to(z.device, z.dtype) * sensitive


class PICCLForwardModel(nn.Module):
    def __init__(self, featurizer, sensitive_subspace, causal_mediator, classifier, piccl_beta):
        super().__init__()
        self.featurizer = featurizer
        self.sensitive_subspace = sensitive_subspace
        self.causal_mediator = causal_mediator
        self.classifier = classifier
        self.register_buffer("piccl_beta", piccl_beta.detach().clone())

    def predict_embed(self, x):
        z = self.featurizer(x)
        return self.causal_mediator(z, self.sensitive_subspace, self.piccl_beta)

    def predict(self, x):
        return self.classifier(self.predict_embed(x))

    forward = predict


class PICCL(DCCL):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Read this before creating any PICCL module: disabled PICCL is exact DCCL.
        self.use_piccl = parse_bool(hparams["use_piccl"])
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if not self.use_piccl:
            return
        for p in self.pre_featurizer.parameters():
            p.requires_grad = False
        self.pre_featurizer.eval()
        dim = self.featurizer.n_outputs
        self.pire = PairedInterventionResponseEstimator()
        self.residual_bank = ClassDomainResidualBank(num_classes, num_domains, dim)
        self.sensitive_subspace = InterventionSensitiveSubspace(dim, hparams["piccl_rank"])
        self.causal_mediator = CausalMediatorProjection()
        self.register_buffer("piccl_beta", torch.tensor(0.0))
        self.optimizer = get_optimizer(hparams["optimizer"], self._optimizer_groups())

    def _dccl_optimizer_groups(self):
        return [
            {"params": self.featurizer.parameters(), "lr": self.hparams["lr"], "weight_decay": self.hparams["weight_decay"]},
            {"params": self.classifier.parameters(), "lr": self.hparams["lr"] / .1, "weight_decay": self.hparams["weight_decay"]},
            {"params": self.proj_head.parameters(), "lr": self.hparams["lr"] / 10, "weight_decay": self.hparams["weight_decay"]},
            {"params": self.mean_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.var_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.pre_proj_head.parameters(), "lr": self.hparams["lr"] / 10},
        ]

    def _optimizer_groups(self):
        groups = self._dccl_optimizer_groups()
        if self.use_piccl:
            groups.append({"params": self.sensitive_subspace.parameters(), "lr": self.hparams["lr"] * .25,
                           "weight_decay": self.hparams["weight_decay"]})
        return groups

    def train(self, mode=True):
        super().train(mode)
        if self.use_piccl:
            self.pre_featurizer.eval()
        return self

    def _beta(self, step):
        total = max(int(self.hparams.get("piccl_total_steps", 1)) - 1, 1)
        progress = step / total
        beta = 0.0 if progress <= .10 else min((progress - .10) / .20, 1.0) * self.hparams["piccl_beta_max"]
        self.piccl_beta.fill_(beta)
        return self.piccl_beta

    @staticmethod
    def _domain_ids(x):
        return torch.cat([torch.full((a.shape[0],), i, device=a.device, dtype=torch.long) for i, a in enumerate(x)])

    def _project(self, z, beta):
        return self.causal_mediator(z, self.sensitive_subspace, beta)

    @staticmethod
    def _pooled_features(features):
        """Convert [B,C,H,W] or [B,D] causal features to [B,D]."""
        if features.dim() == 4:
            return F.adaptive_avg_pool2d(features, 1).flatten(1)
        if features.dim() == 2:
            return features
        raise ValueError("causal features must have shape [B,D] or [B,C,H,W]")

    def _causal_reliability(self, h_factual, h_intervened):
        """Detached cosine agreement of factual and intervention features, [B]."""
        factual = F.normalize(self._pooled_features(h_factual), dim=1)
        intervened = F.normalize(self._pooled_features(h_intervened), dim=1)
        reliability = ((factual * intervened).sum(1) + 1.).div(2.).clamp(0., 1.)
        # Reliability is a fixed loss weight: never allow the model to change it
        # through gradient descent, even if a legacy config sets the flag false.
        reliability = reliability.detach()
        if not torch.isfinite(reliability).all():
            raise RuntimeError("causal reliability contains NaN or Inf")
        return reliability

    def _reliability_gamma(self, step):
        total_steps = max(int(self.hparams.get("piccl_total_steps", 1)), 1)
        warmup = total_steps * float(self.hparams["piccl_reliability_warmup_ratio"])
        ramp = total_steps * float(self.hparams["piccl_reliability_ramp_ratio"])
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(max((step - warmup) / ramp, 0.0), 1.0)

    @staticmethod
    def _expand_view_major(values, num_views=2):
        """Expand [B,...] as cat(unbind(features, dim=1)): all view 0 then view 1."""
        return values.repeat((num_views,) + (1,) * (values.dim() - 1))

    def _causal_positive_pair_weights(self, labels, domain_ids, sample_ids, view_ids,
                                      reliability, reliability_min, cross_domain_only=True):
        """Build [B*V,B*V] weights; only cross-domain same-class pairs vary."""
        same_class = labels[:, None].eq(labels[None, :])
        same_domain = domain_ids[:, None].eq(domain_ids[None, :])
        self_augmentation = sample_ids[:, None].eq(sample_ids[None, :]) & view_ids[:, None].ne(view_ids[None, :])
        cross_domain_positive_mask = same_class & ~same_domain if cross_domain_only else torch.zeros_like(same_class)
        pair_reliability = torch.sqrt(reliability[:, None] * reliability[None, :])
        weights = torch.ones_like(pair_reliability)
        cross_weights = reliability_min + (1. - reliability_min) * pair_reliability
        weights = torch.where(cross_domain_positive_mask, cross_weights, weights)
        if not torch.isfinite(weights).all() or not ((weights >= reliability_min) & (weights <= 1.)).all():
            raise RuntimeError("causal positive weights are outside [reliability_min, 1]")
        if not torch.all(weights[self_augmentation] == 1) or not torch.all(weights[same_class & same_domain] == 1):
            raise RuntimeError("non-cross-domain positive weights must be one")
        return weights, cross_domain_positive_mask, self_augmentation

    def _isr_loss(self, delta_pair, delta_domain):
        aug = self.sensitive_subspace.coverage_loss(delta_pair)
        dom_valid = delta_domain.numel() > 0 and bool((delta_domain.pow(2).sum(1) > 1e-8).any())
        if not dom_valid:
            return aug, aug, aug
        dom = self.sensitive_subspace.coverage_loss(delta_domain)
        return .5 * (aug + dom), aug, dom

    def _gt_loss(self, inter_feats, pre_feats):
        from domainbed.lib import misc
        reg_loss = inter_feats[0].new_tensor(0.)
        for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(inter_feats, pre_feats, self.mean_encoders, self.var_encoders):
            var = var_enc(inter_f)
            reg_loss = reg_loss + ((mean_enc(inter_f) - pre_f).pow(2).div(var) + var.log()).mean() / 2
        return reg_loss

    def update(self, x, y, **kwargs):
        if not self.use_piccl:
            return super().update(x, y, **kwargs)
        all_x, all_y, all_x_2 = torch.cat(x), torch.cat(y), torch.cat(kwargs["x_2"])
        beta = self._beta(kwargs.get("step", 0)).to(all_x.device)
        z, inter_feats = self.featurizer(all_x, ret_feats=True)
        z_int, _ = self.featurizer(all_x_2, ret_feats=True)
        with torch.no_grad():
            z_ref, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
            z_int_ref = self.pre_featurizer(all_x_2)
        delta_pair = self.pire(z, z_int, z_ref, z_int_ref).detach()
        self.residual_bank.update((z - z_ref).detach(), all_y, self._domain_ids(x))
        delta_domain = self.residual_bank.domain_responses().to(z.device, z.dtype)
        loss_isr, loss_isr_aug, loss_isr_dom = self._isr_loss(delta_pair, delta_domain)
        m, m_int = self._project(z, beta), self._project(z_int, beta)
        loss_cls = F.cross_entropy(self.classifier(m), all_y)
        if self.two_ce:
            loss_cls = .5 * (loss_cls + F.cross_entropy(self.classifier(m_int), all_y))
        loss_sup_cl = m.new_zeros(())
        loss_sup_cl_unweighted = m.new_zeros(())
        reliability_metrics = {}
        if self.l:
            features = torch.stack([F.normalize(self.proj_head(m)), F.normalize(self.proj_head(m_int))], 1)
            use_reliability = parse_bool(self.hparams["piccl_use_causal_reliability"])
            if use_reliability:
                # z/z_int are factual features; m/m_int are their independently
                # intervened/fused causal representations. All metadata follows
                # SupCon's view-major cat(unbind(..., dim=1)) expansion.
                raw_reliability = torch.cat([self._causal_reliability(z, m),
                                               self._causal_reliability(z_int, m_int)])
                gamma = self._reliability_gamma(kwargs.get("step", 0))
                causal_reliability = (1. - gamma) + gamma * raw_reliability
                base_domains = self._domain_ids(x)
                base_samples = torch.arange(all_y.shape[0], device=all_y.device)
                expanded_labels = self._expand_view_major(all_y)
                expanded_domains = self._expand_view_major(base_domains)
                expanded_samples = self._expand_view_major(base_samples)
                expanded_views = torch.cat([torch.zeros_like(base_samples), torch.ones_like(base_samples)])
                if not (expanded_labels.shape[0] == expanded_domains.shape[0] == expanded_samples.shape[0]
                        == causal_reliability.shape[0] == features.shape[0] * features.shape[1]):
                    raise RuntimeError("SupCon metadata does not match contrast feature expansion")
                weights, cross_mask, _ = self._causal_positive_pair_weights(
                    expanded_labels, expanded_domains, expanded_samples, expanded_views,
                    causal_reliability, float(self.hparams["piccl_reliability_min"]),
                    parse_bool(self.hparams["piccl_reliability_cross_domain_only"]))
                if self.re_w:
                    d = torch.unsqueeze(torch.cat([torch.cat(kwargs["d"]), torch.cat(kwargs["d_2"])]), 1).float()
                    neg_mask = torch.eq(d, d.T).float()
                    loss_sup_cl_unweighted = self.supcon_loss(features, all_y, neg_mask=neg_mask,
                                                               pos_mask=1-neg_mask if self.pos_mask else None)
                    loss_sup_cl = self.supcon_loss(features, all_y, neg_mask=neg_mask,
                                                    pos_mask=1-neg_mask if self.pos_mask else None,
                                                    positive_weights=weights)
                elif self.sample_d:
                    m_2_d = self._project(self.featurizer(torch.cat(kwargs["x_2_d"])), beta)
                    add_pos = F.normalize(self.proj_head(m_2_d))
                    add_pos = torch.cat([add_pos, add_pos])
                    loss_sup_cl_unweighted = self.supcon_loss(features, all_y, add_pos=add_pos)
                    loss_sup_cl = self.supcon_loss(features, all_y, add_pos=add_pos, positive_weights=weights)
                else:
                    loss_sup_cl_unweighted = self.supcon_loss(features, all_y)
                    loss_sup_cl = self.supcon_loss(features, all_y, positive_weights=weights)
                cross_weights = weights[cross_mask]
                reliability_metrics = {
                    "causal_reliability_mean": raw_reliability.mean().item(),
                    "causal_reliability_std": raw_reliability.std(unbiased=False).item(),
                    "causal_reliability_min": raw_reliability.min().item(),
                    "causal_reliability_max": raw_reliability.max().item(),
                    "causal_reliability_gamma": gamma,
                    "cross_domain_positive_pairs": int(cross_mask.sum().item()),
                    "cross_domain_positive_weight_mean": cross_weights.mean().item() if cross_weights.numel() else 1.,
                    "cross_domain_positive_weight_min": cross_weights.min().item() if cross_weights.numel() else 1.,
                    "cross_domain_positive_weight_max": cross_weights.max().item() if cross_weights.numel() else 1.,
                    "loss_sup_cl_unweighted": loss_sup_cl_unweighted.item(),
                    "loss_sup_cl_weighted": loss_sup_cl.item(),
                    "reliability_nan_count": int((~torch.isfinite(raw_reliability)).sum().item()),
                }
            elif self.re_w:
                d = torch.unsqueeze(torch.cat([torch.cat(kwargs["d"]), torch.cat(kwargs["d_2"])]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                loss_sup_cl = self.supcon_loss(features, all_y, neg_mask=neg_mask,
                                                pos_mask=1-neg_mask if self.pos_mask else None)
            elif self.sample_d:
                m_2_d = self._project(self.featurizer(torch.cat(kwargs["x_2_d"])), beta)
                add_pos = F.normalize(self.proj_head(m_2_d))
                loss_sup_cl = self.supcon_loss(features, all_y, add_pos=torch.cat([add_pos, add_pos]))
            else:
                loss_sup_cl = self.supcon_loss(features, all_y)
        pre_cl_loss = m.new_zeros(())
        if self.l_layer:
            features = torch.stack([F.normalize(self.pre_proj_head(m)), F.normalize(self.pre_proj_head(z_ref.detach()))], 1)
            if self.re_w:
                d = torch.unsqueeze(torch.cat([torch.cat(kwargs["d"]), torch.cat(kwargs["d_2"])]), 1).float()
                neg_mask = torch.eq(d, d.T).float()
                pre_cl_loss = self.supcon_loss_pre(features, all_y, neg_mask=neg_mask,
                                                    pos_mask=1-neg_mask if self.pos_mask else None)
            else:
                pre_cl_loss = self.supcon_loss_pre(features, all_y)
        reg_loss = self._gt_loss(inter_feats, pre_feats) if self.l_d else m.new_zeros(())
        loss = loss_cls + self.l * loss_sup_cl + self.l_layer * pre_cl_loss + self.l_d * reg_loss + self.hparams["piccl_isr_weight"] * loss_isr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics = {"loss": loss.item(), "loss_cls": loss_cls.item(), "sup_cl_loss": loss_sup_cl.item(),
                   "pre_cl_loss": pre_cl_loss.item(), "reg_loss": reg_loss.item(), "loss_isr": loss_isr.item(),
                   "loss_isr_aug": loss_isr_aug.item(), "loss_isr_dom": loss_isr_dom.item(), "piccl_beta": beta.item()}
        metrics.update(reliability_metrics)
        return metrics

    def predict_embed(self, x):
        if not self.use_piccl:
            return super().predict_embed(x)
        return self._project(self.featurizer(x), self.piccl_beta)

    def predict(self, x):
        if not self.use_piccl:
            return super().predict(x)
        return self.classifier(self.predict_embed(x))

    def get_forward_model(self):
        if not self.use_piccl:
            return super().get_forward_model()
        return PICCLForwardModel(self.featurizer, self.sensitive_subspace, self.causal_mediator,
                                 self.classifier, self.piccl_beta)

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = get_optimizer(clone.hparams["optimizer"], clone._optimizer_groups())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone
