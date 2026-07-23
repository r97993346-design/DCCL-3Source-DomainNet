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
        if self.l:
            features = torch.stack([F.normalize(self.proj_head(m)), F.normalize(self.proj_head(m_int))], 1)
            if self.re_w:
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
        return {"loss": loss.item(), "loss_cls": loss_cls.item(), "sup_cl_loss": loss_sup_cl.item(),
                "pre_cl_loss": pre_cl_loss.item(), "reg_loss": reg_loss.item(), "loss_isr": loss_isr.item(),
                "loss_isr_aug": loss_isr_aug.item(), "loss_isr_dom": loss_isr_dom.item(), "piccl_beta": beta.item()}

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
