"""PICCL: Paired-Intervention Causal Connectivity Learning."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.algorithms.algorithms import DCCL, ForwardModel
from domainbed.optimizers import get_optimizer


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def parse_bool(value):
    """Backward-compatible alias for PICCL boolean hparams."""
    return _as_bool(value)

# 干预视角的"虚假相关性" — 学生对数据增强的额外敏感度
class PairedInterventionResponseEstimator(nn.Module):
    def forward(self, z, z_int, z_ref, z_int_ref):
        tensors = {"z": z, "z_int": z_int, "z_ref": z_ref, "z_int_ref": z_int_ref}
        for name, tensor in tensors.items():
            if tensor.dim() != 2:
                raise ValueError(f"{name} must be a 2D pooled feature tensor [B,D], got {tuple(tensor.shape)}")
        shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"PIRE inputs must have identical shapes, got {shapes}")
        return (z_int - z) - (z_int_ref - z_ref).detach()

#  "类-域残差原型库"。它的作用是:持续追踪每个类别在每个域上的残差特征,用于提取跨域的响应模式。
class ClassDomainResidualBank(nn.Module):
    def __init__(self, num_classes, num_domains, feature_dim, momentum=0.99, min_valid_domains=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.min_valid_domains = min_valid_domains
        # 类-域残差原型矩阵
        # 每个类别在每个域上的残差特征的均值[num_classes, num_domains, feature_dim]
        self.register_buffer("prototypes", torch.zeros(num_classes, num_domains, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, num_domains, dtype=torch.bool))
        self.register_buffer("counts", torch.zeros(num_classes, num_domains, dtype=torch.long))

    @torch.no_grad()
    def update(self, residual, labels, domains):
        residual = residual.detach()
        labels = labels.detach().long()
        domains = domains.detach().long()
        if residual.numel() == 0:
            return
        for c in labels.unique().tolist():
            if c < 0 or c >= self.num_classes:
                continue
            for d in domains.unique().tolist():
                if d < 0 or d >= self.num_domains:
                    continue
                mask = (labels == c) & (domains == d)
                if not mask.any():
                    continue
                #	从 batch 中选出属于当前组的所有样本
                current = residual[mask].mean(dim=0)
                if self.initialized[c, d]:
                    #	动量平均让原型有**"遗忘"能力**,更适应训练过程中的模型演变。
                    self.prototypes[c, d].mul_(self.momentum).add_(current, alpha=1.0 - self.momentum)
                else:
                    self.prototypes[c, d].copy_(current)
                    self.initialized[c, d] = True
                self.counts[c, d] += int(mask.sum().item())
    #     # 对于猫类别:
    #   center = (prototype[猫, 0] + prototype[猫, 1]) / 2
    #   responses += [prototype[猫, 0] - center, prototype[猫, 1] - center]
    #                ↑ "猫在油画上的特征偏离中心"    ↑ "猫在照片上的特征偏离中心"
    #                ← 这两个向量就是"域相关的虚假特征方向"!
    def domain_responses(self):
        responses = []
        for c in range(self.num_classes):
            valid = self.initialized[c]
            if int(valid.sum().item()) < self.min_valid_domains:
                continue
            center = self.prototypes[c, valid].mean(dim=0)
            #这些偏离向量 = "同一个类别,不同域风格带来的特征差异"
            # = PICCL 想识别并抑制的"域相关虚假特征方向"!
            responses.append(self.prototypes[c, valid] - center)
        if not responses:
            return self.prototypes.new_zeros((0, self.feature_dim))
        return torch.cat(responses, dim=0).detach()


class InterventionSensitiveSubspace(nn.Module):
    def __init__(self, feature_dim, rank=16, eps=1e-8):
        super().__init__()
        rank = min(int(rank), int(feature_dim))
        self.basis = nn.Parameter(torch.randn(feature_dim, rank) * 0.02)
        self.eps = eps

#将可学习的 basis 矩阵正交化,得到一组标准正交基,供后续的投影、覆盖损失等操作使用
    def orthonormal_basis(self, detach=False, dtype=None):
        q = torch.linalg.qr(self.basis.float(), mode="reduced").Q
        if detach:
            q = q.detach()
        if dtype is not None:
            q = q.to(dtype=dtype)
        return q

    def project(self, v, detach_basis=False):
        if v.numel() == 0:
            return v
        q = self.orthonormal_basis(detach=detach_basis, dtype=v.dtype).to(v.device)
        return (v @ q) @ q.t()
    # "覆盖损失" = "干预敏感子空间的响应向量与其在子空间上的投影之间的差异"
    def coverage_loss(self, responses):
        if responses.numel() == 0:
            return self.basis.sum() * 0.0
        responses = responses.detach()
        nonzero = responses.pow(2).sum(dim=1) > self.eps
        if not nonzero.any():
            return self.basis.sum() * 0.0
        v = responses[nonzero]
        proj = self.project(v, detach_basis=False)
        return ((v - proj).pow(2).sum(dim=1) / (v.pow(2).sum(dim=1) + self.eps)).mean()
    # "正交损失" = "干预敏感子空间的基向量之间的正交性差异"
    def orthogonality_loss(self):
        b = F.normalize(self.basis, dim=0, eps=self.eps)
        eye = torch.eye(b.shape[1], device=b.device, dtype=b.dtype)
        return (b.t() @ b - eye).pow(2).sum()

    def capture_ratio(self, responses):
        if responses.numel() == 0:
            return responses.new_tensor(0.0)
        responses = responses.detach()
        nonzero = responses.pow(2).sum(dim=1) > self.eps
        if not nonzero.any():
            return responses.new_tensor(0.0)
        v = responses[nonzero]
        proj = self.project(v, detach_basis=True)
        return (proj.pow(2).sum(dim=1) / (v.pow(2).sum(dim=1) + self.eps)).mean()
# 训练初期 (step 小) alpha = 0.0 → 不做任何抑制 → piccl_m = z (等同于普通 DCCL)

# 训练中期 (step 中等) alpha = 0.5 → 部分抑制虚假方向 → 部分保留内容特征

# 训练后期 (step 大) alpha = 1.0 → 完全去掉虚假方向分量 → piccl_m = z - sensitive (虚假方向完全去除)

class CausalMediatorProjection(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, z, subspace, alpha, detach_basis=True):
        sensitive = subspace.project(z, detach_basis=detach_basis)
        return self.layer_norm(z - alpha * sensitive)

# 简单版用于 `fusion_mode="legacy"`,门控版用于 `fusion_mode="residual_gate"`。
class ResidualGateFusion(nn.Module):
    def __init__(self, feature_dim, gate_bias=-4.0):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.gate_bias = float(gate_bias)

    def forward(self, original, piccl_feature, scale, alpha):
           # Step 1: 根据原始特征生成门控信号
        gate = torch.sigmoid(self.linear(original) + self.gate_bias)
        # Step 2: 门控融合
    #   fused = original + scale * alpha * gate * (piccl_feature - original)
    #   相当于: 在 original 和 piccl_feature 之间做 gate 控制的插值
        fused = original + float(scale) * alpha.to(original.device, original.dtype) * gate * (piccl_feature - original)
        return fused, gate

# PICCLForwardModel 是一个封装了 featurizer、subspace、mediator 和 classifier 的前向模型,
# 用于在推理阶段使用 PICCL 的特征处理流程。
class PICCLForwardModel(nn.Module):
    def __init__(self, featurizer, subspace, mediator, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.subspace = subspace
        self.mediator = mediator
        self.classifier = classifier
        self.register_buffer("piccl_alpha", torch.tensor(0.0))

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        z = self.featurizer(x)
        m = self.mediator(z, self.subspace, self.piccl_alpha.to(z.device, z.dtype), detach_basis=True)
        return self.classifier(m)

    def predict_embed(self, x):
        z = self.featurizer(x)
        return self.mediator(z, self.subspace, self.piccl_alpha.to(z.device, z.dtype), detach_basis=True)


class PICCL(DCCL):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.piccl_strict_bypass = _as_bool(hparams.get("piccl_strict_bypass", False))
        if self.piccl_strict_bypass:
            self.optimizer = get_optimizer(hparams["optimizer"], self._dccl_optimizer_groups())
            return
        feature_dim = self.featurizer.n_outputs
        for p in self.pre_featurizer.parameters():
            p.requires_grad = False
        self.pre_featurizer.eval()
        self.pire = PairedInterventionResponseEstimator()
        self.residual_bank = ClassDomainResidualBank(
            num_classes, num_domains, feature_dim,
            momentum=hparams.get("piccl_proto_momentum", 0.99),
            min_valid_domains=hparams.get("piccl_min_valid_domains", 2),
        )
        self.sensitive_subspace = InterventionSensitiveSubspace(
            feature_dim, hparams.get("piccl_rank", 16), hparams.get("piccl_eps", 1e-8)
        )
        self.causal_mediator = CausalMediatorProjection(feature_dim)
        self.residual_gate = ResidualGateFusion(feature_dim, hparams.get("piccl_gate_bias", -4.0))
        self.register_buffer("piccl_alpha", torch.tensor(0.0))
        self.optimizer = get_optimizer(hparams["optimizer"], self._optimizer_groups())

    def _dccl_optimizer_groups(self):
        return [
            {"params": self.featurizer.parameters(), "lr": self.hparams["lr"], "weight_decay": self.hparams["weight_decay"]},
            {"params": self.classifier.parameters(), "lr": self.hparams["lr"] / 0.1, "weight_decay": self.hparams["weight_decay"]},
            {"params": self.proj_head.parameters(), "lr": self.hparams["lr"] / 10, "weight_decay": self.hparams["weight_decay"]},
            {"params": self.mean_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.var_encoders.parameters(), "lr": self.hparams["lr"] * 10},
            {"params": self.pre_proj_head.parameters(), "lr": self.hparams["lr"] / 10},
        ]

    def _optimizer_groups(self):
        if _as_bool(self.hparams.get("piccl_strict_bypass", False)):
            return self._dccl_optimizer_groups()
        return self._dccl_optimizer_groups() + [
            {"params": self.sensitive_subspace.parameters(), "lr": self.hparams["lr"] * float(self.hparams.get("piccl_lr_multiplier", 1.0)), "weight_decay": self.hparams["weight_decay"]},
            {"params": self.causal_mediator.parameters(), "lr": self.hparams["lr"] * float(self.hparams.get("piccl_lr_multiplier", 1.0)), "weight_decay": self.hparams["weight_decay"]},
            {"params": self.residual_gate.parameters(), "lr": self.hparams["lr"] * float(self.hparams.get("piccl_lr_multiplier", 1.0)), "weight_decay": self.hparams["weight_decay"]},
        ]

    def train(self, mode=True):
        super().train(mode)
        self.pre_featurizer.eval()
        return self
#     PICCL 通过三套独立的 ramp 调度实现"温和训练":

# 1) alpha 调度: 让"抑制强度"从 0 逐步升到 max_alpha
#    避免训练初期强行抑制还没学好的方向

# 2) loss_scale 调度: 让"子空间学习的损失权重"从 0 逐步升到 1
#    让模型先学好分类特征,再逐步引入"检测虚假方向"的任务

# 3) feature_alpha 调度: 让"门控融合"的强度从 0 逐步升到 1
#    让 residual_gate 的线性层先学好,再逐步介入

# 三者协同,保证 PICCL 的介入是"渐进式"的,
# 不会在训练初期就破坏 DCCL 已经学好的特征表示。
    def _ramp_value(self, step, start_ratio, warmup_ratio, max_value=1.0):
        total = max(int(self.hparams.get("piccl_total_steps", 1)) - 1, 1)
        progress = float(step) / float(total)
        start = float(start_ratio)
        warmup = float(warmup_ratio)
        if progress <= start:
            return 0.0
        if warmup <= 0:
            return float(max_value)
        return float(max_value) * min(max((progress - start) / warmup, 0.0), 1.0)

    def _alpha(self, step):
        total = max(int(self.hparams.get("piccl_total_steps", 1)) - 1, 1)
        progress = float(step) / float(total)
        warm = float(self.hparams.get("piccl_warmup_ratio", 0.10))
        ramp = max(float(self.hparams.get("piccl_ramp_ratio", 0.20)), 1e-12)
        max_alpha = float(self.hparams.get("piccl_alpha_max", 0.5))
        delayed = float(self.hparams.get("piccl_delayed_start_ratio", 0.0))
        if progress <= delayed + warm:
            alpha = 0.0
        elif progress >= delayed + warm + ramp:
            alpha = max_alpha
        else:
            alpha = max_alpha * ((progress - delayed - warm) / ramp)
        self.piccl_alpha.fill_(alpha)
        return self.piccl_alpha

    def _loss_scale(self, step):
        return self._ramp_value(step, self.hparams.get("piccl_delayed_start_ratio", 0.0), self.hparams.get("piccl_loss_warmup_ratio", 0.0), 1.0)

    def _feature_scale(self, step):
        return self._ramp_value(step, self.hparams.get("piccl_delayed_start_ratio", 0.0), self.hparams.get("piccl_feature_warmup_ratio", 0.0), 1.0)

    def _domain_ids(self, x):
        return torch.cat([torch.full((xi.shape[0],), i, dtype=torch.long, device=xi.device) for i, xi in enumerate(x)])
    def _nt_xent(self, q, q_int):
        if q.shape[0] < 2:
            return q.sum() * 0.0
        logits = q @ q_int.t() / self.hparams["t"]
        targets = torch.arange(q.shape[0], device=q.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

    def _cross_domain_supcon(self, q, labels, domains):
        n = q.shape[0]
        if n < 2:
            return q.sum() * 0.0
        logits = q @ q.t() / self.hparams["t"]
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(n, device=q.device, dtype=torch.bool)
        pos = (labels[:, None] == labels[None, :]) & (domains[:, None] != domains[None, :]) & (~self_mask)
        valid = pos.sum(dim=1) > 0
        if not valid.any():
            return q.sum() * 0.0
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        return -(log_prob * pos.float()).sum(dim=1)[valid].div(pos.sum(dim=1)[valid].float()).mean()

    def _gt_loss(self, inter_feats, pre_feats):
        reg_loss = inter_feats[0].new_tensor(0.0)
        from domainbed.lib import misc
        for inter_f, pre_f, mean_enc, var_enc in misc.zip_strict(inter_feats, pre_feats, self.mean_encoders, self.var_encoders):
            mean = mean_enc(inter_f)
            var = var_enc(inter_f)
            reg_loss = reg_loss + (((mean - pre_f).pow(2).div(var) + var.log()).mean() / 2.0)
        return reg_loss

    def update(self, x, y, **kwargs):
        if self.piccl_strict_bypass or not _as_bool(self.hparams.get("use_piccl", True)):
            return super().update(x, y, **kwargs)
        all_x, all_y = torch.cat(x), torch.cat(y)
        all_x_2 = torch.cat(kwargs["x_2"])
        domains = self._domain_ids(x)
        step = kwargs.get("step", 0)
        alpha = self._alpha(step).to(all_x.device)
        detach_basis = not _as_bool(self.hparams.get("piccl_basis_receive_task_grad", False))

        z, inter_feats = self.featurizer(all_x, ret_feats=True)
        z_int, _ = self.featurizer(all_x_2, ret_feats=True)
        with torch.no_grad():
            z_ref, pre_feats = self.pre_featurizer(all_x, ret_feats=True)
            z_int_ref = self.pre_featurizer(all_x_2)
        delta_pair = self.pire(z, z_int, z_ref, z_int_ref)
        delta_pair_target = delta_pair.detach()
        residual = (z - z_ref.detach()).detach()
        self.residual_bank.update(residual, all_y, domains)
        delta_dom = self.residual_bank.domain_responses().to(device=z.device, dtype=z.dtype)
        responses = torch.cat([delta_pair_target, delta_dom], dim=0) if delta_dom.numel() else delta_pair_target

        piccl_m = self.causal_mediator(z, self.sensitive_subspace, alpha.to(dtype=z.dtype), detach_basis=detach_basis)
        piccl_m_int = self.causal_mediator(z_int, self.sensitive_subspace, alpha.to(dtype=z.dtype), detach_basis=detach_basis)
        gate = None
        fusion_mode = self.hparams.get("piccl_fusion_mode", "legacy")
        if fusion_mode == "legacy":
            m, m_int = piccl_m, piccl_m_int
        elif fusion_mode == "residual_gate":
            feature_alpha = z.new_tensor(self._feature_scale(step))
            m, gate = self.residual_gate(z, piccl_m, self.hparams.get("piccl_residual_scale", 0.1), feature_alpha)
            m_int, _ = self.residual_gate(z_int, piccl_m_int, self.hparams.get("piccl_residual_scale", 0.1), feature_alpha)
        else:
            raise ValueError(f"Unknown piccl_fusion_mode={fusion_mode}")
        #logits = self.classifier(z)
        # 这里修改一下2——2都经过因果
        logits = self.classifier(m)
        loss_cls = F.cross_entropy(logits, all_y)
        loss = loss_cls
        loss_sup_cl = m.new_tensor(0.0)
        if self.l:
            embed_2 = self.proj_head(m_int)
            embed_1 = self.proj_head(m)
            view_1 = nn.functional.normalize(embed_1)
            view_2 = nn.functional.normalize(embed_2)
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
                loss_sup_cl = self.supcon_loss(features, all_y, neg_mask=neg_mask, pos_mask=pos_mask)
            else:
                if self.sample_d:
                    all_x_2_d = torch.cat(kwargs["x_2_d"])
                    feature_x_2_d = self.featurizer(all_x_2_d)
                    piccl_x_2_d = self.causal_mediator(feature_x_2_d, self.sensitive_subspace, alpha.to(dtype=feature_x_2_d.dtype), detach_basis=detach_basis)
                    if fusion_mode == "residual_gate":
                        piccl_x_2_d, _ = self.residual_gate(feature_x_2_d, piccl_x_2_d, self.hparams.get("piccl_residual_scale", 0.1), feature_alpha)
                    embed_2_d = self.proj_head(piccl_x_2_d)
                    view_2_d = nn.functional.normalize(embed_2_d)
                    add_pos = torch.cat([view_2_d, view_2_d], 0)
                    loss_sup_cl = self.supcon_loss(features, all_y, add_pos=add_pos)
                else:
                    loss_sup_cl = self.supcon_loss(features, all_y)
            loss = loss + self.l * loss_sup_cl
        pre_cl_loss = m.new_tensor(0.0)
        with torch.no_grad():
            z_ref_det = z_ref.detach()
        if self.l_layer:
            embed_1 = self.pre_proj_head(m)
            embed_2 = self.pre_proj_head(z_ref_det)
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
                    pos_mask = 1 - neg_mask
                else:
                    pos_mask = None
                pre_cl_loss = self.supcon_loss_pre(features, all_y_pre, neg_mask=neg_mask, pos_mask=pos_mask)
            else:
                pre_cl_loss = self.supcon_loss_pre(features, all_y_pre)
            loss = loss + self.l_layer * pre_cl_loss
        loss_gt = z.new_tensor(0.0)
        if self.hparams.get("piccl_gt_mode", "replace") == "keep" and self.l_d:
            loss_gt = self._gt_loss(inter_feats, pre_feats)
            loss = loss + self.l_d * loss_gt
        loss_isr = self.sensitive_subspace.coverage_loss(responses)
        loss_orth = self.sensitive_subspace.orthogonality_loss()
        weighted_isr = float(self.hparams.get("piccl_isr_weight", 0.1)) * loss_isr
        weighted_orth = float(self.hparams.get("piccl_orth_weight", 0.001)) * loss_orth
        loss = loss + weighted_isr + weighted_orth
        self.optimizer.zero_grad()
        loss.backward()
        grad_stats = self._diagnostic_grad_norms()
        self.optimizer.step()
        self.pre_featurizer.eval()
        with torch.no_grad():
            raw_cos = F.cosine_similarity(z, z_int, dim=1).mean()
            med_cos = F.cosine_similarity(m, m_int, dim=1).mean()
            delta_norm = (m - z).norm(dim=1).mean()
            original_norm = z.norm(dim=1).mean().clamp_min(1e-12)
            fused_cos = F.cosine_similarity(z, m, dim=1).mean()
            orth_err = self.sensitive_subspace.orthogonality_loss()
            cap = self.sensitive_subspace.capture_ratio(responses)
        metrics = {
            "loss": float(loss.item()), "loss_cls": float(loss_cls.item()),
            "loss_isr": float(loss_isr.item()), "loss_orth": float(loss_orth.item()),
            "loss_gt": float(loss_gt.item()), "piccl_alpha": float(alpha.item()),
            "paired_response_norm": float(delta_pair_target.norm(dim=1).mean().item()),
            "domain_response_norm": float(delta_dom.norm(dim=1).mean().item()) if delta_dom.numel() else 0.0,
            "valid_domain_response_count": float(delta_dom.shape[0]), "basis_orth_error": float(orth_err.item()),
            "response_capture_ratio": float(cap.item()), "raw_intervention_cosine": float(raw_cos.item()),
            "mediator_intervention_cosine": float(med_cos.item()),
            "ce_loss": float(loss_cls.item()), "sup_cl_loss": float(loss_sup_cl.item()), "pre_cl_loss": float(pre_cl_loss.item()),
            "dccl_total_loss": float((loss_cls + self.l * loss_sup_cl + self.l_layer * pre_cl_loss + self.l_d * loss_gt).item()),
            "weighted_loss_isr": float(weighted_isr.item()), "weighted_loss_orth": float(weighted_orth.item()),
            "piccl_feature_scale": float(self._feature_scale(step)),
            "residual_scale_effective": float(self.hparams.get("piccl_residual_scale", 0.1)),
            "residual_delta_norm": float(delta_norm.item()),
            "fused_original_cosine": float(fused_cos.item()),
            "piccl_executed": 1.0,
            "original_feature_norm": float(original_norm.item()), "piccl_feature_norm": float(piccl_m.norm(dim=1).mean().item()),
            "feature_delta_norm": float(delta_norm.item()), "feature_delta_ratio": float((delta_norm / original_norm).item()),
            "original_fused_cosine": float(fused_cos.item()),
            "gate_mean": float(gate.mean().item()) if gate is not None else 0.0,
            "gate_std": float(gate.std().item()) if gate is not None else 0.0,
            "gate_min": float(gate.min().item()) if gate is not None else 0.0,
            "gate_max": float(gate.max().item()) if gate is not None else 0.0,
            "backbone_grad_norm": grad_stats["backbone_grad_norm"], "piccl_grad_norm": grad_stats["piccl_grad_norm"],
            "classifier_grad_norm": grad_stats["classifier_grad_norm"],
            "has_nan_or_inf": float(any(not torch.isfinite(t).all().item() for t in [loss, z, m, logits])),
        }
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"param_group_lr_{i}"] = float(group["lr"])
        return metrics

    def _diagnostic_grad_norms(self):
        def norm(parameters):
            vals = [p.grad.detach().norm() for p in parameters if p.grad is not None]
            if not vals:
                return 0.0
            return float(torch.stack(vals).norm().item())
        piccl_params = list(self.sensitive_subspace.parameters()) + list(self.causal_mediator.parameters()) + list(self.residual_gate.parameters())
        return {
            "backbone_grad_norm": norm(self.featurizer.parameters()),
            "piccl_grad_norm": norm(piccl_params),
            "classifier_grad_norm": norm(self.classifier.parameters()),
        }

    def predict_embed(self, x):
        if self.piccl_strict_bypass or not _as_bool(self.hparams.get("use_piccl", True)):
            return self.featurizer(x)
        z = self.featurizer(x)
        piccl_m = self.causal_mediator(z, self.sensitive_subspace, self.piccl_alpha.to(z.device, z.dtype), detach_basis=True)
        if self.hparams.get("piccl_fusion_mode", "legacy") == "residual_gate":
            fused, _ = self.residual_gate(z, piccl_m, self.hparams.get("piccl_residual_scale", 0.1), self.piccl_alpha.to(z.device, z.dtype))
            return fused
        return piccl_m
    # 尝试一下都不经过因果模块
    # def predict_embed(self, x):
    #     return self.featurizer(x)


    def predict(self, x):
        return self.classifier(self.predict_embed(x))

    def get_forward_model(self):
        if self.piccl_strict_bypass or not _as_bool(self.hparams.get("use_piccl", True)):
            return super().get_forward_model()
        model = PICCLForwardModel(self.featurizer, self.sensitive_subspace, self.causal_mediator, self.classifier)
        model.piccl_alpha.copy_(self.piccl_alpha.detach().cpu())
        return model

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = get_optimizer(self.hparams["optimizer"], clone._optimizer_groups())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone
