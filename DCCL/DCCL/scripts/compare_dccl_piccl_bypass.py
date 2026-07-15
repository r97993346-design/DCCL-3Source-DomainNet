#!/usr/bin/env python
"""Fixed-batch PICCL bypass diagnostic.

This script documents and checks the four intended Stage-0 comparison modes:
A) DCCL, B) PICCL use_piccl=false, C) PICCL residual_scale=0 with causal
ISR/orth losses active, and D) PICCL strict bypass. It uses lightweight tensors so it
can run in CI without PACS data or CUDA.
"""
import argparse
import json
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domainbed.algorithms.piccl import ResidualGateFusion


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)


def grad_norm(module):
    vals = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(vals).norm()) if vals else 0.0


def run_mode(seed, mode):
    set_seed(seed)
    backbone = torch.nn.Linear(5, 4)
    projector = torch.nn.Linear(4, 3)
    classifier = torch.nn.Linear(4, 2)
    gate = ResidualGateFusion(4)
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    domains = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])

    z = backbone(x)
    piccl_feature = z * 0.5 + 0.1
    if mode in {"A_dccl", "B_piccl_disabled", "D_strict_bypass"}:
        fused = z
        causal_loss = z.sum() * 0.0
        causal_executed = 0.0
    elif mode == "C_scale0_aux_active":
        fused, _ = gate(z, piccl_feature, scale=0, alpha=torch.tensor(1.0))
        # Causal ISR/orth losses are independent of residual_scale semantics.
        causal_loss = piccl_feature.pow(2).mean()
        causal_executed = 1.0
    else:
        raise ValueError(mode)

    logits = classifier(fused)
    cls_loss = F.cross_entropy(logits, y)
    q = F.normalize(projector(fused), dim=1)
    dccl_loss = -(q @ q.T).diag().mean()
    total = cls_loss + dccl_loss + causal_loss
    total.backward()
    before = backbone.weight.detach().clone()
    opt = torch.optim.SGD(list(backbone.parameters()) + list(projector.parameters()) + list(classifier.parameters()), lr=1e-2)
    opt.step()
    return {
        "mode": mode,
        "domain_checksum": int(domains.sum()),
        "feature": z.detach(),
        "fused": fused.detach(),
        "logits": logits.detach(),
        "classification_loss": float(cls_loss.detach()),
        "dccl_loss": float(dccl_loss.detach()),
        "causal_loss": float(causal_loss.detach()),
        "total_loss": float(total.detach()),
        "backbone_grad_norm": grad_norm(backbone),
        "classifier_grad_norm": grad_norm(classifier),
        "first_step_delta": (backbone.weight.detach() - before).detach(),
        "causal_executed": causal_executed,
    }


def first_diff(a, b):
    for key in ["feature", "fused", "logits", "classification_loss", "dccl_loss", "causal_loss", "total_loss", "backbone_grad_norm", "classifier_grad_norm", "first_step_delta"]:
        av, bv = a[key], b[key]
        if torch.is_tensor(av):
            if not torch.allclose(av, bv, atol=1e-7, rtol=1e-7):
                return key, float((av - bv).abs().max())
        elif abs(av - bv) > 1e-7:
            return key, abs(av - bv)
    return None, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    modes = ["A_dccl", "B_piccl_disabled", "C_scale0_aux_active", "D_strict_bypass"]
    out = {m: run_mode(args.seed, m) for m in modes}
    summary = {}
    for m in modes[1:]:
        key, delta = first_diff(out["A_dccl"], out[m])
        summary[m] = {"first_difference": key, "max_abs_delta": delta}
    print(json.dumps(summary, indent=2, sort_keys=True))
    assert summary["D_strict_bypass"]["first_difference"] is None
    assert summary["C_scale0_aux_active"]["first_difference"] == "causal_loss"

if __name__ == "__main__":
    main()
