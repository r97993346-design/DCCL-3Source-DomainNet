"""Check the DirectDiff gate on the previous A16 SWAD checkpoint exports."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from domainbed.algorithms.dicm_gate import DirectionGate, GateError
from domainbed.lib.causal_checkpoints import load_inference, qr_basis
from tools.pacs_directdiff.common import (
    BETA, DOMAINS, PREVIOUS, ROOT, atomic_json, controls, historical_fold,
    source_centroids, squared_norm,
)
from tools.pacs_projection.analyze import hash_path


def check_one(index, domain):
    prior = PREVIOUS / "features" / "A16" / domain / "swad" / "features.npz"
    summary = json.loads((PREVIOUS / "probes" / "A16" / domain / "swad.json").read_text())
    _, checkpoint = historical_fold("A16", index, "swad")
    assert summary["checkpoint_sha256"] == hash_path(checkpoint)
    old, state = load_inference(checkpoint, "cpu")
    assert state["identity"]["test_envs"] == [index]
    with np.load(prior, allow_pickle=False) as data:
        keys = ("z", "Q", "E_beta", "y", "domain", "is_source_train")
        z, q, legacy, y, d, train = (data[key] for key in keys)
    np.testing.assert_allclose(np.linalg.svd(q.T @ qr_basis(old.module).numpy(), compute_uv=False),
                               np.ones(16), atol=1e-5)
    assert q.shape == (2048, 16) and index not in np.unique(d)
    prototypes, valid, sizes = source_centroids(z, y, d, train)
    q = torch.from_numpy(q)
    source_z = torch.from_numpy(z[train])
    gate = DirectionGate(2048, 16)
    gate.refresh(q, prototypes, valid, source_z, step=0)

    sample = source_z[:32].clone().requires_grad_()
    q_probe = q.clone().requires_grad_()
    stable = sample - BETA * gate.suppressible(sample, q_probe)
    assert stable.shape == sample.shape and torch.isfinite(stable).all()
    stable.square().mean().backward()
    assert sample.grad is not None and sample.grad.norm() > 0 and q_probe.grad is None
    assert not list(gate.parameters())
    torch.testing.assert_close(gate.rotation.T @ gate.rotation, torch.eye(16), atol=1e-6, rtol=0)

    head = old.module.network[2]
    with TemporaryDirectory(prefix="pacs-offline-gate-") as tmp, torch.no_grad():
        filename = Path(tmp) / "gate.pth"
        torch.save(gate.state_dict(), filename)
        restored = DirectionGate(2048, 16)
        restored.load_state_dict(torch.load(filename, weights_only=True))
        output = head(sample - BETA * restored.suppressible(sample, q))
        torch.testing.assert_close(output, head(stable), atol=0, rtol=0)

    views, beta_match = controls(z, q.numpy(), gate.rotation.numpy(), gate.gains.numpy(), train)
    views["legacy_A16"] = legacy
    arrays = {key: value[~train] for key, value in views.items()}
    base_energy = squared_norm(z[~train])
    results = {}
    with torch.no_grad():
        for key, array in arrays.items():
            logits = head(torch.from_numpy(array))
            results[key] = dict(
                source_validation_fixed_head=float((logits.argmax(1).numpy() == y[~train]).mean()),
                removed_energy=squared_norm(z[~train] - array) / base_energy,
            )
    a = results["gate"]["removed_energy"]
    b = results["uniform_equal_sample_energy"]["removed_energy"]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    return dict(held_out=domain, checkpoint_sha256=summary["checkpoint_sha256"],
                split_sha256=state["identity"]["split_sha256"], source_train=int(train.sum()),
                source_validation=int((~train).sum()), source_cell_sizes=sizes.tolist(),
                matched_beta=beta_match, gate=gate.snapshot(), views=results)


def run():
    torch.set_num_threads(2)
    rows = []
    for index, domain in enumerate(DOMAINS):
        row = check_one(index, domain)
        rows.append(row)
        print(domain, "gains",
              f'{min(row["gate"]["gains"]):.4f}..{max(row["gate"]["gains"]):.4f}',
              "gate/uniform energy", f'{row["gate"]["relative_removal"]:.3%}',
              "verified detached gradient and identical gate-reload logits", flush=True)
    try:
        DirectionGate(16, 4).refresh(torch.eye(16)[:, :4], torch.zeros(2, 3, 16),
                                      torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 16), 0)
    except GateError:
        pass
    else:
        raise AssertionError("Degenerate source scatter failed to stop estimation")
    atomic_json(ROOT / "offline" / "A16_gate_feasibility.json", dict(
        status="passed", protocol="frozen A16 weights and cached source features, not end-to-end training",
        method="source train centroids; source validation fixed head and removal energy",
        results=rows))
    print("PASS: historical A16 offline gate feasibility, all four source-only folds", flush=True)


if __name__ == "__main__":
    run()
