"""Reload trained DirectDiff16 checkpoints and run paired source-only diagnostics."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader

from domainbed.algorithms.dicm_gate import DirectionGate
from domainbed.lib.causal_checkpoints import load_inference, prediction, qr_basis
from tools.pacs_causal.features import ImageViews, STYLES, source_samples
from tools.pacs_causal.metrics import coverage, geometry
from tools.pacs_directdiff.common import (
    BETA, DOMAINS, GROUPS, PREVIOUS, ROOT, VARIANTS, atomic_json, controls, folds, say,
    source_centroids, squared_norm,
)
from tools.pacs_directdiff.probes import probe_views
from tools.pacs_directdiff.queue import completed, tmux_name, wait_gpu
from tools.pacs_projection.analyze import class_domain_drift, hash_path, protection


SAMPLE_COLUMNS = ("sample_id", "y", "domain", "is_source_train", "domain_train", "domain_validation",
                  "tsne_indices", "intervention_indices")


@torch.inference_mode()
def encode_student(backbone, rows, device, style=None):
    loader = DataLoader(ImageViews(rows, style), batch_size=64, num_workers=2,
                        shuffle=False, pin_memory=device == "cuda")
    features = []
    for batch in loader:
        features.append(backbone(batch.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(features)


def save_features(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("features.tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def describe(group, index, variant):
    domain = DOMAINS[index]
    folder, = folds(group, index)
    path = folder / "checkpoints" / f"{variant}.pth"
    model, saved = load_inference(path, "cuda")
    assert saved["kind"] == variant and saved["identity"]["test_envs"] == [index]
    assert saved["hparams"]["dicm_direct_difference"] and saved["reference"] is None
    assert not any("pre_featurizer" in key or ".reference." in key for key in saved["model"])
    assert hash_path(folder / saved["sample_manifest"]["file"]) == saved["sample_manifest"]["sha256"]
    logits = prediction(model, saved["example_input"].cuda())
    torch.testing.assert_close(logits, saved["example_logits"], rtol=1e-5, atol=1e-6)
    reload_error = float((logits - saved["example_logits"]).abs().max())
    wrapped = model.module if variant == "swad" else model.get_forward_model()
    assert wrapped.reference is None and wrapped.source_bank is not None
    projection = wrapped.network[1]
    native_gate = projection.gate
    active = native_gate is not None and native_gate.last_step.item() >= 0
    q = qr_basis(wrapped).numpy()
    assert q.shape == (2048, 16)
    np.testing.assert_allclose(q.T @ q, np.eye(16), atol=1e-5)
    np.testing.assert_allclose(np.linalg.svd(q.T @ saved["q"].numpy(), compute_uv=False),
                               np.ones(16), atol=1e-5)
    rows, classes = source_samples(saved)
    with np.load(PREVIOUS / "features" / "A16" / domain / variant / "features.npz", allow_pickle=False) as old:
        samples = {key: old[key] for key in SAMPLE_COLUMNS}
    np.testing.assert_array_equal(samples["sample_id"], [row["sample_id"] for row in rows])
    np.testing.assert_array_equal(samples["y"], [row["category"] for row in rows])
    np.testing.assert_array_equal(samples["domain"], [row["domain"] for row in rows])
    np.testing.assert_array_equal(samples["is_source_train"], [row["split"] == "train" for row in rows])
    train = samples["is_source_train"]
    val = ~train
    y, d = samples["y"], samples["domain"]
    assert index not in d and train[samples["domain_train"]].all()
    assert val[samples["domain_validation"]].all() and val[samples["intervention_indices"]].all()

    z = encode_student(wrapped.network[0], rows, "cuda")
    if active:
        gate = native_gate
        gate_source = "checkpoint source EMA"
    else:
        prototypes, valid, _ = source_centroids(z, y, d, train)
        gate = DirectionGate(2048, 16)
        gate.refresh(torch.from_numpy(q), prototypes, valid, torch.from_numpy(z[train]), step=0)
        gate_source = "offline full source-training centers"
    rotation, gains = gate.rotation.cpu().numpy(), gate.gains.cpu().numpy()
    details = gate.snapshot()
    v = q @ rotation
    views, beta_match = controls(z, q, rotation, gains, train)
    actual = "gate" if active else "uniform"
    e = views[actual]
    removed = z - e
    with torch.no_grad():
        expected = projection(torch.from_numpy(z[:16]).cuda()).cpu().numpy()
    np.testing.assert_allclose(e[:16], expected, rtol=1e-5, atol=5e-6)
    head = wrapped.network[2].cpu()
    src_fp = {}
    with torch.no_grad():
        for label, array in views.items():
            score = head(torch.from_numpy(array[val])).argmax(1).numpy()
            src_fp[label] = dict(
                source_validation_top1=float((score == y[val]).mean()),
                train_removed_energy=squared_norm(z[train] - array[train]) / squared_norm(z[train]),
                validation_removed_energy=squared_norm(z[val] - array[val]) / squared_norm(z[val]),
            )
    for split, name in (("train_removed_energy", "uniform_equal_train_energy"),
                        ("validation_removed_energy", "uniform_equal_sample_energy")):
        np.testing.assert_allclose(src_fp["gate"][split], src_fp[name][split], rtol=2e-6, atol=1e-8)

    q_env, protection_rank = protection(z, q, y, d, train)
    interventions = samples["intervention_indices"]
    snapshots, changes = {}, []
    for style in STYLES:
        perturbed = encode_student(wrapped.network[0], [rows[int(i)] for i in interventions], "cuda", style)
        response = perturbed - z[interventions]
        snapshots["delta_z_" + style] = response
        for space, basis in (("Q", q), ("Q_env", q_env)):
            changes.append(dict(style=style, space=space, **coverage(response, basis)))
    basis = projection.subspace.basis.detach().cpu().numpy().copy()
    del model, wrapped, projection, native_gate, gate
    torch.cuda.empty_cache()

    cache = ROOT / "features" / group / domain / variant / "features.npz"
    save_features(cache, z=z, E=e, S=removed, Q=q, Basis=basis, Q_env=q_env,
                  rotation=rotation, gains=gains, domains=np.array(DOMAINS), classes=np.array(classes),
                  **samples, **snapshots)
    payload = dict(samples, train=train)
    metrics, transport = probe_views(
        dict(views, S=removed), z, payload,
        affine_maps={"uniform": (q, np.full(16, BETA)),
                     "gate": (v, BETA * gains),
                     "uniform_equal_train_energy": (q, np.full(16, beta_match))},
    )
    axes = PCA(n_components=48, svd_solver="randomized", random_state=0).fit(z[train])
    geometries = {label: geometry(array[val], y[val], d[val], axes)
                  for label, array in (("z", z), ("E", e), ("S", removed))}
    return dict(group=group, held_out=domain, checkpoint=variant, checkpoint_path=str(path),
                checkpoint_sha256=hash_path(path), split_sha256=saved["identity"]["split_sha256"],
                code_commit=saved["source"]["commit"], reload_logit_error=reload_error,
                source_train=int(train.sum()), source_validation=int(val.sum()),
                actual=actual, gate_fit=gate_source, gate_active=active,
                gate=details, beta_match=beta_match, protection=protection_rank,
                source_class_conditioned_domain_shift_coverage=class_domain_drift(z, q, q_env, y, d, val),
                fixed_head=src_fp, probes=metrics, probe_transport=transport,
                geometry=geometries, coverage=changes)


def has_live_pane(session):
    result = subprocess.run(["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_dead}"],
                            text=True, capture_output=True)
    return result.returncode == 0 and "0" in result.stdout.splitlines()


def await_training(group):
    marker = ROOT / "logs" / f"{group}.complete.json"
    while not marker.is_file():
        if not has_live_pane(tmux_name(group)) and not marker.is_file():
            break
        time.sleep(60)
    receipt = json.loads((ROOT / "logs" / "preflight.passed.json").read_text())
    return receipt["commit"]


def run(group):
    commit = await_training(group)
    analyzed = 0
    for index, domain in enumerate(DOMAINS):
        candidates = folds(group, index)
        if not candidates or not (candidates[0] / ".github-archive").is_dir():
            continue
        folder, = candidates
        completed(folder, group, index, commit)
        analyzed += 1
        for variant in VARIANTS:
            path = ROOT / "probes" / group / domain / f"{variant}.json"
            if path.is_file():
                data = json.loads(path.read_text())
                assert data["checkpoint_sha256"] == hash_path(folder / "checkpoints" / f"{variant}.pth")
                assert (ROOT / "features" / group / domain / variant / "features.npz").is_file()
                continue
            uuid, free = wait_gpu(GROUPS[group])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            say("Analyze", group, domain, variant, f"{free} MiB free")
            subprocess.run([sys.executable, "-B", "-u", "-m", "tools.pacs_directdiff.analyze",
                            group, str(index), variant], env=env, check=True)
    atomic_json(ROOT / "logs" / f"{group}.analysis_complete.json",
                dict(group=group, folds=analyzed, features=2 * analyzed, training_commit=commit))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        group = sys.argv[1]
        try:
            run(group)
        except Exception as error:
            atomic_json(ROOT / "logs" / f"{group}.analysis_failure.json",
                        dict(group=group, error=type(error).__name__, reason=str(error)))
            raise
    else:
        group, index, phase = sys.argv[1:]
        index = int(index)
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        result = describe(group, index, phase)
        path = ROOT / "probes" / group / DOMAINS[index] / f"{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
        say("Verified feature export", group, DOMAINS[index], phase)
