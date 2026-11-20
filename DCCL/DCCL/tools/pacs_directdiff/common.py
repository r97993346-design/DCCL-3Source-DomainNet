"""Paths and source-only controls shared by the two DirectDiff16 runs."""

from pathlib import Path

import numpy as np
import torch

from tools.pacs_causal.queue import DATA, DOMAINS, atomic_json, query, say
from tools.pacs_projection.queue import A8 as CAUSAL, ROOT as PREVIOUS


ROOT = PREVIOUS.parent / "dicm-pacs-directdiff16-20261120"
PROJECT = Path(__file__).resolve().parents[2]
GROUPS = {"Uniform": 0, "Gate": 1}
VARIANTS = ("iid_best", "swad")
BETA = .75


def folds(group, index):
    return sorted((ROOT / group / "PACS").glob(f"*_{group}_env{index}"))


def historical_fold(group, index, checkpoint="iid_best"):
    if group not in ("P0", "A16"):
        raise ValueError(f"Unknown historical control {group}")
    location = CAUSAL if group == "P0" else PREVIOUS
    folder, = (location / group / "PACS").glob(f"*_{group}_env{index}")
    return folder, folder / "checkpoints" / f"{checkpoint}.pth"


def source_centroids(z, y, domains, train, min_count=8):
    """Offline analog of EMA centroids, computed from the source-training side."""
    classes = np.unique(y[train])
    sources = np.unique(domains[train])
    if len(sources) != 3:
        raise ValueError(f"Expected three source domains, found {len(sources)}")
    means = np.zeros((len(classes), len(sources), z.shape[1]), dtype=np.float64)
    sizes = np.zeros(means.shape[:2], dtype=np.int64)
    for i, c in enumerate(classes):
        for j, d in enumerate(sources):
            select = train & (y == c) & (domains == d)
            sizes[i, j] = np.count_nonzero(select)
            if sizes[i, j] >= min_count:
                means[i, j] = z[select].mean(0, dtype=np.float64)
    return torch.from_numpy(means), torch.from_numpy(sizes >= min_count), sizes


def squared_norm(x):
    return float(np.einsum("nd,nd->", x, x, dtype=np.float64))


def controls(z, q, rotation, gains, train):
    """Match total removed energy on source train and per example without labels."""
    v = q @ rotation
    pz = (z @ q) @ q.T
    gz = ((z @ v) * gains) @ v.T
    denom = squared_norm(pz[train])
    if denom <= 1e-12:
        raise ValueError("Cannot match energy without projected source-train signal")
    beta_match = BETA * np.sqrt(squared_norm(gz[train]) / denom)
    per_example = np.linalg.norm(gz, axis=1) / np.maximum(np.linalg.norm(pz, axis=1), 1e-12)
    views = {
        "z": z,
        "uniform": z - BETA * pz,
        "gate": z - BETA * gz,
        "orthogonal": z - pz,
        "uniform_equal_train_energy": z - np.float32(beta_match) * pz,
        "uniform_equal_sample_energy": z - BETA * per_example[:, None] * pz,
    }
    return views, float(beta_match)
