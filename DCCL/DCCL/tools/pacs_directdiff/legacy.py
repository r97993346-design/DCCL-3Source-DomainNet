"""Recompute comparable source probes on verified, already-exported A16 features."""

import json

import numpy as np

from tools.pacs_causal.features import STYLES
from tools.pacs_causal.metrics import coverage
from tools.pacs_directdiff.analyze import SAMPLE_COLUMNS
from tools.pacs_directdiff.common import DOMAINS, PREVIOUS, ROOT, atomic_json, historical_fold, squared_norm
from tools.pacs_directdiff.probes import PRIMARY_RIDGE, probe_views
from tools.pacs_projection.analyze import hash_path


def a16_reference(index, variant):
    domain = DOMAINS[index]
    cache = PREVIOUS / "features" / "A16" / domain / variant / "features.npz"
    prior = json.loads((PREVIOUS / "probes" / "A16" / domain / f"{variant}.json").read_text())
    _, checkpoint = historical_fold("A16", index, variant)
    assert prior["checkpoint_sha256"] == hash_path(checkpoint)
    with np.load(cache, allow_pickle=False) as data:
        sample = {key: data[key] for key in SAMPLE_COLUMNS}
        z = data["z"]
        arrays = {"z": z, "E": data["E_beta"], "S": data["S_removed"]}
        changes = [
            dict(style=style, space=space, **coverage(data["delta_z_" + style], data[key]))
            for style in STYLES for space, key in (("Q", "Q"), ("Q_env", "Q_env"))
        ]
    assert index not in sample["domain"]
    train = sample["is_source_train"]
    scores, _ = probe_views(arrays, z, dict(sample, train=train), sensitivity=())
    return dict(group="A16", held_out=domain, checkpoint=variant,
                split_sha256=prior["split_sha256"], checkpoint_sha256=prior["checkpoint_sha256"],
                probes=scores, coverage=changes, protection=prior["protection"],
                validation_removed_energy=squared_norm(arrays["S"][~train]) / squared_norm(z[~train]),
                geometry={key: prior["geometry"][name]
                          for key, name in (("z", "z"), ("E", "E_beta"))})


def load_legacy(variant):
    output = []
    for i, d in enumerate(DOMAINS):
        path = ROOT / "probes" / "A16" / d / f"{variant}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if str(PRIMARY_RIDGE) in cached["probes"]["z"]["class"]:
                output.append(cached)
                continue
        row = a16_reference(i, variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, row)
        output.append(row)
    return output
