"""Checkpointing and validated transfer for DICM stage-wise experiments.

Multi-worker DataLoader state is not captured; checkpoints do not promise exact
continuation of the minibatch or augmentation stream.
"""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def source_provenance():
    cwd = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(cwd), *args])
    commit = git("rev-parse", "HEAD").decode().strip()
    branch = git("branch", "--show-current").decode().strip()
    digest = hashlib.sha256(git("diff", "--binary", "HEAD"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(bytes([0]))
    repository = Path(git("rev-parse", "--show-toplevel").decode().strip())
    for entry in sorted(filter(None, untracked)):
        path = repository / os.fsdecode(entry)
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
            digest.update(entry + bytes([0]) + path.read_bytes())
    return {"commit": commit, "branch": branch, "working_tree_sha256": digest.hexdigest()}


def split_identity(dataset, in_splits, out_splits, args, test_envs):
    def split(split_items):
        result = []
        for data, _ in split_items:
            base = data.underlying_dataset
            records = getattr(base, "samples", None)
            if records is None and hasattr(base, "dataset"):
                records = getattr(base.dataset, "samples", None)
            result.append({
                "keys": list(map(int, data.keys)),
                "samples": [(str(path), int(label)) for path, label in records] if records is not None else None,
                "length": len(base),
            })
        return result

    data = {
        "dataset": args.dataset, "test_envs": list(test_envs),
        "domains": list(dataset.environments), "trial_seed": args.trial_seed,
        "seed": args.seed, "holdout": args.holdout_fraction,
        "label_ratio": args.label_ratio,
        "in": split(in_splits), "out": split(out_splits),
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return {"dataset": args.dataset, "test_envs": list(test_envs),
            "domains": list(dataset.environments), "trial_seed": args.trial_seed,
            "seed": args.seed, "split_sha256": digest}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def pack_average(model):
    if model is None:
        return None
    return {"state": model.state_dict(), "start_step": model.start_step,
            "end_step": model.end_step, "end_loss": getattr(model, "end_loss", None)}


def pack_swad(swad, segment):
    if swad is None:
        return None
    return {"segment": pack_average(segment),
            "converge_Q": [pack_average(m) for m in swad.converge_Q],
            "smooth_Q": [pack_average(m) for m in swad.smooth_Q],
            "final_model": pack_average(swad.final_model),
            "converge_step": swad.converge_step, "dead_valley": swad.dead_valley,
            "stage_first_dead_step": getattr(swad, "stage_first_dead_step", None),
            "threshold": swad.threshold, "n_converge": swad.n_converge,
            "n_tolerance": swad.n_tolerance, "tolerance_ratio": swad.tolerance_ratio}


def validate_stage(stage, hparams, parent):
    if hparams.get("dicm_reliable_contrast_gamma_max", 0):
        raise ValueError("Stagewise experiments require equal-weight contrastive learning")
    mode, l, pre = hparams["dicm_ablation"], hparams["l"], hparams["l_layer"]
    if stage == 0 and (mode != "A1" or l != 0 or pre != 0 or parent):
        raise ValueError("Stage 0 requires A1, l=0, l_layer=0 and no parent")
    if stage == 1 and (mode not in ("A1", "A2", "A4") or l != 0 or pre != 0 or not parent):
        raise ValueError("Stage 1 requires a parent, no GT and no contrastive losses")
    if stage == 2 and (mode != "A4" or not parent or (l, pre) not in ((0, 0), (1, 0), (0, 1), (1, 1))):
        raise ValueError("Stage 2 requires A4, a parent and contrastive coefficients of zero or one")


def load_parent(path, stage, algorithm, identity, hparams):
    parent = torch.load(path, map_location="cpu", weights_only=False)
    if parent.get("schema_version") != 1 or parent.get("identity") != identity:
        raise ValueError("Parent dataset, target, seed or split does not match")
    if parent["stage"] != stage - 1:
        raise ValueError("Parent checkpoint is from the wrong stage")
    old = parent["hparams"]
    invariant_keys = ["model", "n_layer", "resnet18", "pretrained", "freeze_bn", "resnet_dropout"]
    if stage == 2:
        invariant_keys += ["dicm_rank", "dicm_beta", "dicm_residual_projection",
                           "dicm_proto_momentum", "dicm_min_domain_samples", "dicm_min_domain_updates",
                           "dicm_min_valid_domains", "dicm_min_delta_norm", "dicm_isr_weight",
                           "dicm_aug_weight", "dicm_dom_weight", "dicm_basis_lr", "lr",
                           "weight_decay", "optimizer", "t", "t_pre"]
        if old["dicm_ablation"] not in ("A2", "A4"):
            raise ValueError("Stage 2 requires a pure DICM parent")
    for key in invariant_keys:
        if old.get(key) != hparams.get(key):
            raise ValueError(f"Parent parameter mismatch: {key}")
    state = parent["model"]
    if stage == 1:
        for name in ("featurizer", "classifier", "pre_featurizer"):
            module = getattr(algorithm, name)
            prefix = name + "."
            part = {key[len(prefix):]: value for key, value in state.items()
                    if key.startswith(prefix)}
            module.load_state_dict(part, strict=True)
    else:
        algorithm.load_state_dict(state, strict=True)
    return int(parent["cumulative_updates"]), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, algorithm, hparams, args, identity, stage, stage_updates,
         cumulative_updates, parent_hash, swad, segment, best_score, records=None, provenance=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "stage": stage, "identity": identity,
             "stage_updates": stage_updates, "cumulative_updates": cumulative_updates,
             "parent_sha256": parent_hash, "best_source_val_accuracy": best_score,
             "source_provenance": provenance,
             "hparams": dict(hparams), "model": algorithm.state_dict(),
             "optimizer": algorithm.optimizer.state_dict(), "rng": rng_state(),
             "records": records or [],
             "swad": pack_swad(swad, segment),
             "data_loader_state_saved": False,
             "resume_guarantee": "multi-worker sampling and augmentation stream are not exactly restored"}
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def restore_swad(saved, swad, algorithm):
    from domainbed.lib.swa_utils import AveragedModel

    if (saved is None) != (swad is None):
        raise ValueError("SWAD configuration differs from checkpoint")
    if saved is None:
        return None
    for key in ("n_converge", "n_tolerance", "tolerance_ratio"):
        if saved[key] != getattr(swad, key):
            raise ValueError(f"SWAD parameter mismatch: {key}")

    def unpack(value):
        if value is None:
            return None
        model = AveragedModel(algorithm, device="cpu")
        model.load_state_dict(value["state"], strict=True)
        model.start_step, model.end_step = value["start_step"], value["end_step"]
        if value["end_loss"] is not None:
            model.end_loss = value["end_loss"]
        return model

    swad.converge_Q.extend(unpack(v) for v in saved["converge_Q"])
    swad.smooth_Q.extend(unpack(v) for v in saved["smooth_Q"])
    swad.final_model = unpack(saved["final_model"])
    swad.converge_step = saved["converge_step"]
    swad.dead_valley, swad.threshold = saved["dead_valley"], saved["threshold"]
    swad.stage_first_dead_step = saved.get("stage_first_dead_step")
    return unpack(saved["segment"])


def resume(path, stage, identity, hparams, algorithm, swad):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved.get("stage") != stage or saved.get("identity") != identity:
        raise ValueError("Resume stage, target or split mismatch")
    if saved["hparams"] != dict(hparams):
        raise ValueError("Resume hyperparameters differ from checkpoint")
    algorithm.load_state_dict(saved["model"], strict=True)
    algorithm.optimizer.load_state_dict(saved["optimizer"])
    segment = restore_swad(saved["swad"], swad, algorithm)
    restore_rng(saved["rng"])
    return saved, segment
