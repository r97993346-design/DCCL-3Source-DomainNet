"""The shared CondCov trial protocol and checkpoint validation."""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

import yaml


CODE = Path(__file__).resolve().parents[2]
BASE = CODE.parents[1].parent
OUTPUT = CODE / "train_output" / "condcov_optuna_joint_20261125"
BRANCH = "experiment/condcov-optuna-joint-20261125"
MODEL_COMMIT = "a55914e203d1b3e36706a681550d17e620eb4172"
DATA = Path("/home/hooasia/lgg/data/repro_dccl_data")
CLIP = Path("/home/hooasia/.cache/clip/ViT-B-16.pt")
STEPS = 5000
CHECK = 300
INITIAL_BUDGET = 10
MAX_NEW = 15
SAMPLER_SEED = 42
N_JOBS = 1
EVAL_STEPS = [*range(0, STEPS, CHECK), STEPS - 1]
DIAG_STEPS = [0, 300, 900, 1800, 3000, 4200, 4800, 4999]
DATASETS = {
    "PACS": (("Art Painting", "Cartoon", "Photo", "Sketch"), 0.975, "PACS"),
    "VLCS": (("Caltech", "LabelMe", "Pascal", "SUN"), 0.845, "VLCS"),
    "OfficeHome": (("Art", "Clipart", "Product", "Real World"), 0.86, "office_home"),
}
SPACE = {
    "condcov_weight": (0.5, 1.0, 2.0),
    "rd_weight": (0.5, 1.0, 2.0),
    "contrastive_weight": (0.1, 0.25, 0.5),
    "prompt_length": (4, 8, 16),
    "contrastive_mix": (0.25, 0.5, 0.75),
}
SEARCH_KEYS = dict(zip(SPACE, (
    "cipt_condcov_weight", "cipt_env_loss_weight", "cipt_causal_contrastive_weight",
    "cipt_prompt_length", "cipt_clip_teacher_mix",
)))
FIXED_ARGS = {
    "algorithm": "CIPTDCCL", "source_sampling": "domain_balanced", "steps": STEPS,
    "checkpoint_freq": CHECK, "trial_seed": 1, "hparams_seed": 0, "holdout_fraction": 0.2,
    "aug": 0.0, "evalmode": "all", "num_workers": 2, "eval_num_workers": 0, "model_save": 0,
}
PROTOCOL_ARGS = dict(FIXED_ARGS, epochs=None, source_envs=None, target_env=None,
                     fixed_fold_seed=False, source_only_selection=True, deterministic=True,
                     prebuild_loader=False, label_ratio=1.)
LOSS_KEYS = ("loss_total", "loss_e_cls", "loss_s_uniform", "loss_condcov", "loss_rd",
             "loss_clip_contrast", "tr_outloss")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected):
    return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def atomic_write(path, render):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8",
                                     newline="", delete=False) as stream:
        temp = Path(stream.name)
        try:
            render(stream)
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def write_json(path, obj):
    atomic_write(path, lambda stream: json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False))


def write_text(path, value):
    atomic_write(path, lambda stream: stream.write(value))


def write_csv(path, rows, fields):
    def render(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, render)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def git(*args, cwd=CODE):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def seed_for(dataset, fold):
    return int(hashlib.md5(str((dataset, "CIPTDCCL", [fold], 0, 1)).encode()).hexdigest(), 16) % (2**31)


def signature(params):
    require(set(params) == set(SPACE), f"wrong search parameters: {params}")
    result = []
    for key, choices in SPACE.items():
        require(params[key] in choices, f"{key}={params[key]} is outside search space")
        result.append(choices[choices.index(params[key])])
    return tuple(result)


def fixed_hparams():
    config = yaml.safe_load((CODE / "config_cipt_pacs_ind_ablation.yaml").read_text(encoding="utf-8"))
    config.update(lr=1e-5, cipt_s_uniform_weight=1., cipt_e_cls_weight=1.,
                  cipt_use_ind=False, cipt_gamma=0., ind_mode="condcov", swad=None,
                  cipt_prompt_init="a photo of a", test_batchsize=64, label_ratio=1.,
                  data_augmentation=True, val_augment=False)
    for key in (*SEARCH_KEYS.values(), "cipt_contrastive_weight"):
        config.pop(key, None)
    return config


def hparams_for(params):
    signature(params)
    return dict(fixed_hparams(),
                **{config_key: params[key] for key, config_key in SEARCH_KEYS.items()},
                cipt_contrastive_weight=params["contrastive_weight"])


def command(dataset, fold, trial, config, output, telemetry=None):
    launcher = "train_all.py" if telemetry is None else "scripts/condcov_optuna/instrumented_train.py"
    cmd = [sys.executable, "-B", "-u", launcher]
    if telemetry is not None:
        cmd.extend((str(telemetry), str(read_json(config)["cipt_prompt_length"])))
    cmd.extend([f"{dataset}_trial{trial}_fold{fold}", str(config), "--test_envs", str(fold),
                "--dataset", dataset, "--data_dir", str(DATA), "--cipt_clip_path", str(CLIP), "--batch_size",
                "32", "--deterministic", "--source_only_selection", "--ind_grad_steps",
                *map(str, DIAG_STEPS), "--output_root", str(output)])
    for key, value in FIXED_ARGS.items():
        cmd.extend((f"--{key}", str(value)))
    return cmd


def validate_fold(rows, dataset, fold, params, *, imported=False):
    require([r["step"] for r in rows] == EVAL_STEPS, "expected all 5000 updates through checkpoint 4999")
    expected = hparams_for(params)
    if imported:
        # The legacy flag/weight and generic contrastive coefficient are inactive in this route.
        for key in ("cipt_use_ind", "cipt_gamma", "cipt_contrastive_weight"):
            expected.pop(key)
    effective = {"ind_weight": params["condcov_weight"], "e_cls_w": 1., "s_uni_w": 1.,
                 "env_loss_w": params["rd_weight"], "contrastive_weight_eff": params["contrastive_weight"]}
    for row in rows:
        args, hp = row["args"], row["hparams"]
        require(all(args.get(key) == value for key, value in PROTOCOL_ARGS.items()),
                "training or evaluation protocol mismatch")
        require(args["dataset"] == dataset and args["real_test_envs"] == [fold]
                and args["seed"] == seed_for(dataset, fold), "fold hash seed or dataset mismatch")
        require(Path(args["data_dir"]).resolve() == DATA.resolve()
                and Path(args["cipt_clip_path"]).resolve() == CLIP.resolve(), "dataset or CLIP path mismatch")
        require(all(hp.get(key) == value for key, value in expected.items()), "fixed or search parameter mismatch")
        require(row["ind_mode"] == "condcov" and row["cipt_ind_loss"] == 0., "original IND was active")
        require(all(same_number(row[key], value) for key, value in effective.items()),
                "unexpected effective loss weight")
        require(all(row[key] == 1. for key in ("ablation_de_enabled", "env_anchor_active", "cg_active", "e_only_infer")),
                "required training path was inactive")
        require(all(math.isfinite(row[key]) for key in (*LOSS_KEYS, "train_out", "test_in", "test_out")),
                "loss or metric was not finite")
    selected = max(rows, key=lambda r: r["train_out"])
    final = rows[-1]
    return {
        "fold": fold, "domain": DATASETS[dataset][0][fold], "seed": seed_for(dataset, fold),
        "selected_step": selected["step"], "iid": selected["test_in"], "test_in": selected["test_in"],
        "test_out": selected["test_out"], "source_val": selected["train_out"],
        "final_step": final["step"], "updates": final["step"] + 1,
        "final_iid": final["test_in"], "final_test_out": final["test_out"], "final_source_val": final["train_out"],
    }


def summarize(folds):
    require([fold["fold"] for fold in folds] == list(range(4)), "trial requires all four distinct folds")
    durations = [f.get("elapsed_seconds") for f in folds]
    return {
        "source_val": mean(f["source_val"] for f in folds),
        "iid": mean(f["iid"] for f in folds), "test_in": mean(f["test_in"] for f in folds),
        "test_out": mean(f["test_out"] for f in folds),
        "elapsed_seconds": sum(durations) if all(x is not None for x in durations) else None,
        "folds": folds,
    }
