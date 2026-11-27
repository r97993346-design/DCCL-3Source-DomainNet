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
RUN_NAME = "cipt_clip_supcon_cosine_optuna_mincl_space_v2_20261127"
OUTPUT = CODE / "train_output" / RUN_NAME
BRANCH = "experiment/cipt-clip-supcon-mincl-space-v2-20261127"
BASE_COMMIT = "8e25addc9ae0ba32cd48e1673f023997ae000039"
PAUSE = OUTPUT / "pause_after_trial.request"
MODEL_COMMIT = "a55914e203d1b3e36706a681550d17e620eb4172"
DATA = Path("/home/hooasia/lgg/data/repro_dccl_data")
CLIP = Path("/home/hooasia/.cache/clip/ViT-B-16.pt")
STEPS = 5001
CHECK = 300
BUDGET = 648
SAMPLER_SEED = 42
STARTUP_TRIALS = 5
N_JOBS = 1
EVAL_STEPS = [*range(0, STEPS, CHECK), STEPS - 1]
DATASETS = {
    "PACS": (("Art Painting", "Cartoon", "Photo", "Sketch"), 0.975, "PACS"),
    "VLCS": (("Caltech", "LabelMe", "Pascal", "SUN"), 0.845, "VLCS"),
    "OfficeHome": (("Art", "Clipart", "Product", "Real World"), 0.86, "office_home"),
}
SPACE = {
    "initial_lr": (2e-5, 3e-5),
    "condcov_weight": (0.5, 1.0, 2.0, 3.0),
    "rd_weight": (0.5, 1.0, 2.0, 3.0),
    "contrastive_weight": (0.001, 0.025, 0.05),
    "contrastive_mix": (0.0, 0.25, 0.5, 0.75),
    "prompt_length": (4, 8, 16),
}
OLD_RUN_NAME = "cipt_clip_supcon_cosine_optuna_mincl_20261126"
OLD_OUTPUT = BASE.parent / OLD_RUN_NAME / "code/DCCL/DCCL/train_output" / OLD_RUN_NAME
VLCS_LR = (7.5e-6, 1e-5)
VALIDATION_SPACE = dict(SPACE, initial_lr=(*SPACE["initial_lr"], *VLCS_LR))
INITIAL_PARAMS = {name: min(choices) for name, choices in SPACE.items()}


def space_for(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}")
    return dict(SPACE, initial_lr=VLCS_LR) if dataset == "VLCS" else dict(SPACE)

SEARCH_KEYS = dict(zip(SPACE, (
    "lr", "cipt_condcov_weight", "cipt_env_loss_weight", "cipt_causal_contrastive_weight",
    "cipt_clip_teacher_mix", "cipt_prompt_length",
)))
LR_POLICY = {"name": "CosineAnnealingLR", "eta_min": 0.0, "warmup_steps": 0, "t_max": STEPS,
             "final_update_index": STEPS - 1, "timing": "optimizer_then_scheduler"}
FIXED_ARGS = {
    "algorithm": "CIPTDCCL", "source_sampling": "domain_balanced", "steps": STEPS,
    "checkpoint_freq": CHECK, "trial_seed": 1, "hparams_seed": 0, "holdout_fraction": 0.2,
    "aug": 0.0, "evalmode": "all", "num_workers": 2, "eval_num_workers": 0, "model_save": 0,
}
PROTOCOL_ARGS = dict(FIXED_ARGS, epochs=None, source_envs=None, target_env=None,
                     fixed_fold_seed=False, source_only_selection=True, deterministic=True,
                     prebuild_loader=False, label_ratio=1.)
TERMS = ("e_cls", "s_uniform", "condcov", "rd", "contrastive")


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


def budget_for(root, dataset):
    path = root / "trial_budgets.json"
    return read_json(path).get(dataset, BUDGET) if path.is_file() else BUDGET


def signature(params, space=VALIDATION_SPACE):
    require(set(params) == set(space), f"wrong search parameters: {params}")
    result = []
    for key, choices in space.items():
        require(params[key] in choices, f"{key}={params[key]} is outside search space")
        result.append(choices[choices.index(params[key])])
    return tuple(result)


def fixed_hparams():
    config = yaml.safe_load((CODE / "config_cipt_pacs_ind_ablation.yaml").read_text(encoding="utf-8"))
    config.update(cipt_s_uniform_weight=1., cipt_e_cls_weight=1., cipt_use_ind=False, cipt_gamma=0.,
                  ind_mode="condcov", swad=None, cipt_prompt_init="a photo of a", test_batchsize=64,
                  label_ratio=1., data_augmentation=True, val_augment=False, class_balanced=False,
                  cipt_contrastive_type="clip_guided_supcon", cipt_neighbor_diagnostics=False,
                  cipt_contrastive_warmup_steps=0, cipt_contrastive_temperature=.07,
                  cipt_clip_teacher_temperature=.2, cipt_clip_teacher_topk=0,
                  cipt_lr_schedule="cosine", cipt_lr_min=0.)
    for key in (*SEARCH_KEYS.values(), "cipt_contrastive_weight"):
        config.pop(key, None)
    return config


def hparams_for(params):
    signature(params)
    return dict(fixed_hparams(), **{config_key: params[key] for key, config_key in SEARCH_KEYS.items()},
                cipt_contrastive_weight=params["contrastive_weight"])


def study_name(dataset):
    return f"{dataset}_condcov_rd_clip_allpos_cosine_mincl_space_v2_seed1"


def cosine_lr(index, initial_lr, updates=STEPS):
    return initial_lr * (1.0 + math.cos(math.pi * index / updates)) / 2.0


def command(dataset, fold, trial, config, output, telemetry=None, *, updates=STEPS):
    launcher = "train_all.py" if telemetry is None else "scripts/cipt_clip_supcon_optuna/instrumented_train.py"
    cmd = [sys.executable, "-B", "-u", launcher]
    if telemetry is not None:
        cmd.append(str(telemetry))
    cmd.extend([f"{dataset}_trial{trial}_fold{fold}", str(config), "--test_envs", str(fold),
                "--dataset", dataset, "--data_dir", str(DATA), "--cipt_clip_path", str(CLIP), "--batch_size",
                "32", "--deterministic", "--source_only_selection", "--output_root", str(output)])
    for key, value in dict(FIXED_ARGS, steps=updates).items():
        cmd.extend((f"--{key}", str(value)))
    return cmd


def validate_fold(rows, dataset, fold, params, *, updates=STEPS):
    eval_steps = EVAL_STEPS if updates == STEPS else sorted({*range(0, updates, CHECK), updates - 1})
    require([r["step"] for r in rows] == eval_steps, f"expected {updates} updates and final evaluation")
    expected = hparams_for(params)
    effective = dict(zip(TERMS, (1., 1., params["condcov_weight"], params["rd_weight"], params["contrastive_weight"])))
    for row in rows:
        args, hp, step = row["args"], row["hparams"], row["step"]
        require(all(args.get(k) == v for k, v in dict(PROTOCOL_ARGS, steps=updates).items()),
                "training or evaluation protocol mismatch")
        require(args["dataset"] == dataset and args["real_test_envs"] == [fold]
                and args["seed"] == seed_for(dataset, fold), "fold hash seed or dataset mismatch")
        require(Path(args["data_dir"]).resolve() == DATA.resolve()
                and Path(args["cipt_clip_path"]).resolve() == CLIP.resolve(), "dataset or CLIP path mismatch")
        require(all(hp.get(k) == v for k, v in expected.items()), "fixed or search parameter mismatch")
        require(row["ind_mode"] == "condcov" and row["cipt_ind_loss"] == row["nbr_check"] == 0.
                and not hp.get("ind_grad_steps") and not any(k.endswith("grad_by_step") for k in row),
                "legacy IND, NR or high-cost gradient diagnostics were active")
        require(all(row[k] == 1. for k in ("ablation_de_enabled", "env_anchor_active", "cg_all_class_active", "e_only_infer")),
                "required CLIP SupCon path was inactive")
        require(row["optimizer_updates"] == row["scheduler_updates"] == step + 1
                and row["actual_lr"] > 0
                and math.isclose(row["actual_lr"], cosine_lr(step, params["initial_lr"], updates),
                                 rel_tol=1e-7, abs_tol=1e-16), "bad update index or LR")
        scalars = (*(f"loss_{k}" for k in TERMS), "loss_total", "tr_loss", "tr_outloss",
                   "train_out", "test_in", "test_out", "step_time", "eval_time")
        require(all(math.isfinite(row[k]) for k in scalars), "non-finite loss or metric")
        for term, weight in effective.items():
            require(same_number(row[f"weight_{term}"], weight), f"bad effective {term} weight")
            require(math.isclose(row[f"weighted_{term}"], weight * row[f"loss_{term}"],
                                 rel_tol=2e-6, abs_tol=2e-8), f"double or missing weighting for {term}")
        require(math.isclose(sum(row[f"weighted_{term}"] for term in TERMS), row["loss_total"],
                             rel_tol=2e-6, abs_tol=2e-8), "summed terms differ from total loss")
    selected = max(rows, key=lambda r: r["train_out"])
    final = rows[-1]
    return {"fold": fold, "domain": DATASETS[dataset][0][fold], "seed": seed_for(dataset, fold),
            "selected_step": selected["step"], "iid": selected["test_in"], "test_in": selected["test_in"],
            "test_out": selected["test_out"], "source_val": selected["train_out"], "final_step": final["step"],
            "updates": final["step"] + 1, "scheduler_updates": final["scheduler_updates"], "evaluations": len(rows),
            "final_iid": final["test_in"], "final_test_out": final["test_out"], "final_source_val": final["train_out"]}


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
