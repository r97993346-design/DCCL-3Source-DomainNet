"""Local multiobjective TPE, with serial folds and source-only selection."""
import importlib.util
import json
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "scripts/cipt_clip_supcon_optuna"))
import protocol as p
spec = importlib.util.spec_from_file_location("fold_runner", CODE / "scripts/cipt_clip_supcon_optuna/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
NAME = "oh_stagewise_grid_crossdomain_constant1e5_20261128"
DATASET = "OfficeHome"
ROOT = Path("/root/autodl-tmp/outputs") / NAME
ROOT.mkdir(parents=True, exist_ok=True)
p.DATASETS["TerraIncognita"] = (("L100","L38","L43","L46"), .58, "terra_incognita_fixed")
p.FIXED_ARGS.update(num_workers=8, eval_num_workers=0)
p.PROTOCOL_ARGS.update(num_workers=8, eval_num_workers=0, cipt_cache_eval_visual=True)
p.OUTPUT = runner.OUTPUT = ROOT
runner.PAUSE = ROOT / "pause.request"
runner.GPU_FOR_DATASET = {DATASET: 0}
runner.ACTIVE_GPU = {0: 0}
p.LR_POLICY = {"name":"constant","lr":1e-5,"updates":5001,"scheduler_updates":0}

def hp(params):
    return dict(p.fixed_hparams(), lr=1e-5, cipt_lr_schedule="constant",
                cipt_contrastive_type="clip_guided_cross_domain", cipt_prompt_length=8,
                cipt_clip_teacher_mix=.25, cipt_s_uniform_weight=params["s_weight"],
                cipt_condcov_weight=params["condcov_weight"], cipt_env_loss_weight=params["rd_weight"],
                cipt_causal_contrastive_weight=params["contrastive_weight"],
                cipt_contrastive_weight=params["contrastive_weight"])

p.hparams_for = runner.hparams_for = hp
original_command = p.command
def command(*args, **kw):
    cmd = original_command(*args, **kw)
    cmd[3] = "scripts/motpe_20261128/instrument.py"
    return [*cmd, "--cipt_cache_eval_visual"]
runner.command = command

# Reuse protocol verification with the explicit constant-LR and cross-domain changes.
def validate(rows, dataset, fold, params, *, updates=p.STEPS):
    assert [r["step"] for r in rows] == p.EVAL_STEPS
    expected = hp(params)
    weights = dict(zip(p.TERMS,(1.,params["s_weight"],params["condcov_weight"],params["rd_weight"],params["contrastive_weight"])))
    for row in rows:
        args, h = row["args"], row["hparams"]
        assert all(args.get(k)==v for k,v in p.PROTOCOL_ARGS.items())
        assert args["dataset"]==dataset and args["real_test_envs"]==[fold] and args["seed"]==p.seed_for(dataset,fold)
        assert all(h.get(k)==v for k,v in expected.items())
        assert row["optimizer_updates"]==row["step"]+1 and row["scheduler_updates"]==0 and row["actual_lr"]==1e-5
        assert row["ind_mode"]=="condcov" and row["cipt_ind_loss"]==row["nbr_check"]==0
        assert all(row[k]==1 for k in ("ablation_de_enabled","e_only_infer"))
        assert row["env_anchor_active"] == float(params["rd_weight"] > 0)
        assert (row.get("cg_active",0)>0) == (params["contrastive_weight"] > 0)
        assert row["eval_visual_cache_bytes"] > 0
        if row["step"] > 0: assert row["eval_visual_cache_hit_splits"] == 8
        for term,w in weights.items():
            assert math.isclose(row["weight_"+term], w, rel_tol=1e-9, abs_tol=1e-12)
            assert math.isclose(row["weighted_"+term],w*row["loss_"+term],rel_tol=2e-6,abs_tol=2e-8)
        assert math.isclose(sum(row["weighted_"+k] for k in weights),row["loss_total"],rel_tol=2e-6,abs_tol=2e-8)
        assert all(math.isfinite(row[k]) for k in ("test_in","test_out","train_out","loss_total"))
    selected = max(rows,key=lambda r:r["train_out"])
    return dict(fold=fold,domain=p.DATASETS[dataset][0][fold],seed=p.seed_for(dataset,fold),
                selected_step=selected["step"],source_val=selected["train_out"],iid=selected["test_in"],
                test_in=selected["test_in"],test_out=selected["test_out"],updates=5001, actual_lr=selected["actual_lr"],
                losses={k:v for k,v in selected.items() if k.startswith(("loss_","weighted_","weight_"))},
                code_commit=p.git("rev-parse","HEAD"))

def read_attempt(attempt,dataset,fold,params,*,updates=p.STEPS):
    assert (attempt/"exit_code").read_text().strip()=="0"
    logs=list((attempt/"raw").glob(f"{dataset}/*/results.jsonl"))
    assert len(logs)==1 and not list(attempt.rglob("*.pth"))
    result=validate(p.read_rows(logs[0]),dataset,fold,params)
    trace=p.read_rows(attempt/"learning_rates.jsonl")
    assert len(trace)==5001 and all(r["step"]==i and r["lr_used"]==1e-5 and r["scheduler_updates"]==0 for i,r in enumerate(trace))
    assert "ADAM_UPDATES 5001 CONSTANT_LR 1e-5 SCHEDULER_UPDATES 0" in (attempt/"console.log").read_text()
    return dict(result,log=str(logs[0]),learning_rates=str(attempt/"learning_rates.jsonl"),
                elapsed_seconds=float((attempt/"elapsed_seconds").read_text()))
verified_read_attempt = read_attempt
def read_attempt(*args, **kwargs):
    try:
        return verified_read_attempt(*args, **kwargs)
    except AssertionError as exc:
        raise ValueError("Completed attempt failed protocol validation") from exc
runner.read_attempt = read_attempt
