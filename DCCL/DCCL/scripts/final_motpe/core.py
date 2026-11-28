"""Local multiobjective TPE, with serial folds and source-only selection."""
import importlib.util
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "scripts/cipt_clip_supcon_optuna"))
import protocol as p
spec = importlib.util.spec_from_file_location("fold_runner", CODE / "scripts/cipt_clip_supcon_optuna/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
DATASET = sys.argv[1]
NAME = 'condcov_rd_crossdomain_cosine2e5_prompt4_final_motpe_20261128'
ROOT = Path('/media/hooasia/data/outputs/condcov_rd_crossdomain_cosine2e5_prompt4_final_motpe_20261128') / DATASET
GPUS = {'PACS': 1, 'OfficeHome': 0}
GPU = GPUS[DATASET]
WORKERS = {'PACS': 2, 'OfficeHome': 4}[DATASET]
p.DATASETS["TerraIncognita"] = (("L100","L38","L43","L46"), .58, "terra_incognita_fixed")
p.DATA = Path('/home/hooasia/lgg/data/repro_dccl_data')
p.CLIP = Path('/home/hooasia/.cache/clip/ViT-B-16.pt')
p.FIXED_ARGS.update(num_workers=WORKERS, eval_num_workers=0)
p.PROTOCOL_ARGS.update(num_workers=WORKERS, eval_num_workers=0, cipt_cache_eval_visual=True)
p.OUTPUT = runner.OUTPUT = ROOT
runner.PAUSE = ROOT / "pause.request"
runner.GPU_FOR_DATASET = {DATASET: GPU}
runner.ACTIVE_GPU = {GPU: 0}
LR = 2e-5
p.LR_POLICY = dict(name="CosineAnnealingLR", initial_lr=LR, eta_min=0, t_max=5001, timing="Adam then scheduler")

def hp(params):
    return dict(p.fixed_hparams(), lr=LR, cipt_lr_schedule="cosine",
                cipt_contrastive_type="clip_guided_cross_domain", cipt_prompt_length=4,
                cipt_clip_teacher_mix=.25, cipt_s_uniform_weight=params["s_weight"],
                cipt_condcov_weight=params["condcov_weight"], cipt_env_loss_weight=params["rd_weight"],
                cipt_causal_contrastive_weight=params["contrastive_weight"],
                cipt_contrastive_weight=params["contrastive_weight"], ind_grad_steps=[])

p.hparams_for = runner.hparams_for = hp
original_command = p.command
def command(*args, **kw):
    cmd = original_command(*args, **kw)
    cmd[3] = "scripts/final_motpe/instrument.py"
    return [*cmd, "--cipt_cache_eval_visual"]
runner.command = command

class ProtocolError(ValueError):
    pass


def expect(field, actual, expected, *, close=False):
    matches = math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-12) if close and isinstance(actual, (float,int)) else actual == expected
    if not matches:
        raise ProtocolError(f"{field}: actual={actual!r}, expected={expected!r}")


def expect_lr(field, actual, expected):
    if not isinstance(actual, (float,int)) or not math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-16):
        raise ProtocolError(f"{field}: actual={actual!r}, expected={expected!r}")

def validate(rows, dataset, fold, params):
    expect("checkpoint steps", [r.get("step") for r in rows], p.EVAL_STEPS)
    expected = hp(params)
    weights = dict(zip(p.TERMS, (1., params["s_weight"], params["condcov_weight"],
                               params["rd_weight"], params["contrastive_weight"])))
    for row in rows:
        step = row["step"]
        prefix = f"checkpoint {step}"
        args, h = row["args"], row["hparams"]
        for key, value in dict(p.PROTOCOL_ARGS, dataset=dataset, real_test_envs=[fold],
                               seed=p.seed_for(dataset, fold)).items():
            expect(f"{prefix} args.{key}", args.get(key), value)
        for key, value in expected.items():
            expect(f"{prefix} hparams.{key}", h.get(key), value)
        for key, value in dict(ind_mode="condcov",
                               cipt_ind_loss=0, nbr_check=0, ablation_de_enabled=1, e_only_infer=1).items():
            expect(f"{prefix} {key}", row.get(key), value)
        expect(f"{prefix} RD active", row.get("env_anchor_active"), float(params["rd_weight"]>0))
        expect(f"{prefix} CL active", row.get("cg_active",0)>0, params["contrastive_weight"]>0)
        expect(f"{prefix} cache populated", row.get("eval_visual_cache_bytes",0)>0, True)
        if step: expect(f"{prefix} cache hits", row.get("eval_visual_cache_hit_splits"), 8)
        for term, weight in weights.items():
            expect(f"{prefix} weight_{term}", row.get("weight_"+term), weight, close=True)
            expect(f"{prefix} weighted_{term}", row.get("weighted_"+term), weight * row["loss_"+term], close=True)
        expect(f"{prefix} summed losses", sum(row["weighted_"+k] for k in weights), row["loss_total"], close=True)
        for key in ("train_out","test_in","test_out","loss_total"):
            expect(f"{prefix} finite {key}", math.isfinite(row[key]), True)
    selected = max(rows,key=lambda r:r["train_out"])
    return dict(fold=fold,domain=p.DATASETS[dataset][0][fold],seed=p.seed_for(dataset,fold),
                selected_step=selected["step"], source_val=selected["train_out"], iid=selected["test_in"],
                test_in=selected["test_in"],test_out=selected["test_out"],updates=5001,scheduler_updates=5001,
                actual_lr=p.cosine_lr(selected["step"],LR), initial_lr=LR, final_scheduler_lr=0.,
                losses={k:v for k,v in selected.items() if k.startswith(("loss_","weighted_","weight_"))},
                code_commit=p.git("rev-parse","HEAD"))


def read_attempt(attempt,dataset,fold,params,*,updates=p.STEPS):
    expect("process exit code", (attempt/"exit_code").read_text().strip(), "0")
    logs=list((attempt/"raw").glob(f"{dataset}/*/results.jsonl"))
    expect("result logs", len(logs), 1)
    expect("saved weights", list(attempt.rglob("*.pth")), [])
    result=validate(p.read_rows(logs[0]),dataset,fold,params)
    trace=p.read_rows(attempt/"learning_rates.jsonl")
    expect("Adam trace length", len(trace), updates)
    for index,row in enumerate(trace):
        expect(f"trace[{index}].step",row.get("step"),index)
        expect(f"trace[{index}].optimizer_updates",row.get("optimizer_updates"),index+1)
        expect(f"trace[{index}].scheduler_updates",row.get("scheduler_updates"),index+1)
        expect_lr(f"trace[{index}].lr_used",row.get("lr_used"),p.cosine_lr(index,LR))
        expect_lr(f"trace[{index}].lr_after_scheduler",row.get("lr_after_scheduler"),p.cosine_lr(index+1,LR))
    expect("final scheduler LR",trace[-1]["lr_after_scheduler"],0.)
    console=(attempt/"console.log").read_text()
    expect("prompt length telemetry","PROMPT_CONTEXT_LENGTH 4" in console,True)
    expect("scheduler audit completion","ADAM_UPDATES 5001 COSINE_UPDATES 5001 FINAL_LR 0.0" in console,True)
    return dict(result,last_adam_lr=trace[-1]["lr_used"],log=str(logs[0]),learning_rates=str(attempt/"learning_rates.jsonl"),
                elapsed_seconds=float((attempt/"elapsed_seconds").read_text()))
runner.read_attempt = read_attempt


# This independent study has no dependency on the obsolete handoff protocol.
def start_attempt(attempt, dataset, fold, number, config, gpu, env, lock_fd):
    import shlex
    import subprocess
    import time
    attempt.mkdir()
    args = command(dataset, fold, number, config, attempt/"raw", attempt/"learning_rates.jsonl")
    wrapper = attempt/"command.sh"
    p.write_text(wrapper, "\n".join([
        "#!/bin/bash", "start=$(date +%s)", "rc=0",
        shlex.join(args) + " || rc=$?", "end=$(date +%s)",
        'printf "%s\\n" "$((end-start))" > elapsed_seconds.tmp',
        "mv elapsed_seconds.tmp elapsed_seconds",
        'printf "%s\\n" "$rc" > exit_code.tmp',
        "mv exit_code.tmp exit_code", 'exit "$rc"']) + "\n")
    while runner.available_ram_kib() <= 10 * 1024**2:
        p.write_json(ROOT/"status.json",dict(state="waiting_for_ram",trial=number,fold=fold,gpu=gpu))
        time.sleep(15)
    # Run relative training entrypoints from CODE while emitting exit markers in attempt.
    shell = wrapper.read_text().replace(shlex.join(args), "(cd " + shlex.quote(str(CODE)) + " && " + shlex.join(args) + ")")
    p.write_text(wrapper,shell)
    with (attempt/"console.log").open("wb") as stream:
        child=subprocess.Popen(["bash",str(wrapper)],cwd=attempt,env=env,
                               stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
        p.write_json(ROOT/"status.json",dict(state="running",dataset=dataset,trial=number,fold=fold,
                     domain=p.DATASETS[dataset][0][fold],gpu=gpu,pid=child.pid,controller_pid=__import__("os").getpid()))
        runner.emit("fold_started",dataset=dataset,trial=number,fold=fold,gpu=gpu,pid=child.pid)
        child.wait()

runner.start_attempt = start_attempt
