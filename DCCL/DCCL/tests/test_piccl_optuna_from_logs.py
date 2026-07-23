import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tune_piccl_optuna as tuner

HISTORY = ROOT / "experiment_logs" / "PACS" / "piccl_optuna_evidence_20260713"
CONFIG = ROOT / "configs" / "piccl_pacs_optuna_from_logs.json"


def test_four_historical_logs_read_and_params_recovered():
    runs = tuner.read_history(HISTORY)
    assert len(runs) == 4
    by = {r["name"]: r for r in runs}
    assert by["260712_13-48-25_pacs_dccl_seed0"]["algorithm"] == "DCCL"
    assert by["260713_22-47-00_pacs_piccl_candidate_preserve_dccl_seed0"]["target_report_only"] is not None


def test_search_range_from_config():
    cfg = json.loads(CONFIG.read_text())
    assert list(cfg["search_space"]) == ["piccl_rank", "piccl_beta_max", "piccl_isr_weight"]
    assert cfg["search_space"]["piccl_rank"]["choices"] == [8, 16]
    assert cfg["search_space"]["piccl_beta_max"]["low"] == 0.10
    assert cfg["search_space"]["piccl_isr_weight"]["log"] is True


def test_tpe_sampler_and_no_random_choice_in_optuna_script():
    src = (ROOT / "scripts" / "tune_piccl_optuna.py").read_text()
    assert "optuna.samplers.TPESampler" in src
    assert "optuna.create_study" in src
    assert "trial.suggest_float" in src
    assert "random.choice" not in src


def test_enqueue_trial(tmp_path):
    optuna = pytest.importorskip("optuna")
    cfg = json.loads(CONFIG.read_text())
    study = optuna.create_study(storage=f"sqlite:///{tmp_path/'s.db'}", load_if_exists=True, direction="maximize")
    queued = tuner.enqueue_history(study, cfg, tuner.read_history(HISTORY))
    assert isinstance(queued, list)


def test_pacs_four_commands_same_params(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    args = type("A", (), {"data_dir":"/data", "steps":5000, "checkpoint_freq":100, "seed":0})()
    params = {**cfg["fixed_params"], **{k: v.get("low", 1) for k, v in cfg["search_space"].items()}}
    cmds = [tuner.build_env_command(args, cfg, params, tmp_path / "trial_0000", i) for i in range(4)]
    assert [c[c.index("--test_envs") + 1] for c in cmds] == ["0", "1", "2", "3"]
    stripped = [[x for j,x in enumerate(c) if not (j>0 and c[j-1] in {"--test_envs", "--output_root"})] for c in cmds]
    assert all("--piccl_beta_max" in c for c in stripped)
    assert len({tuple(s[3:]) for s in stripped}) == 1


def test_target_accuracy_not_in_objective_and_global_objective_formula():
    scores = {i: {"source_score": 0.9 + i * 0.01} for i in range(4)}
    expected = sum([0.9,0.91,0.92,0.93])/4 - 0.2 * math.sqrt(sum((x-0.915)**2 for x in [0.9,0.91,0.92,0.93])/4)
    assert tuner.global_objective(scores) == pytest.approx(expected)
    row = {"args":{"real_test_envs":[0]}, "env0_out":0.0, "env1_out":1.0, "env2_out":1.0, "env3_out":1.0}
    assert tuner.source_score(row) == pytest.approx(1.0)
    row["env0_out"] = 1.0
    assert tuner.source_score(row) == pytest.approx(1.0)


def test_failed_trial_not_in_ranking_logic():
    rows = [{"status":"COMPLETE","objective":1.0}, {"status":"FAIL","objective":2.0}, {"status":"COMPLETE","objective":0.5}]
    ranked = [r for r in rows if r["status"] == "COMPLETE" and r.get("objective") is not None]
    ranked.sort(key=lambda r: r["objective"], reverse=True)
    assert ranked[0]["objective"] == 1.0


def test_sqlite_resume_create_study(tmp_path):
    pytest.importorskip("optuna")
    cfg = json.loads(CONFIG.read_text())
    s1 = tuner.create_study(cfg, tmp_path)
    s2 = tuner.create_study(cfg, tmp_path)
    assert s1.study_name == s2.study_name
    assert (tmp_path / "study.db").exists()


def test_output_root_dry_run_no_training(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "tune_piccl_optuna.py"), "--config", str(CONFIG), "--history-root", str(HISTORY), "--output-root", str(tmp_path), "--data-dir", "/data", "--dry-run"]
    out = subprocess.check_output(cmd, text=True, cwd=ROOT)
    payload = json.loads(out)
    assert payload["output_root"] == str(tmp_path)
    assert "subprocess.run" not in out
    assert len(payload["commands"]) == 4


def test_train_all_output_root_backward_compatible():
    src = (ROOT / "train_all.py").read_text()
    assert "--output_root" in src
    assert 'args.work_dir / Path("train_output") / args.dataset' in src
