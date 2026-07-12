import json
from pathlib import Path

from scripts.tune_piccl import best_and_final, infer_target_envs, source_objective


def test_source_objective_excludes_target_domain(tmp_path):
    row = {"args": {"real_test_envs": [0]}, "env0_out": 0.1, "env1_out": 0.8, "env2_out": 0.6, "test_out": 0.1, "step": 1}
    assert infer_target_envs(row) == [0]
    assert source_objective(row) > 0.65


def test_results_parser_identifies_best_and_final_by_source_objective(tmp_path):
    p = tmp_path / "results.jsonl"
    rows = [
        {"args": {"real_test_envs": [0]}, "env0_out": 0.99, "env1_out": 0.4, "env2_out": 0.4, "step": 0},
        {"args": {"real_test_envs": [0]}, "env0_out": 0.10, "env1_out": 0.8, "env2_out": 0.8, "step": 100},
        {"args": {"real_test_envs": [0]}, "env0_out": 0.20, "env1_out": 0.7, "env2_out": 0.7, "step": 200},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    best, final = best_and_final(p)
    assert best["step"] == 100
    assert final["step"] == 200
