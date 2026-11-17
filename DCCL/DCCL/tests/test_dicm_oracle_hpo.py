import copy
import math

import pytest
import torch

from experiment_archive import history_row
from tools.dicm_oracle_hpo.search import FIXED, config_for, fingerprint, fold_result, objective_value


def make_fold(target=0, converged=True):
    score = 0.71 + target / 100
    domain = ("CALTECH", "LABELME", "PASCAL", "SUN")[target]
    hparams = config_for({
        "lr": 0.00005, "dicm_basis_lr": 0.0001, "l": 0.5, "dicm_isr_weight": 0.05,
        "dicm_reliable_contrast_gamma_max": 0.5,
        "weight_decay": 0.0001, "lr_schedule": "fixed", "optimizer": "adam",
    })
    fold = {
        "target": [domain], "args": {"stage": None, "stage_parent": None},
        "hparams": hparams, "selection": {"iid": 0.8, "SWAD": score},
        "iid_checkpoint": {"train_out": 0.89, "step": 199},
        "swad_evaluation": {"converged": converged, "dead_valley": converged,
                            "converge_step": 99 if converged else None,
                            "first_dead_update": 700 if converged else None,
                            "stop_update": 700, "start": 100, "end": 499,
                            "count": 4, "metrics": {"test_in": score, "train_out": 0.88}},
    }
    manifest = {"dataset": "VLCS", "code": {"commit": "abc"},
                "evaluation_protocol": "oracle-guided HPO", "run_id": "f" * 32}
    return fold, manifest


def test_objective_uses_all_four_final_swad_values_and_keeps_iid():
    records = []
    for i in range(4):
        fold, manifest = make_fold(i)
        records.append(fold_result("VLCS", i, fold, manifest, "abc"))
    value, reason = objective_value(records)
    assert value == pytest.approx((0.71 + 0.72 + 0.73 + 0.74) / 4)
    assert reason is None
    assert [r["iid"] for r in records] == [0.8] * 4
    assert records[0]["best_source_validation"] == 0.89


def test_fallback_makes_trial_invalid_without_fabricated_value():
    records = []
    for i in range(4):
        fold, manifest = make_fold(i, converged=(i != 1))
        records.append(fold_result("VLCS", i, fold, manifest, "abc"))
    value, reason = objective_value(records)
    assert value is None and "LABELME" in reason
    assert records[1]["swad"] == 0.72 and records[1]["fallback"] is True
    with pytest.raises(ValueError, match="exactly four"):
        objective_value(records[:3])


def test_mappings_enforce_target_source_and_consistent_swad():
    fold, manifest = make_fold()
    altered = copy.deepcopy(fold)
    altered["swad_evaluation"]["metrics"]["test_in"] = 0.9
    with pytest.raises(ValueError, match="disagree"):
        fold_result("VLCS", 0, altered, manifest, "abc")
    with pytest.raises(ValueError, match="target"):
        fold_result("VLCS", 1, fold, manifest, "abc")
    altered = copy.deepcopy(fold)
    altered["args"]["stage"] = 1
    with pytest.raises(ValueError, match="stagewise"):
        fold_result("VLCS", 0, altered, manifest, "abc")


def test_fallback_is_blank_in_history_and_dedup_is_stable():
    fold, manifest = make_fold(converged=False)
    manifest.update({"completed_utc": "", "algorithm": "DICM", "seed": 0,
                     "fingerprint": "", "code": {"commit": "abc", "branch": "x"}})
    fold["iid_checkpoint"].update(test_in=0.8, test_out=0.8)
    assert history_row(manifest, fold, "")["SWAD"] == ""
    assert fingerprint("PACS", {"lr": 0.1, "l": 0.5}) == fingerprint(
        "PACS", {"l": 0.5, "lr": 0.1})
    assert fingerprint("PACS", {"lr": 0.1}) != fingerprint("VLCS", {"lr": 0.1})


def test_cosine_scheduler_keeps_parameter_group_ratios():
    backbone = torch.nn.Parameter(torch.tensor(1.0))
    basis = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([{"params": [backbone], "lr": 0.01},
                           {"params": [basis], "lr": 0.03}], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    for step in range(4):
        opt.zero_grad()
        (backbone ** 2 + basis ** 2).backward()
        opt.step()
        scheduler.step()
        if step < 3:
            assert opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] == pytest.approx(3)
    assert opt.param_groups[0]["lr"] == pytest.approx(0)
