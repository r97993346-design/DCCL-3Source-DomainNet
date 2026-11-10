"""CLI and real-loader regressions; no images, downloads, GPU or real training."""

import random
import re
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from domainbed.evaluator import Evaluator
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.reproducibility import preserve_rng_state, seed_everything


class SyntheticDataset(Dataset):
    def __init__(self, randomized=False):
        self.randomized = randomized

    def __len__(self):
        return 6

    def __getitem__(self, index):
        x = torch.tensor([float(index)])
        if self.randomized:
            x = x + random.random() + np.random.rand() + torch.rand(1)
        return {"x": x, "y": index % 2, "id": index}


class ProbeModel(nn.Module):
    def predict(self, x):
        return torch.cat((x, -x), dim=1)


def make_evaluator(isolate=False):
    kwargs = {"dataset": SyntheticDataset(randomized=True), "batch_size": 2, "num_workers": 0}
    meta = [(name, kwargs, None) for name in ("env0_out", "env1_in", "env1_out")]
    return Evaluator([1], meta, 2, None, isolate_rng=isolate)


@pytest.mark.parametrize("workers", [0, 2])
def test_fast_loader_honors_requested_workers_and_covers_dataset(workers):
    if workers:
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except PermissionError:
            pytest.skip("Host disallows IPC sockets; worker-count forwarding is tested separately")
        else:
            probe.close()
    loader = FastDataLoader(SyntheticDataset(), batch_size=2, num_workers=workers)
    try:
        assert loader._infinite_iterator._num_workers == workers
        assert torch.cat([batch["id"] for batch in loader]).tolist() == list(range(6))
    finally:
        if workers:
            loader._infinite_iterator._shutdown_workers()


def test_fast_loader_forwards_worker_count_without_requiring_ipc(monkeypatch):
    original = torch.utils.data.DataLoader
    forwarded = []

    def cpu_loader(*args, **kwargs):
        forwarded.append(kwargs["num_workers"])
        kwargs["num_workers"] = 0
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", cpu_loader)
    loader = FastDataLoader(SyntheticDataset(), batch_size=2, num_workers=2)
    assert forwarded == [2]
    assert torch.cat([batch["id"] for batch in loader]).tolist() == list(range(6))


@pytest.mark.parametrize("fails", [False, True])
def test_rng_context_restores_all_cpu_sources_even_on_exception(fails):
    seed_everything(29)
    expected = random.random(), np.random.rand(), torch.rand(4)
    seed_everything(29)
    try:
        with preserve_rng_state():
            random.random()
            np.random.rand(10)
            torch.rand(10)
            if fails:
                raise RuntimeError("probe")
    except RuntimeError:
        assert fails
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], atol=0, rtol=0)


def test_rng_context_restores_initialized_cuda_states_without_allocating_gpu(monkeypatch):
    states = [torch.tensor([1, 2], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    with preserve_rng_state():
        pass
    assert len(restored) == 1 and restored[0] is states


def test_real_evaluation_restores_rng_and_check_frequency_keeps_training_batches():
    def stream(freq, isolate):
        seed_everything(29)
        training = iter(InfiniteDataLoader(SyntheticDataset(), None, batch_size=2, num_workers=0))
        evaluator, model = make_evaluator(isolate), ProbeModel()
        batches = []
        for step in range(12):
            batches.append(next(training)["id"].tolist())
            if step % freq == 0:
                evaluator.evaluate(model)
        return batches

    assert stream(2, True) == stream(5, True)
    # The default retains the original shared-RNG behavior, explicitly tested.
    assert stream(2, False) != stream(5, False)
    seed_everything(29)
    expected = random.random(), np.random.rand(), torch.rand(4)
    seed_everything(29)
    make_evaluator(True).evaluate(ProbeModel())
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], atol=0, rtol=0)


def show_config(monkeypatch, capsys, algorithm, extra=()):
    import train_all
    config = "configs/dccl_pacs_a0.yaml" if algorithm == "DCCL" else "configs/dicm_pacs.yaml"
    monkeypatch.setattr("sys.argv", ["train_all.py", "review", config, "--algorithm", algorithm, "--show", *extra])
    monkeypatch.setattr(train_all, "get_dataset", lambda *a, **k: pytest.fail("review loaded data"))
    monkeypatch.setattr(train_all, "get_writer", lambda *a, **k: pytest.fail("review created outputs"))
    train_all.main()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    values = dict(re.findall(r"^(l|l_d|l_layer|t|t_pre|n_layer): (.+)$", plain, re.M))
    return plain, values


@pytest.mark.parametrize("algorithm,mode", [("DCCL", None), ("DICM", "A4"), ("DICM", "A5")])
def test_shared_core_cli_overrides_match_for_a0_a4_a5(monkeypatch, capsys, algorithm, mode):
    extra = ["--l", "0.4", "--l_layer", "0.3", "--t=0.15", "--t_pre", "0.25", "--n_layer", "2"]
    if mode:
        extra += ["--dicm_ablation", mode]
    _, values = show_config(monkeypatch, capsys, algorithm, extra)
    assert values["l"] == "0.4" and values["l_layer"] == "0.3"
    assert values["t"] == "0.15" and values["t_pre"] == "0.25" and values["n_layer"] == "2"


@pytest.mark.parametrize("algorithm,extra", [("DCCL", []), ("DICM", ["--dicm_ablation", "A5"])])
def test_explicit_gt_override_is_effective_on_both_gt_controls(monkeypatch, capsys, algorithm, extra):
    _, values = show_config(monkeypatch, capsys, algorithm, [*extra, "--l_d=0.03"])
    assert values["l_d"] == "0.03"


@pytest.mark.parametrize("extra,updates", [([], 50), (["--step_count_mode", "legacy"], 1)])
def test_show_reports_budget_rng_and_workers_without_data(monkeypatch, capsys, extra, updates):
    plain, _ = show_config(monkeypatch, capsys, "DCCL", ["--steps", "50", "--checkpoint_freq", "100",
                                                      "--num_workers", "0", "--isolate_eval_rng", *extra])
    assert f"max_updates={updates}; checkpoint_freq=100" in plain
    assert "Evaluation RNG: isolated; num_workers=0" in plain


@pytest.fixture
def main_entry(monkeypatch):
    import train_all

    class Metadata:
        environments = ["A", "C", "P", "S"]
        num_classes = 7

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return [0] * 4

    calls, tables = [], []

    class Sink:
        def nofmt(self, value):
            if hasattr(value, "field_names"):
                tables.append((value.field_names, value.rows))

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def fake_train(target, **kwargs):
        calls.append((target[0], kwargs["n_steps"], kwargs["checkpoint_freq"], kwargs["args"].tb_freq))
        return {"iid": 0.1 * (target[0] + 1)}, []

    monkeypatch.setattr(train_all.Path, "mkdir", lambda *a, **k: None)
    monkeypatch.setattr(train_all, "get_writer", lambda *a, **k: Sink())
    monkeypatch.setattr(train_all.Logger, "get", lambda *a, **k: Sink())
    monkeypatch.setattr(train_all, "get_dataset", lambda *a, **k: (Metadata(), [], []))
    monkeypatch.setattr(train_all, "train", fake_train)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def run(extra=()):
        monkeypatch.setattr("sys.argv", ["train_all.py", "review", "configs/dccl_pacs_a0.yaml",
                                        "--algorithm", "DCCL", *extra])
        train_all.main()
        return SimpleNamespace(calls=calls, tables=tables)

    return run


@pytest.mark.parametrize("targets", [[], [3], [3, 0], [3, 2, 1, 0]])
def test_main_summary_uses_actual_targets_and_order(main_entry, targets):
    extra = ["--test_envs", *map(str, targets)] if targets else []
    run = main_entry(extra)
    targets = targets or [0, 1, 2, 3]
    headers, rows = run.tables[-1]
    assert headers == ["Selection", *[["A", "C", "P", "S"][t] for t in targets], "Avg."]
    assert [row[0] for row in run.calls] == targets
    assert rows[0][1:-1] == [f"{0.1 * (t + 1):.3%}" for t in targets]


@pytest.mark.parametrize("steps,freq,mode,expected", [
    (50, 100, "exact", 50), (300, 100, "exact", 300), (350, 100, "exact", 350),
    (350, 10, "exact", 350), (50, 100, "legacy", 1), (300, 100, "legacy", 301),
])
def test_main_passes_correct_budget_to_each_target(main_entry, steps, freq, mode, expected):
    run = main_entry(["--steps", str(steps), "--checkpoint_freq", str(freq), "--step_count_mode", mode,
                      "--test_envs", "3", "--tb_freq", "2"])
    assert run.calls == [(3, expected, freq, 2)]


@pytest.mark.parametrize("extra", [["--steps", "0"], ["--steps", "-1"], ["--checkpoint_freq", "0"],
                                    ["--test_envs", "3", "3"], ["--test_envs", "4"]])
def test_invalid_budgets_and_target_ids_fail_before_training(main_entry, extra):
    with pytest.raises(ValueError):
        main_entry(extra)
