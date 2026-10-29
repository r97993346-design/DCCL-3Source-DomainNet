"""CPU functional tests; no dataset, CLIP weights, or torchvision required."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "CIPT"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from cipt.engine import cosine_scheduler, make_optimizer, train_one_epoch
from cipt_iid_swad import (
    DeviceLossValley, IIDSelector, SWADTracker, TrainableWeights,
    evaluate_sources, load_trainable_state, preserve_rng, trainable_state,
    write_summary,
)
from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley
from test_cipt_tda_consistent import ToyCIPT
import train_cipt_tda_iid_swad_domainbed as runner

torch.set_num_threads(1)


class ScoresModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))

    def predict(self, x, use_tda=True):
        if not use_tda:
            raise AssertionError("Selections must use TDA inference")
        return x + self.bias


class RandomScores(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        random.random()
        np.random.random()
        torch.rand(1)
        return torch.tensor([3.0, 0.0]), torch.tensor(0)


class SelectionTests(unittest.TestCase):
    def test_source_domains_have_equal_weight_despite_different_sizes(self):
        model = ScoresModel().train()
        sources = {
            "large": DataLoader(TensorDataset(torch.tensor([[10.0, 0.0]] * 9), torch.zeros(9, dtype=torch.long))),
            "small": DataLoader(TensorDataset(torch.tensor([[0.0, 10.0]]), torch.zeros(1, dtype=torch.long))),
        }
        metrics = evaluate_sources(model, sources, "cpu")
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["pooled_accuracy"], 0.9)
        self.assertAlmostEqual(metrics["loss"], 5.0000456, places=5)
        self.assertTrue(model.training)

    def test_source_evaluation_preserves_all_rng_streams_and_mode(self):
        model = ScoresModel().eval()
        runner.seed_everything(42)
        expected = (random.random(), np.random.random(), torch.rand(1))
        runner.seed_everything(42)
        evaluate_sources(model, {"source": DataLoader(RandomScores())}, "cpu")
        actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        self.assertFalse(model.training)

    def test_iid_keeps_first_source_best_and_copies_its_weights(self):
        model, selector = ScoresModel(), IIDSelector()
        for step, (value, accuracy) in enumerate([(1, 0.8), (2, 0.9), (3, 0.9), (4, 0.7)], 1):
            with torch.no_grad():
                model.bias.fill_(value)
            selector.consider(model, {"accuracy": accuracy, "loss": 0.3, "per_domain": {}}, step, step)
        self.assertEqual(selector.metadata["step"], 2)
        load_trainable_state(model, selector.state)
        torch.testing.assert_close(model.bias, torch.full((2,), 2.0))

    def test_swad_snapshots_exclude_frozen_backbone_and_do_not_alias_live_weights(self):
        model = ScoresModel()
        model.backbone = nn.Linear(50, 50).requires_grad_(False)
        original = copy.deepcopy(model.backbone.state_dict())
        tracker = SWADTracker(model, "cpu")
        self.assertEqual(tracker.live.names, ("bias",))
        tracker.update(1)
        with torch.no_grad():
            model.bias.fill_(4)
        tracker.update(2)
        torch.testing.assert_close(tracker.segment.module.values[0], torch.full((2,), 2.0))
        for key, value in original.items():
            torch.testing.assert_close(model.backbone.state_dict()[key], value, atol=0, rtol=0)

    def test_dense_swad_matches_repository_lossvalley(self):
        for losses in ([0.7, 0.4, 0.4, 0.4, 0.5, 0.5] + [1.5] * 9, [0.7, 0.6, 0.5]):
            with self.subTest(losses=losses), redirect_stdout(io.StringIO()):
                model = ScoresModel()
                tracker = SWADTracker(model, "cpu")
                reference = LossValley(SimpleNamespace(logger=mock.Mock()), 3, 6, 0.3)
                live = TrainableWeights(model)
                segment = AveragedModel(live)
                step = 0
                for loss in losses:
                    for _ in range(3):
                        step += 1
                        with torch.no_grad():
                            model.bias.fill_(step)
                        tracker.update(step)
                        if not reference.dead_valley:
                            segment.update_parameters(live, step=step)
                    tracker.checkpoint({"accuracy": 0.5, "loss": loss})
                    if not reference.dead_valley:
                        reference.update_and_evaluate(segment, 0.5, loss, lambda *_: None)
                        if not reference.dead_valley:
                            segment = AveragedModel(live)
                actual, metadata = tracker.finalize()
                with mock.patch.object(nn.Module, "cuda", lambda self, *a, **k: self):
                    expected = reference.get_final_model()
                torch.testing.assert_close(actual["bias"], expected.module.values[0], atol=0, rtol=0)
                self.assertEqual(metadata["start_step"], expected.start_step)
                self.assertEqual(metadata["end_step"], expected.end_step)
                self.assertEqual(metadata["dead_valley"], reference.dead_valley)
                self.assertEqual(metadata["status"], "loss_valley" if reference.is_converged else "not_converged_last_segment")

    def test_dead_valley_stops_averaging_while_base_model_can_continue(self):
        model = ScoresModel()
        tracker = SWADTracker(model, "cpu", 2, 2, 0.3)
        with redirect_stdout(io.StringIO()):
            for step, loss in enumerate([1, 1, 1, 4, 4, 4], 1):
                with torch.no_grad():
                    model.bias.fill_(step)
                tracker.update(step)
                tracker.checkpoint({"accuracy": 0.5, "loss": loss})
            self.assertTrue(tracker.valley.dead_valley)
            end = tracker.valley.final_model.end_step
            for step in range(7, 11):
                with torch.no_grad():
                    model.bias.fill_(step)
                tracker.update(step)
            _, metadata = tracker.finalize()
        self.assertEqual(metadata["end_step"], end)
        self.assertEqual(float(model.bias[0].detach()), 10.0)


class GuardedTarget(Dataset):
    def __init__(self, split, directory):
        self.split, self.directory = split, directory
        self.access_count = 0

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        if not (self.directory / "final_trainable.pth").is_file():
            raise AssertionError("Target accessed before final training checkpoint")
        if not (self.directory / "iid_best_trainable.pth").is_file():
            raise AssertionError("Target accessed before source-only IID selection")
        self.access_count += 1
        return self.split[index]


class RunnerTests(unittest.TestCase):
    def mini_run(self, directory, no_swad=False, epoch_validation=False):
        argv = ["--data-dir", "/unused", "--epochs", "5", "--batch-size", "4",
                "--eval-batch-size", "4", "--num-workers", "0", "--eval-num-workers", "0",
                "--output-root", str(directory), "--checkpoint-freq", "0" if epoch_validation else "2",
                "--eval-every", "2" if epoch_validation else "1"]
        if no_swad:
            argv.append("--no-swad")
        args = runner.parse_args(argv)
        output = Path(directory) / "PACS/target_0_A/seed_0_trial_0"
        generator = torch.Generator().manual_seed(12)
        images = torch.randn(12, 5, generator=generator)
        labels = torch.arange(12) % 3
        train = TensorDataset(images, labels)
        val = ConcatDataset([TensorDataset(images[i:i+2], labels[i:i+2]) for i in (0, 4, 8)])
        target_in = GuardedTarget(TensorDataset(images[:4], labels[:4]), output)
        target_out = GuardedTarget(TensorDataset(images[4:6], labels[4:6]), output)
        dataset = SimpleNamespace(environments=["A", "B", "C", "D"])
        sizes = {"source_train": {"B": 4, "C": 4, "D": 4},
                 "source_val": {"B": 2, "C": 2, "D": 2}, "target_in": 4, "target_out": 2}
        def build(*args, **kwargs):
            model = ToyCIPT()
            model.sample_templates = True
            model.clip_model.logit_scale.requires_grad_(False)
            return model, None
        reference = SimpleNamespace(
            UPSTREAM_COMMIT="test", make_splits=lambda *_: (dataset, ["a", "b", "c"], train, val, target_in, target_out, sizes),
            loader=lambda data, batch_size, workers, shuffle=False: DataLoader(data, batch_size=batch_size, shuffle=shuffle),
        )
        runner.seed_everything(8)
        with redirect_stdout(io.StringIO()):
            result = runner.run_target(args, 0, reference, (build, cosine_scheduler, None, make_optimizer, train_one_epoch))
        self.assertGreater(target_in.access_count, 0)
        self.assertGreater(target_out.access_count, 0)
        return result, output, args

    def test_end_to_end_iid_swad_checkpoint_reload_and_rng_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            with_swad, first, args = self.mini_run(Path(directory) / "swad")
            without_swad, second, _ = self.mini_run(Path(directory) / "iid", no_swad=True)
            a = torch.load(first / "final_trainable.pth", weights_only=True)
            b = torch.load(second / "final_trainable.pth", weights_only=True)
            self.assertEqual(a["metadata"]["step"], 15)
            self.assertEqual(a["trainable_state_dict"].keys(), b["trainable_state_dict"].keys())
            for name in a["trainable_state_dict"]:
                torch.testing.assert_close(a["trainable_state_dict"][name], b["trainable_state_dict"][name], atol=0, rtol=0)
            self.assertEqual(with_swad["selections"]["iid"], without_swad["selections"]["iid"])
            for filename in ("iid_best_trainable.pth", "final_trainable.pth", "swad_trainable.pth"):
                saved = torch.load(first / filename, weights_only=True)
                clone = ToyCIPT()
                clone.clip_model.logit_scale.requires_grad_(False)
                load_trainable_state(clone, saved["trainable_state_dict"])
            history = json.loads((first / "history.json").read_text())
            self.assertEqual([r["step"] for r in history], [2, 4, 6, 8, 10, 12, 14, 15])
            with redirect_stdout(io.StringIO()):
                summary = runner.summarize_saved(args)
            self.assertEqual(summary["tables"]["target_out"]["columns"], ["Selection", "A", "Avg."])
            self.assertEqual([r[0] for r in summary["tables"]["target_out"]["rows"]], ["iid", "last", "SWAD"])

    def test_epoch_validation_includes_final_even_if_not_multiple_of_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, _ = self.mini_run(directory, epoch_validation=True)
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([r["step"] for r in history], [6, 12, 15])

    def test_summary_keeps_target_in_out_distinct_and_averages_target_domains(self):
        results = []
        for i, (out, in_) in enumerate([(0.2, 0.9), (0.6, 0.7)]):
            results.append({"test_env": i, "target": f"domain{i}", "selections": {
                name: {"target_out": out, "target_in": in_, "source_val": 0.95}
                for name in ("iid", "last", "SWAD")}})
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summary = write_summary(directory, "PACS", "consistent", results, 0, 0)
            self.assertAlmostEqual(summary["tables"]["target_out"]["rows"][0][-1], 0.4)
            self.assertAlmostEqual(summary["tables"]["target_in"]["rows"][0][-1], 0.8)
            csv_path = next(Path(directory).glob("*_target_out.csv"))
            with csv_path.open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["Selection", "domain0", "domain1", "Avg."])
            self.assertEqual(rows[1][-1], "40.000000%")


if __name__ == "__main__":
    unittest.main()
