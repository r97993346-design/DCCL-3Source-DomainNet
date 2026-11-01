"""CPU coverage for pooled epochs, domain identity, LR boundaries and selection.

No images, CLIP checkpoint, CUDA or real dataset training are required.
Run: python -m unittest discover -s tests -p 'test_cipt_mixed_training.py'
"""

import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
import random
import socket
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from domainbed.lib.cipt_training import (
    MixedSourceDataLoader, group_mixed_batch_by_domain, repeat_epochs,
    validate_cipt_training_options,
)
from domainbed.lib.cipt_lr_schedule import make_cipt_lr_scheduler
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.query import Q


ROOT = Path(__file__).resolve().parents[1]


class ToySource(torch.utils.data.Dataset):
    classes = ["dog", "horse", "guitar"]

    def __init__(self, domain, size, forbidden=False):
        self.domain, self.size, self.forbidden = domain, size, forbidden

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.forbidden:
            raise AssertionError("target images must not enter the training loader")
        sample_id = 100 * self.domain + index
        return {
            "x": torch.arange(8).float() / 8 + index / 10,
            "y": index % 3, "d": self.domain, "sample_id": sample_id,
        }


def splits(forbid_target=True):
    return [(ToySource(0, 7), None),
            (ToySource(1, 4, forbidden=forbid_target), None),
            (ToySource(2, 6), None)]


def sampled_ids(batches):
    return [int(value) for groups in batches for group in groups for value in group["sample_id"]]


def selected_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


class MixedLoaderTests(unittest.TestCase):
    def test_each_epoch_covers_only_source_in_splits_once_including_tail(self):
        loader = MixedSourceDataLoader(splits(), [1], 6, 0, seed=17)
        self.assertEqual(len(loader), 3)
        expected = list(range(7)) + list(range(200, 206))
        orders = []
        for _ in range(3):
            batches = list(loader)
            ids = sampled_ids(batches)
            self.assertEqual(sorted(ids), expected)
            self.assertEqual(len(set(ids)), 13)
            self.assertEqual([sum(len(g["y"]) for g in b) for b in batches], [6, 6, 1])
            for groups in batches:
                for group in groups:
                    self.assertEqual(len(torch.unique(group["d"])), 1)
                    self.assertTrue(torch.equal(group["d"], group["sample_id"] // 100))
            orders.append(ids)
        self.assertNotEqual(orders[0], orders[1])

    def test_evaluation_rng_does_not_change_later_epoch_permutation(self):
        a = MixedSourceDataLoader(splits(), [1], 5, 0, seed=8)
        b = MixedSourceDataLoader(splits(), [1], 5, 0, seed=8)
        for _ in range(3):
            expected = sampled_ids(list(a))
            torch.rand(123)  # e.g. evaluator constructs loaders between epochs
            self.assertEqual(sampled_ids(list(b)), expected)

    def test_worker_loader_keeps_complete_epoch(self):
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except PermissionError:
            self.skipTest("local worker IPC sockets are disabled in this environment")
        else:
            probe.close()
        loader = MixedSourceDataLoader(splits(), [1], 6, 1, seed=17)
        self.assertEqual(sorted(sampled_ids(list(loader))), list(range(7)) + list(range(200, 206)))

    def test_grouping_preserves_cross_domain_relations_with_missing_domains(self):
        batch = {"d": torch.tensor([2, 2, 0]), "y": torch.tensor([1, 0, 1]),
                 "x": torch.arange(24).view(3, 8)}
        groups = group_mixed_batch_by_domain(batch, (0, 2, 3))
        actual = torch.cat([g["d"] for g in groups])
        inferred = torch.cat([torch.full_like(g["y"], i) for i, g in enumerate(groups)])
        self.assertTrue(torch.equal(actual[:, None].eq(actual), inferred[:, None].eq(inferred)))
        self.assertEqual(sum(len(g["y"]) for g in groups), 3)
        with self.assertRaises(ValueError):
            group_mixed_batch_by_domain(batch, (0,))

    def test_invalid_training_options_fail_instead_of_silently_changing_budget(self):
        base = {"cipt_train_sampling": "mixed", "cipt_train_epochs": 30,
                "cipt_lr_step_unit": "epoch"}
        self.assertEqual(validate_cipt_training_options({}), ("domain_balanced", 0, "step"))
        self.assertEqual(validate_cipt_training_options(base), ("mixed", 30, "epoch"))
        for changes in ({"cipt_train_epochs": 0}, {"cipt_train_epochs": -1},
                        {"cipt_train_epochs": 1.5}, {"cipt_train_epochs": True},
                        {"cipt_train_sampling": "bad"}, {"cipt_lr_step_unit": "bad"},
                        {"class_balanced": True}, {"cipt_train_sampling": "domain_balanced"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_cipt_training_options(dict(base, **changes))
        with self.assertRaises(ValueError):
            validate_cipt_training_options(base, requested_steps=3000)
        for bad_splits, batch_size in ((splits(), 0), (splits(), True), ([], 6),
                                       ([(ToySource(0, 0), None)], 6),
                                       ([(ToySource(0, 2), torch.ones(2))], 6)):
            with self.assertRaises(ValueError):
                MixedSourceDataLoader(bad_splits, [], batch_size, 0, seed=0)

    def test_training_overlay_and_explicit_epoch_cli_preserve_method_choices(self):
        method = yaml.safe_load((ROOT / "config_cipt_b5b_entropy_rd.yaml").read_text())
        overlay = yaml.safe_load((ROOT / "config_cipt_mixed_epoch30.yaml").read_text())
        params = dict(method, **overlay)
        for key in method.keys() - overlay.keys():
            self.assertEqual(params[key], method[key])
        self.assertEqual(params["lr"], .0025)
        self.assertEqual(params["batch_size"], 64)
        self.assertEqual(params["optimizer"], "adam")
        namespace = {}
        selected_definitions(ROOT / "train_all.py", {"apply_explicit_hparam_cli"}, namespace)
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", "--cipt_train_epochs", dest="cipt_train_epochs", type=int)
        args = parser.parse_args(["--epochs", "12"])
        namespace["apply_explicit_hparam_cli"](parser, args, params, ["--epochs", "12"])
        self.assertEqual(params["cipt_train_epochs"], 12)

    def test_real_entrypoint_parser_and_sconf_apply_cli_above_both_yamls(self):
        from sconf import Config
        from domainbed import hparams_registry
        from domainbed.lib.cl_hparams import setup_alg_hparams
        configs = []
        def capture_config(*args, **kwargs):
            config = Config(*args, **kwargs)
            configs.append(config)
            return config
        logger = SimpleNamespace(info=lambda *a: None, nofmt=lambda *a: None)
        namespace = dict(
            argparse=argparse, sys=sys, Path=Path, np=np, torch=torch,
            ExitStack=contextlib.ExitStack,
            PIL=SimpleNamespace(__version__="test"),
            torchvision=SimpleNamespace(__version__="test"),
            hparams_registry=hparams_registry, setup_alg_hparams=setup_alg_hparams,
            Config=capture_config, validate_cipt_training_options=validate_cipt_training_options,
            misc=SimpleNamespace(timestamp=lambda: "test"),
            get_writer=lambda *a: None, Logger=SimpleNamespace(get=lambda *a: logger),
        )
        selected_definitions(ROOT / "train_all.py",
                             {"main", "apply_explicit_hparam_cli", "apply_weak_erm_hparams"}, namespace)
        with tempfile.TemporaryDirectory() as directory:
            command = ["train_all.py", "inspect", "config_cipt_b5b_entropy_rd.yaml",
                       "config_cipt_mixed_epoch30.yaml", "--algorithm", "CIPTDCCL",
                       "--show", "--output_root", directory, "--epochs", "12",
                       "--batch_size", "32", "--lr", "1e-4"]
            with mock.patch.object(sys, "argv", command), self.assertRaises(SystemExit) as result:
                namespace["main"]()
            self.assertIn(result.exception.code, (None, 0))
            self.assertEqual(configs[-1]["cipt_train_epochs"], 12)
            self.assertEqual(configs[-1]["batch_size"], 32)
            self.assertEqual(configs[-1]["lr"], 1e-4)
            self.assertEqual(configs[-1]["cipt_lr_step_unit"], "epoch")
            self.assertEqual(configs[-1]["cipt_env_constraint"], "relative_distance")
            with mock.patch.object(sys, "argv", command + ["--steps", "3000"]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                namespace["main"]()
            self.assertEqual(result.exception.code, 2)


class ToyAlgorithm(torch.nn.Module):
    instances = []

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams["lr"])
        self.calls = []
        self.used_lrs = []
        self.instances.append(self)

    def update(self, x, y, step, **kwargs):
        self.calls.append([g.detach().clone() for g in kwargs["sample_id"]])
        self.used_lrs.append(self.optimizer.param_groups[0]["lr"])
        self.optimizer.zero_grad()
        loss = (self.weight - 2).square()
        loss.backward()
        self.optimizer.step()
        return {"total_loss": loss.item()}


class FakeEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, algorithm):
        updates = len(algorithm.calls)
        # Target performance improves throughout, but source validation peaks
        # at the first complete epoch. IID must follow source validation.
        return {}, {"test_in": updates / 1000, "test_out": updates / 2000,
                    "train_out": .9 if updates == 3 else .1, "tr_outloss": .3}


class HParams(dict):
    __getattr__ = dict.__getitem__


def run_trainer(out_dir, epoch_mode=True, test_env=1):
    misc_namespace = {"hashlib": hashlib}
    selected_definitions(ROOT / "domainbed/lib/misc.py",
                         {"seed_hash", "index_conditional_iterate", "SplitIterator", "merge_dictlist"},
                         misc_namespace)
    misc_namespace["to_row"] = str
    in_splits = splits(forbid_target=False)
    dataset = SimpleNamespace(
        environments=["art", "cartoon", "photo"], N_STEPS=5001,
        datasets=[item[0] for item in in_splits], input_shape=(8,), num_classes=3,
    )
    class Dataset:
        def __len__(self):
            return 3
    dataset_obj = Dataset()
    dataset_obj.__dict__.update(vars(dataset))
    params = HParams(indomain_test=0, val_augment=False, test_batchsize=16,
                     batch_size=6 if epoch_mode else 2, lr=.0025, optimizer="adam", swad=None,
                     freeze_bn=True, cipt_objective="env_anchor", cipt_lr_schedule="cosine",
                     cipt_lr_step_unit="epoch" if epoch_mode else "step",
                     cipt_train_sampling="mixed" if epoch_mode else "domain_balanced",
                     cipt_train_epochs=30 if epoch_mode else 0)
    args = SimpleNamespace(dataset="PACS", algorithm="CIPTDCCL", source_envs=None,
                           target_env=None, trial_seed=0, seed=0, num_workers=0,
                           eval_num_workers=0, prebuild_loader=False, evalmode="fast",
                           debug=False, log=False, model_save=None, tb_freq=1, out_dir=out_dir)
    logger = SimpleNamespace(info=lambda *a: None, debug=lambda *a: None, warning=lambda *a: None)
    writer = SimpleNamespace(add_scalars_with_prefix=lambda *a: None)
    namespace = dict(
        collections=collections, json=json, time=time, copy=copy, random=random,
        Path=Path, np=np, torch=torch, device="cpu", Q=Q,
        misc=SimpleNamespace(**misc_namespace),
        get_dataset=lambda *a: (dataset_obj, in_splits, in_splits),
        algorithms=SimpleNamespace(get_algorithm_class=lambda _: ToyAlgorithm),
        Evaluator=FakeEvaluator, InfiniteDataLoader=InfiniteDataLoader, FastDataLoader=FastDataLoader,
        MixedSourceDataLoader=MixedSourceDataLoader, repeat_epochs=repeat_epochs,
        validate_cipt_training_options=validate_cipt_training_options,
        make_cipt_lr_scheduler=make_cipt_lr_scheduler,
    )
    selected_definitions(ROOT / "domainbed/trainer.py",
                         {"train", "resolve_loader_workers", "json_handler"}, namespace)
    result, records = namespace["train"](
        [test_env], args, params, None if epoch_mode else 5,
        100 if epoch_mode else 2, logger, writer,
    )
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines()]
    return ToyAlgorithm.instances[-1], result, records, rows


class TrainerIntegrationTests(unittest.TestCase):
    def test_exact_30_epochs_epoch_cosine_and_source_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            model, result, records, rows = run_trainer(Path(directory))
        self.assertEqual(len(model.calls), 90)  # 30 * ceil(13 / 6), no +1 rounding
        expected = list(range(7)) + list(range(200, 206))
        for epoch in range(30):
            calls = model.calls[3 * epoch:3 * (epoch + 1)]
            self.assertEqual(sorted(int(x) for groups in calls for group in groups for x in group), expected)
            rate = .0025 * .5 * (1 + math.cos(math.pi * epoch / 30))
            for used in model.used_lrs[3 * epoch:3 * (epoch + 1)]:
                self.assertAlmostEqual(used, rate, places=12)
        self.assertAlmostEqual(model.optimizer.param_groups[0]["lr"], 0, places=12)
        self.assertEqual(len(records), 30)
        self.assertEqual(rows[-1]["updates"], 90)
        self.assertEqual(rows[-1]["epoch"], 30)
        self.assertEqual(rows[-1]["training_plan"]["total_source_train_samples"], 13)
        self.assertAlmostEqual(result["iid"], .003)
        self.assertAlmostEqual(result["last"], .09)

    def test_budget_is_resolved_again_for_a_different_target_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, _, rows = run_trainer(Path(directory), test_env=0)
        self.assertEqual(len(model.calls), 60)  # source sizes 4 + 6, ceil(10 / 6)
        self.assertEqual(rows[-1]["training_plan"]["source_train_samples"], {"1": 4, "2": 6})

    def test_legacy_step_mode_keeps_domain_batches_and_checkpoint_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, records, rows = run_trainer(Path(directory), epoch_mode=False)
        self.assertEqual(len(model.calls), 5)
        self.assertEqual([row["step"] for row in records], [0, 2, 4])
        self.assertTrue(all([len(group) for group in call] == [2, 2] for call in model.calls))
        self.assertNotIn("training_plan", rows[-1])

    def test_real_rd_mcc_and_cross_domain_losses_accept_pooled_partial_batches(self):
        import test_cipt_env_anchor as fixture
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for constraint in ("relative_distance", "mcc"):
                for contrastive in ("neighbor_retention", "clip_guided_cross_domain"):
                    with self.subTest(constraint=constraint, contrastive=contrastive):
                        model = fixture.build(
                            constraint, lr=.0025, cipt_contrastive_type=contrastive,
                            cipt_contrastive_warmup_steps=0,
                        )
                        before = model.causal_decomposition.state_dict()
                        before = {k: v.clone() for k, v in before.items()}
                        for groups in MixedSourceDataLoader(splits(), [1], 6, 0, seed=17):
                            values = {key: [group[key] for group in groups] for key in groups[0]}
                            metrics = model.update(**values)
                            self.assertTrue(math.isfinite(metrics["total_loss"]))
                        self.assertTrue(any(not torch.equal(before[k], v)
                                            for k, v in model.causal_decomposition.state_dict().items()))
                        self.assertTrue(all(parameter.grad is None for parameter in model.clip_model.parameters()))
        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
