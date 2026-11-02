"""Independent-target RNG regression using the real trainer and data loaders.

Only the image datasets and algorithm are replaced by tiny CPU fixtures. The
actual train() body, sampling, evaluation and constant-LR update loop execute.
No CLIP weights, images, CUDA or long-running experiment is required.
"""
import ast
import collections
import copy
import json
import random
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib.cipt_lr_schedule import make_cipt_lr_scheduler
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.query import Q


ROOT = Path(__file__).resolve().parents[1]


class HParams(dict):
    __getattr__ = dict.__getitem__


class NullLogger:
    def info(self, *args, **kwargs):
        pass


class NullWriter:
    def add_scalars_with_prefix(self, *args, **kwargs):
        pass


class TinyDataset(torch.utils.data.Dataset):
    classes = ["class0", "class1"]

    def __init__(self, env, indices):
        self.indices = indices
        self.env = env

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        value = self.indices[index]
        return {"x": torch.tensor([value / 12., self.env / 4.]), "y": value % 2}


class TinyEnvironments:
    environments = ["art", "cartoon", "photo", "sketch"]
    input_shape = (2,)
    num_classes = 2
    N_WORKERS = 0

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)


class TinyAlgorithm(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.linear = nn.Linear(input_shape[0], num_classes)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=hparams["lr"])
        self.initial = self.linear.weight.detach().clone()
        self.history = []

    def predict(self, x):
        return self.linear(x)

    def update(self, x, y, **kwargs):
        # Exercise all three RNGs, including consumption by prior tasks.
        draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
        images = torch.cat(x)
        loss = F.cross_entropy(self.predict(images), torch.cat(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.history.append((images.clone(), draws, self.linear.weight.detach().clone()))
        return {"total_loss": loss.item()}


class TargetSeedIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.models = []
        self.splits = []
        definition = next(
            n for n in ast.parse((ROOT / "domainbed/trainer.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        )

        def algorithm_class(_):
            def construct(*args, **kwargs):
                model = TinyAlgorithm(*args, **kwargs)
                self.models.append(model)
                return model
            return construct

        def get_dataset(test_envs, args, hparams, algorithm):
            ins, outs, signatures = [], [], []
            for env in range(4):
                # Match production's trial/env-local split seed semantics.
                rng = np.random.RandomState(misc.seed_hash(args.trial_seed, env))
                indices = rng.permutation(12).tolist()
                signatures.append(indices)
                ins.append((TinyDataset(env, indices[4:]), None))
                outs.append((TinyDataset(env, indices[:4]), None))
            self.splits.append(signatures)
            return TinyEnvironments([d for d, _ in ins]), ins, outs

        namespace = dict(
            collections=collections, copy=copy, json=json, random=random, time=time,
            np=np, torch=torch, misc=misc, device="cpu", Q=Q,
            algorithms=SimpleNamespace(get_algorithm_class=algorithm_class),
            get_dataset=get_dataset, Evaluator=Evaluator,
            InfiniteDataLoader=InfiniteDataLoader, FastDataLoader=FastDataLoader,
            make_cipt_lr_scheduler=make_cipt_lr_scheduler,
            resolve_loader_workers=lambda args, dataset: (0, 0),
            json_handler=lambda value: str(value),
        )
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "trainer.py", "exec"), namespace)
        self.train = namespace["train"]

    def run_target(self, env, steps, seed=0, trial_seed=0):
        args = SimpleNamespace(
            dataset="PACS", algorithm="CIPTDCCL", seed=seed, trial_seed=trial_seed,
            source_envs=None, target_env=None, prebuild_loader=False,
            evalmode="fast", debug=False, log=False, model_save=None,
            tb_freq=100, out_dir=Path(self.temporary.name),
        )
        hparams = HParams(indomain_test=False, batch_size=2, test_batchsize=4,
                         swad=None, lr=.01, cipt_lr_schedule="constant")
        self.train([env], args, hparams, steps, 2, NullLogger(), NullWriter())
        return self.models[-1]

    def assert_same_trajectory(self, first, second, length=None):
        torch.testing.assert_close(first.initial, second.initial, rtol=0, atol=0)
        length = len(first.history) if length is None else length
        for a, b in zip(first.history[:length], second.history[:length]):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            self.assertEqual(a[1], b[1])
            torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)

    def test_previous_target_budget_cannot_change_next_target(self):
        self.run_target(0, 1)
        after_short = self.run_target(3, 3)
        self.run_target(0, 7)
        after_long = self.run_target(3, 3)
        self.assert_same_trajectory(after_short, after_long)

    def test_target_alone_matches_other_target_orders(self):
        alone = self.run_target(3, 3)
        self.run_target(2, 3)
        self.run_target(0, 1)
        after_other_order = self.run_target(3, 3)
        self.assert_same_trajectory(alone, after_other_order)

    def test_constant_lr_budgets_share_the_same_prefix(self):
        short = self.run_target(3, 3)
        long = self.run_target(3, 7)
        self.assertEqual(len(short.history), 3)
        self.assertEqual(len(long.history), 7)
        self.assert_same_trajectory(short, long, 3)

    def test_training_seed_does_not_change_trial_splits(self):
        first = self.run_target(3, 1, seed=1, trial_seed=5)
        split_first = self.splits[-1]
        second = self.run_target(3, 1, seed=2, trial_seed=5)
        self.assertEqual(split_first, self.splits[-1])
        self.assertFalse(torch.equal(first.initial, second.initial))
        self.run_target(3, 1, seed=1, trial_seed=6)
        self.assertNotEqual(split_first, self.splits[-1])


if __name__ == "__main__":
    unittest.main()
