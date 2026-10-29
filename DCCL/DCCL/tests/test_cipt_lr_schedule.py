"""Tests for optional DomainBed step-based Adam cosine learning-rate schedule.

Run from DCCL/: python -m unittest discover -s tests -p 'test_cipt_lr_schedule.py'
Requires only PyTorch; no images, CLIP or CUDA.
"""
import math
import unittest

import torch

from domainbed.lib.cipt_lr_schedule import make_cipt_lr_scheduler


class ToyAlgorithm:
    def __init__(self, lr=0.0025):
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.Adam([self.parameter], lr=lr)

    def update(self):
        self.optimizer.zero_grad()
        (self.parameter - 2).square().backward()
        self.optimizer.step()


class CIPTCosineSchedulerTests(unittest.TestCase):
    def test_constant_default_preserves_old_learning_rate(self):
        alg = ToyAlgorithm(5e-5)
        scheduler = make_cipt_lr_scheduler(
            alg, {"cipt_objective": "env_anchor"}, 7
        )
        self.assertIsNone(scheduler)
        for _ in range(7):
            alg.update()
        self.assertEqual(alg.optimizer.param_groups[0]["lr"], 5e-5)

    def test_cosine_anneals_after_each_optimizer_update(self):
        lr = 0.0025
        min_lr = 0.0
        total = 8
        alg = ToyAlgorithm(lr)
        scheduler = make_cipt_lr_scheduler(
            alg, {"cipt_objective": "env_anchor",
                  "cipt_lr_schedule": "cosine", "cipt_lr_min": min_lr}, total
        )
        used = []
        for step in range(total):
            used.append(alg.optimizer.param_groups[0]["lr"])
            alg.update()
            scheduler.step()
        for step, rate in enumerate(used):
            expected = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * step / total)
            )
            self.assertAlmostEqual(rate, expected, places=12)
        self.assertAlmostEqual(
            alg.optimizer.param_groups[0]["lr"], min_lr, places=12
        )
        self.assertEqual(scheduler.last_epoch, total)

    def test_scheduler_is_reset_per_target_model(self):
        hparams = {"cipt_objective": "env_anchor", "cipt_lr_schedule": "cosine"}
        for _ in range(2):
            alg = ToyAlgorithm(5e-5)
            scheduler = make_cipt_lr_scheduler(alg, hparams, 4)
            self.assertEqual(alg.optimizer.param_groups[0]["lr"], 5e-5)
            for _ in range(4):
                alg.update()
                scheduler.step()
            self.assertLess(alg.optimizer.param_groups[0]["lr"], 1e-10)

    def test_invalid_options_fail_early(self):
        for hparams, steps in (
            ({"cipt_lr_schedule": "unknown"}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "legacy"}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "env_anchor",
              "cipt_lr_min": -1}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "env_anchor",
              "cipt_lr_min": .0025}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "env_anchor"}, 0),
        ):
            with self.subTest(hparams=hparams, steps=steps):
                with self.assertRaises(ValueError):
                    make_cipt_lr_scheduler(ToyAlgorithm(), hparams, steps)


if __name__ == "__main__":
    unittest.main()
