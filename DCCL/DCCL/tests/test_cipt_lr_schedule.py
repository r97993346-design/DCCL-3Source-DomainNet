"""Tests for optional DomainBed step-based Adam cosine learning-rate schedule.

Run from DCCL/DCCL/: python -m unittest discover -s tests -p 'test_cipt_lr_schedule.py'
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
        for objective in ("legacy", "env_anchor"):
            with self.subTest(objective=objective):
                alg = ToyAlgorithm(5e-5)
                control = ToyAlgorithm(5e-5)
                scheduler = make_cipt_lr_scheduler(
                    alg, {"cipt_objective": objective}, 7
                )
                self.assertIsNone(scheduler)
                for _ in range(7):
                    alg.update()
                    control.update()
                self.assertEqual(alg.optimizer.param_groups[0]["lr"], 5e-5)
                self.assertTrue(torch.equal(alg.parameter, control.parameter))

    def test_cosine_anneals_after_each_optimizer_update(self):
        lr = 0.0025
        min_lr = 0.0
        total = 8
        for objective in ("legacy", "env_anchor"):
            with self.subTest(objective=objective):
                alg = ToyAlgorithm(lr)
                scheduler = make_cipt_lr_scheduler(
                    alg, {"cipt_objective": objective,
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

    def test_requested_legacy_cosine_budgets(self):
        # Actual loop budgets after train_all.py's checkpoint rounding.
        for lr, total in ((0.002, 3001), (1e-5, 3001), (0.002, 5001)):
            with self.subTest(lr=lr, total=total):
                alg = ToyAlgorithm(lr)
                scheduler = make_cipt_lr_scheduler(
                    alg, {"cipt_objective": "legacy",
                          "cipt_lr_schedule": "cosine"}, total
                )
                for step in range(total):
                    if step in (0, total // 2, total - 1):
                        expected = 0.5 * lr * (
                            1.0 + math.cos(math.pi * step / total)
                        )
                        self.assertAlmostEqual(
                            alg.optimizer.param_groups[0]["lr"], expected, places=12
                        )
                    alg.update()
                    scheduler.step()
                self.assertEqual(scheduler.last_epoch, total)
                self.assertAlmostEqual(alg.optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_accepts_default_legacy_objective(self):
        alg = ToyAlgorithm(0.002)
        scheduler = make_cipt_lr_scheduler(
            alg, {"cipt_lr_schedule": "cosine"}, 8
        )
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_scheduler_is_reset_per_target_model(self):
        for objective in ("legacy", "env_anchor"):
            with self.subTest(objective=objective):
                hparams = {"cipt_objective": objective, "cipt_lr_schedule": "cosine"}
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
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "env_kl"}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "clip_ce"}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "legacy",
              "cipt_lr_min": -1}, 10),
            ({"cipt_lr_schedule": "cosine", "cipt_objective": "legacy"}, 0),
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
