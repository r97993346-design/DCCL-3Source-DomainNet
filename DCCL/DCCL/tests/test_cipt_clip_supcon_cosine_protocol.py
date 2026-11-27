"""Check the all-positive teacher and production optimizer/scheduler loop."""
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import test_cipt_env_anchor as anchor
import test_target_seed_isolation as rng

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cipt_clip_supcon_optuna"))
from instrumented_train import CosineTrace, register_optimizer_step_post_hook, register_optimizer_step_pre_hook
from protocol import EVAL_STEPS, STEPS, TERMS, SPACE, cosine_lr, hparams_for, seed_for


class CLIPSupConTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(33)
        self.v = torch.randn(6, 8, generator=gen, requires_grad=True)
        self.e = torch.randn(6, 8, generator=gen, requires_grad=True)
        self.labels = torch.tensor([0, 0, 0, 1, 1, 2])
        self.domains = torch.tensor([0, 0, 1, 1, 0, 1])
        self.params = {k: choices[0] for k, choices in SPACE.items()}

    def model(self, mix):
        config = hparams_for(dict(self.params, contrastive_mix=mix))
        config.pop("cipt_env_constraint")
        return anchor.build("relative_distance", **config)

    def test_masks_include_same_class_same_domain_and_exclude_self(self):
        targets = anchor.clip_guided.cross_domain_targets
        w, positive, denom, similarity = targets(
            self.v, self.labels, self.domains, teacher_temperature=.2, teacher_mix=.75, all_same_class=True,
        )
        expected = self.labels[:, None].eq(self.labels[None, :]) & ~torch.eye(6, dtype=torch.bool)
        self.assertTrue(torch.equal(positive, expected))
        self.assertTrue(positive[0, 1])  # Same domain positive.
        self.assertTrue(positive[0, 2])  # Cross-domain positive.
        self.assertTrue(denom[0, 3] and not positive[0, 3])
        self.assertTrue(denom[0, 4] and not positive[0, 4])
        self.assertFalse(denom.diagonal().any())
        self.assertFalse(w.requires_grad)
        self.assertFalse(similarity.requires_grad)
        self.assertTrue(torch.equal(w[~positive], torch.zeros_like(w[~positive])))
        torch.testing.assert_close(w[:5].sum(1), torch.ones(5))
        teacher = torch.softmax(similarity[0, [1, 2]] / .2, dim=0)
        torch.testing.assert_close(w[0, [1, 2]], .25 / 2 + .75 * teacher)
        replaced = targets(self.v, self.labels, None, .2, .75, all_same_class=True)[0]
        self.assertTrue(torch.equal(replaced, w))

    def test_zero_mix_is_exact_supcon_in_value_and_e_gradient(self):
        model = self.model(0.)
        actual, _, _, _ = model._causal_contrastive_loss(self.v, self.e, self.labels, [self.labels])
        expected, _ = model._single_view_supcon(self.e, self.labels)
        self.assertTrue(torch.equal(actual, expected))
        gx, teacher = torch.autograd.grad(actual, (self.e, self.v), retain_graph=True, allow_unused=True)
        self.assertIsNone(teacher)
        self.assertTrue(torch.equal(gx, torch.autograd.grad(expected, self.e)[0]))

    def test_weighted_path_detaches_teacher_and_handles_no_positives(self):
        model = self.model(.5)
        actual, _, _, _ = model._causal_contrastive_loss(self.v, self.e, self.labels, [self.labels])
        gx, teacher = torch.autograd.grad(actual, (self.e, self.v), retain_graph=True, allow_unused=True)
        self.assertIsNone(teacher)
        self.assertTrue(torch.isfinite(gx).all() and gx.norm() > 0)
        labels = torch.arange(len(self.labels))
        zero, frac, _, _ = model._causal_contrastive_loss(self.v, self.e, labels, [labels])
        self.assertEqual((zero.item(), frac.item()), (0., 0.))
        self.assertTrue(torch.equal(torch.autograd.grad(zero, self.e)[0], torch.zeros_like(self.e)))

    def test_three_cpu_training_updates_use_each_loss_once(self):
        model = self.model(.25)
        self.assertFalse(any(p.requires_grad for p in model.clip_model.parameters()))
        self.assertEqual(model.contrastive_temperature, .07)
        self.assertEqual(model.inference_mode, "e_only")
        for step in range(3):
            r = model.update(*anchor.batches(), step=step)
            self.assertEqual((r["cipt_ind_loss"], r["nbr_check"], r["cg_all_class_active"]), (0., 0., 1.))
            self.assertFalse(any("grad_" in key for key in r))
            self.assertTrue(all(math.isfinite(r[f"loss_{k}"]) for k in TERMS))
            self.assertAlmostEqual(sum(r[f"weighted_{k}"] for k in TERMS), r["loss_total"], places=6)


class CosineTrainerTests(rng.TargetSeedIsolationTests):
    def test_adam_cosine_checkpoint_order_and_exact_budget(self):
        for initial in SPACE["initial_lr"]:
            counts = []
            trace = CosineTrace(initial, STEPS)
            real_eval = rng.Evaluator.evaluate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
            old_step = scheduler.step

            def order(self, *args, **kwargs):
                counted = self.last_epoch >= 0
                if counted:
                    trace.before_scheduler(self)
                answer = old_step(self, *args, **kwargs)
                if counted:
                    trace.after_scheduler(self)
                return answer

            def evaluate(self, model):
                counts.append((trace.optimizer_steps, trace.scheduler_steps))
                return real_eval(self, model)

            a = register_optimizer_step_pre_hook(trace.pre)
            b = register_optimizer_step_post_hook(trace.post)
            path = Path(self.temporary.name) / "results.jsonl"
            path.unlink(missing_ok=True)
            try:
                with patch.object(scheduler, "step", order), patch.object(rng.Evaluator, "evaluate", evaluate):
                    self.run_target(0, STEPS, trial_seed=1, checkpoint=300, fixed_seed=False,
                                    optimizer="adam", lr=initial, cipt_lr_min=0., cipt_lr_schedule="cosine",
                                    cipt_contrastive_type="clip_guided_supcon", cipt_objective="env_anchor")
            finally:
                a.remove()
                b.remove()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(trace.optimizer_steps, STEPS)
            self.assertEqual(trace.scheduler_steps, STEPS)
            self.assertEqual(counts, [(i+1, i+1) for i in EVAL_STEPS])
            self.assertEqual([r["step"] for r in rows], EVAL_STEPS)
            self.assertTrue(all(r["args"]["seed"] == seed_for("PACS", 0) for r in rows))
            self.assertEqual(trace.rows[0]["lr_used"], initial)
            self.assertGreater(trace.rows[-1]["lr_used"], 0.)
            self.assertTrue(math.isclose(trace.rows[-1]["lr_used"], cosine_lr(5000, initial), abs_tol=1e-16))
            self.assertEqual(trace.rows[-1]["lr_after_scheduler"], 0.)


if __name__ == "__main__":
    unittest.main()
