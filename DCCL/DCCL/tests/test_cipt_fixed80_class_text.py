"""Frozen all80 text class prototypes for E/S and e_only inference.

Run from DCCL/: python -m unittest discover -s tests -p 'test_cipt_fixed80_class_text.py'
Uses the existing synthetic frozen-CLIP fixture; no checkpoint download.
"""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_anchor as fixture

class Fixed80ClassTextTests(unittest.TestCase):
    def _build(self, mode="fixed80", **overrides):
        return fixture.build(
            "relative_distance",
            cipt_class_text_mode=mode,
            cipt_env_selection="all",
            cipt_env_topk=80,
            cipt_rd_anchor_mode="full",
            cipt_e_cls_weight=1.0,
            cipt_s_uniform_weight=1.0,
            cipt_env_loss_weight=1.0,
            cipt_use_ind=False,
            cipt_use_contrastive=False,
            **overrides,
        )

    def test_frozen_classifier_matches_normalized_all80_ensemble(self):
        model = self._build()
        bank = model.text_features.full_b5b_text_features()
        expected = F.normalize(
            F.normalize(bank.detach().float(), dim=-1).mean(dim=1),
            dim=-1,
        )
        torch.testing.assert_close(model._class_text_features(), expected)
        torch.testing.assert_close(model.env_class_anchors, expected)
        self.assertFalse(model._class_text_features().requires_grad)
        self.assertFalse(model.text_features.prompt_learner.context.requires_grad)
        optimized = {
            id(p) for group in model.optimizer.param_groups
            for p in group["params"]
        }
        self.assertNotIn(id(model.text_features.prompt_learner.context), optimized)
        self.assertIn(id(model.causal_decomposition.causal_adapter.weight), optimized)
        self.assertIn(id(model.causal_decomposition.spurious_adapter.weight), optimized)

    def test_training_and_inference_do_not_evaluate_prompt_learner(self):
        model = self._build()
        old_centers = model.env_class_anchors.clone()
        images, ys = fixture.batches()
        with (
            mock.patch.object(model.text_features, "class_features",
                              side_effect=AssertionError("learned text invoked")),
            mock.patch.object(model.text_features.prompt_learner, "forward",
                              side_effect=AssertionError("CoOp prompt invoked")),
        ):
            metrics = model.update(images, ys)
            predictions = model.predict(images[0])
            visual = model._visual(images[0])
            causal, _ = model.causal_decomposition(visual)
            expected = model._logits(
                causal[:, None], model.env_class_anchors
            )[:, 0]
        torch.testing.assert_close(predictions, expected)
        torch.testing.assert_close(model.env_class_anchors, old_centers)
        self.assertEqual(metrics["env_topk"], 80.0)
        self.assertEqual(metrics["env_all_templates_active"], 1.0)
        self.assertEqual(metrics["env_rd_loss"], metrics["env_loss"])
        self.assertGreater(metrics["env_rd_loss"], 0.0)
        self.assertGreater(metrics["e_cls_loss"], 0.0)
        self.assertGreater(metrics["s_uni_loss"], 0.0)
        self.assertEqual(metrics["class_text_fixed80_active"], 1.0)
        self.assertEqual(metrics["class_text_learnable_active"], 0.0)
        self.assertIsNone(model.text_features.prompt_learner.context.grad)
        self.assertGreater(
            model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_old_learnable_mode_remains_default(self):
        model = self._build("learnable")
        self.assertTrue(model.text_features.prompt_learner.context.requires_grad)
        text = model._class_text_features()
        self.assertTrue(text.requires_grad)
        self.assertEqual(tuple(text.shape), tuple(model.env_class_anchors.shape))
        optimized = {
            id(p) for group in model.optimizer.param_groups
            for p in group["params"]
        }
        self.assertIn(id(model.text_features.prompt_learner.context), optimized)

    def test_mode_is_independent_of_entropy_selection(self):
        model = fixture.build(
            "relative_distance",
            cipt_class_text_mode="fixed80",
            cipt_env_selection="entropy",
            cipt_env_topk=40,
            cipt_use_ind=False,
            cipt_use_contrastive=False,
        )
        self.assertEqual(model.env_topk, 40)
        torch.testing.assert_close(
            model._class_text_features(), model.env_class_anchors
        )
        metrics = model.update(*fixture.batches())
        self.assertEqual(metrics["entropy_selector_active"], 1.0)
        self.assertEqual(metrics["env_topk"], 40.0)

    def test_invalid_text_mode(self):
        with self.assertRaisesRegex(ValueError, "cipt_class_text_mode"):
            self._build("unknown")

if __name__ == "__main__":
    unittest.main()
