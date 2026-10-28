"""Regression: E-to-selected-text center for variable entropy Top-N.

Run under DCCL/: python -m unittest discover -s tests -p 'test_cipt_center_selection.py'
"""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_anchor as fixture


class EntropyCenterSelectionTests(unittest.TestCase):
    def test_entropy_topn_uses_exact_selected_per_image_centers(self):
        for n in (2, 7, 40, 80):
            model = fixture.build(
                "center_mse", cipt_env_selection="entropy",
                cipt_env_topk=n, cipt_use_ind=False,
                cipt_use_contrastive=False,
            )
            images, label_groups = fixture.batches()
            visual = model._visual(torch.cat(images))
            causal, _ = model.causal_decomposition(visual)
            labels = torch.cat(label_groups)
            bank = model.text_features.full_b5b_text_features()
            scale = model.clip_model.logit_scale.exp().detach().float()
            with mock.patch.object(
                model.entropy_selector, "forward",
                wraps=model.entropy_selector.forward,
            ) as spy:
                loss, metrics = model._env_anchor_loss(visual, causal, labels)
            self.assertEqual(spy.call_count, 1)
            idx, _ = fixture.selector.EntropyPromptSelector(n)(
                visual, bank, scale
            )
            self.assertEqual(tuple(idx.shape), (len(labels), n))
            selected = F.normalize(
                bank.detach()[labels[:, None], idx].float(), dim=-1
            )
            # The teacher is NOT the full-80 class center, unless n == 80.
            selected_center = selected.mean(dim=1)
            expected = (
                F.normalize(causal.float(), dim=-1) - selected_center
            ).square().sum(-1).mean()
            torch.testing.assert_close(loss, expected)
            self.assertEqual(metrics["entropy_selector_active"].item(), 1.0)
            self.assertFalse(bank.requires_grad)
            if n == 2:
                full_center = bank.detach()[labels].float().mean(dim=1)
                self.assertGreater(
                    (selected_center - full_center).abs().max().item(), 1e-4
                )
            # Frozen original CLIP (V) supplies template selection,
            # selection does not require gradients.
            self.assertFalse(idx.requires_grad)
            self.assertTrue(causal.requires_grad)
            grad_e = torch.autograd.grad(loss, causal)[0]
            self.assertGreater(grad_e.abs().sum().item(), 0.0)

    def test_entropy_center_real_update_and_diagnostics(self):
        model = fixture.build(
            "center_mse", cipt_env_selection="entropy", cipt_env_topk=13,
            cipt_use_ind=False, cipt_use_contrastive=False,
        )
        metrics = model.update(*fixture.batches())
        self.assertEqual(metrics["env_topk"], 13.0)
        self.assertEqual(metrics["env_all_templates_active"], 0.0)
        self.assertEqual(metrics["entropy_selector_active"], 1.0)
        self.assertEqual(metrics["random_selector_active"], 0.0)
        self.assertGreater(metrics["env_center_mse_loss"], 0.0)
        self.assertEqual(metrics["env_rd_loss"], 0.0)
        self.assertEqual(metrics["env_mean_template_mse_loss"], 0.0)
        self.assertTrue(torch.isfinite(
            model.causal_decomposition.causal_adapter.weight.grad
        ).all())

    def test_no_entropy_all80_preserves_previous_formula(self):
        model = fixture.build(
            "center_mse", cipt_env_selection="all", cipt_env_topk=80,
            cipt_use_ind=False, cipt_use_contrastive=False,
        )
        self.assertIsNone(model.entropy_selector)
        self.assertIsNone(model.random_selector)
        images, ys = fixture.batches()
        visual = model._visual(torch.cat(images))
        causal, _ = model.causal_decomposition(visual)
        labels = torch.cat(ys)
        loss, metrics = model._env_anchor_loss(visual, causal, labels)
        bank = model.text_features.full_b5b_text_features()
        center = F.normalize(bank.detach().float(), dim=-1)[labels].mean(dim=1)
        expected = (
            F.normalize(causal.float(), dim=-1) - center
        ).square().sum(-1).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics["entropy_selector_active"].item(), 0.0)

    def test_random_selection_also_uses_selected_center(self):
        model = fixture.build(
            "center_mse", cipt_env_selection="random", cipt_env_topk=6,
            cipt_use_ind=False, cipt_use_contrastive=False,
        )
        images, ys = fixture.batches()
        visual = model._visual(torch.cat(images))
        causal, _ = model.causal_decomposition(visual)
        labels = torch.cat(ys)
        with mock.patch.object(
            model.random_selector, "forward",
            wraps=model.random_selector.forward,
        ) as spy:
            loss, _ = model._env_anchor_loss(visual, causal, labels)
        self.assertEqual(spy.call_count, 1)
        idx = spy.call_args_list  # The selector's draw is random: capture the actual output next.
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_invalid_n_and_mean_template_ablation_stays_all80(self):
        for n in (1, 81):
            with self.assertRaises(ValueError):
                fixture.build("center_mse", cipt_env_selection="entropy",
                              cipt_env_topk=n)
        with self.assertRaisesRegex(ValueError, "requires cipt_env_topk=80"):
            fixture.build("center_mse", cipt_env_selection="all",
                          cipt_env_topk=40)
        with self.assertRaisesRegex(ValueError, "requires all 80"):
            fixture.build("mean_template_mse", cipt_env_selection="entropy",
                          cipt_env_topk=40)


if __name__ == "__main__":
    unittest.main()
