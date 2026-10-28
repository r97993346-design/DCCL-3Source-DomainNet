"""All-80 official-template alignment ablations: numerical and real update tests.

Run from DCCL/: python -m unittest discover -s tests -p 'test_cipt_rd_all80_alignment.py'
Uses the tiny frozen CLIP fixture; no GPU or pretrained weights are needed.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_anchor as fixture


losses = fixture.losses


class All80AlignmentTests(unittest.TestCase):
    def _inputs(self, batch=3, dim=8, classes=4):
        torch.manual_seed(204)
        e = torch.randn(batch, dim, requires_grad=True)
        bank = torch.randn(classes, 80, dim, requires_grad=True)
        labels = torch.tensor([0, 2, 1], dtype=torch.long)[:batch]
        indices = torch.arange(80, dtype=torch.long)[None].expand(batch, -1)
        return e, bank, labels, indices

    def test_exact_second_and_fourth_formulas(self):
        e, bank, labels, idx = self._inputs()
        unit_e = F.normalize(e.float(), dim=-1)
        unit_t = F.normalize(bank.detach().float()[labels], dim=-1)
        expected_second = (unit_e - unit_t.mean(1)).square().sum(-1).mean()
        expected_fourth = (
            (unit_e[:, None] - unit_t).square().sum(-1).mean()
        )
        second = losses.template_alignment_loss(e, bank, labels, idx, "center_mse")
        fourth = losses.template_alignment_loss(e, bank, labels, idx, "mean_template_mse")
        torch.testing.assert_close(second, expected_second)
        torch.testing.assert_close(fourth, expected_fourth)
        self.assertFalse(second == fourth)  # They differ by frozen template variance.

    def test_equivalence_of_e_gradients_and_frozen_variance_identity(self):
        e, bank, labels, idx = self._inputs()
        l2 = losses.template_alignment_loss(e, bank, labels, idx, "center_mse")
        l4 = losses.template_alignment_loss(e, bank, labels, idx, "mean_template_mse")
        text = F.normalize(bank.detach()[labels], dim=-1)
        variance = (text - text.mean(1, keepdim=True)).square().sum(-1).mean()
        torch.testing.assert_close(l4, l2 + variance)
        grad2 = torch.autograd.grad(l2, e, retain_graph=True)[0]
        grad4 = torch.autograd.grad(l4, e)[0]
        torch.testing.assert_close(grad2, grad4, rtol=1e-5, atol=1e-6)
        self.assertGreater(grad2.abs().sum().item(), 0.0)
        self.assertIsNone(bank.grad)

    def test_all80_no_entropy_no_random_selector_and_real_update(self):
        for mode in ("center_mse", "mean_template_mse"):
            model = fixture.build(
                mode, cipt_env_selection="all", cipt_env_topk=80,
                cipt_use_ind=False, cipt_use_contrastive=False,
            )
            self.assertIsNone(model.entropy_selector)
            self.assertIsNone(model.random_selector)
            self.assertEqual(model.env_topk, 80)
            self.assertTrue(model.env_active)
            with mock.patch.object(
                fixture.selector.EntropyPromptSelector, "forward",
                side_effect=AssertionError("entropy selector called"),
            ), mock.patch.object(
                fixture.random_selector.RandomPromptSelector, "forward",
                side_effect=AssertionError("random selector called"),
            ):
                metrics = model.update(*fixture.batches())
            self.assertEqual(metrics["env_topk"], 80.0)
            self.assertEqual(metrics["env_all_templates_active"], 1.0)
            self.assertEqual(metrics["entropy_selector_active"], 0.0)
            self.assertEqual(metrics["random_selector_active"], 0.0)
            self.assertTrue(torch.isfinite(
                model.causal_decomposition.causal_adapter.weight.grad
            ).all())
            self.assertGreater(
                model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0
            )
            self.assertGreater(metrics["env_loss"], 0)
            self.assertEqual(
                metrics["env_center_mse_loss"] > 0, mode == "center_mse"
            )
            self.assertEqual(
                metrics["env_mean_template_mse_loss"] > 0, mode == "mean_template_mse"
            )

    def test_reject_partial_bank_or_selector_for_these_controls(self):
        with self.assertRaisesRegex(ValueError, "requires cipt_env_topk=80"):
            fixture.build(
                "center_mse", cipt_env_selection="all", cipt_env_topk=40
            )
        # Center MSE now supports entropy-selected top-k and computes
        # its teacher mean from precisely those image-dependent indices.
        entropy_model = fixture.build(
            "center_mse", cipt_env_selection="entropy", cipt_env_topk=40
        )
        self.assertEqual(entropy_model.env_topk, 40)
        self.assertIsNotNone(entropy_model.entropy_selector)
        with self.assertRaisesRegex(ValueError, "requires all 80"):
            fixture.build(
                "mean_template_mse", cipt_env_selection="entropy", cipt_env_topk=40
            )
        e, bank, labels, idx = self._inputs()
        with self.assertRaisesRegex(ValueError, "requires all 80"):
            losses.template_alignment_loss(
                e, bank, labels, idx[:, :40], mode="mean_template_mse"
            )


if __name__ == "__main__":
    unittest.main()
