"""CPU tests for switchable causal-mask decomposition and XCorr loss."""

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1] / "domainbed" / "algorithms"

modules_spec = importlib.util.spec_from_file_location(
    "cipt_mask_modules", ROOT / "cipt_modules.py"
)
modules = importlib.util.module_from_spec(modules_spec)
modules_spec.loader.exec_module(modules)
CausalDecomposition = modules.CausalDecomposition

losses_spec = importlib.util.spec_from_file_location(
    "cipt_mask_losses", ROOT / "cipt_losses.py"
)
losses = importlib.util.module_from_spec(losses_spec)
losses_spec.loader.exec_module(losses)
cross_correlation_loss = losses.cross_correlation_loss


class CausalMaskTest(unittest.TestCase):
    def test_dual_linear_default_preserves_identity_initialization(self):
        torch.manual_seed(0)
        x = torch.randn(5, 8)
        decomp = CausalDecomposition(8)
        e, s = decomp(x)
        self.assertTrue(torch.allclose(e, x))
        self.assertTrue(torch.allclose(s, x))
        self.assertIsNone(decomp.last_mask)

    def test_causal_mask_is_complementary_and_deterministic_in_eval(self):
        torch.manual_seed(1)
        x = torch.randn(6, 8)
        decomp = CausalDecomposition(
            8, mode="causal_mask", mask_hidden_dim=4,
            mask_temperature=1.0, mask_hard=False,
        )
        decomp.eval()
        e1, s1 = decomp(x)
        mask1 = decomp.last_mask.clone()
        e2, s2 = decomp(x)
        mask2 = decomp.last_mask.clone()

        self.assertTrue(torch.allclose(e1 + s1, x, atol=1e-6))
        self.assertTrue(torch.allclose(e1, e2))
        self.assertTrue(torch.allclose(s1, s2))
        self.assertTrue(torch.allclose(mask1, mask2))
        self.assertGreaterEqual(mask1.min().item(), 0.0)
        self.assertLessEqual(mask1.max().item(), 1.0)

    def test_hard_mask_is_binary_at_eval(self):
        torch.manual_seed(2)
        x = torch.randn(4, 8)
        decomp = CausalDecomposition(
            8, mode="causal_mask", mask_hidden_dim=4,
            mask_temperature=0.5, mask_hard=True,
        )
        decomp.eval()
        e, s = decomp(x)
        mask = decomp.last_mask
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        self.assertTrue(torch.allclose(e + s, x, atol=1e-6))

    def test_mask_path_receives_gradient(self):
        torch.manual_seed(3)
        x = torch.randn(7, 8)
        decomp = CausalDecomposition(
            8, mode="causal_mask", mask_hidden_dim=4,
            mask_temperature=1.0, mask_hard=False,
        )
        decomp.train()
        e, s = decomp(x)
        loss = e.square().mean() + 0.1 * decomp.mask_sparsity_loss()
        loss.backward()
        grad = decomp.mask_generator[2].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_xcorr_is_zero_when_one_branch_is_constant(self):
        torch.manual_seed(4)
        e = torch.randn(16, 8)
        s = torch.zeros_like(e)
        loss = cross_correlation_loss(e, s)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_xcorr_penalizes_shared_batch_structure(self):
        torch.manual_seed(5)
        e = torch.randn(32, 8)
        shared = cross_correlation_loss(e, e)
        independent = cross_correlation_loss(e, torch.randn_like(e))
        self.assertGreater(shared.item(), independent.item())


if __name__ == "__main__":
    unittest.main()
