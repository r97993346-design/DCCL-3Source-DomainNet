"""Numerical and one-update integration tests for the CIPT IND ablation."""

import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from test_cipt_env_anchor import batches, build, ind, losses, namespace


def condcov_reference(e, s, labels):
    value = e.new_zeros(())
    for label in labels.unique():
        ec, sc = e[labels == label], s[labels == label]
        if len(ec) >= 2:
            cross = (ec - ec.mean(0)).T @ (sc - sc.mean(0))
            value = value + len(ec) / len(e) * (cross / (len(ec) - 1)).square().sum()
    return value


class ConditionalLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(12)
        self.e = torch.randn(8, 3, generator=generator, requires_grad=True)
        self.s = torch.randn(8, 5, generator=generator, requires_grad=True)
        self.labels = torch.tensor([0, 0, 1, 0, 2, 1, 0, 1])

    def test_rectangular_covariance_and_changed_labels(self):
        first, count = ind.conditional_covariance_loss(self.e, self.s, self.labels, 5)
        self.assertEqual(count.item(), 2)
        other_labels = self.labels.roll(1)
        second, _ = ind.conditional_covariance_loss(self.e, self.s, other_labels, 5)
        torch.testing.assert_close(first, condcov_reference(self.e, self.s, self.labels))
        torch.testing.assert_close(second, condcov_reference(self.e, self.s, other_labels))
        self.assertFalse(torch.isclose(first, second))

    def test_constant_s_and_all_singletons(self):
        constant = torch.full_like(self.s, .625)
        actual, _ = ind.conditional_covariance_loss(self.e, constant, self.labels, 3)
        self.assertLess(actual.item(), 1e-11)
        singletons = torch.arange(len(self.labels))
        actual, count = ind.conditional_covariance_loss(self.e, self.s, singletons, 8)
        self.assertEqual(count.item(), 0)
        self.assertEqual(actual.item(), 0)
        for gradient in torch.autograd.grad(actual, (self.e, self.s)):
            self.assertEqual(gradient.count_nonzero().item(), 0)
        self.assertEqual(ind.coopood_loss(self.e, self.s, singletons, 8).item(), 0)

    def test_finite_gradients_in_both_branches_and_singleton_gradient(self):
        actual, _ = ind.conditional_covariance_loss(self.e, self.s, self.labels, 3)
        reference = condcov_reference(self.e, self.s, self.labels)
        for gradient, expected in zip(torch.autograd.grad(actual, (self.e, self.s)),
                                      torch.autograd.grad(reference, (self.e, self.s))):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)
            self.assertEqual(gradient[4].count_nonzero().item(), 0)
            torch.testing.assert_close(gradient, expected)

    def test_coopood_uses_pooled_outer_products_and_entrywise_l1(self):
        means = torch.stack([self.s[self.labels == label].mean(0) for label in self.labels])
        cross = torch.stack([torch.outer(e, s) for e, s in zip(self.e, self.s - means)]).mean(0)
        actual = ind.coopood_loss(self.e, self.s, self.labels, 4)
        torch.testing.assert_close(actual, cross.abs().sum())
        self.assertGreater(actual.item(), torch.linalg.matrix_norm(cross, ord=1).item())

    def test_opposite_class_covariances_cannot_cancel_in_condcov(self):
        e = torch.tensor([[-1.], [1.], [-1.], [1.]])
        s = torch.tensor([[-1.], [1.], [1.], [-1.]])
        labels = torch.tensor([0, 0, 1, 1])
        actual, _ = ind.conditional_covariance_loss(e, s, labels, 2)
        self.assertEqual(actual.item(), 4.)
        self.assertEqual(ind.coopood_loss(e, s, labels, 2).item(), 0.)

    def check_autocast(self, device, dtype):
        e = self.e.to(device, dtype).detach().requires_grad_()
        s = self.s.to(device, dtype).detach().requires_grad_()
        labels = self.labels.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            cov, _ = ind.conditional_covariance_loss(e, s, labels, 4)
            pooled = ind.coopood_loss(e, s, labels, 4)
            total = cov + pooled
        self.assertEqual(cov.dtype, torch.float32)
        self.assertEqual(pooled.dtype, torch.float32)
        torch.testing.assert_close(cov, condcov_reference(e.float(), s.float(), labels))
        for value in (*torch.autograd.grad(total, (e, s)), total):
            self.assertTrue(torch.isfinite(value).all())

    def test_cpu_fp16_and_bfloat16_autocast(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_autocast("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_and_bfloat16_autocast(self):
        self.check_autocast("cuda", torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_autocast("cuda", torch.bfloat16)


class IndIntegrationTests(unittest.TestCase):
    def build(self, **options):
        return build("relative_distance", cipt_env_topk=80, cipt_env_selection="all",
                     cipt_gamma=.1, cipt_contrastive_type="clip_guided_cross_domain",
                     cipt_use_contrastive=True, **options)

    def test_original_matches_unchanged_formula_and_implicit_switch(self):
        for legacy_enabled, explicit_mode in ((True, "original"), (False, "none")):
            with self.subTest(mode=explicit_mode):
                implicit = self.build(cipt_use_ind=legacy_enabled)
                explicit = self.build(ind_mode=explicit_mode, cipt_use_ind=not legacy_enabled)
                x, y = batches()
                with torch.no_grad():
                    e, s = implicit.causal_decomposition(implicit._visual(torch.cat(x)))
                    expected = .5 * F.cosine_similarity(e, s, dim=-1, eps=1e-6).square().mean()
                actual = explicit.update(x, y)
                self.assertEqual(actual["cipt_ind_loss"], expected.item() if legacy_enabled else 0.)
                self.assertEqual(actual["total_loss"], implicit.update(x, y)["total_loss"])
                for name, state in implicit.state_dict().items():
                    self.assertTrue(torch.equal(state, explicit.state_dict()[name]), name)

    def test_mode_exclusivity_and_source_batch_smoke(self):
        x, y = batches()
        for mode, function, weight in (
            ("condcov", "conditional_covariance_loss", .05),
            ("coopood", "coopood_loss", .02),
        ):
            with self.subTest(mode=mode):
                opts = {"ind_mode": mode, f"cipt_{mode}_weight": weight}
                model = self.build(ind_grad_steps=[0], **opts)
                control = self.build(**opts)
                with mock.patch.dict(namespace, {"independence_loss": mock.Mock(
                    side_effect=AssertionError("original IND was called")
                )}):
                    with mock.patch.dict(namespace, {function: mock.Mock(wraps=getattr(ind, function))}):
                        actual = model.update(x, y, step=0)
                        e, s, labels, classes = namespace[function].call_args.args
                        self.assertEqual((len(e), len(s), classes), (6, 6, 3))
                        self.assertTrue(torch.equal(labels, torch.cat(y)))
                self.assertTrue(all(math.isfinite(value) for value in actual.values()))
                self.assertIn(f"loss_{mode}", actual)
                self.assertNotIn("loss_original_ind", actual)
                self.assertEqual(actual["ind_weight"], weight)
                self.assertEqual(actual.get("condcov_valid_classes"), 3 if mode == "condcov" else None)
                self.assertGreater(actual["ind_grad_e_selected_norm"], 0)
                self.assertGreater(actual["ind_grad_s_selected_norm"], 0)
                self.assertGreater(actual["s_feature_batch_var"], 0)
                self.assertNotIn("ind_grad_e_selected_norm", control.update(x, y, step=0))
                for name, state in model.state_dict().items():
                    self.assertTrue(torch.equal(state, control.state_dict()[name]), name)

    def test_new_weights_are_required_and_independent(self):
        base = {"cipt_objective": "env_anchor", "cipt_gamma": .1}
        for mode in ("condcov", "coopood"):
            params = dict(base, ind_mode=mode)
            with self.assertRaisesRegex(ValueError, f"cipt_{mode}_weight"):
                ind.resolve_ind_config(params)
            for value in (-1., math.inf, math.nan):
                with self.assertRaises(ValueError):
                    ind.resolve_ind_config(dict(params, **{f"cipt_{mode}_weight": value}))
            self.assertEqual(ind.resolve_ind_config(dict(
                params, **{f"cipt_{mode}_weight": 0.},
            ))[1], 0.)


if __name__ == "__main__":
    unittest.main()
