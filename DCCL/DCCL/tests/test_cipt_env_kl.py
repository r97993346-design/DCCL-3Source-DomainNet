"""CPU numerical/gradient and real-update tests; no dataset or CLIP download."""

import argparse
import ast
import contextlib
import io
import math
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from test_cipt_neighbor_contrastive import (
    ALGORITHMS, ROOT, NAMESPACE, Component, SingleView, load_file, toy_features,
)


losses = load_file("env_losses", ALGORITHMS / "cipt_losses.py")
tree = ast.parse((ALGORITHMS / "cipt_env_kl.py").read_text())
definition = next(n for n in tree.body if isinstance(n, ast.ClassDef))
namespace = dict(
    NAMESPACE, _BaseCIPTDCCL=Component,
    decomposition_loss_terms=losses.decomposition_loss_terms,
    independence_loss=losses.independence_loss,
    prompt_consistency_loss=losses.prompt_consistency_loss,
)
exec(compile(ast.Module(body=[definition], type_ignores=[]), "cipt_env_kl.py", "exec"), namespace)
Environment = namespace["CIPTDCCL"]


def ordered_pair_reference(logits):
    """Independent, explicit mathematical specification, including both sides."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return torch.stack([
        F.kl_div(log_p[:, target], p[:, source], reduction="batchmean")
        for source in range(logits.shape[1])
        for target in range(logits.shape[1]) if source != target
    ]).mean()


class PromptLossTests(unittest.TestCase):
    def test_decomposition_split_preserves_values_and_gradients(self):
        torch.manual_seed(12)
        e = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        s = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 2, 1, 0])
        ce, uniform = losses.decomposition_loss_terms(e, s, labels)
        expected_uniform = (-math.log(3) - F.log_softmax(s, -1).mean(-1)).mean()
        expected = F.cross_entropy(e, labels) + expected_uniform
        torch.testing.assert_close(uniform, expected_uniform)
        torch.testing.assert_close(ce + uniform, expected)
        torch.testing.assert_close(losses.decomposition_loss(e, s, labels), expected)
        actual_grads = torch.autograd.grad(ce + uniform, (e, s), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (e, s))
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference)

    def test_pairwise_kl_matches_explicit_value_and_gradient(self):
        torch.manual_seed(8)
        for k in (2, 4, 9):
            logits = torch.randn(3, k, 5, dtype=torch.float64, requires_grad=True)
            actual = losses.prompt_consistency_loss(logits)
            expected = ordered_pair_reference(logits)
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertTrue((actual_grad.abs().sum(-1) > 0).all())

    def test_gradcheck_and_prompt_permutation(self):
        torch.manual_seed(4)
        logits = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(losses.prompt_consistency_loss, (logits,)))
        expected = losses.prompt_consistency_loss(logits)
        torch.testing.assert_close(expected, losses.prompt_consistency_loss(logits[:, [2, 0, 1]]))
        torch.testing.assert_close(expected, losses.prompt_consistency_loss(logits.repeat(3, 1, 1)))

    def test_identical_predictions_zero_and_extreme_logits_finite(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.tensor([[[80., -80., 0.]]], dtype=dtype).repeat(2, 4, 1)
            logits.requires_grad_()
            loss = losses.prompt_consistency_loss(logits)
            self.assertAlmostEqual(loss.item(), 0., places=5)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
        logits = torch.tensor([[[1000., -1000.], [-1000., 1000.]]], requires_grad=True)
        loss = losses.prompt_consistency_loss(logits)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bad_prompt_shapes_are_rejected(self):
        for shape in ((3, 4), (2, 1, 3), (0, 2, 3), (2, 3, 0)):
            with self.assertRaises(ValueError):
                losses.prompt_consistency_loss(torch.zeros(shape))


class EnvironmentUpdateTests(unittest.TestCase):
    def build(self, cls=Environment, **overrides):
        params = dict(
            cipt_objective="env_kl", cipt_env_kl_weight=.1,
            cipt_pure=False, cipt_template_mode="b5c", cipt_neutral_subject="subject",
            cipt_k=2, cipt_tda_heads=1, cipt_beta=4., cipt_gamma=5.,
            cipt_selector_mode="adaptive", cipt_selector_candidates=4,
            cipt_contrastive_type="neighbor_retention", cipt_neighbor_k=1,
            cipt_neighbor_alpha=.5, cipt_causal_contrastive_weight=1.5,
            cipt_contrastive_warmup_steps=500, cipt_contrastive_temperature=.07,
            t=.1, optimizer="adam", lr=1e-3, weight_decay=0.,
        )
        params.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return cls((4,), 2, 2, params)

    def batches(self):
        v, y, _ = toy_features(.25)
        return [v[:2], v[2:]], [y[:2], y[2:]]

    def test_weighted_total_no_augmented_ce_and_label_free_selection(self):
        model = self.build()
        with mock.patch.object(model, "_select_interventions", wraps=model._select_interventions) as select:
            with mock.patch.object(F, "cross_entropy", wraps=F.cross_entropy) as ce:
                metrics = model.update(*self.batches())
        self.assertIsNone(select.call_args.kwargs["labels"])
        self.assertEqual(ce.call_count, 1)  # Only E's [B,C] label CE, never [B*K,C].
        self.assertEqual(tuple(ce.call_args.args[0].shape), (4, 2))
        expected_base = (metrics["e_cls_loss"] + 4 * metrics["s_uni_loss"]
                         + 5 * metrics["cipt_ind_loss"])
        expected = (expected_base + metrics["contrastive_weight_eff"]
                    * metrics["dccl_contrastive_loss"] + .1 * metrics["env_kl_loss"])
        self.assertAlmostEqual(metrics["cipt_base_loss"], expected_base, places=5)
        self.assertAlmostEqual(metrics["total_loss"], expected, places=5)
        self.assertEqual(metrics["e_cls_w"], 1.)
        self.assertEqual(metrics["s_uni_w"], 4.)
        self.assertEqual(metrics["prompt_selector_active"], 1.)
        self.assertEqual(metrics["env_active"], 1.)
        self.assertEqual(model.visual_calls, 1)
        self.assertIsNone(model.encoder.weight.grad)

    def test_kl_alone_backpropagates_through_fusion_and_e(self):
        torch.manual_seed(3)
        model = self.build(cipt_e_cls_weight=0., cipt_s_uniform_weight=0.,
                           cipt_use_ind=False, cipt_use_contrastive=False,
                           cipt_env_kl_weight=1.)
        metrics = model.update(*self.batches())
        self.assertGreater(metrics["env_kl_loss"], 0.)
        self.assertAlmostEqual(metrics["total_loss"], metrics["env_kl_loss"], places=6)
        self.assertGreater(model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.tda.attention.in_proj_weight.grad[8:].abs().sum().item(), 0.)
        self.assertGreater(model.text_features.classes.grad.abs().sum().item(), 0.)
        self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)

    def test_explicit_old_weights_preserve_source_no_tda_objective_and_update(self):
        torch.manual_seed(33)
        source = self.build(Component, cipt_use_tda=False)
        torch.manual_seed(33)
        changed = self.build(cipt_env_kl_weight=0., cipt_e_cls_weight=5.)
        with mock.patch.object(changed, "_select_interventions", side_effect=AssertionError("disabled KL ran TDA")):
            a = source.update(*self.batches())
            b = changed.update(*self.batches())
        self.assertAlmostEqual(a["total_loss"], b["total_loss"], places=5)
        self.assertEqual(a["contrastive_weight_eff"], b["contrastive_weight_eff"])
        for old, new in zip(source.parameters(), changed.parameters()):
            torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-5)
        self.assertEqual(b["env_active"], 0.)
        self.assertEqual(b["prompt_count"], 0.)

    def test_legacy_mode_delegates_unchanged(self):
        torch.manual_seed(9)
        source = self.build(Component)
        torch.manual_seed(9)
        delegated = self.build(cipt_objective="legacy")
        a = source.update(*self.batches())
        b = delegated.update(*self.batches())
        self.assertEqual(a, b)
        for old, new in zip(source.parameters(), delegated.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_e_only_inference_never_selects_or_fuses(self):
        model = self.build()
        model.eval()
        images = self.batches()[0][0]
        with torch.no_grad():
            e, _ = model.causal_decomposition(model._visual(images))
            expected = model._logits(e[:, None], model.text_features.class_features())[:, 0]
            with mock.patch.object(model, "_select_interventions", side_effect=AssertionError("inference selection")):
                with mock.patch.object(model.tda, "forward", side_effect=AssertionError("inference fusion")):
                    actual = model.predict(images)
        torch.testing.assert_close(actual, expected)

    def test_component_switches_explicit_weights_and_all_bank(self):
        model = self.build(cipt_beta=7., cipt_e_cls_weight=None)
        self.assertEqual(model.e_cls_weight, 1.)
        self.assertEqual(model.s_uniform_weight, 7.)
        model = self.build(cipt_use_tda=False, cipt_use_de=False,
                           cipt_use_ind=False, cipt_use_contrastive=False)
        metrics = model.update(*self.batches())
        self.assertEqual(metrics["e_cls_w"], 1.)
        self.assertEqual(metrics["s_uni_w"], 0.)
        self.assertEqual(metrics["env_kl_loss"], 0.)
        self.assertAlmostEqual(metrics["total_loss"], metrics["e_cls_loss"], places=6)
        model = self.build(cipt_e_cls_weight=4., cipt_s_uniform_weight=2.,
                           cipt_selector_mode="all", cipt_k=1)
        metrics = model.update(*self.batches())
        self.assertEqual(metrics["e_cls_w"], 4.)
        self.assertEqual(metrics["s_uni_w"], 2.)
        self.assertEqual(metrics["prompt_count"], 6.)

    def test_invalid_configuration_fails_before_training(self):
        for key in ("cipt_e_cls_weight", "cipt_s_uniform_weight", "cipt_env_kl_weight"):
            for value in (-1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.build(**{key: value})
        for options in (dict(cipt_objective="unknown"), dict(cipt_k=1),
                        dict(cipt_template_mode="b5b", cipt_selector_mode="random")):
            with self.assertRaises(ValueError):
                self.build(**options)


class ConfigurationTests(unittest.TestCase):
    def test_yaml_and_both_kinds_of_cli_override(self):
        from sconf import Config

        parser = argparse.ArgumentParser()
        parser.add_argument("--cipt_selector_mode", default="random")
        parser.add_argument("--cipt_beta", type=float, default=4.)
        argv = ["--cipt_selector_mode=random", "--cipt_beta", "6",
                "--cipt_env_kl_weight", "0.2", "--cipt_e_cls_weight", "4"]
        args, remaining = parser.parse_known_args(argv)
        with (ROOT / "config.yaml").open() as base, (ROOT / "config_cipt_env_kl.yaml").open() as experiment:
            config = Config(base, experiment)
        self.assertEqual(config.cipt_e_cls_weight, 1.)
        config.argv_update(remaining)
        tree = ast.parse((ROOT / "train_all.py").read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                          and n.name == "apply_explicit_hparam_cli")
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "train_all.py", "exec"), namespace)
        apply = namespace["apply_explicit_hparam_cli"]
        # Real training includes beta in registry defaults.
        config["cipt_beta"] = 4.
        apply(parser, args, config, argv)
        self.assertEqual(config.cipt_selector_mode, "random")
        self.assertEqual(config.cipt_beta, 6.)
        self.assertEqual(config.cipt_env_kl_weight, .2)
        self.assertEqual(config.cipt_e_cls_weight, 4.)
        self.assertEqual(config.cipt_objective, "env_kl")
        # Unspecified argparse defaults must not overwrite the adaptive preset.
        config.cipt_selector_mode = "adaptive"
        apply(parser, parser.parse_args([]), config, [])
        self.assertEqual(config.cipt_selector_mode, "adaptive")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
