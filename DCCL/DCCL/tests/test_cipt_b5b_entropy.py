"""CPU numerical, class-conditioning, and train/inference entropy checks."""

import argparse
import ast
import inspect
import math
from unittest import mock
import unittest

import torch
from torch import nn
import torch.nn.functional as F

import test_cipt_env_kl as fixture
import test_cipt_neighbor_contrastive as base


Selector = base.entropy.EntropyPromptSelector


class EntropySelectionTests(unittest.TestCase):
    def bank(self):
        bank = torch.zeros(2, 80, 2)
        bank[0, :40, 0], bank[1, :40, 0] = 1., -1.
        bank[0, 40:, 1], bank[1, 40:, 1] = 1., -1.
        return bank

    def test_per_image_low_entropy_and_stable_ties(self):
        images = torch.eye(2, requires_grad=True)
        bank = self.bank().requires_grad_()
        scale = torch.tensor(1000., requires_grad=True)
        indices, metrics = Selector(4)(images, bank, scale)
        self.assertEqual(indices.tolist(), [[0, 1, 2, 3], [40, 41, 42, 43]])
        self.assertFalse(metrics["env_selected_entropy_mean"].requires_grad)
        self.assertTrue(all(torch.isfinite(x) for x in metrics.values()))
        self.assertLess(metrics["env_selected_entropy_mean"], metrics["env_entropy_mean"])
        self.assertIsNone(images.grad)
        self.assertIsNone(bank.grad)
        self.assertIsNone(scale.grad)

    def test_all_classes_one_shared_index_set_and_no_label_input(self):
        images, bank = torch.eye(2), self.bank()
        selected, _ = Selector(7)(images, bank, 10.)
        swapped, _ = Selector(7)(images, bank.flip(0), 10.)
        torch.testing.assert_close(selected, swapped)
        self.assertNotIn("labels", inspect.signature(Selector.forward).parameters)
        all_indices, _ = Selector(80)(images, bank, 10.)
        torch.testing.assert_close(all_indices.sort(1).values, torch.arange(80).expand(2, -1))

    def test_invalid_n_and_bank_fail_early(self):
        for n in (0, 81, True, 2.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                Selector(n)
        for bank in (torch.zeros(2, 79, 2), torch.zeros(1, 80, 2), torch.zeros(2, 80, 3)):
            with self.assertRaises(ValueError):
                Selector(2)(torch.eye(2), bank, 10.)
        for scale in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                Selector(2)(torch.eye(2), self.bank(), scale)


class B5bTrainingInferenceTests(unittest.TestCase):
    def build(self, cls=fixture.Environment, **overrides):
        original_init = base.TinyBase.__init__

        def with_official_bank(model, *args, **kwargs):
            original_init(model, *args, **kwargs)
            model.clip_model = nn.Module()
            model.clip_model.logit_scale = nn.Parameter(torch.tensor(math.log(2.)), requires_grad=False)
            generator = torch.Generator().manual_seed(179)
            model.text_features.register_buffer("b5b_text_bank", torch.randn(2, 80, 4, generator=generator))
            model.text_features.full_b5b_text_features = lambda: model.text_features.b5b_text_bank.detach()

        options = dict(cipt_template_mode="b5b", cipt_selector_mode="entropy",
                       cipt_entropy_topk=3, cipt_k=1, cipt_e_cls_weight=1.,
                       cipt_s_uniform_weight=1., cipt_use_ind=False,
                       cipt_use_contrastive=False, cipt_env_kl_weight=0.,
                       cipt_fused_cls_weight=1.)
        options.update(overrides)
        with mock.patch.object(base.TinyBase, "__init__", with_official_bank):
            return fixture.EnvironmentUpdateTests().build(cls=cls, **options)

    def batches(self):
        return fixture.EnvironmentUpdateTests().batches()

    def test_selection_does_not_use_labels_but_preserves_b5b_training_context(self):
        model = self.build()
        x, y = self.batches()
        visual = model._visual(torch.cat(x))
        causal, _ = model.causal_decomposition(visual)
        labels = torch.cat(y)
        indices, _ = model.entropy_selector(visual, model.text_features.b5b_text_bank, 2.)
        for target in (labels, 1 - labels):
            with mock.patch.object(model.tda, "forward", wraps=model.tda.forward) as fusion:
                z, metrics = model._select_interventions(
                    visual, causal, model.text_features.class_features(), labels=target)
            expected_contexts = model.text_features.b5b_text_bank[target[:, None], indices]
            torch.testing.assert_close(fusion.call_args.args[1], expected_contexts)
            self.assertEqual(tuple(z.shape), (4, 3, 4))
            self.assertEqual(metrics["prompt_count"].item(), 3.)

    def test_update_selected_n_gradients_and_unchanged_loss_weights(self):
        model = self.build(cipt_env_kl_weight=.2)
        metrics = model.update(*self.batches())
        self.assertEqual(model.visual_calls, 1)
        self.assertEqual(metrics["prompt_count"], 3.)  # Not cipt_k=1.
        self.assertEqual(metrics["entropy_selector_active"], 1.)
        expected = (metrics["e_cls_loss"] + metrics["s_uni_loss"]
                    + metrics["fused_cls_loss"] + .2 * metrics["env_kl_loss"])
        self.assertAlmostEqual(metrics["total_loss"], expected, places=5)
        self.assertGreater(model.causal_decomposition.causal_adapter.weight.grad.abs().sum(), 0.)
        self.assertGreater(model.text_features.classes.grad.abs().sum(), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.tda.parameters()))
        self.assertIsNone(model.encoder.weight.grad)
        self.assertIsNone(model.clip_model.logit_scale.grad)
        self.assertIsNone(model.text_features.b5b_text_bank.grad)

    def test_candidate_inference_matches_full_class_vectorization(self):
        model = self.build()
        model.eval()
        images = torch.cat(self.batches()[0])
        with torch.no_grad():
            v = model._visual(images)
            e, _ = model.causal_decomposition(v)
            text = model.text_features.class_features()
            bank, indices, _ = model._entropy_b5b_indices(v)
            contexts = bank[:, indices].permute(1, 0, 2, 3)
            expanded_e = e[:, None].expand(-1, 2, -1).reshape(-1, 4)
            z = model.tda(expanded_e, contexts.reshape(-1, 3, 4)).reshape(4, 2, 3, 4)
            reference = 2. * torch.einsum("bcnd,cd->bnc", F.normalize(z, dim=-1), text)
            actual = model._b5b_entropy_prompt_logits(v, e, text)
            torch.testing.assert_close(actual, reference)
            for mode in ("mean_logits", "mean_probs"):
                model.inference_mode = mode
                expected = (reference.mean(1) if mode == "mean_logits" else
                            torch.logsumexp(F.log_softmax(reference, -1), 1) - math.log(3))
                calls = model.visual_calls
                torch.testing.assert_close(model.predict(images), expected)
                self.assertEqual(model.visual_calls, calls + 1)

    def test_e_only_and_disabled_training_skip_entropy_and_fusion(self):
        model = self.build(cipt_use_tda=False, cipt_env_inference="e_only")
        with mock.patch.object(model.entropy_selector, "forward", side_effect=AssertionError("selection disabled")), \
                mock.patch.object(model.tda, "forward", side_effect=AssertionError("fusion disabled")):
            metrics = model.update(*self.batches())
            result = model.predict(torch.cat(self.batches()[0]))
        self.assertEqual(tuple(result.shape), (4, 2))
        self.assertEqual(metrics["prompt_count"], 0.)
        self.assertEqual(metrics["env_active"], 0.)

    def test_legacy_b5b_uses_selected_templates_in_update_and_predict(self):
        model = self.build(cls=base.Component, cipt_objective="legacy", cipt_use_ind=False)
        metrics = model.update(*self.batches())
        self.assertEqual(metrics["prompt_count"], 3.)
        model.eval()
        images = torch.cat(self.batches()[0])
        with torch.no_grad():
            v = model._visual(images)
            e, _ = model.causal_decomposition(v)
            expected = model._b5b_entropy_prompt_logits(v, e, model.text_features.class_features()).mean(1)
            torch.testing.assert_close(model.predict(images), expected)

    def test_n_one_allowed_for_ce_but_not_mcc_and_wrong_bank_rejected(self):
        model = self.build(cipt_entropy_topk=1)
        self.assertEqual(model.update(*self.batches())["prompt_count"], 1.)
        with self.assertRaises(ValueError):
            self.build(cipt_entropy_topk=1, cipt_env_kl_weight=.1)
        with self.assertRaises(ValueError):
            self.build(cipt_template_mode="b5c")

    def test_cli_exposes_entropy_n_and_yaml_is_selector_only(self):
        tree = ast.parse((base.ROOT / "train_all.py").read_text())
        parser = argparse.ArgumentParser()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in ("--cipt_selector_mode", "--cipt_entropy_topk")):
                statement = ast.fix_missing_locations(ast.Module(body=[ast.Expr(node)], type_ignores=[]))
                exec(compile(statement, "cli", "exec"),
                     {"parser": parser, "int": int})
        args = parser.parse_args(["--cipt_selector_mode", "entropy", "--cipt_entropy_topk", "17"])
        self.assertEqual((args.cipt_selector_mode, args.cipt_entropy_topk), ("entropy", 17))
        import yaml
        config = yaml.safe_load((base.ROOT / "config_cipt_b5b_entropy.yaml").read_text())
        self.assertEqual(config, {"cipt_template_mode": "b5b", "cipt_selector_mode": "entropy",
                                  "cipt_entropy_topk": 40})
        from sconf import Config
        with (base.ROOT / "config.yaml").open() as defaults, \
                (base.ROOT / "config_cipt_sigm_mcc.yaml").open() as objective, \
                (base.ROOT / "config_cipt_b5b_entropy.yaml").open() as overlay:
            merged = Config(defaults, objective, overlay)
        apply_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                          and n.name == "apply_explicit_hparam_cli")
        namespace = {}
        exec(compile(ast.Module(body=[apply_node], type_ignores=[]), "cli", "exec"), namespace)
        apply = namespace["apply_explicit_hparam_cli"]
        apply(parser, parser.parse_args([]), merged, [])
        self.assertEqual(merged.cipt_selector_mode, "entropy")
        apply(parser, args, merged, ["--cipt_selector_mode", "entropy", "--cipt_entropy_topk", "17"])
        self.assertEqual(merged.cipt_entropy_topk, 17)


if __name__ == "__main__":
    unittest.main()
