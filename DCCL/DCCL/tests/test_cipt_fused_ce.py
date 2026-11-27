"""Fused label supervision: objective, gradient paths, switches and config."""

import contextlib
from unittest import mock
import unittest

import torch
import torch.nn.functional as F

import test_cipt_env_kl as fixture


class FusedClassificationTests(unittest.TestCase):
    def build(self, **overrides):
        params = dict(cipt_fused_cls_weight=1., cipt_env_kl_weight=0.,
                      cipt_use_ind=False, cipt_use_contrastive=False)
        params.update(overrides)
        return fixture.EnvironmentUpdateTests().build(**params)

    def batches(self):
        return fixture.EnvironmentUpdateTests().batches()

    def test_per_environment_label_ce_and_weighted_total(self):
        torch.manual_seed(23)
        model = self.build(cipt_e_cls_weight=1.3, cipt_s_uniform_weight=.6,
                           cipt_fused_cls_weight=.7, cipt_env_kl_weight=.2)
        x, y = self.batches()
        with torch.no_grad():
            visual = model._visual(torch.cat(x))
            e, _ = model.causal_decomposition(visual)
            classes = model.text_features.class_features()
            z, _ = model._select_interventions(visual, e, classes, labels=None)
            logits = model._logits(z, classes)
            targets = torch.cat(y)[:, None, None].expand(-1, logits.shape[1], 1)
            expected_ce = -F.log_softmax(logits, -1).gather(-1, targets).mean()
        with mock.patch.object(model, "_select_interventions", wraps=model._select_interventions) as select:
            metrics = model.update(x, y)
        self.assertEqual(select.call_count, 1)  # CE and MCC share one fusion pass.
        self.assertIsNone(select.call_args.kwargs["labels"])
        self.assertAlmostEqual(metrics["fused_cls_loss"], expected_ce.item(), places=6)
        expected_base = (1.3 * metrics["e_cls_loss"] + .6 * metrics["s_uni_loss"]
                         + .7 * expected_ce.item())
        self.assertAlmostEqual(metrics["cipt_base_loss"], expected_base, places=5)
        self.assertAlmostEqual(metrics["total_loss"],
                               expected_base + .2 * metrics["env_kl_loss"], places=5)
        self.assertAlmostEqual(metrics["fused_cls_term"], .7 * expected_ce.item(), places=6)
        self.assertEqual(metrics["fused_cls_w"], .7)
        self.assertEqual(metrics["env_kl_active"], 1.)

    def test_fused_ce_alone_trains_fusion_e_and_class_text_with_k_one(self):
        for fusion in ("tda", "sigm"):
            with self.subTest(fusion=fusion):
                torch.manual_seed(41)
                model = self.build(cipt_fusion_mode=fusion, cipt_k=1,
                                   cipt_e_cls_weight=0., cipt_s_uniform_weight=0.)
                # K=1 is valid for label CE; pairwise MCC must never execute.
                with mock.patch.dict(model.update.__globals__, {
                    "prompt_consistency_loss": mock.Mock(side_effect=AssertionError("MCC disabled")),
                }):
                    metrics = model.update(*self.batches())
                self.assertEqual(metrics["prompt_count"], 1.)
                self.assertEqual(metrics["env_active"], 1.)
                self.assertEqual(metrics["env_kl_active"], 0.)
                self.assertEqual(metrics["env_kl_loss"], 0.)
                self.assertGreater(metrics["fused_cls_loss"], 0.)
                self.assertAlmostEqual(metrics["total_loss"], metrics["fused_cls_loss"], places=6)
                self.assertGreater(model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(model.text_features.classes.grad.abs().sum().item(), 0.)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0.
                                    for p in model.tda.parameters()))
                self.assertIsNone(model.encoder.weight.grad)
                self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)

    def test_master_switch_disables_fused_ce_and_mcc(self):
        model = self.build(cipt_use_tda=False, cipt_env_kl_weight=1.,
                           cipt_s_uniform_weight=0.)
        with mock.patch.object(model, "_select_interventions", side_effect=AssertionError("fusion disabled")):
            metrics = model.update(*self.batches())
        self.assertEqual(metrics["fused_cls_loss"], 0.)
        self.assertEqual(metrics["fused_cls_w"], 0.)
        self.assertEqual(metrics["fused_cls_term"], 0.)
        self.assertEqual(metrics["env_active"], 0.)
        self.assertEqual(metrics["fused_cls_active"], 0.)
        self.assertAlmostEqual(metrics["total_loss"], metrics["e_cls_loss"], places=6)

    def test_zero_ce_and_zero_mcc_skip_fusion(self):
        model = self.build(cipt_fused_cls_weight=0.)
        with mock.patch.object(model, "_select_interventions", side_effect=AssertionError("no environment loss")):
            metrics = model.update(*self.batches())
        self.assertEqual(metrics["prompt_count"], 0.)
        self.assertEqual(metrics["env_active"], 0.)

    def test_mcc_still_requires_two_prompts(self):
        with self.assertRaises(ValueError):
            self.build(cipt_k=1, cipt_env_kl_weight=.1)

    def test_invalid_ce_weights_and_class_conditioned_bank_are_rejected(self):
        for value in (-1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(cipt_fused_cls_weight=value)
        with self.assertRaises(ValueError):
            self.build(cipt_template_mode="b5b", cipt_selector_mode="random")

    def test_e_only_inference_with_fused_training_keeps_direct_e_path(self):
        model = self.build(cipt_env_inference="e_only")
        model.eval()
        images = self.batches()[0][0]
        with torch.no_grad():
            e, _ = model.causal_decomposition(model._visual(images))
            expected = model._logits(e[:, None], model.text_features.class_features())[:, 0]
            with mock.patch.object(model, "_select_interventions", side_effect=AssertionError("e_only inference")):
                actual = model.predict(images)
        torch.testing.assert_close(actual, expected)


class FusedConfigurationTests(unittest.TestCase):
    def test_preset_and_cli_control_ce_and_mcc_independently(self):
        from sconf import Config

        with contextlib.ExitStack() as stack:
            configs = [stack.enter_context((fixture.ROOT / name).open()) for name in (
                "config.yaml", "config_cipt_sigm_mcc.yaml", "config_cipt_fused_ce.yaml")]
            config = Config(*configs)
        self.assertEqual(config.cipt_objective, "env_kl")
        self.assertEqual(config.cipt_fusion_mode, "sigm")
        self.assertEqual(config.cipt_env_inference, "e_only")
        self.assertEqual(config.cipt_e_cls_weight, 1.)
        self.assertEqual(config.cipt_fused_cls_weight, 1.)
        self.assertEqual(config.cipt_env_kl_weight, 0.)
        self.assertTrue(config.cipt_use_tda)
        config.argv_update(["--cipt_fused_cls_weight", "0.5", "--cipt_env_kl_weight", "0.1"])
        self.assertEqual(config.cipt_fused_cls_weight, .5)
        self.assertEqual(config.cipt_env_kl_weight, .1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
