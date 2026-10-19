"""CPU ablations of SIGM/TDA, MCC gradients, and inference aggregation.

Uses the existing CLIP-free fake algorithm fixture. This deliberately does not
claim PACS/VLCS benchmark or SWAD validation results.
"""

import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_kl as fixture


class SIGMModuleTests(unittest.TestCase):
    def test_shared_and_batched_text_modulation_match_formula(self):
        torch.manual_seed(5)
        module = fixture.sigm.SemanticInteractionGuidance(
            4, hidden_dim=6, scale=0.7, init_std=0.2
        )
        e = torch.randn(3, 4, requires_grad=True)
        prompts = torch.randn(5, 4)
        gamma, beta = module.projection(prompts).chunk(2, -1)
        expected = ((1 + 0.7 * gamma[None]) * e[:, None]
                    + 0.7 * beta[None])
        actual = module(e, prompts)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, module(e, prompts[None].expand(3, -1, -1)))
        self.assertEqual(tuple(actual.shape), (3, 5, 4))
        actual.sum().backward()
        self.assertGreater(e.grad.abs().sum().item(), 0)
        self.assertGreater(module.projection[-1].weight.grad.abs().sum().item(), 0)

    def test_invalid_config_and_shapes_fail_fast(self):
        for options in (dict(scale=0), dict(init_std=0), dict(hidden_dim=0)):
            with self.assertRaises(ValueError):
                fixture.sigm.SemanticInteractionGuidance(4, **options)
        m = fixture.sigm.SemanticInteractionGuidance(4)
        for e, txt in (
            (torch.rand(2, 4, 1), torch.rand(3, 4)),
            (torch.rand(2, 4), torch.rand(3, 5)),
            (torch.rand(2, 4), torch.rand(3, 3, 4)),
        ):
            with self.assertRaises(ValueError):
                m(e, txt)

    def test_exact_identity_is_a_consistency_loss_dead_end(self):
        model = fixture.EnvironmentUpdateTests().build(
            cipt_fusion_mode="sigm", cipt_selector_mode="all"
        )
        with torch.no_grad():
            model.tda.projection[-1].weight.zero_()
            model.tda.projection[-1].bias.zero_()
        x, y = fixture.EnvironmentUpdateTests().batches()
        causal, _ = model.causal_decomposition(model._visual(torch.cat(x)))
        z = model.tda(causal, model.text_features.full_intervention_features())
        torch.testing.assert_close(z, causal[:, None, :].expand_as(z))
        kl = fixture.losses.prompt_consistency_loss(
            model._logits(z, model.text_features.class_features())
        )
        self.assertLess(abs(kl.item()), 1e-6)


class SIGMTrainingTests(unittest.TestCase):
    def build(self, **overrides):
        params = dict(cipt_fusion_mode="sigm", cipt_sigm_init_std=0.15)
        params.update(overrides)
        return fixture.EnvironmentUpdateTests().build(**params)

    def test_sigm_params_in_optimizer_and_prompt_only_gradient(self):
        torch.manual_seed(71)
        model = self.build(
            cipt_e_cls_weight=0., cipt_s_uniform_weight=0.,
            cipt_use_ind=False, cipt_use_contrastive=False,
            cipt_env_kl_weight=1., cipt_selector_mode="all"
        )
        params = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in params for p in model.tda.parameters()))
        # Stronger starting offsets in this small synthetic dimension avoid
        # tiny near-zero KL dominated by float32 rounding.
        with torch.no_grad():
            model.tda.projection[-1].weight.normal_(std=.3)
        m = model.update(*fixture.EnvironmentUpdateTests().batches())
        self.assertGreater(m["env_kl_loss"], 0.)
        self.assertAlmostEqual(m["total_loss"], m["env_kl_loss"], places=5)
        self.assertGreater(model.tda.projection[-1].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0.)
        self.assertGreaterEqual(m["env_shift"], 0.)
        self.assertGreater(m["env_l2_shift"], 0.)
        self.assertGreaterEqual(m["env_spread"], 0.)
        self.assertGreaterEqual(m["env_spread_l2"], 0.)
        self.assertGreaterEqual(m["env_pred_agreement"], 0.)
        self.assertLessEqual(m["env_pred_agreement"], 1.)
        self.assertEqual(m["sigm_active"], 1.)
        self.assertEqual(m["inference_fused"], 0.)

    def test_tda_control_keeps_original_operator(self):
        model = fixture.EnvironmentUpdateTests().build(cipt_fusion_mode="tda")
        self.assertNotIsInstance(model.tda, fixture.sigm.SemanticInteractionGuidance)
        self.assertEqual(model.fusion_mode, "tda")
        m = model.update(*fixture.EnvironmentUpdateTests().batches())
        self.assertEqual(m["sigm_active"], 0.)
        self.assertIn("env_kl_loss", m)

    def test_zero_kl_weight_bypasses_fusion_training(self):
        model = self.build(cipt_env_kl_weight=0.)
        with mock.patch.object(model.tda, "forward", side_effect=AssertionError("unexpected fusion")):
            m = model.update(*fixture.EnvironmentUpdateTests().batches())
        self.assertEqual(m["env_active"], 0.)
        self.assertEqual(m["env_kl_loss"], 0.)
        self.assertEqual(m["env_spread"], 0.)

    def test_inference_same_checkpoint_three_aggregation_modes(self):
        model = self.build()
        model.eval()
        x = fixture.EnvironmentUpdateTests().batches()[0][0]
        with torch.no_grad():
            visual = model._visual(x)
            e, _ = model.causal_decomposition(visual)
            class_features = model.text_features.class_features()
            logits_e = model._logits(e[:, None, :], class_features)[:, 0]
            fused, _ = model._select_interventions(visual, e, class_features)
            prompt_logits = model._logits(fused, class_features)
            expected_probs = torch.logsumexp(
                F.log_softmax(prompt_logits, -1), dim=1
            ) - math.log(prompt_logits.shape[1])
            for mode, expected in (
                ("e_only", logits_e),
                ("mean_logits", prompt_logits.mean(1)),
                ("mean_probs", expected_probs),
            ):
                model.inference_mode = mode
                torch.testing.assert_close(model.predict(x), expected, atol=2e-6, rtol=1e-5)

    def test_bad_modes_and_bad_context_at_inference(self):
        for key, value in (
            ("cipt_fusion_mode", "invalid"),
            ("cipt_env_inference", "invalid"),
        ):
            with self.assertRaises(ValueError):
                self.build(**{key: value})
        with self.assertRaises(ValueError):
            self.build(cipt_template_mode="b5b", cipt_env_inference="mean_probs")
        model = self.build(cipt_env_inference="mean_logits")
        self.assertEqual(model.inference_mode, "mean_logits")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

