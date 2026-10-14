"""Position ablations using the real CIPT/TDA/selector training methods on CPU.

Only the heavyweight frozen CLIP parent is replaced by the existing tiny
fixture. These tests do not measure PACS/VLCS accuracy or GPU throughput.
"""

import itertools
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_neighbor_contrastive as fixtures
from test_cipt_neighbor_contrastive import (
    Component, SingleView, neighbor, per_anchor_ce, toy_features,
)


class ContrastivePositionTests(unittest.TestCase):
    build = fixtures.UpdateTests.build
    batches = fixtures.UpdateTests.batches

    def make(self, cls=Component, **options):
        options.setdefault("cipt_contrastive_position", "post_tda_mean")
        return self.build(cls, **options)

    def forward_contrastive(self, model):
        images, labels = self.batches()
        visual = model._visual(torch.cat(images))
        causal, _ = model.causal_decomposition(visual)
        interventions, _ = model._select_interventions(
            visual, causal, model.text_features.class_features(),
            labels=torch.cat(labels),
        )
        result = model._causal_contrastive_loss(
            visual, causal, torch.cat(labels), labels,
            interventions=interventions,
        )
        return result, causal, interventions

    def assert_same_parameters(self, first, second):
        self.assertEqual(first.state_dict().keys(), second.state_dict().keys())
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_normalizes_each_intervention_before_mean(self):
        z = torch.tensor([[[3., 0., 0., 0.], [0., 1., 0., 0.]]])
        pooled, coherence = SingleView._pool_post_tda(z)
        torch.testing.assert_close(
            pooled, torch.tensor([[1., 1., 0., 0.]]) / math.sqrt(2)
        )
        torch.testing.assert_close(coherence, torch.tensor([1 / math.sqrt(2)]))
        self.assertFalse(torch.allclose(pooled, F.normalize(z.mean(1), dim=-1)))
        # Permuting the prompt axis or rescaling a prompt must not change its
        # contribution to the mean direction used by the classifier.
        rescaled = z[:, [1, 0]] * torch.tensor([[[7.], [.2]]])
        torch.testing.assert_close(SingleView._pool_post_tda(rescaled)[0], pooled)

    def test_post_space_sets_weights_and_matches_independent_loss_gradient(self):
        visual, labels, domains = toy_features(.25)
        visual.requires_grad_()
        causal = visual.detach().clone().requires_grad_()
        changed, _, _ = toy_features(2.)
        z = changed[:, None].repeat(1, 2, 1).requires_grad_()
        batches = [labels[:2], labels[2:]]
        model = self.make(cipt_contrastive_warmup_steps=0)
        loss, valid, scale, stats = model._causal_contrastive_loss(
            visual, causal, labels, batches, interventions=z,
        )
        pooled = F.normalize(F.normalize(z, dim=-1).mean(1), dim=-1)
        weights, _ = neighbor.neighbor_retention_weights(
            visual, pooled, labels, domains, k=1, alpha=.5,
        )
        expected = (per_anchor_ce(pooled, labels, .1) * weights).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(loss, z, retain_graph=True)[0],
            torch.autograd.grad(expected, z, retain_graph=True)[0],
        )
        # E equals V, so comparing V/E by mistake would produce purity 1 and
        # unit weights, even though the post-TDA neighborhood has degraded.
        self.assertEqual(stats["nbr_pur_e"].item(), .75)
        self.assertEqual(stats["nbr_wmean"].item(), 1.125)
        self.assertEqual(stats["con_post_active"].item(), 1.)
        self.assertEqual(valid.item(), 1.)
        self.assertEqual(scale, 1.)
        self.assertFalse(any(value.requires_grad for value in stats.values()))
        loss.backward()
        self.assertIsNone(visual.grad)
        self.assertIsNone(causal.grad)

    def test_contrastive_only_gradients_reach_tda_and_e_but_not_s_or_text(self):
        for position in ("e", "post_tda_mean"):
            for selector in ("random", "all", "adaptive"):
                with self.subTest(position=position, selector=selector):
                    torch.manual_seed(5)
                    model = self.make(
                        cipt_contrastive_position=position,
                        cipt_selector_mode=selector, cipt_selector_candidates=4,
                        cipt_contrastive_type="supcon",
                    )
                    (loss, _, _, _), _, _ = self.forward_contrastive(model)
                    loss.backward()
                    self.assertGreater(
                        model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0.
                    )
                    self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)
                    self.assertIsNone(model.encoder.weight.grad)
                    self.assertIsNone(model.text_features.classes.grad)
                    if position == "post_tda_mean":
                        # With one key per intervention, the value projection
                        # and output projection are the relevant attention path.
                        self.assertGreater(
                            model.tda.attention.in_proj_weight.grad[8:].abs().sum().item(), 0.
                        )
                        self.assertGreater(model.tda.attention.out_proj.weight.grad.abs().sum().item(), 0.)
                    else:
                        self.assertTrue(all(p.grad is None for p in model.tda.parameters()))

    def test_update_reuses_selected_outputs_and_keeps_b_anchors(self):
        for cls, mode, selector, kind in itertools.product(
            (SingleView, Component), ("b5a", "b5c"),
            ("random", "all", "adaptive"), ("supcon", "neighbor_retention"),
        ):
            with self.subTest(cls=cls.__name__, mode=mode, selector=selector, kind=kind):
                model = self.make(
                    cls, cipt_template_mode=mode, cipt_selector_mode=selector,
                    cipt_selector_candidates=4, cipt_contrastive_type=kind,
                )
                with mock.patch.object(model.tda, "forward", wraps=model.tda.forward) as tda:
                    with mock.patch.object(model, "_single_view_supcon", wraps=model._single_view_supcon) as con:
                        metrics = model.update(*self.batches())
                self.assertEqual(model.visual_calls, 1)
                self.assertEqual(tda.call_count, 1)
                self.assertEqual(con.call_args.args[0].shape, (4, 4))
                self.assertEqual(metrics["con_post_active"], 1.)
                self.assertEqual(metrics["con_z_check"], 1.)
                self.assertEqual(metrics["contrastive_weight_eff"], .002)
                self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_class_conditioned_and_constant_modes_train(self):
        for mode in ("b5b", "bconst", "sconst"):
            with self.subTest(mode=mode):
                model = self.make(cipt_template_mode=mode)
                self.assertEqual(model.update(*self.batches())["con_post_active"], 1.)

    def test_all_component_switches_preserve_training(self):
        for de, ind, tda, con in itertools.product((False, True), repeat=4):
            with self.subTest(de=de, ind=ind, tda=tda, con=con):
                model = self.make(
                    cipt_use_de=de, cipt_use_ind=ind,
                    cipt_use_tda=tda, cipt_use_contrastive=con,
                )
                metrics = model.update(*self.batches())
                self.assertEqual(metrics["con_post_active"], float(tda and con))
                self.assertEqual(metrics["con_tda_fallback"], float(con and not tda))
                if not con:
                    self.assertEqual(metrics["dccl_contrastive_loss"], 0.)
                    self.assertEqual(model._causal_contrastive_step.item(), 0)
                self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_tda_off_falls_back_to_exact_e_objective_and_update(self):
        for kind in ("supcon", "neighbor_retention"):
            models = []
            for position in ("e", "post_tda_mean"):
                torch.manual_seed(9)
                models.append(self.make(
                    cipt_use_tda=False, cipt_contrastive_position=position,
                    cipt_contrastive_type=kind, cipt_selector_mode="adaptive",
                ))
            metrics = [model.update(*self.batches()) for model in models]
            self.assertEqual(metrics[0]["dccl_contrastive_loss"], metrics[1]["dccl_contrastive_loss"])
            self.assertEqual(metrics[1]["con_tda_fallback"], 1.)
            self.assertEqual(metrics[1]["con_z_check"], 0.)
            self.assert_same_parameters(*models)

    def test_disabled_paths_skip_pooling_and_match_e_updates(self):
        for options in (
            dict(cipt_use_contrastive=False), dict(cipt_pure=True),
            dict(cipt_causal_contrastive_weight=0.),
        ):
            models = []
            for position in ("e", "post_tda_mean"):
                torch.manual_seed(17)
                model = self.make(cipt_contrastive_position=position, **options)
                with mock.patch.object(model, "_pool_post_tda", side_effect=AssertionError("unused pooling")):
                    metrics = model.update(*self.batches())
                self.assertEqual(metrics["con_z_check"], 0.)
                self.assertEqual(metrics["dccl_contrastive_loss"], 0.)
                models.append(model)
            self.assert_same_parameters(*models)

    def test_alpha_zero_keeps_supcon_loss_and_update_at_new_position(self):
        models = []
        for kind in ("supcon", "neighbor_retention"):
            torch.manual_seed(13)
            models.append(self.make(cipt_contrastive_type=kind, cipt_neighbor_alpha=0.))
        metrics = [model.update(*self.batches()) for model in models]
        self.assertEqual(metrics[0]["dccl_contrastive_loss"], metrics[1]["dccl_contrastive_loss"])
        self.assert_same_parameters(*models)

    def test_diagnostics_measure_post_space_without_changing_update(self):
        for enabled in (False, True):
            models = []
            for diagnostics in (False, True):
                torch.manual_seed(23)
                models.append(self.make(
                    cipt_contrastive_type="supcon", cipt_use_contrastive=enabled,
                    cipt_neighbor_diagnostics=diagnostics,
                ))
            metrics = [model.update(*self.batches()) for model in models]
            self.assertEqual(metrics[1]["nbr_check"], 1.)
            self.assertEqual(metrics[1]["con_z_check"], 1.)
            self.assertEqual(metrics[1]["con_post_active"], float(enabled))
            self.assertEqual(metrics[0]["total_loss"], metrics[1]["total_loss"])
            self.assert_same_parameters(*models)

    def test_position_does_not_change_prediction_or_state_dict(self):
        for selector in ("random", "all", "adaptive"):
            models = []
            for position in ("e", "post_tda_mean"):
                torch.manual_seed(29)
                models.append(self.make(
                    cipt_contrastive_position=position, cipt_template_mode="b5c",
                    cipt_selector_mode=selector, cipt_selector_candidates=4,
                ).eval())
            self.assert_same_parameters(*models)
            # Prediction must never compute training-time contrastive pooling.
            with mock.patch.object(models[1], "_pool_post_tda", side_effect=AssertionError("inference pooling")):
                torch.testing.assert_close(
                    models[0].predict(self.batches()[0][0]),
                    models[1].predict(self.batches()[0][0]),
                )

    def test_single_view_half_precision_and_no_positive_are_finite(self):
        for batch in (0, 1, 3):
            z = torch.randn(batch, 1, 4, dtype=torch.float16, requires_grad=True)
            pooled, coherence = SingleView._pool_post_tda(z)
            self.assertEqual(pooled.dtype, torch.float32)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertTrue(torch.isfinite(coherence).all())
            model = self.make()
            loss, valid = model._single_view_supcon(pooled, torch.arange(batch))
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(valid.item(), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())

    def test_invalid_position_and_missing_or_malformed_outputs_fail(self):
        with self.assertRaisesRegex(ValueError, "cipt_contrastive_position"):
            self.make(cipt_contrastive_position="after_logits")
        model = self.make()
        images, labels = self.batches()
        v = torch.cat(images)
        with self.assertRaisesRegex(ValueError, "requires the selected TDA"):
            model._causal_contrastive_loss(v, v, torch.cat(labels), labels)
        for z in (torch.randn(4, 4), torch.randn(4, 0, 4),
                  torch.randn(3, 2, 4), torch.randn(4, 2, 5)):
            with self.assertRaises(ValueError):
                model._causal_contrastive_loss(v, v, torch.cat(labels), labels, interventions=z)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
