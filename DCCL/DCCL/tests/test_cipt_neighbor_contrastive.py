"""CPU tests; no CLIP weights or datasets required.

The update smoke tests execute the real algorithm classes with a small frozen
encoder standing in for CLIP. They are not PACS/VLCS performance experiments.
"""

import ast
import contextlib
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ROOT / "domainbed" / "algorithms"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


neighbor = load_file("neighbor_loss", ALGORITHMS / "cipt_neighbor_contrastive.py")
losses = load_file("base_losses", ALGORITHMS / "cipt_losses.py")
modules = load_file("base_modules", ALGORITHMS / "cipt_modules.py")
optimizer = load_file("base_optimizer", ROOT / "domainbed" / "optimizers.py")


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = nn.Parameter(torch.eye(4)[:2].clone())
        self.register_buffer("irrelevant_text_features", torch.randn(2, 4))

    def set_template_mode(self, mode):
        self.mode = mode

    def set_neutral_subject(self, subject):
        self.neutral_subject = subject

    def intervention_features(self, labels=None):
        if labels is None:
            return self.irrelevant_text_features
        return self.irrelevant_text_features[None, :, :].expand(
            labels.shape[0], -1, -1
        )

    def full_intervention_features(self):
        bank = self.irrelevant_text_features
        return torch.cat((bank, bank.roll(1, dims=1), -bank), dim=0)

    def class_features(self):
        return F.normalize(self.classes, dim=-1)


class TinyBase(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.encoder.weight.requires_grad_(False)
        self.causal_decomposition = modules.CausalDecomposition(4)
        self.text_features = TinyText()
        self.tda = modules.TextDiversityAugmentation(4, 1)
        self.beta, self.gamma = hparams["cipt_beta"], hparams["cipt_gamma"]
        self.debug_shapes = False
        self.visual_calls = 0

    def _visual(self, images):
        self.visual_calls += 1
        with torch.no_grad():
            return self.encoder(images).float()

    def _logits(self, features, text):
        return 2 * F.normalize(features, dim=-1) @ text.t()

    def predict(self, images):
        e, _ = self.causal_decomposition(self._visual(images))
        z = self.tda(e, self.text_features.irrelevant_text_features)
        return self._logits(z, self.text_features.class_features()).mean(1)


NAMESPACE = {
    "math": math, "torch": torch, "F": F, "_BaseCIPTDCCL": TinyBase,
    "get_optimizer": optimizer.get_optimizer,
    "cipt_classification_loss": losses.classification_loss,
    "cipt_decomposition_loss": losses.decomposition_loss,
    "cipt_independence_loss": losses.independence_loss,
    "empty_neighbor_stats": neighbor.empty_neighbor_stats,
    "neighbor_retention_weights": neighbor.neighbor_retention_weights,
    "validate_neighbor_options": neighbor.validate_neighbor_options,
    "CLASS_CONDITIONED_TDA_MODES": {"b5b", "bconst"},
    "SELECTABLE_TDA_MODES": {"b5a", "b5c"},
    "SafeDiversePromptSelector": modules.SafeDiversePromptSelector,
}


def load_algorithm_class(filename, parent):
    # Avoid importing unrelated CUDA / torchvision algorithms. Execute the
    # actual class body, replacing only the heavy pretrained-model parent.
    tree = ast.parse((ALGORITHMS / filename).read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CIPTDCCL")
    namespace = dict(NAMESPACE, _BaseCIPTDCCL=parent)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), filename, "exec"), namespace)
    return namespace["CIPTDCCL"]


SingleView = load_algorithm_class("cipt_dccl_ablation.py", TinyBase)
Component = load_algorithm_class("cipt_dccl_component_ablation.py", SingleView)


def toy_features(b):
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 0, 1, 1])
    features = torch.zeros(4, 4)
    features[torch.arange(4), labels] = 1
    features[:, 2] = b * torch.tensor([1., .9, -1., .8])
    return features, labels, domains


def per_anchor_ce(features, labels, temperature):
    """Independent specification: average CE over each positive target."""
    z = F.normalize(features.float(), dim=-1)
    rows = []
    for i in range(len(labels)):
        others = torch.arange(len(labels), device=labels.device) != i
        logits = z[i] @ z[others].t() / temperature
        positives = torch.nonzero(labels[others] == labels[i]).flatten()
        if positives.numel():
            rows.append(torch.stack([
                F.cross_entropy(logits[None, :], p.reshape(1)) for p in positives
            ]).mean())
    return torch.stack(rows)


class NeighborhoodTests(unittest.TestCase):
    def test_known_degradation_weights_and_detachment(self):
        v, y, d = toy_features(.25)
        e, _, _ = toy_features(2.)
        v.requires_grad_()
        e.requires_grad_()
        w, stats = neighbor.neighbor_retention_weights(v, e, y, d, k=1, alpha=.5)
        torch.testing.assert_close(w, torch.tensor([1.5, 1., 1., 1.]))
        self.assertFalse(w.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))
        self.assertEqual(stats["nbr_pur_v"].item(), 1.)
        self.assertEqual(stats["nbr_pur_e"].item(), .75)
        self.assertEqual(stats["nbr_drop"].item(), .25)

    def test_equal_features_and_improvement_keep_unit_weights(self):
        v, y, d = toy_features(.25)
        e, _, _ = toy_features(2.)
        for first, second in ((v, v), (e, v)):
            w, _ = neighbor.neighbor_retention_weights(first, second, y, d, k=1)
            torch.testing.assert_close(w, torch.ones(4))

    def test_domain_average_and_clip_after_average(self):
        # Anchor 0 loses its correct nearest neighbor in domain 1. Domain 2
        # has more gallery samples, but must receive the same domain weight.
        v = torch.tensor([[1., 0.], [1., .1], [0., 1.], [0., 1.],
                          [1., .1], [1., .2], [1., .3], [1., .4]])
        e = v.clone()
        e[1], e[2] = v[2], v[1]
        y = torch.tensor([0, 0, 1, 0, 1, 1, 1, 1])
        d = torch.tensor([0, 1, 1, 2, 2, 2, 2, 2])
        w, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=1)
        self.assertEqual(w[0].item(), 1.25)
        e[3] = torch.tensor([1., 0.])  # domain 2 improves: net gap becomes zero
        w, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=1)
        self.assertEqual(w[0].item(), 1.)

    def test_same_domain_candidates_do_not_set_anchor_weight(self):
        v, y, d = toy_features(.25)
        e, _, _ = toy_features(2.)
        before, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=1)
        v[1], e[1] = v[0], e[0]  # change another image in anchor 0's own domain
        after, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=1)
        self.assertEqual(before[0].item(), after[0].item())

    def test_degenerate_galleries_fall_back_to_one(self):
        v, y, d = toy_features(.25)
        for labels, domains in ((y, torch.zeros_like(d)), (torch.zeros_like(y), d),
                                 (torch.tensor([0, 0, 1, 1]), d), (torch.arange(4), d)):
            w, stats = neighbor.neighbor_retention_weights(v, -v, labels, domains)
            torch.testing.assert_close(w, torch.ones(4))
            self.assertEqual(stats["nbr_cover"].item(), 0.)
        # k larger than each gallery safely uses all gallery members.
        e, _, _ = toy_features(2.)
        w, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=100)
        torch.testing.assert_close(w, torch.ones(4))
        for size in (0, 1):
            w, stats = neighbor.neighbor_retention_weights(v[:size], v[:size], y[:size], d[:size])
            self.assertEqual(w.numel(), size)
            self.assertTrue(all(torch.isfinite(t).all() for t in stats.values()))

    def test_permutation_and_scale_invariance(self):
        torch.manual_seed(19)
        v, e = torch.randn(12, 7), torch.randn(12, 7)
        y, d = torch.tensor([0, 0, 1, 1] * 3), torch.arange(3).repeat_interleave(4)
        w, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=2)
        order = torch.randperm(12)
        changed, _ = neighbor.neighbor_retention_weights(v[order] * 3, e[order] * 7, y[order], d[order], k=2)
        torch.testing.assert_close(changed, w[order])
        self.assertTrue(((w >= 1) & (w <= 1.5)).all())

    def test_invalid_options(self):
        for k, alpha in ((0, .5), (2.5, .5), (True, .5), (5, -1), (5, float("nan"))):
            with self.assertRaises(ValueError):
                neighbor.validate_neighbor_options(k, alpha)


class LossTests(unittest.TestCase):
    def test_original_denominator_positives_and_weighted_gradient(self):
        torch.manual_seed(4)
        e = torch.randn(6, 4, requires_grad=True)
        y = torch.tensor([0, 0, 0, 1, 1, 2])  # one invalid anchor remains a negative
        w = torch.tensor([1.2, 1., 1.5, 1., 1.3, 1.], requires_grad=True)
        obj = SimpleNamespace(contrastive_temperature=.1)
        actual, fraction = SingleView._single_view_supcon(obj, e, y, w)
        expected = (per_anchor_ce(e, y, .1) * w.detach()[:5]).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, e, retain_graph=True)[0],
                                   torch.autograd.grad(expected, e)[0])
        self.assertAlmostEqual(fraction.item(), 5 / 6, places=6)
        self.assertIsNone(torch.autograd.grad(actual, w, allow_unused=True)[0])

    def test_alpha_zero_is_exact_supcon_in_value_and_gradient(self):
        v, y, d = toy_features(.25)
        e, _, _ = toy_features(2.)
        e.requires_grad_()
        w, _ = neighbor.neighbor_retention_weights(v, e, y, d, k=1, alpha=0.)
        obj = SimpleNamespace(contrastive_temperature=.1)
        original, _ = SingleView._single_view_supcon(obj, e, y)
        weighted, _ = SingleView._single_view_supcon(obj, e, y, w)
        self.assertTrue(torch.equal(original, weighted))
        self.assertTrue(torch.equal(torch.autograd.grad(original, e, retain_graph=True)[0],
                                    torch.autograd.grad(weighted, e)[0]))

    def test_no_positive_zero_is_differentiable(self):
        for size in (0, 1, 3):
            e = torch.randn(size, 4, requires_grad=True)
            loss, valid = SingleView._single_view_supcon(
                SimpleNamespace(contrastive_temperature=.1), e, torch.arange(size))
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(valid.item(), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(e.grad).all())


class UpdateTests(unittest.TestCase):
    def build(self, cls=Component, **overrides):
        params = dict(cipt_pure=False, cipt_template_mode="b5a",
                      cipt_neutral_subject="subject", cipt_k=2,
                      cipt_tda_heads=1, cipt_beta=4., cipt_gamma=5.,
                      cipt_contrastive_type="neighbor_retention", cipt_neighbor_k=1,
                      cipt_neighbor_alpha=.5, cipt_neighbor_diagnostics=False,
                      cipt_causal_contrastive_weight=1., cipt_contrastive_warmup_steps=500,
                      cipt_contrastive_temperature=.1,
                      t=.2, optimizer="adam", lr=1e-3, weight_decay=0.)
        params.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return cls((4,), 2, 2, params)

    def batches(self):
        v, y, _ = toy_features(.25)
        return [v[:2], v[2:]], [y[:2], y[2:]]

    def test_both_update_paths_and_frozen_encoder(self):
        for cls in (SingleView, Component):
            torch.manual_seed(2)
            model = self.build(cls)
            before = model.causal_decomposition.causal_adapter.weight.detach().clone()
            metrics = model.update(*self.batches())
            self.assertEqual(model.visual_calls, 1)
            self.assertIsNone(model.encoder.weight.grad)
            self.assertFalse(torch.equal(before, model.causal_decomposition.causal_adapter.weight))
            self.assertEqual(metrics["nbr_check"], 1.)
            self.assertEqual(metrics["contrastive_weight_eff"], .002)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))

    def test_diagnostics_work_with_contrastive_off(self):
        model = self.build(cipt_use_contrastive=False, cipt_neighbor_diagnostics=True)
        metrics = model.update(*self.batches())
        self.assertEqual(metrics["dccl_contrastive_loss"], 0.)
        self.assertEqual(metrics["contrastive_weight_eff"], 0.)
        self.assertEqual(model._causal_contrastive_step.item(), 0)
        self.assertEqual(metrics["nbr_check"], 1.)
        self.assertEqual(metrics["nbr_wmean"], 1.)

    def test_unneeded_diagnostics_are_skipped(self):
        for options in (dict(cipt_contrastive_type="supcon"), dict(cipt_neighbor_alpha=0.),
                        dict(cipt_use_contrastive=False), dict(cipt_pure=True),
                        dict(cipt_causal_contrastive_weight=0.)):
            model = self.build(**options)
            with mock.patch.dict(SingleView._causal_contrastive_loss.__globals__, {
                "neighbor_retention_weights": mock.Mock(side_effect=AssertionError("unexpected kNN"))
            }):
                metrics = model.update(*self.batches())
            self.assertEqual(metrics["nbr_check"], 0.)

    def test_standard_supcon_switch_is_preserved(self):
        torch.manual_seed(41)
        standard = self.build(cipt_contrastive_type="supcon")
        torch.manual_seed(41)
        alpha_zero = self.build(
            cipt_contrastive_type="neighbor_retention",
            cipt_neighbor_alpha=0.0,
        )
        standard_metrics = standard.update(*self.batches())
        alpha_zero_metrics = alpha_zero.update(*self.batches())
        self.assertEqual(
            standard_metrics["dccl_contrastive_loss"],
            alpha_zero_metrics["dccl_contrastive_loss"],
        )
        for a, b in zip(standard.parameters(), alpha_zero.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_method_specific_temperature_overrides_generic_t(self):
        model = self.build(cipt_contrastive_temperature=.07, t=.3)
        self.assertEqual(model.contrastive_temperature, .07)

    def test_invalid_contrastive_hparams_fail_early(self):
        for value in (0.0, -0.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.build(cipt_contrastive_temperature=value)
        for value in (-0.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.build(cipt_causal_contrastive_weight=value)

    def test_component_switches_and_inference(self):
        model = self.build(cipt_use_de=False, cipt_use_ind=False, cipt_use_tda=False)
        metrics = model.update(*self.batches())
        self.assertEqual(metrics["cipt_de_loss"], 0.)
        self.assertEqual(metrics["cipt_ind_loss"], 0.)
        self.assertEqual(metrics["nbr_check"], 1.)
        with mock.patch.dict(SingleView._causal_contrastive_loss.__globals__, {
            "neighbor_retention_weights": mock.Mock(side_effect=AssertionError("inference kNN"))
        }):
            prediction = model.predict(self.batches()[0][0])
        self.assertEqual(prediction.shape, (2, 2))

    def test_random_selector_preserves_existing_eval_predictions(self):
        model = self.build(cipt_selector_mode="random", cipt_use_contrastive=False)
        model.eval()
        images = self.batches()[0][0]
        torch.testing.assert_close(model.predict(images), TinyBase.predict(model, images))
        self.assertEqual(model.update(*self.batches())["prompt_selector_active"], 0.)

    def test_adaptive_paired_bank_runs_in_train_and_eval(self):
        for cls in (SingleView, Component):
            model = self.build(cls, cipt_selector_mode="adaptive",
                               cipt_selector_candidates=4, cipt_use_contrastive=False)
            metrics = model.update(*self.batches())
            self.assertEqual(metrics["prompt_selector_active"], 1.)
            self.assertEqual(metrics["prompt_count"], 2.)
            self.assertEqual(metrics["prompt_selector_candidates"], 4.)
            model.eval()
            with mock.patch.object(model.prompt_selector, "select", wraps=model.prompt_selector.select) as select:
                logits = model.predict(self.batches()[0][0])
                select.assert_called_once()
            self.assertEqual(logits.shape, (2, 2))
            torch.testing.assert_close(logits, model.predict(self.batches()[0][0]))

    def test_tda_off_skips_selector_even_if_configured(self):
        model = self.build(cipt_selector_mode="adaptive", cipt_use_tda=False)
        with mock.patch.object(model.prompt_selector, "shortlist",
                               side_effect=AssertionError("selector ran with TDA disabled")):
            self.assertEqual(model.update(*self.batches())["prompt_selector_active"], 0.)
            self.assertEqual(model.predict(self.batches()[0][0]).shape, (2, 2))

    def test_adaptive_selection_rejects_class_conditioned_modes(self):
        for mode in ("b5b", "bconst", "sconst"):
            with self.assertRaisesRegex(ValueError, "b5a or b5c"):
                self.build(cipt_template_mode=mode, cipt_selector_mode="adaptive")

    def test_diagnostics_do_not_change_training(self):
        for contrastive in (False, True):
            models = []
            for diagnostics in (False, True):
                torch.manual_seed(32)
                models.append(self.build(cipt_contrastive_type="supcon",
                                         cipt_use_contrastive=contrastive,
                                         cipt_neighbor_diagnostics=diagnostics))
            metrics = [m.update(*self.batches()) for m in models]
            self.assertEqual(metrics[0]["total_loss"], metrics[1]["total_loss"])
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
