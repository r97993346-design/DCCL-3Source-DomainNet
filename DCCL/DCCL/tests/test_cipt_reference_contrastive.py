"""CPU loss/gradient and training-route checks; no CLIP download or dataset.

Run from DCCL/DCCL: python -m unittest discover -s tests -v
The integration fixture replaces the frozen CLIP/text encoders with tiny
modules, while executing the real TDA, prompt sampling and update wrappers.
It is not a substitute for a GPU experiment with pretrained CLIP.
"""

import contextlib
import importlib.util
import io
import itertools
from pathlib import Path
import sys
import types
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


losses = load_file("reference_test_losses", ALGORITHMS / "cipt_losses.py")
modules = load_file("reference_test_modules", ALGORITHMS / "cipt_modules.py")


def scalar_definition(features, labels, scores=None, floor=0.1, temperature=0.1):
    """Independent per-anchor definition, including the original uniform loss."""
    u = F.normalize(features.float(), dim=-1)
    values = []
    for i in range(len(labels)):
        other = [j for j in range(len(labels)) if j != i]
        positive = [j for j in other if labels[j] == labels[i]]
        if not positive:
            continue
        denominator = torch.logsumexp(torch.stack([
            torch.dot(u[i], u[j]) / temperature for j in other
        ]), dim=0)
        weights = torch.ones(len(positive)) if scores is None else (
            floor + (1 - floor) * scores.detach()[positive]
        )
        terms = torch.stack([
            denominator - torch.dot(u[i], u[j]) / temperature for j in positive
        ])
        values.append(torch.dot(weights / weights.sum(), terms))
    return torch.stack(values).mean() if values else features.sum() * 0


class ReferenceLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.e = torch.randn(8, 7, requires_grad=True)
        self.y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3])

    def test_reliability_counts_correct_labels_and_detaches(self):
        predictions = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        logits = F.one_hot(predictions, 3).float().requires_grad_()
        scores = losses.intervention_reference_reliability(logits, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(scores, torch.tensor([1.0, 0.25, 0.0]))
        self.assertFalse(scores.requires_grad)

    def test_confidence_uses_true_class_not_maximum_probability(self):
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
        scores = losses.confidence_reference_reliability(logits, torch.tensor([0, 1]))
        torch.testing.assert_close(scores, torch.full((2,), 1 / (1 + torch.exp(torch.tensor(2.0)))))
        self.assertFalse(scores.requires_grad)

    def test_uniform_loss_and_gradient_match_original_definition(self):
        actual, _ = losses.reference_weighted_causal_contrastive_loss(self.e, self.y)
        expected = scalar_definition(self.e, self.y)
        torch.testing.assert_close(actual, expected)
        grad_actual = torch.autograd.grad(actual, self.e, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected, self.e)[0]
        torch.testing.assert_close(grad_actual, grad_expected, atol=2e-6, rtol=2e-5)

    def test_only_reference_weights_change_not_anchor_average(self):
        scores = torch.tensor([0., 1., 0.25, 0., 0.5, 1., 0., 1.], requires_grad=True)
        actual, metrics = losses.reference_weighted_causal_contrastive_loss(self.e, self.y, scores)
        expected = scalar_definition(self.e, self.y, scores)
        torch.testing.assert_close(actual, expected)
        grad_actual = torch.autograd.grad(actual, self.e, retain_graph=True)[0]
        grad_expected = torch.autograd.grad(expected, self.e, retain_graph=True)[0]
        torch.testing.assert_close(grad_actual, grad_expected, atol=2e-6, rtol=2e-5)
        self.assertIsNone(torch.autograd.grad(actual, scores, allow_unused=True)[0])
        self.assertGreater(grad_actual[0].norm().item(), 0)
        self.assertEqual(metrics["valid_anchor_fraction"].item(), 0.75)
        self.assertLess(metrics["irc_positive_ess_fraction"].item(), 1)

    def test_all_zero_scores_keep_anchors_and_recover_uniform(self):
        weighted, metrics = losses.reference_weighted_causal_contrastive_loss(
            self.e, self.y, torch.zeros(8))
        uniform, _ = losses.reference_weighted_causal_contrastive_loss(self.e, self.y)
        torch.testing.assert_close(weighted, uniform)
        weighted.backward()
        self.assertTrue(torch.isfinite(self.e.grad).all())
        self.assertGreater(self.e.grad.norm().item(), 0)
        self.assertEqual(metrics["irc_weighted_anchor_fraction"].item(), 0)

    def test_single_positive_cannot_be_reweighted(self):
        y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        actual, metrics = losses.reference_weighted_causal_contrastive_loss(
            self.e, y, torch.linspace(0, 1, 8))
        expected, _ = losses.reference_weighted_causal_contrastive_loss(self.e, y)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metrics["irc_positive_ess_fraction"].item(), 1)
        self.assertEqual(metrics["irc_weighted_anchor_fraction"].item(), 0)

    def test_empty_singleton_and_missing_positives_are_differentiable_zero(self):
        for batch in (0, 1, 5):
            with self.subTest(batch=batch):
                e = torch.randn(batch, 4, requires_grad=True)
                loss, metrics = losses.reference_weighted_causal_contrastive_loss(e, torch.arange(batch))
                self.assertEqual(loss.item(), 0)
                self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values()))
                loss.backward()
                torch.testing.assert_close(e.grad, torch.zeros_like(e))

    def test_small_temperature_and_half_input_stay_finite(self):
        e = self.e.detach().half().requires_grad_()
        loss, _ = losses.reference_weighted_causal_contrastive_loss(e, self.y, temperature=0.001)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(e.grad).all())

    def test_permutation_does_not_change_objective(self):
        scores = torch.rand(8)
        permutation = torch.randperm(8)
        first, _ = losses.reference_weighted_causal_contrastive_loss(self.e, self.y, scores)
        second, _ = losses.reference_weighted_causal_contrastive_loss(
            self.e[permutation], self.y[permutation], scores[permutation])
        torch.testing.assert_close(first, second)

    def test_invalid_shapes_and_hyperparameters_fail_clearly(self):
        for kwargs in ({"temperature": 0}, {"temperature": float("nan")},
                       {"reference_floor": 0}, {"reference_floor": 1.1},
                       {"reference_reliability": torch.ones(3)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                losses.reference_weighted_causal_contrastive_loss(self.e, self.y, **kwargs)


@contextlib.contextmanager
def tiny_training_route():
    """Import the real wrappers and prompt sampling with a tiny frozen encoder."""
    old_path = list(sys.path)
    try:
        with mock.patch.dict(sys.modules, {"clip": types.ModuleType("clip")}):
            prompts = load_file("reference_test_prompts", ALGORITHMS / "cipt_prompt.py")
    finally:
        sys.path[:] = old_path

    class TinyTextFeatures(prompts.CIPTTextFeatures):
        def __init__(self, classes, dim, k):
            nn.Module.__init__(self)
            self.k = k
            self.template_mode = "b5a"
            self.class_vectors = nn.Parameter(torch.randn(classes, dim))
            self.register_buffer("b5a_text_bank", torch.randn(4, dim))
            self.register_buffer("b5c_text_bank", torch.randn(80, dim))
            self.register_buffer("b5b_text_bank", torch.randn(classes, 80, dim))

        def class_features(self):
            return F.normalize(self.class_vectors, dim=-1)

    class TinyBase(nn.Module):
        def __init__(self, input_shape, num_classes, num_domains, hparams):
            super().__init__()
            dim = input_shape[0]
            self.hparams = hparams
            self.clip_model = nn.Linear(dim, dim)
            self.clip_model.register_parameter("logit_scale", nn.Parameter(torch.tensor(1.0)))
            self.clip_model.requires_grad_(False)
            self.causal_decomposition = modules.CausalDecomposition(dim)
            self.tda = modules.TextDiversityAugmentation(dim)
            self.text_features = TinyTextFeatures(num_classes, dim, hparams["cipt_k"])
            self.proj_head = nn.Linear(dim, dim)
            self.pre_proj_head = nn.Linear(dim, dim)
            self.reg_log_variance = nn.Parameter(torch.zeros(dim))
            self.beta, self.gamma = 4., 5.
            self.debug_shapes = False
            self.visual_calls = 0

        def _visual(self, x):
            self.visual_calls += 1
            with torch.no_grad():
                return self.clip_model(x).float()

        def _logits(self, embeddings, class_features):
            return self.clip_model.logit_scale.exp().detach() * torch.einsum(
                "bkd,cd->bkc", F.normalize(embeddings, dim=-1), class_features)

        def predict(self, x):
            e, _ = self.causal_decomposition(self._visual(x))
            z = self.tda(e, self.text_features.irrelevant_text_features)
            return self._logits(z, self.text_features.class_features()).mean(1)

    package = types.ModuleType("domainbed")
    package.__path__ = [str(ROOT / "domainbed")]
    subpackage = types.ModuleType("domainbed.algorithms")
    subpackage.__path__ = [str(ALGORITHMS)]
    base_module = types.ModuleType("domainbed.algorithms.algorithms")
    base_module.CIPTDCCL = TinyBase
    with mock.patch.dict(sys.modules, {
        "domainbed": package, "domainbed.algorithms": subpackage,
        "domainbed.algorithms.algorithms": base_module,
        "domainbed.algorithms.cipt_losses": losses,
    }):
        parent = load_file("domainbed.algorithms.cipt_dccl_ablation", ALGORITHMS / "cipt_dccl_ablation.py")
        sys.modules["domainbed.algorithms.cipt_dccl_ablation"] = parent
        wrapper = load_file("reference_test_wrapper", ALGORITHMS / "cipt_dccl_component_ablation.py")
        yield wrapper.CIPTDCCL, parent.CIPTDCCL


class TrainingRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = tiny_training_route()
        cls.algorithm, cls.parent_algorithm = cls.context.__enter__()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        torch.set_num_threads(cls.old_threads)

    def make_model(self, algorithm=None, **overrides):
        torch.manual_seed(23)
        hparams = dict(cipt_k=4, cipt_tda_heads=1, cipt_template_mode="b5c",
                       cipt_causal_contrastive_weight=1., cipt_contrastive_warmup_steps=2,
                       optimizer="sgd", lr=0.01, weight_decay=0., t=0.1)
        hparams.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return (algorithm or self.algorithm)((8,), 3, 3, hparams)

    def batch(self):
        return [torch.randn(3, 8) for _ in range(3)], [torch.arange(3) for _ in range(3)]

    def test_all_reference_and_template_modes_reuse_one_tda_forward(self):
        for reference, template in itertools.product(("intervention", "confidence", "uniform"),
                                                       ("b5a", "b5b", "b5c")):
            with self.subTest(reference=reference, template=template):
                model = self.make_model(cipt_contrastive_reference=reference, cipt_template_mode=template)
                with mock.patch.object(model.tda, "forward", wraps=model.tda.forward) as forward:
                    result = model.update(*self.batch())
                self.assertEqual(forward.call_count, 1)
                self.assertEqual(model.visual_calls, 1)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
                self.assertEqual(result["contrastive_weight_eff"], 0.5)
                self.assertEqual(result["irc_intervention_count"], 4 if reference == "intervention" else 0)
                self.assertFalse(hasattr(model, "proj_head"))

    def test_all_independent_component_switches(self):
        for de, ind, tda, con in itertools.product((False, True), repeat=4):
            with self.subTest(de=de, ind=ind, tda=tda, con=con):
                model = self.make_model(cipt_use_de=de, cipt_use_ind=ind,
                                        cipt_use_tda=tda, cipt_use_contrastive=con)
                with mock.patch.object(model.tda, "forward", wraps=model.tda.forward) as forward:
                    result = model.update(*self.batch())
                self.assertEqual(forward.call_count, int(tda))
                self.assertEqual(result["irc_tda_fallback"], float(con and not tda))
                self.assertEqual(result["irc_intervention_active"], float(con and tda))
                self.assertEqual(model._causal_contrastive_step.item(), int(con))
                self.assertTrue(torch.isfinite(torch.tensor(result["total_loss"])))

    def test_confidence_control_works_without_de_or_tda(self):
        for tda in (False, True):
            model = self.make_model(cipt_contrastive_reference="confidence", cipt_use_de=False, cipt_use_tda=tda)
            result = model.update(*self.batch())
            self.assertGreater(result["irc_reliability_mean"], 0)
            self.assertEqual(result["irc_tda_fallback"], 0)

    def test_contrastive_gradients_reach_only_causal_adapter(self):
        model = self.make_model()
        x, y = self.batch()
        labels = torch.cat(y)
        e, s = model.causal_decomposition(model._visual(torch.cat(x)))
        text = model.text_features.class_features()
        z = model.tda(e, model._intervention_features(labels))
        logits = model._logits(z, text)
        loss, _ = model._causal_reference_contrastive(e, labels, logits)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("causal_decomposition.causal_adapter."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.norm().item(), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_pure_zero_weight_and_warmup_behavior(self):
        for overrides in ({"cipt_pure": True}, {"cipt_causal_contrastive_weight": 0.}):
            model = self.make_model(**overrides)
            with mock.patch.object(model, "_causal_reference_contrastive", side_effect=AssertionError):
                result = model.update(*self.batch())
            self.assertEqual(result["dccl_contrastive_loss"], 0)
            self.assertEqual(model._causal_contrastive_step.item(), 0)
        model = self.make_model()
        weights = [model.update(*self.batch())["contrastive_weight_eff"] for _ in range(3)]
        self.assertEqual(weights, [0.5, 1., 1.])
        model = self.make_model(cipt_contrastive_warmup_steps=0)
        self.assertEqual(model.update(*self.batch())["contrastive_weight_eff"], 1.)

    def test_reference_modes_do_not_change_prediction(self):
        for template in ("b5a", "b5b", "b5c"):
            models = [self.make_model(cipt_template_mode=template, cipt_contrastive_reference=ref).eval()
                      for ref in ("intervention", "confidence", "uniform")]
            x = torch.randn(2, 8)
            with torch.no_grad():
                expected = models[0].predict(x)
                for model in models[1:]:
                    with mock.patch.object(model, "_causal_reference_contrastive", side_effect=AssertionError):
                        torch.testing.assert_close(model.predict(x), expected)

    def test_direct_parent_update_also_uses_new_loss(self):
        model = self.make_model(algorithm=self.parent_algorithm)
        result = model.update(*self.batch())
        self.assertEqual(result["irc_intervention_active"], 1)
        self.assertEqual(model.visual_calls, 1)


if __name__ == "__main__":
    unittest.main()
