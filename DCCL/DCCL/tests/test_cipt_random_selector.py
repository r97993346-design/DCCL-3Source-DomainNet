"""Random official-B5b selection and real MCC/RD training integration tests."""

import contextlib
import copy
import json
import math
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_anchor as fixture


RandomPromptSelector = fixture.random_selector.RandomPromptSelector


def build(constraint="relative_distance", **options):
    params = dict(cipt_env_selection="random", cipt_use_ind=False,
                  cipt_use_contrastive=False, cipt_causal_contrastive_weight=0.)
    params.update(options)
    return fixture.build(constraint, **params)


def run_smoke(steps=5):
    report = []
    for constraint in ("mcc", "relative_distance"):
        model = build(constraint)
        before = model.causal_decomposition.causal_adapter.weight.detach().clone()
        averaged = fixture.swa.AveragedModel(model)
        history = []
        selected = []
        handle = model.random_selector.register_forward_hook(
            lambda module, args, output: selected.append(output[0].clone()))
        for step in range(steps):
            metrics = model.update(*fixture.batches())
            assert all(math.isfinite(value) for value in metrics.values())
            assert metrics["entropy_selector_active"] == 0.
            assert metrics["random_selector_active"] == 1.
            assert metrics["env_topk"] == 40.
            assert metrics["dccl_contrastive_loss"] == metrics["contrastive_weight_eff"] == 0.
            averaged.update_parameters(model, step=step)
            history.append({k: metrics[k] for k in (
                "total_loss", "env_loss", "e_cls_loss", "s_uni_loss")})
        handle.remove()
        assert not torch.equal(selected[0], selected[1])
        assert not torch.equal(before, model.causal_decomposition.causal_adapter.weight)
        averaged.eval()
        with torch.no_grad():
            images = fixture.batches()[0][0]
            expected = averaged.predict(images)
            assert expected.shape == (2, 3) and torch.isfinite(expected).all()
            restored = build(constraint)
            restored.load_state_dict(averaged.module.state_dict(), strict=True)
            restored.eval()
            torch.testing.assert_close(restored.predict(images), expected)
        report.append(dict(constraint=constraint, steps=steps,
                           swa_averaged=int(averaged.n_averaged), history=history))
    return report


class RandomSelectorTests(unittest.TestCase):
    def test_per_image_sampling_without_replacement_and_fresh_draws(self):
        visual = torch.ones(4, 8)
        bank = torch.zeros(3, 80, 8)
        torch.manual_seed(15)
        for n in (2, 40, 80):
            indices, metrics = RandomPromptSelector(n)(visual, bank)
            self.assertEqual(tuple(indices.shape), (4, n))
            self.assertEqual(indices.dtype, torch.long)
            self.assertTrue(((indices >= 0) & (indices < 80)).all())
            self.assertTrue(all(len(row.unique()) == n for row in indices))
            self.assertFalse(torch.equal(indices[0], indices[1]))
            self.assertFalse(indices.requires_grad)
            if n == 80:
                torch.testing.assert_close(indices.sort(dim=1).values,
                                           torch.arange(80)[None].expand(4, -1))
            self.assertEqual(metrics["entropy_selector_active"], 0.)
            self.assertEqual(metrics["random_selector_active"], 1.)
            self.assertTrue(all(v.dtype == torch.float32 and not v.requires_grad
                                for v in metrics.values()))
        sampler = RandomPromptSelector()
        first, _ = sampler(visual, bank)
        second, _ = sampler(visual, bank)
        self.assertFalse(torch.equal(first, second))

    def test_feature_values_do_not_influence_selection_and_seed_repeats(self):
        sampler = RandomPromptSelector()
        visual = torch.zeros(3, 8, dtype=torch.float16, requires_grad=True)
        bank = torch.zeros(5, 80, 8, requires_grad=True)
        torch.manual_seed(91)
        first, _ = sampler(visual, bank)
        torch.manual_seed(91)
        changed, _ = sampler(visual + 100, bank + 200)
        torch.testing.assert_close(first, changed)
        torch.manual_seed(92)
        different, _ = sampler(visual, bank)
        self.assertFalse(torch.equal(first, different))
        self.assertIsNone(visual.grad)
        self.assertIsNone(bank.grad)

    def test_invalid_topk_and_input_shapes(self):
        for n in (1, 81, True, 2.5, "40"):
            with self.assertRaises(ValueError):
                RandomPromptSelector(n)
        for visual, bank in ((torch.zeros(0, 8), torch.zeros(3, 80, 8)),
                             (torch.zeros(1, 0), torch.zeros(3, 80, 0)),
                             (torch.zeros(1, 8), torch.zeros(3, 79, 8)),
                             (torch.zeros(1, 8), torch.zeros(3, 80, 7)),
                             (torch.zeros(1, 8), torch.zeros(0, 80, 8))):
            with self.assertRaises(ValueError):
                RandomPromptSelector()(visual, bank)

    def test_labels_and_e_changes_cannot_affect_indices(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint)
            images, labels = fixture.batches()
            visual = model._visual(torch.cat(images))
            e, _ = model.causal_decomposition(visual)
            selected = []
            def capture(module, args, output):
                self.assertEqual(len(args), 2)  # V/bank only; no y, E/S or logits.
                torch.testing.assert_close(args[0], visual)
                self.assertEqual(tuple(args[1].shape), (3, 80, 8))
                selected.append(output[0].clone())
            handle = model.random_selector.register_forward_hook(capture)
            torch.manual_seed(53)
            model._env_anchor_loss(visual, e, torch.cat(labels))
            torch.manual_seed(53)
            model._env_anchor_loss(visual, -e * 9, torch.cat(labels).roll(1))
            handle.remove()
            torch.testing.assert_close(*selected)

    def test_no_entropy_or_old_selectors_and_mcc_uses_e_only(self):
        for constraint in ("mcc", "relative_distance"):
            with mock.patch.dict(fixture.namespace, {
                "EntropyPromptSelector": mock.Mock(side_effect=AssertionError("entropy construction")),
            }):
                model = build(constraint)
            self.assertIsNone(model.entropy_selector)
            selected = []
            e_features = []
            handle = model.random_selector.register_forward_hook(
                lambda module, args, output: selected.append(output[0]))
            e_handle = model.causal_decomposition.register_forward_hook(
                lambda module, args, output: e_features.append(output[0]))
            logits = mock.Mock(wraps=fixture.selector.class_prompt_logits)
            with contextlib.ExitStack() as stack:
                for obj, method in ((model, "_select_interventions"),
                                    (model.text_features, "_select_indices"),
                                    (model.prompt_selector, "shortlist"),
                                    (model.prompt_selector, "select"), (model.tda, "forward")):
                    stack.enter_context(mock.patch.object(obj, method, side_effect=AssertionError(method)))
                stack.enter_context(mock.patch.dict(fixture.namespace, {"class_prompt_logits": logits}))
                metrics = model.update(*fixture.batches())
            handle.remove()
            e_handle.remove()
            self.assertEqual(metrics["entropy_selector_active"], 0.)
            self.assertEqual(metrics["random_selector_active"], 1.)
            self.assertEqual(metrics["prompt_count"], 40.)
            if constraint == "mcc":
                self.assertEqual(logits.call_count, 1)
                self.assertIs(logits.call_args.args[0], e_features[0])
                all_logits = fixture.selector.class_prompt_logits(*logits.call_args.args)
                chosen = all_logits.gather(1, selected[0][..., None].expand(-1, -1, 3))
                self.assertEqual(tuple(chosen.shape), (6, 40, 3))
            else:
                self.assertEqual(logits.call_count, 0)

    def test_all_n_batch_one_and_environment_only_adapter_gradients(self):
        for constraint in ("mcc", "relative_distance"):
            for n in (2, 40, 80):
                for batch_one in (False, True):
                    model = build(constraint, cipt_env_topk=n,
                                  cipt_e_cls_weight=0., cipt_s_uniform_weight=0.)
                    images, labels = fixture.batches()
                    if batch_one:
                        images, labels = [images[0][:1]], [labels[0][:1]]
                    bank_before = model.text_features.b5b_text_bank.clone()
                    metrics = model.update(images, labels)
                    self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
                    self.assertAlmostEqual(metrics["total_loss"], metrics["env_loss"], places=6)
                    grad = model.causal_decomposition.causal_adapter.weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0.)
                    self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)
                    torch.testing.assert_close(bank_before, model.text_features.b5b_text_bank)
                    self.assertIsNone(model.text_features.b5b_text_bank.grad)
                    self.assertIsNone(model.env_class_anchors.grad)
                    self.assertTrue(all(p.grad is None for p in model.clip_model.parameters()))

    def test_learnable_prompt_ce_gradient_and_prediction_without_randomness(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint, cipt_s_uniform_weight=0.)
            images, labels = fixture.batches()
            context = model.text_features.prompt_learner.context
            expected_grad = torch.autograd.grad(
                F.cross_entropy(model.predict(torch.cat(images)), torch.cat(labels)), context)[0]
            model.update(images, labels)
            self.assertGreater(expected_grad.abs().sum().item(), 0.)
            torch.testing.assert_close(context.grad, expected_grad)
            model.eval()
            state = torch.get_rng_state().clone()
            with mock.patch.object(model.random_selector, "forward", side_effect=AssertionError("test sampling")):
                actual = model.predict(images[0])
                e, _ = model.causal_decomposition(model._visual(images[0]))
                text = model.text_features.class_features()
                expected = model.clip_model.logit_scale.exp() * F.normalize(e, dim=-1) @ text.t()
                torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_disabled_environment_does_not_sample_or_advance_rng(self):
        for options in (dict(cipt_env_constraint="none"), dict(cipt_env_loss_weight=0.)):
            model = build(**options)
            state = torch.get_rng_state().clone()
            with mock.patch.object(model.random_selector, "forward", side_effect=AssertionError("disabled sampling")):
                metrics = model.update(*fixture.batches())
            self.assertEqual(metrics["env_loss"], 0.)
            self.assertEqual(metrics["entropy_selector_active"], 0.)
            self.assertEqual(metrics["random_selector_active"], 0.)
            self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_seed_determinism_and_rng_checkpoint_resume(self):
        deterministic_before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for constraint in ("mcc", "relative_distance"):
                runs = []
                for _ in range(2):
                    model = build(constraint, seed=23)
                    history = [model.update(*fixture.batches()) for _ in range(3)]
                    runs.append((history, copy.deepcopy(model.state_dict())))
                self.assertEqual(runs[0][0], runs[1][0])
                for key in runs[0][1]:
                    torch.testing.assert_close(runs[0][1][key], runs[1][1][key], rtol=0., atol=0.)
                weights = copy.deepcopy(model.state_dict())
                optimizer = copy.deepcopy(model.optimizer.state_dict())
                rng = torch.get_rng_state().clone()
                expected = model.update(*fixture.batches())
                restored = build(constraint)
                restored.load_state_dict(weights, strict=True)
                restored.optimizer.load_state_dict(optimizer)
                torch.set_rng_state(rng)
                actual = restored.update(*fixture.batches())
                self.assertEqual(actual, expected)
                for a, b in zip(model.parameters(), restored.parameters()):
                    torch.testing.assert_close(a, b, rtol=0., atol=0.)
        finally:
            torch.use_deterministic_algorithms(deterministic_before)

    def test_cli_default_invalid_mode_and_legacy_compatibility(self):
        from sconf import Config
        defaults = fixture.registry.default_hparams("CIPTDCCL", "PACS")
        self.assertEqual(defaults["cipt_env_selection"], "entropy")
        self.assertIsInstance(fixture.build().entropy_selector, fixture.selector.EntropyPromptSelector)
        with (fixture.ROOT / "config.yaml").open() as base, \
                (fixture.ROOT / "config_cipt_b5b_entropy_rd.yaml").open() as preset:
            cfg = Config(base, preset, default=defaults)
        cfg.argv_update(["--cipt_env_selection", "random", "--cipt_env_topk", "40",
                         "--cipt_use_ind", "false", "--cipt_use_contrastive", "false",
                         "--cipt_causal_contrastive_weight", "0"])
        self.assertEqual(cfg.cipt_env_selection, "random")
        self.assertEqual(cfg.cipt_env_topk, 40)
        self.assertFalse(cfg.cipt_use_ind)
        self.assertFalse(cfg.cipt_use_contrastive)
        self.assertEqual(cfg.cipt_causal_contrastive_weight, 0.)
        with self.assertRaises(ValueError):
            build(cipt_env_selection="unknown")
        for objective in ("legacy", "env_kl", "clip_ce"):
            model = fixture.build(cipt_objective=objective, cipt_template_mode="b5c",
                                  cipt_selector_mode="adaptive", cipt_use_tda=True,
                                  cipt_env_kl_weight=.1, cipt_env_selection="ignored")
            metrics = model.update(*fixture.batches())
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            self.assertEqual(tuple(model.predict(fixture.batches()[0][0]).shape), (2, 3))

    def test_multi_step_training_swa_and_state_reload(self):
        report = run_smoke()
        self.assertEqual(len(report), 2)
        self.assertTrue(all(row["steps"] == row["swa_averaged"] == 5 for row in report))


if __name__ == "__main__":
    torch.set_num_threads(1)
    if "--smoke" in sys.argv:
        print(json.dumps(run_smoke(), indent=2))
    else:
        unittest.main()
