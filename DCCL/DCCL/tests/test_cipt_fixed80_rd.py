"""Fixed B5b ensemble + all-80 RD, using the production CPU CLIP fixture.

Run: python -m unittest discover -s tests -p 'test_cipt_*.py'
     python tests/test_cipt_fixed80_rd.py --smoke
"""

import contextlib
import json
import math
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

import test_cipt_env_anchor as fixture


def build_fixed(**overrides):
    preset = yaml.safe_load((fixture.ROOT / "config_cipt_b5b_fixed80_rd.yaml").read_text())
    preset.update(overrides)
    return fixture.build("relative_distance", **preset)


def run_smoke(steps=5):
    model = build_fixed()
    original = model.causal_decomposition.causal_adapter.weight.detach().clone()
    averaged = fixture.swa.AveragedModel(model)
    history = []
    for step in range(steps):
        metrics = model.update(*fixture.batches())
        assert all(math.isfinite(value) for value in metrics.values())
        assert metrics["entropy_selector_active"] == 0.
        assert metrics["env_all_templates_active"] == 1.
        assert metrics["class_text_ensemble_active"] == 1.
        assert metrics["env_topk"] == 80.
        assert metrics["dccl_contrastive_loss"] == metrics["contrastive_weight_eff"] == 0.
        averaged.update_parameters(model, step=step)
        history.append({k: metrics[k] for k in (
            "total_loss", "e_cls_loss", "s_uni_loss", "env_rd_loss")})
    assert not torch.equal(original, model.causal_decomposition.causal_adapter.weight)
    averaged.eval()
    with torch.no_grad():
        images = fixture.batches()[0][0]
        expected = averaged.predict(images)
        assert expected.shape == (2, 3) and torch.isfinite(expected).all()
        restored = build_fixed()
        restored.load_state_dict(averaged.module.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(restored.predict(images), expected)
    return dict(steps=steps, swa_averaged=int(averaged.n_averaged), history=history)


class Fixed80RDTests(unittest.TestCase):
    def test_all80_formula_without_entropy_random_selection_or_learnable_text(self):
        with mock.patch.dict(fixture.namespace, {
            "EntropyPromptSelector": mock.Mock(side_effect=AssertionError("entropy construction")),
        }):
            model = build_fixed()
        self.assertIsNone(model.entropy_selector)
        images, labels = fixture.batches()
        y = torch.cat(labels)
        visual = model._visual(torch.cat(images))
        e, s = model.causal_decomposition(visual)
        bank = model.text_features.full_b5b_text_features()
        anchors = F.normalize(bank.mean(1), dim=-1)
        scale = model.clip_model.logit_scale.exp()
        logits = scale * F.normalize(e, dim=-1) @ anchors.t()
        s_logits = scale * F.normalize(s, dim=-1) @ anchors.t()
        expected_ce = F.cross_entropy(logits, y)
        expected_s = F.kl_div(F.log_softmax(s_logits, dim=-1),
                              torch.full_like(s_logits, 1. / 3), reduction="batchmean")
        expected_rd = torch.stack([
            (F.cosine_similarity(e[i], bank[y[i], j], dim=0)
             - F.cosine_similarity(anchors[y[i]], bank[y[i], j], dim=0)).square()
            for i in range(len(y)) for j in range(80)
        ]).mean()
        rd = mock.Mock(wraps=fixture.losses.relative_distance_loss)
        with contextlib.ExitStack() as stack:
            for obj, method in (
                (model.text_features, "class_features"),
                (model.text_features.prompt_learner, "forward"),
                (model.text_features.text_encoder, "forward"),
                (model.text_features, "_select_indices"),
                (model, "_select_interventions"),
                (model.prompt_selector, "shortlist"),
                (model.prompt_selector, "select"),
                (model.tda, "forward"),
            ):
                stack.enter_context(mock.patch.object(obj, method, side_effect=AssertionError(method)))
            stack.enter_context(mock.patch.dict(fixture.namespace, {
                "relative_distance_loss": rd,
                "class_prompt_logits": mock.Mock(side_effect=AssertionError("entropy logits")),
            }))
            metrics = model.update(images, labels)
        self.assertEqual(rd.call_count, 1)
        indices = rd.call_args.args[4]
        torch.testing.assert_close(indices, torch.arange(80)[None].expand(6, -1))
        self.assertAlmostEqual(metrics["e_cls_loss"], expected_ce.item(), places=6)
        self.assertAlmostEqual(metrics["s_uni_loss"], expected_s.item(), places=6)
        self.assertAlmostEqual(metrics["env_rd_loss"], expected_rd.item(), places=6)
        self.assertAlmostEqual(metrics["total_loss"],
                               (expected_ce + expected_s + expected_rd).item(), places=6)
        self.assertEqual(metrics["entropy_selector_active"], 0.)
        self.assertEqual(metrics["prompt_count"], 80.)

    def test_frozen_bank_and_prompts_optimizer_and_rd_adapter_gradients(self):
        for only_rd in (False, True):
            with self.subTest(only_rd=only_rd):
                weights = dict(cipt_e_cls_weight=0., cipt_s_uniform_weight=0.) if only_rd else {}
                model = build_fixed(**weights)
                context = model.text_features.prompt_learner.context
                snapshots = {"bank": model.text_features.b5b_text_bank.clone(),
                             "anchors": model.env_class_anchors.clone(),
                             "context": context.detach().clone()}
                optimized = {id(p) for g in model.optimizer.param_groups for p in g["params"]}
                adapters = {id(p) for p in model.causal_decomposition.parameters()}
                self.assertEqual(optimized, adapters)
                metrics = model.update(*fixture.batches())
                grad = model.causal_decomposition.causal_adapter.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0.)
                if only_rd:
                    self.assertAlmostEqual(metrics["total_loss"], metrics["env_rd_loss"], places=6)
                else:
                    s_grad = model.causal_decomposition.spurious_adapter.weight.grad
                    self.assertTrue(torch.isfinite(s_grad).all())
                    self.assertGreater(s_grad.abs().sum().item(), 0.)
                for module in (model.clip_model, model.tda, model.text_features.prompt_learner):
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
                self.assertTrue(torch.equal(snapshots["bank"], model.text_features.b5b_text_bank))
                self.assertTrue(torch.equal(snapshots["anchors"], model.env_class_anchors))
                self.assertTrue(torch.equal(snapshots["context"], context))
                self.assertIsNone(model.env_class_anchors.grad)
                self.assertIsNone(model.text_features.b5b_text_bank.grad)

    def test_prediction_is_e_against_the_frozen_80_template_mean(self):
        model = build_fixed()
        for _ in range(3):
            model.update(*fixture.batches())
        model.eval()
        with contextlib.ExitStack() as stack, torch.no_grad():
            for obj, method in ((model.text_features, "class_features"),
                                (model.text_features, "full_b5b_text_features"),
                                (model.text_features.prompt_learner, "forward"),
                                (model, "_env_anchor_loss")):
                stack.enter_context(mock.patch.object(obj, method, side_effect=AssertionError(method)))
            for batch in (fixture.batches()[0][0][:1], torch.cat(fixture.batches()[0])):
                visual = model._visual(batch)
                e, _ = model.causal_decomposition(visual)
                self.assertFalse(torch.equal(e, visual))
                expected = model.clip_model.logit_scale.exp() * F.normalize(e, dim=-1) @ model.env_class_anchors.t()
                torch.testing.assert_close(model.predict(batch), expected)

    def test_class_text_and_selection_switches_are_independent(self):
        for selection in ("entropy", "all"):
            for text_mode in ("learnable", "b5b_ensemble"):
                model = build_fixed(cipt_env_selection=selection,
                                    cipt_env_topk=40 if selection == "entropy" else 80,
                                    cipt_class_text_mode=text_mode)
                metrics = model.update(*fixture.batches())
                self.assertEqual(metrics["entropy_selector_active"], float(selection == "entropy"))
                self.assertEqual(metrics["class_text_ensemble_active"], float(text_mode == "b5b_ensemble"))
                context = model.text_features.prompt_learner.context
                if text_mode == "learnable":
                    self.assertGreater(context.grad.abs().sum().item(), 0.)
                else:
                    self.assertIsNone(context.grad)

    def test_invalid_modes_and_full_bank_count(self):
        for options in (dict(cipt_env_selection="random"),
                        dict(cipt_class_text_mode="unknown"),
                        *(dict(cipt_env_topk=k) for k in (40, 79, 80., True))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_fixed(**options)

    def test_yaml_cli_and_original_defaults(self):
        from sconf import Config
        defaults = fixture.registry.default_hparams("CIPTDCCL", "PACS")
        self.assertEqual(defaults["cipt_class_text_mode"], "learnable")
        self.assertEqual(defaults["cipt_env_selection"], "entropy")
        with (fixture.ROOT / "config.yaml").open() as base, \
                (fixture.ROOT / "config_cipt_b5b_fixed80_rd.yaml").open() as experiment:
            cfg = Config(base, experiment, default=defaults)
        self.assertFalse(cfg.cipt_use_contrastive)
        self.assertEqual(cfg.cipt_causal_contrastive_weight, 0.)
        self.assertEqual(cfg.cipt_class_text_mode, "b5b_ensemble")
        self.assertEqual(cfg.cipt_env_selection, "all")
        cfg.argv_update(["--cipt_class_text_mode", "learnable", "--cipt_env_selection", "entropy",
                         "--cipt_env_topk", "40", "--cipt_env_loss_weight", "0.5"])
        self.assertEqual(cfg.cipt_class_text_mode, "learnable")
        self.assertEqual(cfg.cipt_env_selection, "entropy")
        self.assertEqual(cfg.cipt_env_topk, 40)
        self.assertEqual(cfg.cipt_env_loss_weight, .5)

    def test_seed_determinism_and_batch_one(self):
        models = [build_fixed() for _ in range(2)]
        histories = [[m.update(*fixture.batches()) for _ in range(3)] for m in models]
        self.assertEqual(*histories)
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            self.assertTrue(torch.equal(a, b))
        metrics = build_fixed().update([fixture.batches()[0][0][:1]],
                                       [fixture.batches()[1][0][:1]])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_multi_step_training_swa_and_strict_state_reload(self):
        result = run_smoke()
        self.assertEqual(result["steps"], 5)
        self.assertEqual(result["swa_averaged"], 5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    if "--smoke" in sys.argv:
        print(json.dumps(run_smoke(), indent=2))
    else:
        unittest.main()
