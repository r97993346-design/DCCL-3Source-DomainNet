import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from domainbed.evaluator import Evaluator
from test_cipt_env_anchor import build
from test_cipt_clip_supcon_cosine_protocol import SPACE, hparams_for
from run import PRE_CACHE_COMMIT, record_eval_cache_protocol


def model_for_test():
    params = {key: choices[0] for key, choices in SPACE.items()}
    config = hparams_for(dict(params, contrastive_mix=0.25))
    return build(**config)



def evaluators():
    generator = torch.Generator().manual_seed(10)
    meta = []
    for name, count in (("env0_in", 7), ("env0_out", 2), ("env1_in", 5), ("env1_out", 4)):
        data = [{"x": torch.randn(8, generator=generator), "y": i % 3} for i in range(count)]
        options = {"dataset": data, "batch_size": 3, "num_workers": 0}
        meta.append((name, options, None))
    return [Evaluator([0], meta, 2, None, evalmode="all", cache_visual=value) for value in (False, True)]


class EvalVisualCacheTests(unittest.TestCase):
    def test_score_parity_lifecycle_and_current_trainable_modules(self):
        model = model_for_test()
        baseline, cached = evaluators()
        snapshots = []

        def logits_from(evaluator, method):
            normal = getattr(model, method)
            parts = []

            def record(x):
                logits = normal(x)
                parts.append(logits.detach().clone())
                return logits

            with patch.object(model, method, record):
                metrics = evaluator.evaluate(model, ret_losses=True)
            return torch.cat(parts), metrics

        for changed in (False, True):
            if changed:
                with torch.no_grad():
                    model.causal_decomposition.causal_adapter.bias.add_(0.7)
                    model.text_features.prompt_learner.context[0, 0].add_(0.8)
            predicted, expected = logits_from(baseline, "predict")
            with patch.object(model.clip_model, "encode_image", wraps=model.clip_model.encode_image) as encode:
                result, actual = logits_from(cached, "predict_from_eval_visual")
            self.assertEqual(encode.call_count, 0 if changed else 8)
            self.assertEqual(cached.visual_cache.hit_splits, 4 if changed else 0)
            torch.testing.assert_close(result, predicted, rtol=0, atol=0)
            self.assertEqual(actual, expected)
            self.assertTrue(torch.equal(result.argmax(1), predicted.argmax(1)))
            snapshots.append(result)
        self.assertFalse(torch.equal(*snapshots))
        self.assertEqual(cached.visual_cache.size_bytes, 18 * (8 * 4 + 8))
        self.assertTrue(model.training)
        self.assertFalse(model.clip_model.visual.training)
        with self.assertRaisesRegex(ValueError, "only supported"):
            model.predict_from_eval_visual(torch.randn(2, 8))

    def test_encoder_weight_change_invalidates_cache(self):
        baseline, cached = evaluators()
        model = model_for_test()
        cached.evaluate(model)
        with torch.no_grad():
            model.clip_model.visual.weight[0].mul_(2)
        with patch.object(model.clip_model, "encode_image", wraps=model.clip_model.encode_image) as encode:
            self.assertEqual(cached.evaluate(model), baseline.evaluate(model))
        self.assertEqual(encode.call_count, 16)
        model.clip_model.visual.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "requires a frozen"):
            cached.evaluate(model)

    def test_no_change_to_real_cipt_update_with_tiny_frozen_clip(self):
        def exercise(cache):
            model = model_for_test()
            evaluator = evaluators()[int(cache)]
            generator = torch.Generator().manual_seed(13)
            losses, batches = [], []
            for step in range(3):
                indices = torch.randint(30, (6,), generator=generator)
                random_view = torch.randn(6, 8)
                x = random_view + indices.float()[:, None] * .01
                y = torch.tensor([0, 1, 2, 0, 1, 2])
                d = torch.tensor([0, 0, 1, 1, 2, 2])
                batches.append((indices, x))
                result = model.update(x.split(2), y.split(2), d=d.split(2), step=step)
                self.assertEqual(result["cg_all_class_active"], 1.)
                self.assertGreater(result["loss_contrastive"], 0.)
                losses.append(result)
                with torch.random.fork_rng(devices=[]):
                    evaluator.evaluate(model)
            return losses, batches, model.state_dict()

        baseline, baseline_batches, original = exercise(False)
        cached, cached_batches, candidate = exercise(True)
        self.assertEqual(baseline, cached)
        for (a_indices, a_x), (b_indices, b_x) in zip(baseline_batches, cached_batches):
            self.assertTrue(torch.equal(a_indices, b_indices))
            self.assertTrue(torch.equal(a_x, b_x))
        for key, value in original.items():
            torch.testing.assert_close(value, candidate[key], rtol=0, atol=0)


    def test_default_is_scoped_to_eligible_supcon_evaluation(self):
        entry = Path(__file__).resolve().parents[1] / "train_all.py"
        policy = next(node for node in ast.parse(entry.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "default_cipt_eval_visual_cache")
        namespace = {}
        exec(compile(ast.Module(body=[policy], type_ignores=[]), str(entry), "exec"), namespace)
        enabled = namespace[policy.name]
        params = hparams_for({key: choices[0] for key, choices in SPACE.items()})
        args = dict(dataset="PACS", algorithm="CIPTDCCL", source_only_selection=True, debug=False)
        for dataset in ("PACS", "VLCS", "OfficeHome"):
            self.assertTrue(enabled(SimpleNamespace(**dict(args, dataset=dataset)), params))
        self.assertFalse(enabled(SimpleNamespace(**dict(args, dataset="TerraIncognita")), params))
        self.assertFalse(enabled(SimpleNamespace(**dict(args, source_only_selection=False)), params))
        for overrides in (dict(val_augment=True), dict(swad="LossValley"),
                          dict(cipt_contrastive_type="neighbor_retention")):
            self.assertFalse(enabled(SimpleNamespace(**args), dict(params, **overrides)))

    def test_existing_study_manifest_preserves_original_and_rejects_protocol_changes(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            original = {"code_commit": PRE_CACHE_COMMIT, "study_names": {"PACS": "existing_study"}, "steps": 5001}
            path = output / "protocol.json"
            path.write_text(json.dumps(original))
            plan = dict(original, code_commit="cache-commit")
            record_eval_cache_protocol(output, plan)
            record_eval_cache_protocol(output, plan)
            self.assertEqual(json.loads(path.read_text()), original)
            rollout = json.loads((output / "eval_visual_cache_protocol.json").read_text())
            self.assertEqual(rollout["code_commit"], plan["code_commit"])
            self.assertEqual(rollout["study_names"], original["study_names"])
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                record_eval_cache_protocol(output, dict(plan, steps=9000))
