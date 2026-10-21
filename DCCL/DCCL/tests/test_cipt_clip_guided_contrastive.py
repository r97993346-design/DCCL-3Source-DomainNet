"""CPU math/gradient and production-constructor/update regression tests.

No dataset or CLIP checkpoint is downloaded. The integration fixture replaces
only CLIP weights/tokenization; actual prompt learner, E/S, constructors,
optimizer and update methods run. This is not a PACS performance experiment.

python -m unittest discover -s tests -p test_cipt_clip_guided_contrastive.py -v
python tests/test_cipt_clip_guided_contrastive.py --cuda-smoke
"""

import contextlib
import io
import math
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from test_cipt_neighbor_contrastive import ROOT, clip_guided as cg
import test_cipt_env_anchor as fixture


def sample_batch():
    visual = torch.tensor([[1., 0., 0.], [1., .1, 0.], [.9, .1, 0.],
                           [0., 1., 0.], [0., 0., 1.], [0., .1, 1.]])
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    domains = torch.tensor([0, 0, 1, 2, 0, 1])
    generator = torch.Generator().manual_seed(37)
    causal = torch.randn(6, 5, generator=generator)
    return visual, causal, labels, domains


def reference_loss(visual, causal, labels, domains, temperature=.1,
                   teacher_temperature=.2, teacher_mix=.5, topk=0):
    """Independent specification: weighted CE for each selected positive index."""
    rows = []
    for i in range(len(labels)):
        candidates = [j for j in range(len(labels))
                      if i != j and labels[i] == labels[j] and domains[i] != domains[j]]
        if not candidates:
            continue
        with torch.no_grad():
            cosines = F.cosine_similarity(visual[i:i + 1].float(), visual[candidates].float())
            if topk:
                selected = cosines.argsort(descending=True)[:topk]
                candidates = [candidates[j] for j in selected.tolist()]
                cosines = cosines[selected]
            q = (cosines / teacher_temperature).softmax(dim=0)
            weights = (1.0 - teacher_mix) / len(candidates) + teacher_mix * q
        negatives = [j for j in range(len(labels)) if labels[i] != labels[j]]
        denominator = candidates + negatives
        logits = F.cosine_similarity(causal[i:i + 1].float(), causal[denominator].float()) / temperature
        rows.append(sum(
            weights[j] * F.cross_entropy(logits[None], labels.new_tensor([j]))
            for j in range(len(candidates))
        ))
    return torch.stack(rows).mean() if rows else causal.float().sum() * 0.0


def build_model(seed=0, **overrides):
    params = fixture.hparams()
    params.update(yaml.safe_load((ROOT / "config_cipt_clip_guided.yaml").read_text()))
    params.update(overrides)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fixture.Environment((8,), 3, 3, params)


class CrossDomainLossTests(unittest.TestCase):
    def test_positive_negative_and_neutral_masks(self):
        v, e, y, d = sample_batch()
        w, p, denominator, _ = cg.cross_domain_targets(v, y, d)
        self.assertEqual(torch.where(p[0])[0].tolist(), [2, 3])
        self.assertEqual(torch.where(denominator[0])[0].tolist(), [2, 3, 4, 5])
        self.assertFalse(denominator[0, 0])
        self.assertFalse(denominator[0, 1])  # Same domain, same class is neutral.
        self.assertFalse((p & y[:, None].ne(y[None])).any())
        self.assertTrue((denominator & y[:, None].ne(y[None])).sum() == y[:, None].ne(y[None]).sum())
        self.assertTrue(torch.equal(w > 0, p))
        self.assertTrue(torch.equal(p, p.t()))

    def test_matches_independent_formula_and_e_gradient(self):
        v, e, y, d = sample_batch()
        e.requires_grad_()
        for mix in (0., .5, 1.):
            for k in (0, 1, 100):
                actual, _ = cg.clip_guided_cross_domain_loss(v, e, y, d, teacher_mix=mix, topk=k)
                expected = reference_loss(v, e, y, d, teacher_mix=mix, topk=k)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                a = torch.autograd.grad(actual, e, retain_graph=True)[0]
                b = torch.autograd.grad(expected, e)[0]
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_rho_zero_is_equal_weight_cross_domain_supcon(self):
        v, e, y, d = sample_batch()
        actual, _ = cg.clip_guided_cross_domain_loss(v, e, y, d, teacher_mix=0., topk=0)
        expected = reference_loss(v, e, y, d, teacher_mix=0., topk=0)
        torch.testing.assert_close(actual, expected)
        weights, p, _, _ = cg.cross_domain_targets(v, y, d, teacher_mix=0.)
        torch.testing.assert_close(weights, p.float() / p.sum(1, keepdim=True).clamp_min(1))

    def test_weights_formula_normalization_and_finiteness(self):
        v, _, y, d = sample_batch()
        for mix in (0., .5, 1.):
            for k in (0, 1, 100):
                for temp in (.0001, .2, 100.):
                    weights, p, denominator, similarity = cg.cross_domain_targets(v, y, d, temp, mix, k)
                    valid = p.any(dim=1)
                    torch.testing.assert_close(weights.sum(1), valid.float())
                    self.assertTrue(torch.isfinite(weights).all())
                    self.assertTrue((weights >= 0).all())
                    self.assertTrue((weights[~p] == 0).all())
                    self.assertFalse(weights.requires_grad)
                    self.assertTrue((p <= denominator).all())
                    if temp == .2:
                        q = (similarity[0, p[0]] / temp).softmax(0)
                        expected = (1 - mix) / p[0].sum() + mix * q
                        torch.testing.assert_close(weights[0, p[0]], expected)

    def test_topk_uses_visual_and_leaves_unselected_class_neutral(self):
        v, e, y, d = sample_batch()
        _, p, denominator, _ = cg.cross_domain_targets(v, y, d, topk=1)
        self.assertEqual(torch.where(p[0])[0].tolist(), [2])
        self.assertEqual(torch.where(denominator[0])[0].tolist(), [2, 4, 5])
        self.assertFalse(denominator[0, 3])
        # The unselected candidate is deliberately the closest one in E.
        e[3] = e[0]
        actual, _ = cg.clip_guided_cross_domain_loss(v, e, y, d, topk=1)
        torch.testing.assert_close(actual, reference_loss(v, e, y, d, topk=1))
        _, unchanged, _, _ = cg.cross_domain_targets(v, y, d, topk=1)
        self.assertTrue(torch.equal(p, unchanged))

    def test_degenerate_batches_return_differentiable_zero(self):
        v, e, y, d = sample_batch()
        cases = [(v[:n], e[:n], y[:n], d[:n]) for n in (0, 1)]
        cases += [(v, e, y, torch.zeros_like(d)), (v, e, torch.arange(6), d)]
        for vv, ee, yy, dd in cases:
            ee = ee.detach().clone().requires_grad_()
            loss, stats = cg.clip_guided_cross_domain_loss(vv, ee, yy, dd)
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(loss.requires_grad)
            loss.backward()
            self.assertTrue(torch.isfinite(ee.grad).all())
            self.assertTrue((ee.grad == 0).all())
            self.assertTrue(all(x.item() == 0. for x in stats.values()))

    def test_some_invalid_anchors_are_skipped_without_nan_gradients(self):
        v, e, y, d = sample_batch()
        y = torch.tensor([0, 2, 0, 3, 4, 5])
        e.requires_grad_()
        loss, stats = cg.clip_guided_cross_domain_loss(v, e, y, d)
        torch.testing.assert_close(loss, reference_loss(v, e, y, d))
        self.assertAlmostEqual(stats["cg_valid_anchor_fraction"].item(), 2 / 6)
        loss.backward()
        self.assertTrue(torch.isfinite(e.grad).all())

    def test_no_teacher_gradient_but_e_has_gradient(self):
        v, e, y, d = sample_batch()
        v.requires_grad_()
        e.requires_grad_()
        loss, stats = cg.clip_guided_cross_domain_loss(v, e, y, d)
        loss.backward()
        self.assertIsNone(v.grad)
        self.assertTrue(torch.isfinite(e.grad).all())
        self.assertGreater(e.grad.abs().sum().item(), 0.)
        self.assertTrue(all(not x.requires_grad for x in stats.values()))
        self.assertEqual(stats["cg_active"].item(), 1.)
        torch.testing.assert_close(stats["cg_contrastive_loss"], loss.detach())

    def test_joint_permutation_invariance_with_and_without_topk(self):
        torch.manual_seed(59)
        v, e = torch.randn(18, 7), torch.randn(18, 9)
        y, d = torch.arange(18) % 3, torch.arange(3).repeat_interleave(6)
        order = torch.randperm(18)
        for k in (0, 2):
            a, sa = cg.clip_guided_cross_domain_loss(v, e, y, d, topk=k)
            b, sb = cg.clip_guided_cross_domain_loss(v[order], e[order], y[order], d[order], topk=k)
            torch.testing.assert_close(a, b)
            for name in sa:
                torch.testing.assert_close(sa[name], sb[name])

    def test_float32_loss_under_autocast_zero_and_large_features(self):
        v, e, y, d = sample_batch()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for scale in (0., 1., 1000.):
                vv, ee = (v * scale).to(dtype), (e * scale).to(dtype).requires_grad_()
                expected, _ = cg.clip_guided_cross_domain_loss(vv, ee, y, d)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    loss, stats = cg.clip_guided_cross_domain_loss(vv, ee, y, d)
                self.assertEqual(loss.dtype, torch.float32)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertTrue(torch.isfinite(ee.grad).all())
                self.assertTrue(all(torch.isfinite(x).all() for x in stats.values()))

    def test_positive_count_is_not_a_loss_multiplier(self):
        e, v = torch.ones(9, 3), torch.ones(9, 5)
        y, d = torch.zeros(9, dtype=torch.long), torch.arange(3).repeat_interleave(3)
        loss, stats = cg.clip_guided_cross_domain_loss(v, e, y, d)
        self.assertAlmostEqual(loss.item(), math.log(6), places=6)
        self.assertEqual(stats["cg_cross_domain_pos_count"].item(), 6.)
        self.assertAlmostEqual(stats["cg_teacher_weight_entropy"].item(), math.log(6), places=6)

    def test_source_domain_ids_follow_uneven_source_minibatches(self):
        batches = [torch.tensor([0, 1]), torch.tensor([1]), torch.tensor([2, 0, 1])]
        labels = torch.cat(batches)
        self.assertEqual(cg.source_domain_ids(labels, batches).tolist(), [0, 0, 1, 2, 2, 2])
        with self.assertRaises(ValueError):
            cg.source_domain_ids(labels.roll(1), batches)

    def test_invalid_options_and_shapes_fail_early(self):
        v, e, y, d = sample_batch()
        for options in (dict(teacher_temperature=0.), dict(teacher_temperature=float("nan")),
                        dict(teacher_mix=-.1), dict(teacher_mix=1.1), dict(teacher_mix=float("inf")),
                        dict(topk=-1), dict(topk=1.5), dict(topk=True), dict(temperature=0.)):
            with self.assertRaises(ValueError):
                cg.clip_guided_cross_domain_loss(v, e, y, d, **options)
        for vv, ee, yy, dd in ((v, e[:-1], y, d), (v, e, y.float(), d),
                               (v, e, y, d[:, None]), (v[:, None], e, y, d)):
            with self.assertRaises(ValueError):
                cg.clip_guided_cross_domain_loss(vv, ee, yy, dd)


class AlgorithmIntegrationTests(unittest.TestCase):
    def test_exact_three_term_objective_and_only_one_visual_call(self):
        model = build_model()
        for adapter in (model.causal_decomposition.causal_adapter,
                        model.causal_decomposition.spurious_adapter):
            torch.testing.assert_close(adapter.weight, torch.eye(8))
            torch.testing.assert_close(adapter.bias, torch.zeros(8))
        with mock.patch.object(model, "_select_interventions", side_effect=AssertionError("fusion")), \
             mock.patch.object(model, "_env_anchor_loss", side_effect=AssertionError("MCC/RD")), \
             mock.patch.object(model.tda, "forward", side_effect=AssertionError("TDA")), \
             mock.patch.dict(model.update.__func__.__globals__, {
                 "independence_loss": mock.Mock(side_effect=AssertionError("IND"))}), \
             mock.patch.object(model.clip_model, "encode_image", wraps=model.clip_model.encode_image) as encode, \
             mock.patch.object(F, "cross_entropy", wraps=F.cross_entropy) as ce:
            metrics = model.update(*fixture.batches(), x_2=mock.sentinel.unused_second_view)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(ce.call_count, 1)
        self.assertEqual(metrics["contrastive_weight_eff"], 1.)
        expected = metrics["e_cls_loss"] + metrics["s_uni_loss"] + metrics["dccl_contrastive_loss"]
        self.assertAlmostEqual(metrics["total_loss"], expected, places=5)
        self.assertAlmostEqual(metrics["cipt_base_loss"], metrics["e_cls_loss"] + metrics["s_uni_loss"], places=5)
        self.assertEqual(metrics["cg_contrastive_loss"], metrics["dccl_contrastive_loss"])
        self.assertEqual(metrics["cg_valid_anchor_fraction"], metrics["contrastive_valid_anchor_fraction"])
        for name in ("cipt_ind_loss", "env_kl_loss", "fused_cls_loss", "pre_cl_loss", "reg_loss"):
            self.assertEqual(metrics[name], 0.)
        self.assertTrue(all(p.grad is None for p in model.clip_model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.tda.parameters()))
        self.assertIsNone(model.reg_log_variance.grad)
        self.assertGreater(model.text_features.prompt_learner.context.grad.abs().sum().item(), 0.)
        self.assertGreater(model.causal_decomposition.causal_adapter.weight.grad.abs().sum().item(), 0.)
        self.assertFalse(hasattr(model, "proj_head"))
        self.assertFalse(hasattr(model, "pre_proj_head"))

    def test_disabled_contrastive_leaves_e_s_updates_unchanged_in_all_modes(self):
        models = [build_model(cipt_contrastive_type=mode, cipt_use_contrastive=False)
                  for mode in ("supcon", "neighbor_retention", "cross_domain_supcon", "clip_guided_cross_domain")]
        outputs = [model.update(*fixture.batches()) for model in models]
        for model, metrics in zip(models, outputs):
            self.assertEqual(metrics, outputs[0])
            self.assertEqual(metrics["contrastive_weight_eff"], 0.)
            self.assertEqual(metrics["cg_active"], 0.)
            self.assertEqual(model._causal_contrastive_step.item(), 0)
            self.assertAlmostEqual(metrics["total_loss"], metrics["e_cls_loss"] + metrics["s_uni_loss"], places=6)
            for a, b in zip(models[0].parameters(), model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_rho_zero_modes_match_loss_gradients_and_optimizer_update(self):
        plain = build_model(cipt_contrastive_type="cross_domain_supcon")
        guided = build_model(cipt_clip_teacher_mix=0., cipt_clip_teacher_topk=0)
        for model in (plain, guided):
            self.assertEqual(len(model._causal_contrastive_loss(
                *self.route_inputs(model))), 4)
        a, b = plain.update(*fixture.batches()), guided.update(*fixture.batches())
        self.assertEqual(a, b)
        for old, new in zip(plain.parameters(), guided.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            if old.grad is not None:
                torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)

    @staticmethod
    def route_inputs(model):
        x, y = fixture.batches()
        v = model._visual(torch.cat(x))
        e, _ = model.causal_decomposition(v)
        return v, e, torch.cat(y), y

    def test_contrastive_only_changes_e_gradient_and_not_s_or_class_prompt(self):
        active, disabled = build_model(), build_model(cipt_use_contrastive=False)
        active.update(*fixture.batches())
        disabled.update(*fixture.batches())
        torch.testing.assert_close(active.causal_decomposition.spurious_adapter.weight.grad,
                                   disabled.causal_decomposition.spurious_adapter.weight.grad)
        torch.testing.assert_close(active.text_features.prompt_learner.context.grad,
                                   disabled.text_features.prompt_learner.context.grad)
        self.assertFalse(torch.allclose(active.causal_decomposition.causal_adapter.weight.grad,
                                        disabled.causal_decomposition.causal_adapter.weight.grad))

    def test_e_only_prediction_does_not_select_teacher_pairs_or_fuse(self):
        model = build_model().eval()
        images = fixture.batches()[0][0]
        with torch.no_grad():
            e, _ = model.causal_decomposition(model._visual(images))
            expected = model._logits(e[:, None], model.text_features.class_features())[:, 0]
            with mock.patch.object(model, "_causal_contrastive_loss", side_effect=AssertionError("teacher at test")), \
                 mock.patch.object(model.tda, "forward", side_effect=AssertionError("test fusion")):
                torch.testing.assert_close(model.predict(images), expected)

    def test_startup_logs_report_effective_coefficient_and_teacher_options(self):
        for enabled in (True, False):
            params = fixture.hparams()
            params.update(yaml.safe_load((ROOT / "config_cipt_clip_guided.yaml").read_text()))
            params.update(cipt_use_contrastive=enabled, cipt_causal_contrastive_weight=.7)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                fixture.Environment((8,), 3, 3, params)
            printed = output.getvalue()
            for text in ("E_CE_weight=1.0", "S_uniform_weight=1.0", "IND_weight=0.0",
                         "prompt_KL_active=False", "fused_CE_active=False", "inference_mode=e_only",
                         "teacher_temperature=0.2", "teacher_mix_eff=0.5", "teacher_topk=0",
                         "warmup_steps=0", "contrastive_weight_eff_start=" + ("0.7" if enabled else "0.0")):
                self.assertIn(text, printed)

    def test_new_registry_yaml_and_cli_overrides(self):
        from sconf import Config

        registry = fixture.registry.default_hparams("CIPTDCCL", "PACS")
        self.assertEqual(registry["cipt_contrastive_type"], "neighbor_retention")
        for key, expected in (("temperature", .2), ("mix", .5), ("topk", 0)):
            self.assertEqual(registry["cipt_clip_teacher_" + key], expected)
        with (ROOT / "config.yaml").open() as base, (ROOT / "config_cipt_clip_guided.yaml").open() as preset:
            config = Config(base, preset, default=registry)
        self.assertEqual(config.cipt_contrastive_warmup_steps, 0)
        self.assertEqual(config.cipt_s_uniform_weight, 1.)
        self.assertFalse(config.cipt_use_ind)
        self.assertFalse(config.cipt_use_tda)
        config.argv_update(["--cipt_clip_teacher_temperature", "0.3", "--cipt_clip_teacher_mix", "0.8",
                            "--cipt_clip_teacher_topk", "2", "--cipt_contrastive_type", "cross_domain_supcon"])
        self.assertEqual(config.cipt_clip_teacher_temperature, .3)
        self.assertEqual(config.cipt_clip_teacher_mix, .8)
        self.assertEqual(config.cipt_clip_teacher_topk, 2)
        self.assertEqual(config.cipt_contrastive_type, "cross_domain_supcon")


def cuda_smoke():
    """Five CUDA updates with synthetic source batches and tiny frozen CLIP.

    Fail explicitly when no GPU is available; never present a skip as a pass.
    Real ViT-B/16 + PACS smoke commands are provided in the method document.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test NOT RUN: no CUDA device available")
    torch.manual_seed(0)
    model = build_model().cuda().train()
    x = [torch.randn(32, 8, device="cuda") for _ in range(3)]
    y = [torch.arange(32, device="cuda") % 3 for _ in range(3)]
    for _ in range(5):
        metrics = model.update(x, y)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise AssertionError("Non-finite CUDA training metrics")
        if not all(p.grad is None for p in model.clip_model.parameters()):
            raise AssertionError("CLIP received a gradient")
        gradient = model.causal_decomposition.causal_adapter.weight.grad
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise AssertionError("E gradient missing or non-finite")
    # Test the loss kernel under CUDA autocast independently of the model.
    v = torch.randn(96, 512, device="cuda", requires_grad=True)
    e = torch.randn(96, 512, device="cuda", requires_grad=True)
    labels = torch.cat(y)
    domains = cg.source_domain_ids(labels, y)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, _ = cg.clip_guided_cross_domain_loss(v, e, labels, domains, topk=2)
    loss.backward()
    if loss.dtype != torch.float32 or not torch.isfinite(e.grad).all() or v.grad is not None:
        raise AssertionError("CUDA float32/teacher-detach contract failed")
    torch.cuda.synchronize()
    print("CUDA smoke PASS: 5 production updates with tiny frozen CLIP, B=3x32; "
          "float16-autocast loss kernel at D=512, finite E gradients and frozen teacher.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    if sys.argv[1:] == ["--cuda-smoke"]:
        cuda_smoke()
    else:
        unittest.main()
