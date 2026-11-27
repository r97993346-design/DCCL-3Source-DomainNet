"""Selected-template RD centers: numerical specification and real updates."""

import contextlib
import io
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import test_cipt_env_anchor as fixture


losses = fixture.losses


class SelectedAnchorTests(unittest.TestCase):
    def inputs(self, batch=3, topk=40):
        generator = torch.Generator().manual_seed(80)
        bank = torch.randn(3, 80, 8, generator=generator, requires_grad=True)
        anchors = bank.detach().mean(1).requires_grad_()
        e = torch.randn(batch, 8, generator=generator, requires_grad=True)
        labels = torch.tensor([2, 0, 2])[:batch]
        indices = torch.stack([
            torch.randperm(80, generator=generator)[:topk] for _ in range(batch)
        ])
        return e, anchors, bank, labels, indices

    def test_per_image_selected_center_matches_explicit_value_and_e_gradient(self):
        e, anchors, bank, labels, indices = self.inputs()
        actual = losses.relative_distance_loss(
            e, anchors, bank, labels, indices, anchor_mode="selected"
        )
        terms = []
        for i in range(len(e)):
            texts = bank.detach()[labels[i], indices[i]]
            center = texts.mean(0)
            for text in texts:
                terms.append((F.cosine_similarity(e[i], text, dim=0)
                              - F.cosine_similarity(center, text, dim=0)).square())
        expected = torch.stack(terms).mean()
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, e, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, e)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertGreater(actual_grad.abs().sum().item(), 0.)
        self.assertIsNone(anchors.grad)
        self.assertIsNone(bank.grad)

    def test_unselected_templates_and_full_centers_cannot_change_selected_target(self):
        e, anchors, bank, labels, indices = self.inputs(batch=1)
        expected = losses.relative_distance_loss(
            e, anchors, bank, labels, indices, anchor_mode="selected"
        )
        changed = bank.detach().clone()
        mask = torch.ones(80, dtype=torch.bool)
        mask[indices[0]] = False
        changed[labels[0], mask] += 100.
        actual = losses.relative_distance_loss(
            e, anchors + 50., changed, labels, indices, anchor_mode="selected"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_80_agrees_with_full_and_selected_center_gives_zero(self):
        for topk in (2, 40, 80):
            e, anchors, bank, labels, indices = self.inputs(topk=topk)
            center = bank.detach()[labels[:, None], indices].mean(1)
            zero = losses.relative_distance_loss(
                7 * center, anchors, bank, labels, indices, anchor_mode="selected"
            )
            self.assertLess(zero.item(), 1e-12)
            if topk == 80:
                full = losses.relative_distance_loss(e, anchors, bank, labels, indices)
                selected = losses.relative_distance_loss(
                    e, anchors, bank, labels, indices, anchor_mode="selected"
                )
                torch.testing.assert_close(selected, full)

    def test_entropy_indices_are_identical_in_both_modes(self):
        observed = []
        for mode in ("full", "selected"):
            model = fixture.build("relative_distance", cipt_rd_anchor_mode=mode)
            x, y = fixture.batches()
            visual = model._visual(torch.cat(x))
            e, _ = model.causal_decomposition(visual)
            with mock.patch.object(model.entropy_selector, "forward",
                                   wraps=model.entropy_selector.forward) as select:
                value, _ = model._env_anchor_loss(visual, e, torch.cat(y))
                args = select.call_args.args
                indices, _ = fixture.selector.EntropyPromptSelector(40)(*args)
            observed.append(indices)
            expected = losses.relative_distance_loss(
                e, model.env_class_anchors, model.text_features.full_b5b_text_features(),
                torch.cat(y), indices, anchor_mode=mode
            )
            torch.testing.assert_close(value, expected)
        self.assertTrue(torch.equal(*observed))

    def test_selected_rd_alone_updates_e_and_keeps_teacher_frozen(self):
        model = fixture.build(
            "relative_distance", cipt_rd_anchor_mode="selected",
            cipt_e_cls_weight=0., cipt_s_uniform_weight=0.,
            cipt_use_ind=False, cipt_use_contrastive=False, cipt_env_loss_weight=2.
        )
        bank_before = model.text_features.full_b5b_text_features().clone()
        metrics = model.update(*fixture.batches())
        self.assertAlmostEqual(metrics["total_loss"], 2 * metrics["env_rd_loss"], places=6)
        self.assertEqual(metrics["env_rd_selected_anchor"], 1.)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        grad = model.causal_decomposition.causal_adapter.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)
        self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.clip_model.parameters()))
        self.assertIsNone(model.env_class_anchors.grad)
        self.assertIsNone(model.text_features.b5b_text_bank.grad)
        self.assertTrue(torch.equal(bank_before, model.text_features.full_b5b_text_features()))

    def test_default_full_and_mcc_compatibility(self):
        default = fixture.build("relative_distance")
        self.assertEqual(default.rd_anchor_mode, "full")
        explicit = fixture.build("relative_distance", cipt_rd_anchor_mode="full")
        self.assertEqual(default.update(*fixture.batches()), explicit.update(*fixture.batches()))
        a = fixture.build("mcc", cipt_rd_anchor_mode="full")
        b = fixture.build("mcc", cipt_rd_anchor_mode="selected")
        self.assertEqual(a.update(*fixture.batches()), b.update(*fixture.batches()))
        for old, new in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_cli_overrides_and_invalid_modes(self):
        from sconf import Config
        cfg = Config(default=fixture.hparams("relative_distance"))
        self.assertEqual(cfg.cipt_rd_anchor_mode, "full")
        cfg.argv_update(["--cipt_rd_anchor_mode", "selected", "--cipt_env_topk", "40",
                         "--cipt_env_loss_weight", "2.0"])
        self.assertEqual(cfg.cipt_rd_anchor_mode, "selected")
        self.assertEqual(cfg.cipt_env_topk, 40)
        self.assertEqual(cfg.cipt_env_loss_weight, 2.)
        with contextlib.redirect_stdout(io.StringIO()):
            model = fixture.Environment((8,), 3, 3, cfg)
        self.assertEqual(model.rd_anchor_mode, "selected")
        for mode in ("all40", "", True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                fixture.build("relative_distance", cipt_rd_anchor_mode=mode)
        with self.assertRaises(ValueError):
            losses.relative_distance_loss(*self.inputs(), anchor_mode="unknown")

    def test_selected_mode_swa_reload_and_inference_without_selection(self):
        options = dict(cipt_rd_anchor_mode="selected", cipt_use_ind=False,
                       cipt_env_loss_weight=2., cipt_contrastive_warmup_steps=500)
        model = fixture.build("relative_distance", **options)
        averaged = fixture.swa.AveragedModel(model)
        for step in range(5):
            metrics = model.update(*fixture.batches())
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
            averaged.update_parameters(model, step=step)
        averaged.eval()
        restored = fixture.build("relative_distance", **options)
        restored.load_state_dict(averaged.module.state_dict(), strict=True)
        restored.eval()
        images = fixture.batches()[0][0]
        with torch.no_grad(), mock.patch.object(
            restored.entropy_selector, "forward", side_effect=AssertionError("test-time selection")
        ):
            torch.testing.assert_close(restored.predict(images), averaged.predict(images))
        self.assertEqual(int(averaged.n_averaged), 5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
