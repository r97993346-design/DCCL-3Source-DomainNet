"""CPU numerical tests and real-CIPTDCCL update smoke tests.

Only CLIP weights/tokenization are replaced by a tiny frozen encoder. Actual
PromptLearner, text-bank construction, E/S, algorithm constructors/updates,
optimizer, contrastive loss and SWA averaging execute without GPU/downloads.
Like the existing tests, AST loading avoids unrelated torchvision algorithms.
This is execution/gradient coverage, not a PACS accuracy experiment.

  python -m unittest discover -s tests -p 'test_cipt_*.py'
  python tests/test_cipt_env_anchor.py --smoke
"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import sys
from types import ModuleType
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from test_cipt_neighbor_contrastive import ROOT, ALGORITHMS, load_file


losses = load_file("anchor_losses", ALGORITHMS / "cipt_losses.py")
selector = load_file("entropy_selector", ALGORITHMS / "cipt_entropy_selector.py")
random_selector = load_file("random_selector", ALGORITHMS / "cipt_random_selector.py")
modules = load_file("anchor_modules", ALGORITHMS / "cipt_modules.py")
neighbor = load_file("anchor_neighbor", ALGORITHMS / "cipt_neighbor_contrastive.py")
optimizers = load_file("anchor_optimizer", ROOT / "domainbed" / "optimizers.py")
registry = load_file("anchor_registry", ROOT / "domainbed" / "hparams_registry.py")
swa = load_file("anchor_swa", ROOT / "domainbed" / "lib" / "swa_utils.py")
with mock.patch.dict(sys.modules, {"clip": ModuleType("clip")}):
    prompts = load_file("anchor_prompts", ALGORITHMS / "cipt_prompt.py")


def tiny_tokenize(texts):
    if isinstance(texts, str):
        texts = [texts]
    tokens = torch.zeros(len(texts), 64, dtype=torch.long)
    for row, text in enumerate(texts):
        words = text.split()
        ids = [1 + int.from_bytes(hashlib.sha256(w.encode()).digest()[:2], "big") % 509
               for w in words]
        tokens[row, :len(ids) + 2] = torch.tensor([510] + ids + [511])
    return tokens


class TinyTransformer(nn.Module):
    def forward(self, x):
        # Let the end-of-text feature depend on the learnable context tokens.
        count = torch.arange(1, len(x) + 1, device=x.device)[:, None, None]
        return x + x.cumsum(dim=0) / count


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(8, 8, bias=False)
        self.visual.output_dim = 8
        self.token_embedding = nn.Embedding(512, 8)
        self.positional_embedding = nn.Parameter(torch.randn(64, 8) * .01)
        self.transformer = TinyTransformer()
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.eye(8))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(3.0)))
        self.dtype = torch.float32

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens):
        return prompts.CLIPTextEncoder(self)(self.token_embedding(tokens), tokens)


def tiny_frozen_clip(*_):
    return TinyCLIP().eval().requires_grad_(False), tiny_tokenize


def execute_classes(filename, names, namespace):
    tree = ast.parse((ALGORITHMS / filename).read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace)


# Execute every real constructor in the inheritance chain, including the
# production base that creates CLIP/text banks and the original projection heads.
base_namespace = dict(
    torch=torch, nn=nn, F=F, copy=copy,
    load_frozen_clip=tiny_frozen_clip, CIPTTextFeatures=prompts.CIPTTextFeatures,
    CausalDecomposition=modules.CausalDecomposition,
    TextDiversityAugmentation=modules.TextDiversityAugmentation,
    get_optimizer=optimizers.get_optimizer,
)
execute_classes("algorithms.py", ("Algorithm", "CIPTDCCL", "SupConLoss"), base_namespace)
namespace = dict(
    base_namespace, math=math, _BaseCIPTDCCL=base_namespace["CIPTDCCL"],
    cipt_classification_loss=losses.classification_loss,
    cipt_decomposition_loss=losses.decomposition_loss,
    cipt_independence_loss=losses.independence_loss,
    empty_neighbor_stats=neighbor.empty_neighbor_stats,
    neighbor_retention_weights=neighbor.neighbor_retention_weights,
    validate_neighbor_options=neighbor.validate_neighbor_options,
    SafeDiversePromptSelector=modules.SafeDiversePromptSelector,
    CLASS_CONDITIONED_TDA_MODES={"b5b", "bconst"},
    SELECTABLE_TDA_MODES={"b5a", "b5c"},
)
execute_classes("cipt_dccl_ablation.py", ("CIPTDCCL",), namespace)
namespace = dict(namespace, _BaseCIPTDCCL=namespace["CIPTDCCL"])
execute_classes("cipt_dccl_component_ablation.py", ("CIPTDCCL",), namespace)
namespace = dict(
    namespace, _BaseCIPTDCCL=namespace["CIPTDCCL"],
    EntropyPromptSelector=selector.EntropyPromptSelector,
    RandomPromptSelector=random_selector.RandomPromptSelector,
    class_prompt_logits=selector.class_prompt_logits,
    decomposition_loss_terms=losses.decomposition_loss_terms,
    independence_loss=losses.independence_loss,
    prompt_consistency_loss=losses.prompt_consistency_loss,
    relative_distance_loss=losses.relative_distance_loss,
    CLIPTextEncoder=prompts.CLIPTextEncoder, PromptLearner=prompts.PromptLearner,
)
execute_classes("cipt_env_kl.py", ("CIPTDCCL",), namespace)
Environment = namespace["CIPTDCCL"]


def hparams(constraint="mcc", **overrides):
    params = registry.default_hparams("CIPTDCCL", "PACS")
    params.update(yaml.safe_load((ROOT / "config.yaml").read_text()))
    suffix = "rd" if constraint == "relative_distance" else "mcc"
    params.update(yaml.safe_load((ROOT / ("config_cipt_b5b_entropy_" + suffix + ".yaml")).read_text()))
    params.update(cipt_env_constraint=constraint, cipt_class_names=["dog", "horse", "guitar"],
                  n_layer=1, t=.1, t_pre=.2, l_d=.01, l_layer=1.)
    params.update(overrides)
    return params


def build(constraint="mcc", seed=0, **overrides):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return Environment((8,), 3, 3, hparams(constraint, **overrides))


def batches():
    generator = torch.Generator().manual_seed(24)
    images = torch.randn(6, 8, generator=generator)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    return list(images.split(2)), list(labels.split(2))


class EntropySelectorTests(unittest.TestCase):
    def test_full_official_bank_and_all_class_template_alignment(self):
        model = build()
        text = model.text_features
        with mock.patch.object(text, "_select_indices", side_effect=AssertionError("random K")):
            bank = text.full_b5b_text_features()
        self.assertEqual(tuple(bank.shape), (3, 80, 8))
        self.assertFalse(bank.requires_grad)
        self.assertEqual(len(prompts.B5B_IMAGENET_TEMPLATES), 80)
        for c, name in enumerate(text.class_names):
            strings = [t.format(name) for t in prompts.B5B_IMAGENET_TEMPLATES]
            self.assertTrue(all(name in s and "subject" not in s for s in strings))
            with torch.no_grad():
                expected = F.normalize(model.clip_model.encode_text(tiny_tokenize(strings)), dim=-1)
            torch.testing.assert_close(bank[c], expected)
        expected_anchors = F.normalize(bank.mean(dim=1), dim=-1)
        torch.testing.assert_close(model.env_class_anchors, expected_anchors)
        self.assertFalse(model.env_class_anchors.requires_grad)

    def test_independent_per_image_topk_matches_explicit_all_class_entropy(self):
        torch.manual_seed(10)
        visual, bank = torch.randn(4, 8), torch.randn(3, 80, 8)
        indices, metrics = selector.EntropyPromptSelector()(visual, bank, 7.)
        self.assertEqual(tuple(indices.shape), (4, 40))
        self.assertTrue(((indices >= 0) & (indices < 80)).all())
        self.assertFalse(torch.equal(indices[0], indices[1]))
        for i in range(4):
            entropies = []
            for j in range(80):
                logits = 7 * F.cosine_similarity(visual[i][None], bank[:, j], dim=-1)
                p = logits.softmax(dim=-1)
                entropies.append(-(p * logits.log_softmax(dim=-1)).sum())
            expected = torch.stack(entropies).topk(40, largest=False).indices
            torch.testing.assert_close(indices[i], expected)
        self.assertTrue(all(v.dtype == torch.float32 and not v.requires_grad for v in metrics.values()))
        self.assertLessEqual(metrics["env_selected_entropy_mean"], metrics["env_entropy_mean"])

    def test_labels_and_e_s_changes_cannot_affect_selection(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint)
            images, labels = batches()
            visual = model._visual(torch.cat(images))
            e, _ = model.causal_decomposition(visual)
            calls = []
            def record(module, args, output):
                self.assertEqual(len(args), 3)  # V, all-class bank, scale; never y.
                torch.testing.assert_close(args[0], visual)
                calls.append(output[0].clone())
            handle = model.entropy_selector.register_forward_hook(record)
            model._env_anchor_loss(visual, e, torch.cat(labels))
            model._env_anchor_loss(visual, -e * 3, torch.cat(labels).roll(1))
            handle.remove()
            self.assertTrue(torch.equal(calls[0], calls[1]))

    def test_invalid_topk_shapes_and_scale(self):
        for k in (1, 81, True, 2.5, "40"):
            with self.assertRaises(ValueError):
                selector.EntropyPromptSelector(k)
        for bank, scale in ((torch.zeros(3, 79, 8), 1.),
                            (torch.zeros(3, 80, 7), 1.),
                            (torch.zeros(3, 80, 8), float("inf")),
                            (torch.zeros(3, 80, 8), 0.)):
            with self.assertRaises(ValueError):
                selector.EntropyPromptSelector()(torch.ones(1, 8), bank, scale)


class AnchorLossTests(unittest.TestCase):
    def test_mcc_shape_formula_and_identical_predictions(self):
        torch.manual_seed(14)
        e, bank = torch.randn(3, 8), torch.randn(4, 80, 8)
        indices, _ = selector.EntropyPromptSelector()(e, bank, 3.)
        logits = selector.class_prompt_logits(e, bank, 3.).gather(1, indices[..., None].expand(-1, -1, 4))
        self.assertEqual(tuple(logits.shape), (3, 40, 4))
        # Independent ordered-pair specification, including both KL arguments.
        small = logits[:, :4].double().requires_grad_()
        log_p = small.log_softmax(dim=-1)
        expected = torch.stack([F.kl_div(log_p[:, j], log_p[:, i].exp(), reduction="batchmean")
                                for i in range(4) for j in range(4) if i != j]).mean()
        actual = losses.prompt_consistency_loss(small)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, small, retain_graph=True)[0],
                                   torch.autograd.grad(expected, small)[0])
        identical_bank = bank[:, :1].expand(-1, 80, -1)
        identical = selector.class_prompt_logits(e, identical_bank, 3.)
        self.assertLess(abs(losses.prompt_consistency_loss(identical).item()), 1e-6)

    def test_rd_exact_formula_and_anchor_zero(self):
        torch.manual_seed(7)
        bank = torch.randn(3, 80, 8)
        anchors = F.normalize(bank.mean(1), dim=-1)
        labels, e = torch.tensor([0, 2]), torch.randn(2, 8)
        indices = torch.stack((torch.arange(40), torch.arange(40, 80)))
        actual = losses.relative_distance_loss(e, anchors, bank, labels, indices)
        expected = torch.stack([
            (F.cosine_similarity(e[i], bank[labels[i], j], dim=0)
             - F.cosine_similarity(anchors[labels[i]], bank[labels[i], j], dim=0)).square()
            for i in range(2) for j in indices[i]
        ]).mean()
        torch.testing.assert_close(actual, expected)
        for k in (2, 40, 80):
            selected = torch.arange(k)[None].expand(2, -1)
            self.assertLess(losses.relative_distance_loss(
                anchors[labels], anchors, bank, labels, selected).item(), 1e-12)
            # Cosine relations are scale invariant; this cannot be raw MSE(E,A).
            self.assertLess(losses.relative_distance_loss(
                7 * anchors[labels], anchors, bank, labels, selected).item(), 1e-12)

    def test_all_topk_batch_one_mixed_precision_and_nonzero_e_gradients(self):
        torch.manual_seed(31)
        for batch in (1, 5):
            for k in (2, 40, 80):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    bank = torch.randn(3, 80, 8, dtype=dtype, requires_grad=True)
                    anchors = bank.detach().float().mean(1).requires_grad_()
                    visual = torch.randn(batch, 8, dtype=dtype, requires_grad=True)
                    scale = torch.tensor(100., requires_grad=True)
                    idx, _ = selector.EntropyPromptSelector(k)(visual, bank, scale)
                    self.assertFalse(idx.requires_grad)
                    for constraint in ("mcc", "relative_distance"):
                        e = torch.randn(batch, 8, dtype=dtype, requires_grad=True)
                        if constraint == "mcc":
                            logits = selector.class_prompt_logits(e, bank, scale)
                            loss = losses.prompt_consistency_loss(logits.gather(1, idx[..., None].expand(-1, -1, 3)))
                        else:
                            loss = losses.relative_distance_loss(e, anchors, bank, torch.arange(batch) % 3, idx)
                        self.assertEqual(loss.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(torch.isfinite(e.grad).all())
                        self.assertGreater(e.grad.abs().sum().item(), 0.)
                        for frozen in (bank, anchors, visual, scale):
                            self.assertIsNone(frozen.grad)


class AnchorUpdateTests(unittest.TestCase):
    def test_environment_only_has_valid_e_and_adapter_gradients(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint, cipt_e_cls_weight=0., cipt_s_uniform_weight=0.,
                          cipt_use_ind=False, cipt_use_contrastive=False)
            captured = []
            def keep_e(module, args, output):
                output[0].retain_grad()
                captured.append(output[0])
            handle = model.causal_decomposition.register_forward_hook(keep_e)
            original_bank = model.text_features.b5b_text_bank.clone()
            metrics = model.update(*batches())
            handle.remove()
            self.assertAlmostEqual(metrics["total_loss"], metrics["env_loss"], places=6)
            self.assertGreater(captured[0].grad.abs().sum().item(), 0.)
            grad = model.causal_decomposition.causal_adapter.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)
            self.assertIsNone(model.causal_decomposition.spurious_adapter.weight.grad)
            self.assertTrue(torch.equal(model.text_features.b5b_text_bank, original_bank))
            self.assertIsNone(model.text_features.b5b_text_bank.grad)
            self.assertIsNone(model.env_class_anchors.grad)
            self.assertTrue(all(p.grad is None for p in model.clip_model.parameters()))

    def test_ce_count_weights_prompt_gradient_and_optimizer(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint, cipt_e_cls_weight=1.3, cipt_s_uniform_weight=.7,
                          cipt_gamma=2., cipt_env_loss_weight=.4)
            before = model.text_features.prompt_learner.context.detach().clone()
            with mock.patch.object(F, "cross_entropy", wraps=F.cross_entropy) as ce:
                metrics = model.update(*batches())
            self.assertEqual(ce.call_count, 1)
            self.assertEqual(tuple(ce.call_args.args[0].shape), (6, 3))
            expected = (1.3 * metrics["e_cls_loss"] + .7 * metrics["s_uni_loss"]
                        + 2 * metrics["cipt_ind_loss"]
                        + metrics["contrastive_weight_eff"] * metrics["dccl_contrastive_loss"]
                        + .4 * metrics["env_loss"])
            self.assertAlmostEqual(metrics["total_loss"], expected, places=5)
            context = model.text_features.prompt_learner.context
            self.assertGreater(context.grad.abs().sum().item(), 0.)
            self.assertFalse(torch.equal(before, context.detach()))
            optimized = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
            self.assertIn(id(context), optimized)
            for module in (model.clip_model, model.tda):
                self.assertTrue(all(id(p) not in optimized for p in module.parameters()))
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            self.assertEqual(metrics["env_kl_loss"], 0.)
            self.assertEqual(metrics["fused_cls_loss"], 0.)
            self.assertEqual(metrics["env_rd_loss"] if constraint == "mcc" else metrics["env_mcc_loss"], 0.)

    def test_fusion_old_selectors_and_test_labels_never_used(self):
        for constraint in ("mcc", "relative_distance"):
            # Inherited SIGM settings must not instantiate or execute SIGM.
            with mock.patch.dict(namespace, {"SemanticInteractionGuidance": mock.Mock(side_effect=AssertionError("SIGM"))}):
                model = build(constraint, cipt_fusion_mode="sigm", cipt_use_contrastive=False)
            with contextlib.ExitStack() as stack:
                for obj, method in ((model.tda, "forward"), (model, "_select_interventions"),
                                    (model.prompt_selector, "shortlist"), (model.prompt_selector, "select"),
                                    (model.text_features, "_select_indices")):
                    stack.enter_context(mock.patch.object(obj, method, side_effect=AssertionError(method)))
                metrics = model.update(*batches())
                self.assertEqual(metrics["dccl_contrastive_loss"], 0.)
                self.assertEqual(metrics["contrastive_weight_eff"], 0.)
                self.assertEqual(metrics["sigm_active"], 0.)
                self.assertEqual(metrics["entropy_selector_active"], 1.)
                model.eval()
                images = batches()[0][0]
                e, _ = model.causal_decomposition(model._visual(images))
                expected = model._logits(e[:, None], model.text_features.class_features())[:, 0]
                with mock.patch.object(model.entropy_selector, "forward", side_effect=AssertionError("test selection")):
                    torch.testing.assert_close(model.predict(images), expected)

    def test_original_prompt_learner_gets_exact_ce_gradient(self):
        for constraint in ("mcc", "relative_distance"):
            model = build(constraint, cipt_s_uniform_weight=0.,
                          cipt_use_ind=False, cipt_use_contrastive=False)
            images, labels = batches()
            context = model.text_features.prompt_learner.context
            logits = model.predict(torch.cat(images))
            expected = torch.autograd.grad(F.cross_entropy(logits, torch.cat(labels)), context)[0]
            model.update(images, labels)
            self.assertGreater(expected.abs().sum().item(), 0.)
            # Neither official-text regularizer introduces a prompt gradient.
            torch.testing.assert_close(context.grad, expected)

    def test_old_objectives_ignore_new_options_with_real_constructors(self):
        for objective in ("legacy", "env_kl", "clip_ce"):
            model = build(cipt_objective=objective, cipt_template_mode="b5c",
                          cipt_selector_mode="adaptive", cipt_use_tda=True,
                          cipt_env_kl_weight=.1, cipt_env_topk=1,
                          cipt_env_constraint="ignored", cipt_env_loss_weight=-1.)
            result = model.update(*batches())
            self.assertTrue(all(math.isfinite(v) for v in result.values()))
            model.eval()
            predicted = model.predict(batches()[0][0])
            self.assertEqual(tuple(predicted.shape), (2, 3))
            self.assertTrue(torch.isfinite(predicted).all())

    def test_none_zero_weight_and_de_ind_switches(self):
        for options in (dict(cipt_env_constraint="none"), dict(cipt_env_loss_weight=0.)):
            model = build(cipt_use_de=False, cipt_use_ind=False, cipt_use_contrastive=False, **options)
            with mock.patch.object(model.entropy_selector, "forward", side_effect=AssertionError("disabled selector")):
                metrics = model.update(*batches())
            self.assertEqual(metrics["env_loss"], 0.)
            self.assertEqual(metrics["s_uni_loss"], 0.)
            self.assertEqual(metrics["cipt_ind_loss"], 0.)
            self.assertAlmostEqual(metrics["total_loss"], metrics["e_cls_loss"], places=6)

    def test_invalid_configuration(self):
        options = [dict(cipt_env_constraint="mcc,relative_distance"),
                   dict(cipt_use_tda=True), dict(cipt_env_kl_weight=.1),
                   dict(cipt_fused_cls_weight=1.), dict(cipt_env_inference="mean_probs"),
                   dict(cipt_template_mode="b5c"), dict(cipt_selector_mode="adaptive")]
        for key in ("cipt_e_cls_weight", "cipt_s_uniform_weight", "cipt_gamma", "cipt_env_loss_weight"):
            options += [{key: v} for v in (-1., float("nan"), float("inf"))]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(**kwargs)

    def test_yaml_only_mode_differs_and_cli_overrides(self):
        from sconf import Config
        a = yaml.safe_load((ROOT / "config_cipt_b5b_entropy_mcc.yaml").read_text())
        b = yaml.safe_load((ROOT / "config_cipt_b5b_entropy_rd.yaml").read_text())
        self.assertEqual(a.pop("cipt_env_constraint"), "mcc")
        self.assertEqual(b.pop("cipt_env_constraint"), "relative_distance")
        self.assertEqual(a, b)
        with (ROOT / "config.yaml").open() as base, (ROOT / "config_cipt_b5b_entropy_mcc.yaml").open() as experiment:
            cfg = Config(base, experiment, default=registry.default_hparams("CIPTDCCL", "PACS"))
        cfg.argv_update(["--cipt_env_constraint", "relative_distance", "--cipt_env_topk", "80",
                         "--cipt_env_loss_weight", "0.5", "--cipt_e_cls_weight", "2",
                         "--cipt_s_uniform_weight", "0.5", "--cipt_gamma", "1",
                         "--cipt_use_ind", "false", "--cipt_causal_contrastive_weight", "2",
                         "--cipt_contrastive_warmup_steps", "0"])
        self.assertEqual(cfg.cipt_env_constraint, "relative_distance")
        self.assertEqual(cfg.cipt_env_topk, 80)
        self.assertEqual(cfg.cipt_env_loss_weight, .5)
        self.assertEqual(cfg.cipt_e_cls_weight, 2.)
        self.assertEqual(cfg.cipt_s_uniform_weight, .5)
        self.assertFalse(cfg.cipt_use_ind)
        self.assertEqual(cfg.cipt_causal_contrastive_weight, 2.)
        self.assertEqual(cfg.cipt_contrastive_warmup_steps, 0)

    def test_seed_determinism_and_batch_one_update(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for constraint in ("mcc", "relative_distance"):
                models = [build(constraint, seed=12) for _ in range(2)]
                histories = [[m.update(*batches()) for _ in range(3)] for m in models]
                self.assertEqual(*histories)
                for a, b in zip(models[0].parameters(), models[1].parameters()):
                    self.assertTrue(torch.equal(a, b))
                for k in (2, 40, 80):
                    model = build(constraint, cipt_env_topk=k)
                    result = model.update([batches()[0][0][:1]], [batches()[1][0][:1]])
                    self.assertTrue(all(math.isfinite(v) for v in result.values()))
        finally:
            torch.use_deterministic_algorithms(previous)


def run_smoke(steps=5):
    """Multiple optimizer steps + production SWA averaging + state reload."""
    report = []
    for constraint in ("mcc", "relative_distance"):
        model = build(constraint)
        original = model.causal_decomposition.causal_adapter.weight.detach().clone()
        averaged = swa.AveragedModel(model)
        history = []
        for step in range(steps):
            metrics = model.update(*batches())
            assert all(math.isfinite(value) for value in metrics.values())
            averaged.update_parameters(model, step=step)
            history.append({k: metrics[k] for k in ("total_loss", "env_loss", "e_cls_loss", "s_uni_loss", "contrastive_weight_eff")})
        assert not torch.equal(original, model.causal_decomposition.causal_adapter.weight)
        averaged.eval()
        images = batches()[0][0]
        with torch.no_grad():
            expected = averaged.predict(images)
            assert expected.shape == (2, 3) and torch.isfinite(expected).all()
            restored = build(constraint)
            restored.load_state_dict(averaged.module.state_dict(), strict=True)
            restored.eval()
            torch.testing.assert_close(restored.predict(images), expected)
        report.append(dict(constraint=constraint, steps=steps,
                           swa_averaged=int(averaged.n_averaged), history=history))
    return report


if __name__ == "__main__":
    torch.set_num_threads(1)
    if "--smoke" in sys.argv:
        print(json.dumps(run_smoke(), indent=2))
    else:
        unittest.main()
