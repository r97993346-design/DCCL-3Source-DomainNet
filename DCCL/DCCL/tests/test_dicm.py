"""CPU-only contracts using a tiny encoder; no images or pretrained downloads."""

import copy
import json
import random
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn
from sconf import Config

from domainbed import hparams_registry, networks
from domainbed.algorithms.algorithms import DCCL
from domainbed.algorithms.dicm import DICM
from domainbed.algorithms.dicm_components import (
    ClassDomainResidualBank, InterventionSensitiveSubspace,
    PairedInterventionResponseEstimator, StableFeatureProjection,
)
from domainbed.lib import swa_utils
from domainbed.lib.cl_hparams import setup_alg_hparams
from domainbed.lib.dicm_hparams import finalize_dicm_hparams
from domainbed.lib.dicm_selection import select_source_iid
from domainbed.lib.reproducibility import seed_everything

REAL_FEATURIZER = networks.Featurizer


class TinyFeaturizer(nn.Module):
    n_outputs = 12

    def __init__(self, freeze=0):
        super().__init__()
        self.linear = nn.Linear(6, self.n_outputs, bias=False)
        with torch.no_grad():
            # Both encoder constructions start from identical reference weights.
            self.linear.weight.copy_(torch.sin(torch.arange(72).reshape(12, 6).float()) / 3)
        if freeze == "all":
            self.requires_grad_(False)
        self._features = []

    def clear_features(self):
        self._features.clear()

    def forward(self, x, ret_feats=False):
        self.clear_features()
        z = self.linear(x)
        self._features.append(z[:, :, None, None])
        return (z, self._features) if ret_feats else z


@pytest.fixture(autouse=True)
def tiny_backbone(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(networks, "Featurizer", lambda shape, hp, freeze=0, pre=False: TinyFeaturizer(freeze))


def hparams(mode="A4", **overrides):
    hp = hparams_registry.default_hparams("DICM", "PACS")
    hp.update(l=1.0, l_layer=1.0, l_d=0.0, t=0.1, t_pre=0.2, n_layer=1,
              aug=0, mix=0, TN=False, pos_mask=False, re_w=False,
              two_ce=False, sample_d=False, model="resnet50", label_ratio=1,
              lamda=5,
              dicm_rank=2, dicm_ablation=mode, dicm_min_domain_samples=1,
              dicm_min_domain_updates=1, dicm_min_valid_domains=2)
    hp.update(overrides)
    return hp


def batches():
    generator = torch.Generator().manual_seed(17)
    x = [torch.randn(4, 6, generator=generator) for _ in range(2)]
    aug = [v + 0.2 * torch.randn(4, 6, generator=generator) for v in x]
    labels = [torch.tensor([0, 1, 2, 0]) for _ in x]
    return x, labels, aug


def test_seed_reset_restores_python_numpy_torch_and_cudnn_policy():
    seed_everything(23, deterministic=True)
    expected = random.random(), np.random.rand(), torch.rand(5)
    random.random()
    np.random.rand(100)
    torch.rand(100)
    seed_everything(23, deterministic=True)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(5), expected[2], atol=0, rtol=0)
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    seed_everything(23, deterministic=False)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark


def test_pire_uses_matching_reference_difference_without_encoder_gradient():
    z = torch.randn(5, 12, requires_grad=True)
    aug = z + 2
    reference = torch.randn(5, 12, requires_grad=True)
    delta = PairedInterventionResponseEstimator()(z, aug, reference, reference + 0.5)
    torch.testing.assert_close(delta, torch.full_like(delta, 1.5))
    assert not delta.requires_grad


def test_projection_is_additive_orthogonal_and_has_no_task_basis_gradient():
    projection = StableFeatureProjection(12, rank=2)
    z = torch.randn(5, 12, requires_grad=True)
    e, s = projection.decompose(z)
    torch.testing.assert_close(e + s, z)
    torch.testing.assert_close((e * s).sum(1), torch.zeros(5), atol=2e-6, rtol=0)
    e.square().mean().backward()
    assert z.grad is not None and z.grad.norm() > 0
    assert projection.subspace.basis.grad is None
    with torch.no_grad():
        projection.beta.zero_()
    assert projection(z) is z


def test_isr_updates_basis_only_and_ignores_zero_responses():
    subspace = InterventionSensitiveSubspace(12, 2)
    responses = torch.randn(5, 12, requires_grad=True)
    loss, count = subspace.coverage_loss(responses)
    loss.backward()
    assert count == 5 and responses.grad is None
    assert subspace.basis.grad is not None and subspace.basis.grad.norm() > 0
    empty_loss, count = subspace.coverage_loss(torch.zeros(3, 12))
    assert count == 0 and not empty_loss.requires_grad


def test_qr_and_coverage_remain_fp32_under_autocast():
    subspace = InterventionSensitiveSubspace(12, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        q = subspace.orthonormal_basis()
        loss, _ = subspace.coverage_loss(torch.randn(5, 12), q=q)
    assert q.dtype == torch.float32 and loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(subspace.basis.grad).all()


def test_isr_balances_group_means_instead_of_observation_counts():
    model = DICM((6,), 3, 2, hparams())
    with torch.no_grad():
        model.projection.subspace.basis.copy_(torch.eye(12)[:, :2])
    q = model.projection.subspace.orthonormal_basis()
    aug = torch.eye(12)[2:3].repeat(20, 1)  # unexplained energy = 1
    dom = torch.eye(12)[:1]  # unexplained energy = 0
    loss, _, _, _, _, aw, dw = model._isr_loss(aug, dom, q)
    assert loss.item() == pytest.approx(0.25)
    assert (aw, dw) == (0.25, 0.75)
    fallback, *_rest, aw, dw = model._isr_loss(aug, dom[:0], q)
    assert fallback.item() == pytest.approx(1.0)
    assert (aw, dw) == (1.0, 0.0)
    model.dom_weight = 0
    only_aug, *_ = model._isr_loss(aug, dom, q)
    only_aug.backward()


def test_prototypes_require_samples_updates_and_all_valid_sources():
    bank = ClassDomainResidualBank(1, 2, 4, momentum=0.5, min_count=2, min_updates=2)
    labels, domains = torch.zeros(4, dtype=torch.long), torch.tensor([0, 0, 1, 1])
    residual = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0], [0., 2, 0, 0], [0., 2, 0, 0]], requires_grad=True)
    bank.update(residual, labels, domains, step=0)
    assert bank.domain_responses().shape == (0, 4)
    bank.update(residual * 2, labels, domains, step=1)
    response = bank.domain_responses()
    assert response.shape == (2, 4) and not response.requires_grad
    torch.testing.assert_close(response.mean(0), torch.zeros(4))
    torch.testing.assert_close(bank.prototypes[0, 0], torch.tensor([1.5, 0, 0, 0]))
    assert bank.update_counts.tolist() == [[2, 2]]
    assert bank.last_update_step.tolist() == [[1, 1]]
    with pytest.raises(ValueError, match="local source IDs"):
        bank.update(residual, labels, torch.tensor([0, 0, 3, 3]), step=2)


@pytest.mark.parametrize("mode,sc,pre", [("A1", True, True), ("A2", False, False), ("A3", True, False), ("A4", True, True)])
def test_presets_have_no_gt_and_all_enabled_student_heads_receive_e(mode, sc, pre):
    model = DICM((6,), 3, 2, hparams(mode))
    assert not hasattr(model, "mean_encoders") and not hasattr(model, "var_encoders")
    assert all("encoders" not in key for key in model.state_dict())
    teacher_ids = {id(p) for p in model.pre_featurizer.parameters()}
    optimized_ids = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
    assert not (teacher_ids & optimized_ids)
    x, y, aug = batches()
    with torch.no_grad():
        z = model.featurizer(torch.cat(x))
        e = model.projection(z).clone()
        ref = model.pre_featurizer(torch.cat(x)).clone()
    received = {"ce": [], "sc": [], "pre": []}
    handles = [head.register_forward_pre_hook(lambda module, inp, name=name: received[name].append(inp[0].detach().clone()))
               for name, head in [("ce", model.classifier), ("sc", model.proj_head), ("pre", model.pre_proj_head)]]
    metrics = model.update(x, y, x_2=aug, step=0, d=[torch.zeros(4), torch.full((4,), 3)])
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(received["ce"][0], e)
    assert bool(received["sc"]) is sc and bool(received["pre"]) is pre
    if sc:
        torch.testing.assert_close(received["sc"][1], e)
    if pre:
        torch.testing.assert_close(received["pre"][0], e)
        torch.testing.assert_close(received["pre"][1], ref)
    assert metrics["gt_enabled"] == 0 and metrics["reg_loss"] == 0
    assert metrics["supcon_enabled"] == sc and metrics["precl_enabled"] == pre
    assert all(p.grad is None for p in model.pre_featurizer.parameters())
    assert not model.pre_featurizer.training


def test_no_gt_identity_control_matches_original_initialization_rng_and_update():
    hp = hparams("A1")
    torch.manual_seed(31)
    legacy = DCCL((6,), 3, 2, hp)
    legacy.re_w = False
    rng_after_legacy = torch.random.get_rng_state()
    torch.manual_seed(31)
    model = DICM((6,), 3, 2, hp)
    assert torch.equal(rng_after_legacy, torch.random.get_rng_state())
    for name in ("featurizer", "classifier", "proj_head", "pre_proj_head"):
        for key, value in getattr(legacy, name).state_dict().items():
            torch.testing.assert_close(value, getattr(model, name).state_dict()[key], atol=0, rtol=0)
    x, y, aug = batches()
    np.random.seed(31)
    expected = legacy.update(x, y, x_2=aug)
    rng_np = np.random.get_state()
    np.random.seed(31)
    actual = model.update(x, y, x_2=aug)
    assert np.array_equal(rng_np[1], np.random.get_state()[1])
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for name in ("featurizer", "classifier", "proj_head", "pre_proj_head"):
        for key, value in getattr(legacy, name).state_dict().items():
            torch.testing.assert_close(value, getattr(model, name).state_dict()[key], atol=1e-7, rtol=1e-6)


def test_a5_preserves_a4_common_initialization_rng_and_original_gt_optimizer():
    torch.manual_seed(31)
    a4 = DICM((6,), 3, 2, hparams())
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(31)
    a5 = DICM((6,), 3, 2, hparams("A5"))
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for name in ("featurizer", "classifier", "proj_head", "pre_proj_head", "projection"):
        for key, value in getattr(a4, name).state_dict().items():
            torch.testing.assert_close(value, getattr(a5, name).state_dict()[key], atol=0, rtol=0)
    assert len(a5.mean_encoders) == len(a5.var_encoders) == 1
    assert a5.gt_weight == a5.hparams["l_d"] == 0.01
    gt_ids = {id(p) for p in a5.var_encoders.parameters()}
    groups = [g for g in a5.optimizer.param_groups if gt_ids & {id(p) for p in g["params"]}]
    assert len(groups) == 1 and groups[0]["lr"] == pytest.approx(10 * a5.hparams["lr"])
    assert groups[0]["weight_decay"] == 0
    optimized_ids = {id(p) for g in a5.optimizer.param_groups for p in g["params"]}
    assert not optimized_ids & {id(p) for p in a5.pre_featurizer.parameters()}


def test_a5_identity_control_matches_original_gt_loss_gradients_and_update():
    hp = finalize_dicm_hparams(hparams("A5", dicm_beta=0, dicm_isr_weight=0))
    torch.manual_seed(37)
    legacy = DCCL((6,), 3, 2, hp)
    legacy.re_w = False
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(37)
    model = DICM((6,), 3, 2, hp)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    x, y, aug = batches()
    np.random.seed(37)
    expected = legacy.update(x, y, x_2=aug)
    np.random.seed(37)
    actual = model.update(x, y, x_2=aug)
    # Legacy does not log GT separately; recover it from its four-term total.
    legacy_gt = (expected["loss"] - expected["ce_loss"] - hp["l"] * expected["sup_cl_loss"]
                 - hp["l_layer"] * expected["pre_cl_loss"]) / hp["l_d"]
    assert actual["reg_loss"] == pytest.approx(legacy_gt, abs=2e-4)
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for name in ("featurizer", "classifier", "proj_head", "pre_proj_head", "var_encoders"):
        for key, value in getattr(legacy, name).state_dict().items():
            torch.testing.assert_close(value, getattr(model, name).state_dict()[key], atol=1e-7, rtol=1e-6)
    assert model.projection.subspace.basis.grad is None


def test_a5_tasks_use_e_and_gt_keeps_snapshots_across_reference_and_extra_views():
    model = DICM((6,), 3, 2, hparams("A5", sample_d=True))
    x, y, aug = batches()
    extra = [v.flip(0) + 3 for v in x]
    with torch.no_grad():
        z = model.featurizer(torch.cat(x))
        z_aug = model.featurizer(torch.cat(aug))
        ref = model.pre_featurizer(torch.cat(x))
        expected_e, expected_aug = model.projection(z), model.projection(z_aug)
        expected_extra = model.projection(model.featurizer(torch.cat(extra)))
        variance = model.var_encoders[0](z_aug[:, :, None, None])
        expected_gt = ((z_aug[:, :, None, None] - ref[:, :, None, None]).square() / variance
                       + variance.log()).mean().item() / 2
    received = {"ce": [], "sc": [], "pre": []}
    handles = [head.register_forward_pre_hook(lambda module, inp, name=name: received[name].append(inp[0].detach().clone()))
               for name, head in [("ce", model.classifier), ("sc", model.proj_head), ("pre", model.pre_proj_head)]]
    before_variance = model.var_encoders[0].b.detach().clone()
    result = model.update(x, y, x_2=aug, x_2_d=extra, step=0)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(received["ce"][0], expected_e)
    assert len(received["sc"]) == 3 and len(received["pre"]) == 2
    for actual, expected in zip(received["sc"], (expected_aug, expected_e, expected_extra)):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(received["pre"][0], expected_e)
    torch.testing.assert_close(received["pre"][1], ref)
    assert result["reg_loss"] == pytest.approx(expected_gt, abs=1e-6)
    assert result["loss"] == pytest.approx(result["task_loss"] + result["weighted_gt_loss"]
                                            + result["weighted_loss_isr"], abs=1e-6)
    assert result["gt_enabled"] == result["gt_layer_count"] == 1
    assert result["gt_grad_norm"] > 0 and result["backbone_grad_norm"] > 0
    assert not torch.equal(model.var_encoders[0].b, before_variance)
    assert all(p.grad is None for p in model.pre_featurizer.parameters())
    # Identical initial encoders give no valid differential response; GT cannot train B.
    assert result["valid_aug_response_count"] == result["valid_dom_response_count"] == 0
    assert model.projection.subspace.basis.grad is None


def test_a5_gt_has_only_student_intermediate_and_variance_gradients():
    model = DICM((6,), 3, 2, hparams("A5"))
    student = torch.randn(4, 12, 1, 1, requires_grad=True)
    reference = torch.randn_like(student, requires_grad=True)
    model._gt_loss((student,), (reference,)).backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert model.var_encoders[0].b.grad is not None
    assert reference.grad is None and model.projection.subspace.basis.grad is None
    assert all(p.grad is None for p in model.classifier.parameters())


def test_private_basis_init_preserves_global_rng_and_beta_zero_matches_a1():
    torch.manual_seed(41)
    identity = DICM((6,), 3, 2, hparams("A1"))
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(41)
    model = DICM((6,), 3, 2, hparams(dicm_beta=0))
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    model.eval()
    identity.eval()
    x = batches()[0][0]
    torch.testing.assert_close(model.predict(x), identity.predict(x), atol=0, rtol=0)


def test_b_stays_frozen_when_isr_disabled_but_encoder_and_tasks_train():
    model = DICM((6,), 3, 2, hparams(dicm_isr_weight=0))
    initial_basis = model.projection.subspace.basis.detach().clone()
    x, y, aug = batches()
    for step in range(3):
        metrics = model.update(x, y, x_2=aug, step=step)
        assert model.projection.subspace.basis.grad is None
    torch.testing.assert_close(initial_basis, model.projection.subspace.basis, atol=0, rtol=0)
    assert metrics["backbone_grad_norm"] > 0


@pytest.mark.parametrize("mode", ["A4", "A5"])
def test_forward_export_state_restore_and_swad_use_the_identical_projection(mode):
    model = DICM((6,), 3, 2, hparams(mode))
    x, y, aug = batches()
    for step in range(3):
        metrics = model.update(x, y, x_2=aug, step=step)
    assert metrics["basis_grad_norm"] > 0
    assert metrics["valid_dom_response_count"] > 0
    model.eval()
    bank_before = copy.deepcopy(model.residual_bank.state_dict())
    with torch.no_grad():
        output = model.predict(x[0])
        torch.testing.assert_close(output, model.classifier(model.predict_embed(x[0])))
        torch.testing.assert_close(output, model.get_forward_model().predict(x[0]))
    for key, value in bank_before.items():
        torch.testing.assert_close(value, model.residual_bank.state_dict()[key])
    restored = DICM((6,), 3, 2, hparams(mode)).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(output, restored.predict(x[0]))
    averaged = swa_utils.AveragedModel(model)
    averaged.update_parameters(model, step=0)
    first_basis = model.projection.subspace.basis.detach().clone()
    with torch.no_grad():
        model.projection.subspace.basis.add_(0.01 * torch.randn_like(first_basis))
    second_basis = model.projection.subspace.basis.detach().clone()
    averaged.update_parameters(model, step=1)
    averaged.eval()
    avg_projection = averaged.module.network[1]
    torch.testing.assert_close(avg_projection.subspace.basis, (first_basis + second_basis) / 2)
    torch.testing.assert_close(averaged.predict(x[0]), averaged.module.predict(x[0]))
    torch.testing.assert_close(averaged.predict_embed(x[0]), avg_projection(averaged.module.network[0](x[0])))
    assert not any(any(part in name for part in ("residual_bank", "pre_featurizer", "encoders"))
                   for name in averaged.state_dict())
    assert avg_projection.beta.item() == 1


def test_missing_observations_skip_existing_adam_momentum_on_basis():
    model = DICM((6,), 3, 2, hparams())
    x, y, aug = batches()
    for step in range(3):
        model.update(x, y, x_2=aug, step=step)
    basis = model.projection.subspace.basis
    assert basis.grad is not None
    before = basis.detach().clone()
    before_step = model.optimizer.state[basis]["step"].clone()
    model.min_norm = 1e9
    result = model.update(x, y, x_2=aug, step=3)
    assert result["valid_aug_response_count"] == result["valid_dom_response_count"] == 0
    assert basis.grad is None
    torch.testing.assert_close(basis, before, atol=0, rtol=0)
    torch.testing.assert_close(model.optimizer.state[basis]["step"], before_step)


@pytest.mark.parametrize("mode", ["A4", "A5"])
def test_clone_after_update_rebinds_optimizer_to_copied_parameters(mode):
    model = DICM((6,), 3, 2, hparams(mode))
    x, y, aug = batches()
    model.update(x, y, x_2=aug)
    cloned = model.clone().eval()
    model.eval()
    torch.testing.assert_close(cloned.predict(x[0]), model.predict(x[0]))
    assert cloned.featurizer.linear.weight.data_ptr() != model.featurizer.linear.weight.data_ptr()
    owned = {id(p) for p in cloned.parameters()}
    assert all(id(p) in owned for group in cloned.optimizer.param_groups for p in group["params"])
    if mode == "A5":
        copied_variance, original_variance = cloned.var_encoders[0].b, model.var_encoders[0].b
        assert copied_variance.data_ptr() != original_variance.data_ptr()
        assert copied_variance in cloned.optimizer.state
        torch.testing.assert_close(cloned.optimizer.state[copied_variance]["exp_avg"],
                                   model.optimizer.state[original_variance]["exp_avg"])


def test_two_ce_and_extra_positive_also_use_projected_features():
    model = DICM((6,), 3, 2, hparams(two_ce=True, sample_d=True))
    x, y, aug = batches()
    result = model.update(x, y, x_2=aug, x_2_d=[v.flip(0) for v in x])
    assert result["sup_cl_loss"] != 0 and result["loss_cls"] > 0


@pytest.mark.parametrize("mode", ["A4", "A5"])
def test_real_resnet50_inference_without_pretrained_download_or_training(monkeypatch, mode):
    monkeypatch.setattr(networks, "Featurizer", REAL_FEATURIZER)
    model = DICM((3, 224, 224), 3, 2, hparams(mode, pretrained=False, dicm_rank=8)).eval()
    # ResNet's global pooling accepts this small synthetic image size as well.
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        e = model.predict_embed(x)
        output = model.predict(x)
        torch.testing.assert_close(output, model.get_forward_model().predict(x))
        q = model.projection.subspace.orthonormal_basis(detach=True)
        assert (e @ q).abs().max().item() < 1e-4
    assert e.shape == (1, 2048) and output.shape == (1, 3)
    assert not model.residual_bank.initialized.any()


def test_source_selection_ignores_targets_and_keeps_first_tie():
    records = [
        {"step": 0, "train_out": 0.6, "test_in": 1.0, "test_out": 1.0},
        {"step": 100, "train_out": 0.8, "test_in": 0.1, "test_out": 0.1},
        {"step": 200, "train_out": 0.8, "test_in": 0.99, "test_out": 0.99},
    ]
    assert select_source_iid(records)["step"] == 100
    assert select_source_iid([{"step": 3, "train_out": 0.9}])["step"] == 3


def test_cli_core_overrides_are_effective_for_dicm_only():
    args = Namespace(l=0.4, l_layer=0.3, l_d=0.0, t=0.15, t_pre=0.25, n_layer=2)
    resolved = finalize_dicm_hparams(hparams(), args, ["--l", "0.4", "--l_layer=0.3", "--t", "0.15", "--n_layer", "2"])
    assert resolved["l"] == 0.4 and resolved["l_layer"] == 0.3
    assert resolved["t"] == 0.15 and resolved["n_layer"] == 2
    assert resolved["t_pre"] == 0.2  # argparse default must not overwrite dataset config
    with pytest.raises(ValueError, match="disables PreCL"):
        finalize_dicm_hparams(hparams("A3"), args, ["--l_layer", "0.3"])


def test_a5_gt_cli_alias_and_config_are_idempotent_and_scoped_to_a5():
    args = Namespace(l_d=0.03)
    resolved = finalize_dicm_hparams(hparams("A5"), args, ["--l_d", "0.03"])
    assert resolved["l_d"] == resolved["dicm_gt_weight"] == 0.03
    assert finalize_dicm_hparams(dict(resolved)) == resolved
    with pytest.raises(ValueError, match="A1-A4 remove GT"):
        finalize_dicm_hparams(hparams(), args, ["--l_d", "0.03"])
    with pytest.raises(ValueError, match="set GT with dicm_gt_weight"):
        finalize_dicm_hparams(hparams("A5", l_d=0.2))


@pytest.mark.parametrize("overrides", [
    {"dicm_gt_weight": 0}, {"dicm_gt_weight": -0.1}, {"dicm_gt_weight": float("nan")},
    {"l": 0}, {"l_layer": 0},
])
def test_a5_rejects_disabled_or_invalid_full_model_components(overrides):
    with pytest.raises(ValueError, match="A5"):
        DICM((6,), 3, 2, hparams("A5", **overrides))


@pytest.mark.parametrize("key,value", [("TN", True), ("aug", 0.2), ("re_w", True), ("l_d", 0.01)])
def test_unsupported_paths_fail_before_silently_bypassing_dicm(key, value):
    with pytest.raises(ValueError):
        DICM((6,), 3, 2, hparams(**{key: value}))


def test_legacy_dataset_defaults_remain_unchanged_and_dicm_gt_is_zero():
    args = Namespace(dataset="PACS", algorithm="DCCL", model="resnet50", sup=True,
                     two_ce=False, sample_d=False, re_w=False, pos_mask=False,
                     mix=0, aug=0, label_ratio=1, TN=False, lamda=5, start_epoch=1000, log=False)
    legacy = setup_alg_hparams(hparams_registry.default_hparams("DCCL", "PACS"), args)
    assert legacy["l_d"] == 0.01
    args.algorithm = "DICM"
    current = setup_alg_hparams(hparams_registry.default_hparams("DICM", "PACS"), args)
    assert current["l_d"] == 0 and current["l"] == legacy["l"]
    assert current["dicm_gt_weight"] == legacy["l_d"]


@pytest.mark.parametrize("dataset,backbone,expected", [
    ("PACS", "resnet50", 0.01), ("VLCS", "resnet50", 0.05),
    ("OfficeHome", "clip_vit-b16", 0.1),
])
def test_a5_restores_effective_dataset_and_backbone_gt_defaults(dataset, backbone, expected):
    args = Namespace(dataset=dataset, algorithm="DICM", model=backbone, sup=True,
                     two_ce=False, sample_d=False, re_w=False, pos_mask=False,
                     mix=0, aug=0, label_ratio=1, TN=False, lamda=5, start_epoch=1000, log=False)
    hp = setup_alg_hparams(hparams_registry.default_hparams("DICM", dataset), args)
    hp["dicm_ablation"] = "A5"
    resolved = finalize_dicm_hparams(hp)
    assert resolved["l_d"] == resolved["dicm_gt_weight"] == expected


def test_show_reviews_dicm_without_cuda_dataset_or_output_creation(monkeypatch, capsys):
    import train_all
    monkeypatch.setattr("sys.argv", ["train_all.py", "review", "configs/dicm_pacs.yaml",
                                    "--algorithm", "DICM", "--show", "--l", "0.4"])
    monkeypatch.setattr(train_all, "get_dataset", lambda *a, **k: pytest.fail("--show accessed data"))
    monkeypatch.setattr(train_all, "get_writer", lambda *a, **k: pytest.fail("--show created outputs"))
    train_all.main()
    output = capsys.readouterr().out
    assert "dicm_beta: 1.0" in output and "l: 0.4" in output and "l_d: 0.0" in output


@pytest.mark.parametrize("options,expected", [
    (["configs/dicm_pacs_a5.yaml"], 0.01),
    (["--dicm_ablation", "A5"], 0.01),
    (["--dicm_ablation", "A5", "--dicm_gt_weight", "0.02"], 0.02),
    (["--dicm_ablation", "A5", "--l_d", "0.03"], 0.03),
])
def test_show_resolves_a5_gt_overlay_and_cli_without_running_training(monkeypatch, capsys, options, expected):
    import train_all
    monkeypatch.setattr("sys.argv", ["train_all.py", "a5_review", "configs/dicm_pacs.yaml",
                                    *options, "--algorithm", "DICM", "--show"])
    monkeypatch.setattr(train_all, "get_dataset", lambda *a, **k: pytest.fail("--show accessed data"))
    monkeypatch.setattr(train_all, "get_writer", lambda *a, **k: pytest.fail("--show created outputs"))
    train_all.main()
    output = capsys.readouterr().out
    assert "dicm_ablation: A5" in output
    assert f"dicm_gt_weight: {expected}" in output and f"l_d: {expected}" in output
    assert "dicm_aug_weight: 0.25" in output and "dicm_dom_weight: 0.75" in output


def test_a0_config_reviews_original_four_loss_baseline_without_data_or_gpu(monkeypatch, capsys):
    import train_all
    monkeypatch.setattr("sys.argv", ["train_all.py", "a0_review", "configs/dccl_pacs_a0.yaml",
                                    "--algorithm", "DCCL", "--show"])
    monkeypatch.setattr(train_all, "get_dataset", lambda *a, **k: pytest.fail("A0 review accessed data"))
    monkeypatch.setattr(train_all, "get_writer", lambda *a, **k: pytest.fail("A0 review created outputs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("A0 review checked GPU"))
    train_all.main()
    output = capsys.readouterr().out
    assert "Algorithm: DCCL" in output and "Seed policy: per_target; seed=0; trial_seed=0" in output
    assert "l_d: 0.01" in output and "l: 1.0" in output and "l_layer: 1.0" in output
    assert "dicm_" not in output


@pytest.fixture
def trainer_harness(monkeypatch, tmp_path):
    """Real algorithms/trainer/SWAD on four synthetic domains; no PACS access."""
    from domainbed import trainer
    # Legacy LossValley returns .cuda(); intercept only this device handoff.
    # Its real averaging, source-val logic, and the DICM export remain exercised.
    monkeypatch.setattr(swa_utils.AveragedModel, "cuda", lambda self, *a, **k: self)

    class FakeSplit:
        def __init__(self, env):
            self.env = env
            self.keys = [4 * env + i for i in range(4)]

        def __len__(self):
            return len(self.keys)

    class FakeDataset:
        input_shape, num_classes, N_WORKERS = (6,), 3, 0
        environments = ["A", "C", "P", "S"]

        def __len__(self):
            return 4

    dataset_draws = []
    splits = [(FakeSplit(i), None) for i in range(4)]

    def get_dataset(*args, **kwargs):
        # Verify that even dataset construction starts from reset RNGs.
        dataset_draws.append((random.random(), np.random.rand(), torch.rand(3)))
        return FakeDataset(), splits, splits

    monkeypatch.setattr(trainer, "get_dataset", get_dataset)

    created = []
    get_algorithm_class = trainer.algorithms.get_algorithm_class

    def recording_class(name):
        original = get_algorithm_class(name)

        class RecordedAlgorithm(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.test_initial_state = copy.deepcopy(self.state_dict())
                self.test_metrics = []
                self.test_batches = []
                created.append(self)

            def update(self, x, y, **kwargs):
                self.test_batches.append(torch.cat([*x, *kwargs["x_2"]]).clone())
                result = super().update(x, y, **kwargs)
                self.test_metrics.append(dict(result))
                return result

        return RecordedAlgorithm

    monkeypatch.setattr(trainer.algorithms, "get_algorithm_class", recording_class)

    class Loader:
        def __init__(self, dataset, **kwargs):
            self.env = dataset.env

        def __iter__(self):
            while True:
                # Consume all three RNGs each step to expose task-length coupling.
                x = torch.randn(4, 6) + self.env / 10 + random.random() / 10
                aug = x + torch.randn_like(x) / 10 + np.random.rand() / 10
                yield {"x": x, "x_2": aug, "y": torch.tensor([0, 1, 2, 0]),
                       "d": torch.full((4,), self.env)}

    monkeypatch.setattr(trainer, "InfiniteDataLoader", Loader)

    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def evaluate(self, model):
            model.eval()
            with torch.no_grad():
                assert model.predict(torch.ones(4, 6)).shape == (4, 3)
            model.train()
            # Best target is deliberately at the worst source checkpoint.
            index = min(self.calls, 3)
            self.calls += 1
            return {}, {"test_in": [0.99, 0.2, 0.8, 0.3][index],
                        "test_out": [0.99, 0.1, 0.8, 0.4][index], "train_in": 0.0,
                        "train_out": [0.5, 0.9, 0.7, 0.8][index],
                        "tr_outloss": [1.0, 0.8, 0.9, 0.85][index]}

    monkeypatch.setattr(trainer, "Evaluator", Evaluator)

    class Sink:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def run(mode, target=1, steps=3, seed=0):
        algorithm = "DCCL" if mode == "A0" else "DICM"
        hp = hparams(mode if mode != "A0" else "A4")
        if mode == "A0":
            hp = {k: v for k, v in hp.items() if not k.startswith("dicm_")}
            hp["l_d"] = 0.01
        hp.update(swad="LossValley", swad_kwargs={"n_converge": 1, "n_tolerance": 1, "tolerance_ratio": 0.3},
                  indomain_test=False, test_batchsize=4)
        out_dir = tmp_path / f"job_{len(created)}"
        out_dir.mkdir()
        args = Namespace(algorithm=algorithm, prebuild_loader=False, trial_seed=0, seed=seed,
                         deterministic=True, out_dir=out_dir, evalmode="fast", debug=False, log=False,
                         model_save=None, tb_freq=1)
        result, records = trainer.train([target], args, Config(default=hp), steps, 1, Sink(), Sink())
        name = FakeDataset.environments[target]
        selection = json.loads((out_dir / f"selection_te_{name}.json").read_text())
        return Namespace(result=result, records=records, selection=selection, model=created[-1],
                         dataset_draw=dataset_draws[-1])

    return run


@pytest.mark.parametrize("mode", ["A0", "A4", "A5"])
def test_trainer_source_only_selection_swad_and_split_manifest(trainer_harness, mode):
    job = trainer_harness(mode)
    result, records, selection = job.result, job.records, job.selection
    assert "oracle" not in result
    assert result["iid"] == 0.2 and result["iid (test_out)"] == 0.1
    assert result["SWAD"] == 0.3 and result["SWAD (test_out)"] == 0.4
    assert selection["iid"]["step"] == 1 and selection["source_envs"] == [0, 2, 3]
    assert len(selection["source_split_sha256"]) == 64
    assert selection["swad"]["criterion"] == "source validation LossValley"
    assert selection["seed_policy"] == "per_target" and selection["seed"] == 0
    assert selection["deterministic"] and selection["ablation"] == mode
    assert selection["algorithm"] == ("DCCL" if mode == "A0" else "DICM")
    assert len(records) == 3


@pytest.mark.parametrize("mode", ["A0", "A1", "A2", "A3", "A4", "A5"])
def test_target_initialization_and_trajectory_ignore_prior_target_order_and_steps(trainer_harness, mode):
    standalone = trainer_harness(mode, target=3, steps=3)
    trainer_harness(mode, target=0, steps=3)
    after_short = trainer_harness(mode, target=3, steps=3)
    trainer_harness(mode, target=2, steps=5)
    trainer_harness(mode, target=1, steps=4)
    after_long = trainer_harness(mode, target=3, steps=3)
    for repeated in (after_short, after_long):
        assert standalone.dataset_draw[:2] == repeated.dataset_draw[:2]
        torch.testing.assert_close(standalone.dataset_draw[2], repeated.dataset_draw[2], atol=0, rtol=0)
        for key, value in standalone.model.test_initial_state.items():
            torch.testing.assert_close(value, repeated.model.test_initial_state[key], atol=0, rtol=0)
        for expected, actual in zip(standalone.model.test_batches, repeated.model.test_batches):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert standalone.model.test_metrics == repeated.model.test_metrics
        assert standalone.selection == repeated.selection and standalone.result == repeated.result
        if mode in {"A4", "A5"}:
            assert not standalone.model.test_initial_state["residual_bank.initialized"].any()
    # Same target, longer budget: the common updates remain identical.
    longer = trainer_harness(mode, target=3, steps=5)
    assert standalone.model.test_metrics == longer.model.test_metrics[:3]
