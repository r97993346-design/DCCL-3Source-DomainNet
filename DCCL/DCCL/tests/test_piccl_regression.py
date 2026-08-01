import inspect

import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import PICCL
from domainbed.algorithms.piccl_components import InterventionSensitiveSubspace


def _dummy_piccl():
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.hparams = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "piccl_basis_lr_multiplier": 1.0,
        "piccl_basis_weight_decay": 0.0,
        "piccl_min_delta_norm": 1e-4,
        "piccl_isr_aug_weight": 0.25,
        "piccl_isr_dom_weight": 0.75,
    }
    obj.featurizer = torch.nn.Linear(8, 8)
    obj.classifier = torch.nn.Linear(8, 2)
    obj.proj_head = torch.nn.Linear(8, 4)
    obj.mean_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.var_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.pre_proj_head = torch.nn.Linear(8, 4)
    obj.sensitive_subspace = InterventionSensitiveSubspace(8, 2)
    obj.use_residual_gate = False
    return obj


def test_only_basis_is_added_when_residual_gate_is_disabled():
    obj = _dummy_piccl()
    groups = PICCL._piccl_optimizer_groups(obj)
    assert len(groups) == 7
    assert groups[-1]["lr"] == pytest.approx(1e-3)
    assert groups[-1]["weight_decay"] == 0.0


def test_scalar_gate_has_separate_backbone_lr_optimizer_group():
    obj = _dummy_piccl()
    obj.use_residual_gate = True
    obj.gate_logit = torch.nn.Parameter(torch.tensor(-2.0))
    groups = PICCL._piccl_optimizer_groups(obj)
    assert len(groups) == 8
    assert list(groups[-1]["params"]) == [obj.gate_logit]
    assert groups[-1]["lr"] == pytest.approx(obj.hparams["lr"])
    assert groups[-1]["weight_decay"] == 0.0


def test_isr_keeps_aug_and_domain_losses_separate():
    obj = _dummy_piccl()
    obj.sensitive_subspace.coverage_loss = (
        lambda values, *args, **kwargs: torch.tensor(float(values.shape[0]))
    )
    total, aug, dom = PICCL._isr_loss(
        obj, torch.ones(4, 8), torch.ones(2, 8)
    )
    assert aug.item() == 4
    assert dom.item() == 2
    assert total.item() == pytest.approx(2.5)


def test_domain_absence_falls_back_to_full_augmentation_isr():
    obj = _dummy_piccl()
    obj.sensitive_subspace.coverage_loss = (
        lambda values, *args, **kwargs: torch.tensor(float(values.shape[0]))
    )
    total, aug, dom = PICCL._isr_loss(
        obj, torch.ones(4, 8), torch.empty(0, 8)
    )
    assert total.item() == aug.item() == 4
    assert dom.item() == 0


def test_no_large_gate_layer_norm_or_semantic_filter_is_added():
    source = inspect.getsource(PICCL)
    for forbidden in (
        "piccl_fusion_mode",
        "LayerNorm",
        "semantic_confidence",
        "piccl_strict_bypass",
        "_nt_xent",
        "_cross_domain_supcon",
    ):
        assert forbidden not in source


def test_original_dccl_optional_branches_are_present_in_piccl_update():
    source = inspect.getsource(PICCL.update)
    for required in ("self.TN", "self.aug", "self.two_ce", "self.re_w", "self.sample_d"):
        assert required in source
