import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, ResidualGateFusion, parse_bool


def _ids(groups):
    return [id(p) for g in groups for p in g["params"]]


def _dummy_piccl(strict=False):
    obj = object.__new__(PICCL)
    obj.hparams = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "piccl_lr_multiplier": 0.5,
        "piccl_strict_bypass": strict,
    }
    obj.featurizer = torch.nn.Linear(8, 8)
    obj.classifier = torch.nn.Linear(8, 2)
    obj.proj_head = torch.nn.Linear(8, 4)
    obj.mean_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.var_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.pre_proj_head = torch.nn.Linear(8, 4)
    obj.sensitive_subspace = torch.nn.Linear(8, 2)
    obj.causal_mediator = CausalMediatorProjection(8)
    obj.residual_gate = ResidualGateFusion(8)
    return obj


def test_residual_gate_accepts_zero_scale_without_defaulting():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    piccl = torch.randn(4, 8)
    gate = ResidualGateFusion(8, gate_bias=-4.0)
    fused, values = gate(z, piccl, scale=0, alpha=torch.tensor(1.0))
    torch.testing.assert_close(fused, z)
    assert values.min() >= 0 and values.max() <= 1


def test_scale_zero_but_auxiliary_loss_can_update_shared_feature():
    torch.manual_seed(0)
    backbone = torch.nn.Linear(4, 4)
    gate = ResidualGateFusion(4)
    classifier = torch.nn.Linear(4, 2)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    z = backbone(x)
    piccl_feature = z * 0.5
    fused, _ = gate(z, piccl_feature, scale=0, alpha=torch.tensor(1.0))
    torch.testing.assert_close(fused, z)
    cls_loss = torch.nn.functional.cross_entropy(classifier(fused), y)
    aux_loss = piccl_feature.pow(2).mean()
    (cls_loss + aux_loss).backward()
    assert backbone.weight.grad is not None
    assert backbone.weight.grad.norm().item() > 0


def test_piccl_optimizer_has_no_duplicate_parameters_with_dummy_instance():
    obj = _dummy_piccl(strict=False)
    groups = PICCL._optimizer_groups(obj)
    ids = _ids(groups)
    assert len(ids) == len(set(ids))
    assert len(groups) == 9
    assert groups[-1]["lr"] == 5e-4


def test_strict_bypass_optimizer_groups_match_dccl_groups():
    obj = _dummy_piccl(strict=True)
    groups = PICCL._optimizer_groups(obj)
    dccl_groups = PICCL._dccl_optimizer_groups(obj)
    assert len(groups) == len(dccl_groups) == 6
    assert _ids(groups) == _ids(dccl_groups)
    piccl_ids = {id(p) for m in (obj.sensitive_subspace, obj.causal_mediator, obj.residual_gate) for p in m.parameters()}
    assert not (set(_ids(groups)) & piccl_ids)


def test_piccl_parameter_gradients_can_flow_through_residual_gate():
    torch.manual_seed(0)
    gate = ResidualGateFusion(4, gate_bias=-2.0)
    z = torch.randn(3, 4, requires_grad=True)
    piccl = torch.randn(3, 4, requires_grad=True)
    fused, _ = gate(z, piccl, scale=0.5, alpha=torch.tensor(1.0))
    fused.pow(2).mean().backward()
    assert gate.linear.weight.grad is not None
    assert gate.linear.weight.grad.norm().item() > 0


def test_piccl_diagnostics_do_not_return_string_lr_list():
    import inspect
    source = inspect.getsource(PICCL.update)
    assert "param_group_lrs" not in source
    assert "param_group_lr_" in source
    assert "piccl_executed" in source


def test_parse_bool_alias_handles_false_and_zero_values():
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool(0) is False
    assert parse_bool("true") is True
