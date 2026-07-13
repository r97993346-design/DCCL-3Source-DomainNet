import pytest

torch = pytest.importorskip("torch")

<<<<<<< ours
from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, ResidualGateFusion
=======
from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, ResidualGateFusion, parse_bool
>>>>>>> theirs


def test_residual_gate_zero_scale_restores_original_feature():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    piccl = torch.randn(4, 8)
    gate = ResidualGateFusion(8, gate_bias=-4.0)
    fused, values = gate(z, piccl, scale=0.0, alpha=torch.tensor(1.0))
    torch.testing.assert_close(fused, z)
    assert values.min() >= 0 and values.max() <= 1


def test_piccl_optimizer_has_no_duplicate_parameters_with_dummy_instance():
    obj = object.__new__(PICCL)
    obj.hparams = {"lr": 1e-3, "weight_decay": 0.0, "piccl_lr_multiplier": 0.5}
    obj.featurizer = torch.nn.Linear(8, 8)
    obj.classifier = torch.nn.Linear(8, 2)
    obj.proj_head = torch.nn.Linear(8, 4)
    obj.mean_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.var_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.pre_proj_head = torch.nn.Linear(8, 4)
    obj.sensitive_subspace = torch.nn.Linear(8, 2)
    obj.causal_mediator = CausalMediatorProjection(8)
    obj.residual_gate = ResidualGateFusion(8)
    groups = PICCL._optimizer_groups(obj)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert groups[-1]["lr"] == 5e-4


def test_piccl_parameter_gradients_can_flow_through_residual_gate():
    torch.manual_seed(0)
    gate = ResidualGateFusion(4, gate_bias=-2.0)
    z = torch.randn(3, 4, requires_grad=True)
    piccl = torch.randn(3, 4, requires_grad=True)
    fused, _ = gate(z, piccl, scale=0.5, alpha=torch.tensor(1.0))
    fused.pow(2).mean().backward()
    assert gate.linear.weight.grad is not None
    assert gate.linear.weight.grad.norm().item() > 0
<<<<<<< ours
=======


def test_piccl_diagnostics_do_not_return_string_lr_list():
    source = __import__("inspect").getsource(PICCL.update)
    assert "param_group_lrs" not in source
    assert "param_group_lr_" in source


def test_parse_bool_alias_handles_false_string():
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("true") is True
>>>>>>> theirs
