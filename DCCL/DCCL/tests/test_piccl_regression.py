import copy

import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import DCCL
from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, ResidualGateFusion, parse_bool


class TinyFeaturizer(torch.nn.Module):
    n_outputs = 8

    def __init__(self, input_shape, hparams, freeze=0, pre=False):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], self.n_outputs)
        self.bn = torch.nn.BatchNorm1d(self.n_outputs, affine=False, track_running_stats=True)
        self.dropout = torch.nn.Dropout(0.0)
        if freeze == "all":
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x, ret_feats=False):
        z = self.dropout(self.bn(self.linear(self.flatten(x))))
        if ret_feats:
            return z, [z]
        return z


def hparams(use_piccl=False):
    return {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "aug": 0,
        "n_layer": 1,
        "l_layer": 1,
        "l": 1,
        "l_d": 0.01,
        "two_ce": False,
        "pos_mask": False,
        "TN": False,
        "lamda": 5,
        "sample_d": False,
        "t": 0.1,
        "t_pre": 0.2,
        "re_w": False,
        "freeze_bn": True,
        "resnet_dropout": 0.0,
        "model": "resnet50",
        "pretrained": False,
        "use_piccl": use_piccl,
    }


def fixed_minibatches():
    g = torch.Generator().manual_seed(123)
    x = [torch.randn(3, 3, 4, 4, generator=g), torch.randn(3, 3, 4, 4, generator=g)]
    x2 = [t + 0.01 for t in x]
    y = [torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0])]
    d = [torch.full((3,), i, dtype=torch.long) for i in range(2)]
    return x, y, {"x_2": x2, "d": d, "d_2": d, "step": 0}


def named_trainable_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def optimizer_param_names(model):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [[by_id[id(p)] for p in group["params"]] for group in model.optimizer.param_groups]


@pytest.fixture(autouse=True)
def tiny_featurizer(monkeypatch):
    import domainbed.algorithms.algorithms as algorithms_module

    monkeypatch.setattr(algorithms_module.networks, "Featurizer", TinyFeaturizer)


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


@pytest.mark.parametrize("value", [False, "false", "False", "0", 0, "off", "no"])
def test_use_piccl_false_values_parse_to_false(value):
    assert parse_bool(value) is False


def test_use_piccl_false_constructs_exact_dccl_bypass():
    torch.manual_seed(7)
    dccl = DCCL((3, 4, 4), 3, 2, hparams(False))
    torch.manual_seed(7)
    piccl_off = PICCL((3, 4, 4), 3, 2, hparams("false"))

    assert piccl_off.use_piccl is False
    assert not hasattr(piccl_off, "sensitive_subspace")
    assert not hasattr(piccl_off, "causal_mediator")
    assert not hasattr(piccl_off, "residual_gate")

    assert [len(g["params"]) for g in dccl.optimizer.param_groups] == [len(g["params"]) for g in piccl_off.optimizer.param_groups]
    assert len(dccl.optimizer.param_groups) == len(piccl_off.optimizer.param_groups)
    assert optimizer_param_names(dccl) == optimizer_param_names(piccl_off)

    for name, expected in dccl.named_parameters():
        actual = dict(piccl_off.named_parameters())[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    x, y, kwargs = fixed_minibatches()
    with torch.no_grad():
        torch.testing.assert_close(piccl_off.featurizer(torch.cat(x)), dccl.featurizer(torch.cat(x)), rtol=0, atol=0)
        torch.testing.assert_close(piccl_off.predict(torch.cat(x)), dccl.predict(torch.cat(x)), rtol=0, atol=0)

    dccl_before = named_trainable_params(dccl)
    piccl_before = named_trainable_params(piccl_off)
    assert dccl_before.keys() == piccl_before.keys()

    dccl_result = dccl.update(copy.deepcopy(x), copy.deepcopy(y), **copy.deepcopy(kwargs))
    piccl_result = piccl_off.update(copy.deepcopy(x), copy.deepcopy(y), **copy.deepcopy(kwargs))
    for key in ["loss", "ce_loss", "sup_cl_loss", "pre_cl_loss"]:
        torch.testing.assert_close(torch.tensor(piccl_result[key]), torch.tensor(dccl_result[key]), rtol=0, atol=0)

    for name, expected in dccl.named_parameters():
        actual = dict(piccl_off.named_parameters())[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
