import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, PICCLForwardModel, parse_bool


def _ids(groups):
    return [id(p) for g in groups for p in g["params"]]


def _dummy_piccl(strict=False):
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
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
    return obj


def test_piccl_optimizer_has_no_duplicate_parameters_with_dummy_instance():
    obj = _dummy_piccl(strict=False)
    groups = PICCL._optimizer_groups(obj)
    ids = _ids(groups)
    assert len(ids) == len(set(ids))
    assert len(groups) == 8
    assert groups[-1]["lr"] == 5e-4


def test_strict_bypass_optimizer_groups_match_dccl_groups():
    obj = _dummy_piccl(strict=True)
    groups = PICCL._optimizer_groups(obj)
    dccl_groups = PICCL._dccl_optimizer_groups(obj)
    assert len(groups) == len(dccl_groups) == 6
    assert _ids(groups) == _ids(dccl_groups)
    piccl_ids = {id(p) for m in (obj.sensitive_subspace, obj.causal_mediator) for p in m.parameters()}
    assert not (set(_ids(groups)) & piccl_ids)


def test_predict_embed_and_swad_return_causal_mediator_features():
    torch.manual_seed(0)
    featurizer = torch.nn.Linear(4, 4)
    subspace = torch.nn.Linear(4, 2)
    subspace.project = lambda z, detach_basis=True: torch.zeros_like(z)
    mediator = CausalMediatorProjection(4)
    classifier = torch.nn.Linear(4, 2)

    piccl = object.__new__(PICCL)
    torch.nn.Module.__init__(piccl)
    piccl.hparams = {"use_piccl": True}
    piccl.piccl_strict_bypass = False
    piccl.featurizer = featurizer
    piccl.sensitive_subspace = subspace
    piccl.causal_mediator = mediator
    piccl.classifier = classifier
    piccl.register_buffer("piccl_alpha", torch.tensor(0.5))

    swad = PICCLForwardModel(featurizer, subspace, mediator, classifier)
    swad.piccl_alpha.copy_(piccl.piccl_alpha)
    x = torch.randn(3, 4)
    z = featurizer(x)
    piccl_feature = mediator(z, subspace, piccl.piccl_alpha, detach_basis=True)
    expected = piccl_feature

    torch.testing.assert_close(piccl.predict_embed(x), expected)
    torch.testing.assert_close(swad.predict_embed(x), expected)
    torch.testing.assert_close(swad.predict(x), classifier(expected))


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
