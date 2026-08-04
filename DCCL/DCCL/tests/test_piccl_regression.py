import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import CausalMediatorProjection, PICCL, PICCLForwardModel, ResidualGateFusion, parse_bool


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
    obj.residual_gate = ResidualGateFusion()
    return obj


def test_residual_gate_uses_fixed_scale_formula_and_endpoints():
    torch.manual_seed(0)
    original = torch.randn(4, 8)
    piccl_feature = torch.randn(4, 8)
    fusion = ResidualGateFusion()

    scale = 0.1
    fused = fusion(original, piccl_feature, scale)
    expected = original + scale * (piccl_feature - original)
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(fusion(original, piccl_feature, 0), original)
    torch.testing.assert_close(fusion(original, piccl_feature, 1), piccl_feature)
    assert list(fusion.parameters()) == []


def test_scale_zero_but_auxiliary_loss_can_update_shared_feature():
    torch.manual_seed(0)
    backbone = torch.nn.Linear(4, 4)
    fusion = ResidualGateFusion()
    classifier = torch.nn.Linear(4, 2)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    z = backbone(x)
    piccl_feature = z * 0.5
    fused = fusion(z, piccl_feature, scale=0)
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


def test_predict_embed_and_swad_use_identical_fixed_fusion():
    torch.manual_seed(0)
    featurizer = torch.nn.Linear(4, 4)
    subspace = torch.nn.Linear(4, 2)
    subspace.project = lambda z, detach_basis=True: torch.zeros_like(z)
    mediator = CausalMediatorProjection(4)
    classifier = torch.nn.Linear(4, 2)
    fusion = ResidualGateFusion()
    scale = 0.1

    piccl = object.__new__(PICCL)
    torch.nn.Module.__init__(piccl)
    piccl.hparams = {"use_piccl": True, "piccl_residual_scale": scale}
    piccl.piccl_strict_bypass = False
    piccl.featurizer = featurizer
    piccl.sensitive_subspace = subspace
    piccl.causal_mediator = mediator
    piccl.classifier = classifier
    piccl.residual_gate = fusion
    piccl.register_buffer("piccl_alpha", torch.tensor(0.5))

    swad = PICCLForwardModel(featurizer, subspace, mediator, classifier, fusion, scale)
    swad.piccl_alpha.copy_(piccl.piccl_alpha)
    x = torch.randn(3, 4)
    z = featurizer(x)
    piccl_feature = mediator(z, subspace, piccl.piccl_alpha, detach_basis=True)
    expected = z + scale * (piccl_feature - z)

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
