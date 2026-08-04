import inspect

import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import SupConLoss
from domainbed.algorithms.piccl import PICCL, PICCLForwardModel
from domainbed.algorithms.piccl_components import (
    CausalMediatorProjection,
    ClassDomainResidualBank,
    InterventionSensitiveSubspace,
    PairedInterventionResponseEstimator,
)
from domainbed.lib.swa_utils import AveragedModel


def test_original_supcon_api_is_untouched():
    assert "positive_weights" not in inspect.signature(SupConLoss.forward).parameters


def test_beta_zero_is_strict_identity():
    z = torch.randn(4, 8)
    mediator = CausalMediatorProjection()
    subspace = InterventionSensitiveSubspace(8, 2)
    assert mediator(z, subspace, torch.tensor(0.0)) is z


def test_projection_is_low_rank_qr_and_blocks_task_gradients():
    z = torch.randn(4, 8, requires_grad=True)
    subspace = InterventionSensitiveSubspace(8, 2)
    m = CausalMediatorProjection()(z, subspace, torch.tensor(0.2))
    m.sum().backward()
    assert subspace.basis.grad is None
    assert z.grad is not None
    q = subspace.orthonormal_basis(detach=True)
    assert torch.allclose(q.T @ q, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_isr_updates_basis_but_not_response_features():
    subspace = InterventionSensitiveSubspace(8, 2)
    response = torch.randn(6, 8, requires_grad=True)
    subspace.coverage_loss(response).backward()
    assert subspace.basis.grad is not None
    assert response.grad is None


def test_residual_bank_requires_minimum_samples_in_two_domains():
    bank = ClassDomainResidualBank(2, 2, 3, min_count=8, min_valid_domains=2)
    bank.update(torch.ones(7, 3), torch.zeros(7), torch.zeros(7))
    bank.update(torch.ones(8, 3), torch.zeros(8), torch.ones(8))
    assert bank.domain_responses().numel() == 0
    bank.update(torch.ones(1, 3), torch.zeros(1), torch.zeros(1))
    assert bank.domain_responses().shape == (2, 3)


def test_pire_reference_response_is_detached():
    z = torch.tensor([[1.0, 2.0]], requires_grad=True)
    zi = torch.tensor([[3.0, 5.0]], requires_grad=True)
    zr = torch.tensor([[0.5, 1.0]], requires_grad=True)
    zir = torch.tensor([[2.0, 2.0]], requires_grad=True)
    PairedInterventionResponseEstimator()(z, zi, zr, zir).sum().backward()
    assert z.grad is not None and zi.grad is not None
    assert zr.grad is None and zir.grad is None


def test_forward_model_uses_exported_subspace_and_beta():
    featurizer = torch.nn.Linear(3, 8)
    classifier = torch.nn.Linear(8, 2)
    subspace = InterventionSensitiveSubspace(8, 2)

    model = PICCLForwardModel(
        featurizer,
        subspace,
        classifier,
        torch.tensor(0.2),
    )

    x = torch.randn(5, 3)
    z = featurizer(x)
    q = subspace.orthonormal_basis(detach=True)
    expected = z - 0.2 * ((z @ q) @ q.T)

    assert torch.allclose(model.predict_embed(x), expected)

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    assert "sensitive_subspace.basis" in parameters
    assert "piccl_beta" in parameters
    assert parameters["piccl_beta"].requires_grad is False
    assert "basis_q" not in buffers
    assert not hasattr(model, "swa_latest_buffer_names")


def _projection_only_piccl(use_gate, gate_bias=-2.0):
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.use_piccl = True
    obj.use_residual_gate = use_gate
    obj.sensitive_subspace = InterventionSensitiveSubspace(8, 2)
    obj.causal_mediator = CausalMediatorProjection()
    if use_gate:
        obj.gate_logit = torch.nn.Parameter(torch.tensor(gate_bias))
    return obj


def test_disabled_residual_gate_is_exact_v6_projection():
    obj = _projection_only_piccl(False)
    z = torch.randn(4, 8, requires_grad=True)
    beta = torch.tensor(0.2)
    expected = obj.causal_mediator(z, obj.sensitive_subspace, beta)
    actual = PICCL._project(obj, z, beta)
    assert torch.equal(actual, expected)
    assert not hasattr(obj, "gate_logit")


def test_gated_projection_beta_zero_is_strict_identity():
    obj = _projection_only_piccl(True)
    z = torch.randn(4, 8, requires_grad=True)
    assert PICCL._project(obj, z, torch.tensor(0.0)) is z


def test_scalar_gate_gets_task_gradient_while_q_stays_detached():
    obj = _projection_only_piccl(True)
    assert obj.gate_logit.numel() == 1
    z = torch.randn(4, 8, requires_grad=True)
    PICCL._project(obj, z, torch.tensor(0.2)).square().sum().backward()
    assert obj.gate_logit.grad is not None
    assert torch.isfinite(obj.gate_logit.grad)
    assert obj.sensitive_subspace.basis.grad is None


def test_gated_training_and_forward_models_match_and_are_finite():
    torch.manual_seed(7)
    featurizer = torch.nn.Linear(3, 8)
    classifier = torch.nn.Linear(8, 2)
    obj = _projection_only_piccl(True, gate_bias=-1.5)
    beta = torch.tensor(0.2)
    model = PICCLForwardModel(
        featurizer,
        obj.sensitive_subspace,
        classifier,
        beta,
        use_residual_gate=True,
        gate_logit=obj.gate_logit,
    )
    x = torch.randn(5, 3)
    z = featurizer(x)
    train_embed = PICCL._project(obj, z, beta)
    inference_embed = model.predict_embed(x)
    task_loss = model(x).square().mean()
    assert torch.allclose(train_embed, inference_embed)
    assert torch.isfinite(train_embed).all()
    assert torch.isfinite(model(x)).all()
    assert torch.isfinite(task_loss)


def test_swad_copies_only_explicit_latest_piccl_buffers():
    source = torch.nn.Module()
    source.get_forward_model = lambda: source
    source.linear = torch.nn.Linear(2, 2)
    source.register_buffer("basis_q", torch.tensor([[1.0], [0.0]]))
    source.register_buffer("piccl_beta", torch.tensor(0.0))
    source.register_buffer("ordinary_buffer", torch.tensor(3.0))
    source.swa_latest_buffer_names = ("basis_q", "piccl_beta")
    averaged = AveragedModel(source)
    source.piccl_beta.fill_(0.2)
    source.ordinary_buffer.fill_(9.0)
    averaged.update_parameters(source)
    assert averaged.module.piccl_beta.item() == pytest.approx(0.2)
    assert averaged.module.ordinary_buffer.item() == pytest.approx(3.0)


def test_beta_warmup_then_ramp_does_not_gate_isr():
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.hparams = {
        "piccl_total_steps": 100,
        "piccl_warmup_steps": 10,
        "piccl_ramp_steps": 20,
        "piccl_beta_max": 0.1,
    }
    obj.register_buffer("piccl_beta", torch.tensor(0.0))
    assert PICCL._beta(obj, 0).item() == 0.0
    assert PICCL._beta(obj, 20).item() == pytest.approx(0.05)
    assert PICCL._beta(obj, 30).item() == pytest.approx(0.1)


def test_piccl_disabled_update_is_exact_dccl_delegation(monkeypatch):
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.use_piccl = False
    expected = {"loss": 1.0, "ce_loss": 0.5}

    def dccl_update(self, *args, **kwargs):
        return expected

    monkeypatch.setattr("domainbed.algorithms.piccl.DCCL.update", dccl_update)
    actual = PICCL.update(
        obj,
        [torch.randn(3, 4)],
        [torch.tensor([0, 1, 0])],
        x_2=[torch.randn(3, 4)],
    )
    assert actual is expected
    assert not hasattr(obj, "residual_bank")
    assert not hasattr(obj, "gate_logit")
