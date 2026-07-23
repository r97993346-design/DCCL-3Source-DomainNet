import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import (
    CausalMediatorProjection, ClassDomainResidualBank, InterventionSensitiveSubspace,
    PairedInterventionResponseEstimator, PICCLForwardModel,
)
from domainbed.lib.swa_utils import AveragedModel


def test_beta_zero_is_strict_identity():
    torch.manual_seed(0)
    z = torch.randn(4, 8)
    mediator = CausalMediatorProjection()
    subspace = InterventionSensitiveSubspace(8, 2)
    assert mediator(z, subspace, torch.tensor(0.0)) is z


def test_projection_is_low_rank_qr_and_blocks_task_gradients():
    z = torch.randn(4, 8, requires_grad=True)
    subspace = InterventionSensitiveSubspace(8, 2)
    m = CausalMediatorProjection()(z, subspace, torch.tensor(.2))
    m.sum().backward()
    assert subspace.basis.grad is None
    assert z.grad is not None


def test_residual_bank_requires_eight_samples_in_two_domains():
    bank = ClassDomainResidualBank(2, 2, 3)
    bank.update(torch.ones(7, 3), torch.zeros(7), torch.zeros(7))
    bank.update(torch.ones(8, 3), torch.zeros(8), torch.ones(8))
    assert bank.domain_responses().numel() == 0
    bank.update(torch.ones(1, 3), torch.zeros(1), torch.zeros(1))
    assert bank.domain_responses().shape == (2, 3)


def test_pire_reference_response_is_detached():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    zi = torch.tensor([[3., 5.]], requires_grad=True)
    zr = torch.tensor([[.5, 1.]], requires_grad=True)
    zir = torch.tensor([[2., 2.]], requires_grad=True)
    PairedInterventionResponseEstimator()(z, zi, zr, zir).sum().backward()
    assert z.grad is not None and zi.grad is not None
    assert zr.grad is None and zir.grad is None


def test_forward_model_predict_embed_matches_projection_path():
    featurizer, classifier = torch.nn.Linear(3, 8), torch.nn.Linear(8, 2)
    subspace, mediator = InterventionSensitiveSubspace(8, 2), CausalMediatorProjection()
    model = PICCLForwardModel(featurizer, subspace, mediator, classifier, torch.tensor(.2))
    x = torch.randn(5, 3)
    assert torch.equal(model.predict_embed(x), mediator(featurizer(x), subspace, torch.tensor(.2)))


def test_swad_copies_piccl_beta_buffer():
    source = torch.nn.Module()
    source.get_forward_model = lambda: source
    source.register_buffer("piccl_beta", torch.tensor(.2))
    source.linear = torch.nn.Linear(2, 2)
    averaged = AveragedModel(source)
    averaged.update_parameters(source)
    assert averaged.module.piccl_beta.item() == pytest.approx(.2)
