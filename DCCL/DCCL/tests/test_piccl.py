import torch

from domainbed.algorithms.piccl import (
    PairedInterventionResponseEstimator,
    ClassDomainResidualBank,
    InterventionSensitiveSubspace,
    CausalMediatorProjection,
)


def test_paired_intervention_response_formula():
    estimator = PairedInterventionResponseEstimator()
    adapted = torch.tensor([[1.0, 2.0]])
    adapted_int = torch.tensor([[4.0, 8.0]])
    reference = torch.tensor([[0.5, 1.0]])
    reference_int = torch.tensor([[1.5, 3.0]])
    expected = torch.tensor([[2.0, 4.0]])
    actual = estimator(adapted, adapted_int, reference, reference_int)
    assert torch.allclose(actual, expected)


def test_low_rank_projection_shape_and_capture():
    module = InterventionSensitiveSubspace(feature_dim=6, rank=2)
    vectors = torch.randn(5, 6)
    projected = module.project(vectors)
    assert projected.shape == vectors.shape
    assert module.basis.shape == (6, 2)
    loss, capture = module.coverage_loss([vectors.detach()])
    assert loss.ndim == 0
    assert capture.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(capture)


def test_empty_and_zero_responses_are_safe():
    module = InterventionSensitiveSubspace(feature_dim=4, rank=2)
    empty_loss, empty_capture = module.coverage_loss([])
    zero_loss, zero_capture = module.coverage_loss([torch.zeros(3, 4)])
    assert torch.isfinite(empty_loss)
    assert torch.isfinite(empty_capture)
    assert torch.isfinite(zero_loss)
    assert torch.isfinite(zero_capture)


def test_basis_task_gradient_can_be_blocked():
    subspace = InterventionSensitiveSubspace(feature_dim=4, rank=2)
    mediator = CausalMediatorProjection(feature_dim=4)
    features = torch.randn(3, 4, requires_grad=True)
    output = mediator(features, subspace, alpha=0.5, receive_basis_grad=False)
    output.sum().backward()
    assert features.grad is not None
    assert subspace.basis.grad is None


def test_isr_updates_basis():
    subspace = InterventionSensitiveSubspace(feature_dim=4, rank=2)
    response = torch.randn(8, 4)
    loss, _ = subspace.coverage_loss([response])
    loss.backward()
    assert subspace.basis.grad is not None
    assert torch.isfinite(subspace.basis.grad).all()


def test_residual_bank_requires_two_domains():
    bank = ClassDomainResidualBank(
        num_classes=3,
        num_domains=2,
        feature_dim=4,
        momentum=0.9,
        min_valid_domains=2,
    )
    residuals = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
    )
    labels = torch.tensor([1, 1])
    domains = torch.tensor([0, 0])
    bank.update(residuals, labels, domains)
    response, mask = bank.responses_for(labels, domains)
    assert response.shape == (0, 4)
    assert not mask.any()

    bank.update(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([1]),
        torch.tensor([1]),
    )
    response, mask = bank.responses_for(
        torch.tensor([1, 1]), torch.tensor([0, 1])
    )
    assert response.shape == (2, 4)
    assert mask.all()
    assert torch.allclose(response[0], -response[1])


def test_residual_bank_state_dict_round_trip():
    bank = ClassDomainResidualBank(2, 2, 3)
    bank.update(
        torch.randn(4, 3),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    clone = ClassDomainResidualBank(2, 2, 3)
    clone.load_state_dict(bank.state_dict())
    assert torch.equal(bank.initialized, clone.initialized)
    assert torch.equal(bank.counts, clone.counts)
    assert torch.allclose(bank.prototypes, clone.prototypes)


def test_cmp_alpha_zero_is_layer_norm():
    subspace = InterventionSensitiveSubspace(feature_dim=5, rank=2)
    mediator = CausalMediatorProjection(feature_dim=5)
    features = torch.randn(6, 5)
    actual = mediator(features, subspace, alpha=0.0, receive_basis_grad=False)
    expected = mediator.norm(features)
    assert torch.allclose(actual, expected)
