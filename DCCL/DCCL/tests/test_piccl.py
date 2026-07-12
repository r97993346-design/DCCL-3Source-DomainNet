import pytest
import torch
import torch.nn.functional as F

from domainbed.algorithms import get_algorithm_class
from domainbed.algorithms.piccl import (
    PairedInterventionResponseEstimator,
    ClassDomainResidualBank,
    InterventionSensitiveSubspace,
    CausalMediatorProjection,
    PICCL,
)


def test_pire_manual_and_detach():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    zi = torch.tensor([[3., 5.]], requires_grad=True)
    zr = torch.tensor([[.5, 1.]], requires_grad=True)
    zir = torch.tensor([[2., 2.]], requires_grad=True)
    out = PairedInterventionResponseEstimator()(z, zi, zr, zir)
    assert torch.allclose(out, torch.tensor([[0.5, 2.0]]))
    out.sum().backward()
    assert z.grad is not None and zi.grad is not None
    assert zr.grad is None and zir.grad is None


def test_pire_shape_mismatch():
    with pytest.raises(ValueError):
        PairedInterventionResponseEstimator()(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 3))
    with pytest.raises(ValueError):
        PairedInterventionResponseEstimator()(torch.zeros(2, 3, 1), torch.zeros(2, 3, 1), torch.zeros(2, 3, 1), torch.zeros(2, 3, 1))


def test_residual_bank_first_update_ema_missing_min_valid_state_dict():
    bank = ClassDomainResidualBank(2, 3, 2, momentum=0.5, min_valid_domains=2)
    r = torch.tensor([[1., 1.], [3., 3.], [10., 10.]])
    y = torch.tensor([0, 0, 1])
    d = torch.tensor([0, 0, 2])
    bank.update(r, y, d)
    assert bank.initialized[0, 0]
    assert torch.allclose(bank.prototypes[0, 0], torch.tensor([2., 2.]))
    assert bank.counts[0, 0].item() == 2
    bank.update(torch.tensor([[4., 4.]]), torch.tensor([0]), torch.tensor([0]))
    assert torch.allclose(bank.prototypes[0, 0], torch.tensor([3., 3.]))
    assert bank.domain_responses().shape[0] == 0
    bank.update(torch.tensor([[5., 1.]]), torch.tensor([0]), torch.tensor([1]))
    resp = bank.domain_responses()
    assert resp.shape == (2, 2)
    clone = ClassDomainResidualBank(2, 3, 2)
    clone.load_state_dict(bank.state_dict())
    assert torch.allclose(clone.prototypes, bank.prototypes)


def test_subspace_projection_no_dd_empty_zero_and_grad():
    sub = InterventionSensitiveSubspace(5, rank=2)
    assert sub.project(torch.randn(4, 5)).shape == (4, 5)
    for name, tensor in list(sub.named_parameters()) + list(sub.named_buffers()):
        assert tuple(tensor.shape) != (5, 5), name
    empty_loss = sub.coverage_loss(torch.zeros(0, 5))
    zero_loss = sub.coverage_loss(torch.zeros(3, 5))
    assert torch.isfinite(empty_loss) and torch.isfinite(zero_loss)
    loss = sub.coverage_loss(torch.randn(6, 5))
    loss.backward()
    assert sub.basis.grad is not None


def test_task_grad_false_blocks_basis_from_cmp_task_loss():
    sub = InterventionSensitiveSubspace(4, rank=2)
    cmp = CausalMediatorProjection(4)
    z = torch.randn(3, 4, requires_grad=True)
    loss = cmp(z, sub, torch.tensor(0.5), detach_basis=True).sum()
    loss.backward()
    assert sub.basis.grad is None
    assert z.grad is not None


def test_cmp_alpha_zero_and_positive():
    sub = InterventionSensitiveSubspace(4, rank=2)
    cmp = CausalMediatorProjection(4)
    z = torch.randn(2, 4)
    assert torch.allclose(cmp(z, sub, torch.tensor(0.0)), F.layer_norm(z, (4,)), atol=1e-6)
    assert cmp(z, sub, torch.tensor(0.5)).shape == z.shape


def test_nt_xent_and_cross_domain_edge_cases():
    obj = object.__new__(PICCL)
    obj.hparams = {"t": 0.1}
    q = F.normalize(torch.randn(1, 3), dim=1)
    assert torch.isfinite(PICCL._nt_xent(obj, q, q)).item()
    q2 = F.normalize(torch.randn(3, 3), dim=1)
    labels = torch.tensor([0, 1, 2])
    domains = torch.tensor([0, 0, 0])
    assert torch.isfinite(PICCL._cross_domain_supcon(obj, q2, labels, domains)).item()


def test_registry_piccl_and_dccl():
    assert get_algorithm_class("PICCL").__name__ == "PICCL"
    assert get_algorithm_class("DCCL").__name__ == "DCCL"
