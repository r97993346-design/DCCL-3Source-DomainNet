import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import DCCL


def test_cross_domain_only_excludes_same_domain_and_has_differentiable_zero():
    algorithm = object.__new__(DCCL)
    algorithm.hparams = {"t": 0.1}
    q = torch.randn(3, 4, requires_grad=True)
    q = torch.nn.functional.normalize(q, dim=1)
    loss = algorithm._cross_domain_supcon(q, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]))
    assert loss.item() == 0.0
    loss.backward()
    q2 = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    assert torch.isfinite(algorithm._cross_domain_supcon(q2, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])))
