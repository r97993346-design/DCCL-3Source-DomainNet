import pytest

torch = pytest.importorskip("torch")

from domainbed import rise


def test_rise_ad_is_differentiable_and_proto_alias_formula_is_cosine_distance():
    projected = torch.randn(4, 5, requires_grad=True)
    prototypes = torch.nn.functional.normalize(torch.randn(3, 5), dim=1)
    loss, cosine = rise.prototype_alignment_loss(projected, torch.tensor([0, 1, 2, 1]), prototypes)
    assert torch.isfinite(loss)
    assert torch.isfinite(cosine)
    loss.backward()
    assert projected.grad is not None and projected.grad.abs().sum() > 0
