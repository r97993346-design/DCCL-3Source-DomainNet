import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import SupConLoss
from domainbed.algorithms.piccl import PICCL


def _metadata():
    # View-major order: original samples 0..3, followed by their second views.
    labels = torch.tensor([0, 0, 1, 1]).repeat(2)
    domains = torch.tensor([0, 1, 0, 1]).repeat(2)
    samples = torch.arange(4).repeat(2)
    views = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)])
    reliability = torch.tensor([1.0, .25, .75, 0.0, 1.0, .25, .75, 0.0])
    return labels, domains, samples, views, reliability


def test_causal_pair_weights_scope_and_formula():
    labels, domains, samples, views, reliability = _metadata()
    obj = object.__new__(PICCL)
    weights, cross, self_aug = PICCL._causal_positive_pair_weights(
        obj, labels, domains, samples, views, reliability, .2)
    # Same sample across views and same-domain same-class pairs remain exactly one.
    assert torch.all(weights[self_aug] == 1)
    same_domain_positive = labels[:, None].eq(labels[None, :]) & domains[:, None].eq(domains[None, :])
    assert torch.all(weights[same_domain_positive] == 1)
    # sample 0/class 0/domain 0 vs sample 1/class 0/domain 1 is cross-domain.
    expected = .2 + .8 * (.25 ** .5)
    assert cross[0, 1] and weights[0, 1].item() == pytest.approx(expected)
    assert not cross[0, 2]  # different classes are never positive.
    assert torch.all((weights >= .2) & (weights <= 1))


def test_weighted_supcon_matches_unweighted_at_one_and_backpropagates():
    torch.manual_seed(4)
    features = torch.randn(4, 2, 5, requires_grad=True)
    labels, domains, samples, views, _ = _metadata()
    obj = object.__new__(PICCL)
    weights, _, _ = PICCL._causal_positive_pair_weights(
        obj, labels, domains, samples, views, torch.ones(8), .2)
    loss_fn = SupConLoss(.1)
    plain = loss_fn(features, labels[:4])
    weighted = loss_fn(features, labels[:4], positive_weights=weights)
    assert torch.allclose(plain, weighted, atol=1e-6, rtol=1e-5)
    weighted.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_reliability_is_finite_bounded_and_detached():
    obj = object.__new__(PICCL)
    obj.hparams = {"piccl_reliability_detach": True}
    factual = torch.randn(4, 3, requires_grad=True)
    intervened = torch.randn(4, 3, requires_grad=True)
    reliability = PICCL._causal_reliability(obj, factual, intervened)
    assert not reliability.requires_grad
    assert torch.isfinite(reliability).all()
    assert torch.all((reliability >= 0) & (reliability <= 1))


def test_gamma_schedule_and_cross_domain_switch():
    obj = object.__new__(PICCL)
    obj.hparams = {"piccl_total_steps": 100, "piccl_reliability_warmup_ratio": .1,
                   "piccl_reliability_ramp_ratio": .2}
    assert PICCL._reliability_gamma(obj, 0) == 0
    assert PICCL._reliability_gamma(obj, 20) == pytest.approx(.5)
    assert PICCL._reliability_gamma(obj, 30) == 1
    labels, domains, samples, views, reliability = _metadata()
    weights, cross, _ = PICCL._causal_positive_pair_weights(
        obj, labels, domains, samples, views, reliability, .2, cross_domain_only=False)
    assert not cross.any() and torch.all(weights == 1)
