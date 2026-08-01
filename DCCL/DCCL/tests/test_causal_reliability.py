import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import SupConLoss
from domainbed.algorithms.piccl import PICCL
from domainbed.algorithms.piccl_components import (
    InterventionSensitiveSubspace,
    causal_pair_reliability,
    reliable_positive_weights,
)
from domainbed.algorithms.reliable_supcon import ReliableSupConLoss


def _x_axis_subspace():
    subspace = InterventionSensitiveSubspace(3, rank=1)
    with torch.no_grad():
        subspace.basis.copy_(torch.tensor([[1.0], [0.0], [0.0]]))
    return subspace


def test_pair_reliability_uses_raw_feature_difference_and_is_detached():
    subspace = _x_axis_subspace()
    z = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1])
    reliability, cross, raw, _ = causal_pair_reliability(
        z, labels, domains, subspace, min_delta_norm=1e-6
    )
    assert not reliability.requires_grad and not raw.requires_grad
    assert torch.allclose(reliability, reliability.T, atol=1e-6)
    assert reliability[0, 1] == pytest.approx(0.0)  # entirely sensitive
    assert reliability[2, 3] == pytest.approx(1.0)  # near-zero pair is safe
    assert torch.all(reliability[cross].isfinite())


def test_pair_reliability_rewards_invariant_not_sensitive_directions():
    subspace = _x_axis_subspace()
    z = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = torch.tensor([0, 0])
    domains = torch.tensor([0, 1])
    reliability, _, raw, _ = causal_pair_reliability(
        z, labels, domains, subspace
    )
    assert reliability[0, 1] == pytest.approx(1.0)
    assert raw.tolist() == pytest.approx([1.0, 1.0])


def test_only_cross_domain_same_class_positives_are_weighted():
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1])
    pair = torch.tensor(
        [
            [1.0, 0.6, 1.0, 1.0],
            [0.6, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 0.5, 1.0],
        ]
    )
    weights, cross, self_aug, expanded = reliable_positive_weights(
        pair, labels, domains, gamma=0.3, min_weight=0.5
    )
    assert weights.shape == (8, 8)
    assert expanded[0, 1] == pytest.approx(0.6)
    assert expanded[0, 5] == pytest.approx(0.6)  # view-major expansion
    assert weights[0, 1] == pytest.approx(0.88)
    assert torch.all(weights[self_aug] == 1)
    assert not cross[0, 2] and weights[0, 2] == 1  # different class
    assert weights[0, 4] == 1  # self augmentation


def test_all_one_reliable_weights_equal_original_dccl_supcon():
    torch.manual_seed(4)
    features = torch.randn(4, 2, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(8, 8)
    plain = SupConLoss(0.1)(features, labels)
    reliable = ReliableSupConLoss(0.1)(features, labels, weights)
    assert torch.allclose(plain, reliable, atol=1e-6, rtol=1e-6)
    reliable.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_reliability_gamma_has_independent_late_schedule():
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.hparams = {
        "piccl_use_reliable_contrast": True,
        "piccl_total_steps": 100,
        "piccl_reliable_contrast_warmup_steps": 30,
        "piccl_reliable_contrast_ramp_steps": 10,
        "piccl_reliable_contrast_gamma_max": 0.3,
    }
    assert PICCL._reliable_contrast_gamma(obj, 20) == 0.0
    assert PICCL._reliable_contrast_gamma(obj, 35) == pytest.approx(0.15)
    assert PICCL._reliable_contrast_gamma(obj, 40) == pytest.approx(0.3)
