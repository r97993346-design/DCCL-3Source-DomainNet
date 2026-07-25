import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.algorithms import SupConLoss
from domainbed.algorithms.piccl import InterventionSensitiveSubspace, PICCL


def _piccl_for_reliable_contrast():
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.sensitive_subspace = InterventionSensitiveSubspace(3, rank=1)
    with torch.no_grad():
        obj.sensitive_subspace.basis.copy_(torch.tensor([[1.], [0.], [0.]]))
    obj.hparams = {"piccl_reliable_contrast_min_delta_norm": 1e-6,
                   "piccl_reliable_contrast_min_weight": .5,
                   "piccl_reliable_contrast_gamma_max": .3,
                   "piccl_reliable_contrast_warmup_steps": 0,
                   "piccl_reliable_contrast_ramp_steps": 10,
                   "piccl_total_steps": 100, "piccl_warmup_steps": 0,
                   "piccl_ramp_steps": 0, "piccl_warmup_ratio": 0., "piccl_ramp_ratio": 0.}
    return obj


def test_pair_reliability_is_symmetric_detached_and_handles_near_zero_delta():
    obj = _piccl_for_reliable_contrast()
    z = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 0.]], requires_grad=True)
    labels, domains = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    reliability, cross, raw, _ = PICCL._pair_reliability(obj, z, labels, domains)
    assert not reliability.requires_grad and not raw.requires_grad
    assert torch.allclose(reliability, reliability.T, atol=1e-6)
    assert reliability[0, 1] == pytest.approx(1.)  # difference lies in P_s
    assert reliability[2, 3] == pytest.approx(1.)  # near-zero delta is reliable
    assert torch.all(reliability[cross].isfinite())


def test_masks_weights_and_view_major_expansion():
    obj = _piccl_for_reliable_contrast()
    labels, domains = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    pair = torch.tensor([[1., .6, 1., 1.], [.6, 1., 1., 1.], [1., 1., 1., .5], [1., 1., .5, 1.]])
    weights, cross, self_aug, expanded = PICCL._reliable_positive_pair_weights(obj, pair, labels, domains, .3)
    assert weights.shape == (8, 8)
    assert expanded[0, 1] == pytest.approx(.6)
    assert expanded[0, 5] == pytest.approx(.6)  # view-major sample ids
    assert weights[0, 1] == pytest.approx(.88)
    assert torch.all(weights[self_aug] == 1)
    assert not cross[0, 2] and weights[0, 2] == 1  # different class
    logits_mask = 1 - torch.eye(8, dtype=weights.dtype)
    assert torch.all((weights * logits_mask).diag() == 0)


@pytest.mark.parametrize("gamma,min_weight", [(0., .5), (.3, 1.)])
def test_weighted_supcon_degenerates_to_original(gamma, min_weight):
    torch.manual_seed(4)
    obj = _piccl_for_reliable_contrast()
    obj.hparams["piccl_reliable_contrast_min_weight"] = min_weight
    features = torch.randn(4, 2, 5, requires_grad=True)
    labels, domains = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    weights, _, _, _ = PICCL._reliable_positive_pair_weights(obj, torch.full((4, 4), .5), labels, domains, gamma)
    loss_fn = SupConLoss(.1)
    plain, weighted = loss_fn(features, labels), loss_fn(features, labels, positive_weights=weights)
    assert torch.allclose(plain, weighted, atol=1e-6, rtol=1e-6)
    weighted.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_gamma_respects_piccl_warmup_and_ramp():
    obj = _piccl_for_reliable_contrast()
    obj.hparams.update({"piccl_warmup_steps": 5, "piccl_ramp_steps": 1,
                        "piccl_reliable_contrast_warmup_steps": 5,
                        "piccl_reliable_contrast_ramp_steps": 10})
    assert PICCL._reliable_contrast_gamma(obj, 0) == 0.
    assert PICCL._reliable_contrast_gamma(obj, 5) == 0.
    assert PICCL._reliable_contrast_gamma(obj, 10) == pytest.approx(.15)
    assert PICCL._reliable_contrast_gamma(obj, 15) == pytest.approx(.3)
