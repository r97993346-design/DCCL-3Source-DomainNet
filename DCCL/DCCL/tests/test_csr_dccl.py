import torch

from domainbed.algorithms.algorithms import SupConLoss, DCCL


def _make_features(batch_size=8, dim=16):
    torch.manual_seed(7)
    v1 = torch.randn(batch_size, dim)
    v2 = torch.randn(batch_size, dim)
    return torch.stack([torch.nn.functional.normalize(v1, dim=1), torch.nn.functional.normalize(v2, dim=1)], dim=1)


def test_supcon_pair_weight_none_keeps_original():
    features = _make_features()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    loss_fn = SupConLoss(temperature=0.1)
    base = loss_fn(features, labels)
    no_weight = loss_fn(features, labels, pair_weight=None)
    assert torch.allclose(base, no_weight, atol=1e-6, rtol=1e-6)


def test_supcon_all_one_pair_weight_equivalent():
    features = _make_features()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    loss_fn = SupConLoss(temperature=0.1)
    base = loss_fn(features, labels)
    ones = torch.ones(features.shape[0] * 2, features.shape[0] * 2)
    weighted = loss_fn(features, labels, pair_weight=ones)
    assert torch.allclose(base, weighted, atol=1e-6, rtol=1e-6)


def test_supcon_no_positive_not_nan():
    features = _make_features(batch_size=6, dim=8)
    labels = torch.arange(6, dtype=torch.long)
    loss_fn = SupConLoss(temperature=0.1)
    out = loss_fn(features, labels)
    assert torch.isfinite(out)


def test_csr_pair_reliability_range_and_backward(tmp_path):
    torch.manual_seed(3)
    num_classes = 4
    text_dim = 12
    causal = torch.randn(num_classes, text_dim)
    spurious = torch.randn(num_classes, text_dim)
    c_path = tmp_path / "causal.pt"
    s_path = tmp_path / "spurious.pt"
    torch.save(causal, c_path)
    torch.save({"embeddings": spurious}, s_path)

    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.causal_beta = 0.5
    algo.causal_temperature = 1.0
    algo.reliability_min_weight = 0.05
    algo.causal_text_proj = torch.nn.Linear(10, text_dim)
    algo.causal_embeddings = torch.nn.functional.normalize(causal, dim=1)
    algo.spurious_embeddings = torch.nn.functional.normalize(spurious, dim=1)

    embed = torch.randn(5, 10, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    d1 = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    d2 = torch.tensor([1, 1, 0, 2, 0], dtype=torch.long)
    w, stats = algo._build_pair_reliability(embed, y, d1, d2)
    assert w.min() >= algo.reliability_min_weight - 1e-6
    assert w.max() <= 1.0 + 1e-6
    assert 0.0 <= stats["min"] <= stats["max"] <= 1.0

    loss = (w.mean() + embed.pow(2).mean())
    loss.backward()
    assert torch.isfinite(embed.grad).all()
