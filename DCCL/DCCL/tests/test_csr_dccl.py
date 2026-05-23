import torch

from domainbed.algorithms import algorithms as alg_module
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
    class DummyClipImage(torch.nn.Module):
        output_dim = text_dim
        def forward(self, x):
            # [B,3,H,W] -> [B,text_dim]
            pooled = x.mean(dim=(2, 3))
            rep = pooled.mean(dim=1, keepdim=True).repeat(1, text_dim)
            return rep
    algo.csr_clip_image_encoder = DummyClipImage()
    algo.causal_embeddings = torch.nn.functional.normalize(causal, dim=1)
    algo.spurious_embeddings = torch.nn.functional.normalize(spurious, dim=1)

    all_x = torch.randn(5, 3, 8, 8, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    d1 = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    d2 = torch.tensor([1, 1, 0, 2, 0], dtype=torch.long)
    w, stats = algo._build_pair_reliability(all_x, y, d1, d2)
    assert w.min() >= algo.reliability_min_weight - 1e-6
    assert w.max() <= 1.0 + 1e-6
    assert 0.0 <= stats["min"] <= stats["max"] <= 1.0

    loss = (w.mean() + all_x.pow(2).mean())
    loss.backward()
    assert torch.isfinite(all_x.grad).all()


def test_csr_disabled_does_not_init_clip_encoder():
    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.use_causal_reliability = False
    algo.csr_clip_image_encoder = None
    assert algo.csr_clip_image_encoder is None


def test_csr_init_uses_clip_load(monkeypatch, tmp_path):
    num_classes, dim = 3, 4
    c_path = tmp_path / "causal.pt"
    s_path = tmp_path / "spurious.pt"
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "ViT-B/32"}, c_path)
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "ViT-B/32"}, s_path)

    called = {"v": False}
    class DummyVisual(torch.nn.Module):
        output_dim = dim
        def forward(self, x):
            return torch.ones(x.shape[0], dim)
    class DummyClipModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = DummyVisual()
    def fake_load(name, device="cpu"):
        called["v"] = True
        return DummyClipModel(), None
    monkeypatch.setattr(alg_module.clip, "load", fake_load)

    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.num_classes = num_classes
    algo.hparams = {"lr": 1e-3, "weight_decay": 0.0}
    algo._extract_embedding_tensor = DCCL._extract_embedding_tensor.__get__(algo, DCCL)
    algo._init_causal_reliability_modules(
        {"causal_embedding_path": str(c_path), "spurious_embedding_path": str(s_path), "csr_clip_model": ""},
        default_visual_dim=dim,
    )
    assert called["v"] is True
    assert algo.csr_clip_image_encoder is not None


def test_csr_init_fails_on_encoder_name_mismatch(monkeypatch, tmp_path):
    num_classes, dim = 3, 4
    c_path = tmp_path / "causal.pt"
    s_path = tmp_path / "spurious.pt"
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "ViT-B/32", "class_names": ["a", "b", "c"]}, c_path)
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "RN50", "class_names": ["a", "b", "c"]}, s_path)

    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.num_classes = num_classes
    algo.hparams = {"lr": 1e-3, "weight_decay": 0.0}
    algo._extract_embedding_tensor = DCCL._extract_embedding_tensor.__get__(algo, DCCL)
    try:
        algo._init_causal_reliability_modules(
            {"causal_embedding_path": str(c_path), "spurious_embedding_path": str(s_path), "csr_clip_model": ""},
            default_visual_dim=dim,
        )
        assert False, "Expected ValueError for encoder_name mismatch."
    except ValueError as e:
        assert "encoder_name mismatch" in str(e)


def test_csr_init_fails_on_class_order_mismatch(tmp_path):
    num_classes, dim = 3, 4
    c_path = tmp_path / "causal.pt"
    s_path = tmp_path / "spurious.pt"
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "ViT-B/32", "class_names": ["a", "b", "c"]}, c_path)
    torch.save({"embeddings": torch.randn(num_classes, dim), "encoder_name": "ViT-B/32", "class_names": ["a", "c", "b"]}, s_path)

    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.num_classes = num_classes
    algo.hparams = {"lr": 1e-3, "weight_decay": 0.0}
    algo._extract_embedding_tensor = DCCL._extract_embedding_tensor.__get__(algo, DCCL)
    try:
        algo._init_causal_reliability_modules(
            {"causal_embedding_path": str(c_path), "spurious_embedding_path": str(s_path), "csr_clip_model": ""},
            default_visual_dim=dim,
        )
        assert False, "Expected ValueError for class_names mismatch."
    except ValueError as e:
        assert "class_names order mismatch" in str(e)


def test_csr_init_fails_on_shape_mismatch(tmp_path):
    c_path = tmp_path / "causal.pt"
    s_path = tmp_path / "spurious.pt"
    torch.save({"embeddings": torch.randn(3, 4), "encoder_name": "ViT-B/32", "class_names": ["a", "b", "c"]}, c_path)
    torch.save({"embeddings": torch.randn(3, 5), "encoder_name": "ViT-B/32", "class_names": ["a", "b", "c"]}, s_path)

    algo = object.__new__(DCCL)
    torch.nn.Module.__init__(algo)
    algo.num_classes = 3
    algo.hparams = {"lr": 1e-3, "weight_decay": 0.0}
    algo._extract_embedding_tensor = DCCL._extract_embedding_tensor.__get__(algo, DCCL)
    try:
        algo._init_causal_reliability_modules(
            {"causal_embedding_path": str(c_path), "spurious_embedding_path": str(s_path), "csr_clip_model": ""},
            default_visual_dim=4,
        )
        assert False, "Expected ValueError for embedding shape mismatch."
    except ValueError as e:
        assert "embedding shape mismatch" in str(e)
