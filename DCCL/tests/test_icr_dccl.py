import torch
from domainbed.lib.intervention import denormalize_imagenet, normalize_imagenet, fourier_amplitude_intervention


def test_fourier_shape_finite_and_cross_domain_donor():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    d = torch.tensor([0,0,1,1,2,2,0,1])
    img = denormalize_imagenet(x)
    out, donor_idx, fallback = fourier_amplitude_intervention(img, d, cross_domain_only=True)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
    assert donor_idx.shape[0] == x.shape[0]
    assert fallback >= 0
    for i in range(d.shape[0]):
        assert d[donor_idx[i]] != d[i]


def test_fourier_fallback_single_domain():
    torch.manual_seed(1)
    x = torch.rand(4, 3, 16, 16)
    d = torch.tensor([0,0,0,0])
    out, donor_idx, fallback = fourier_amplitude_intervention(x, d, cross_domain_only=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert fallback == 4
    assert (donor_idx >= 0).all()


def test_norm_denorm_roundtrip_finite():
    x = torch.randn(2, 3, 8, 8)
    y = normalize_imagenet(denormalize_imagenet(x))
    assert torch.isfinite(y).all()
    assert torch.allclose(x, y, atol=1e-6)


def test_reliability_bounds():
    sim = torch.tensor([-1.0, 0.0, 1.0])
    r = torch.sigmoid((sim - 0.0) / 0.1)
    assert ((r >= 0) & (r <= 1)).all()
    R = torch.clamp(r.unsqueeze(1) * r.unsqueeze(0), 0.05, 1.0)
    assert ((R >= 0.05) & (R <= 1.0)).all()


def test_weighted_equals_original_when_weights_one():
    torch.manual_seed(0)
    pos = torch.randn(6, 6)
    mask = (torch.randn(6, 6) > 0).float()
    orig = (mask * pos).sum(1) / torch.clamp_min(mask.sum(1), 1.0)
    w = torch.ones(6, 6)
    weighted = (mask * w * pos).sum(1) / torch.clamp_min((mask * w).sum(1), 1.0)
    assert torch.allclose(orig, weighted, atol=1e-6)


def test_no_positive_pair_no_nan_and_backward_finite():
    torch.manual_seed(0)
    logits = torch.randn(4, 4, requires_grad=True)
    mask = torch.zeros(4, 4)
    mean = (mask * logits).sum(1) / torch.clamp_min(mask.sum(1), 1.0)
    loss = mean.mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
