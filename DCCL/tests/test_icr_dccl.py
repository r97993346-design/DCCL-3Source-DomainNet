import torch
import pytest
import sys
import types
from argparse import Namespace

if "clip" not in sys.modules:
    sys.modules["clip"] = types.SimpleNamespace(load=lambda *args, **kwargs: (None, None))

from domainbed import hparams_registry
from domainbed.algorithms.algorithms import DCCL
from domainbed.lib.cl_hparams import setup_alg_hparams
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


def _build_args(**overrides):
    base = dict(
        dataset="DomainNet", model="resnet50", sup=True, two_ce=False, sample_d=False,
        re_w=False, pos_mask=False, mix=0.0, aug=0.0, label_ratio=1.0, TN=False,
        lamda=5.0, start_epoch=1000, log=False, use_fourier_intervention=False,
        use_intervention_reliability=False, fourier_mix_alpha=0.5, fourier_mix_min=0.1,
        fourier_mix_max=0.9, fourier_donor_cross_domain_only=True, intervention_mu=0.0,
        intervention_temperature=0.1, reliability_min_weight=0.05, reliability_loss_weight=1.0,
        detach_reliability_score=True, log_intervention_stats=True
        ,use_factorization_loss=False, factorization_loss_weight=0.01,
        factorization_offdiag_weight=0.005, factorization_eps=1e-4,
        factorization_feature_source="contrastive", log_factorization_stats=True,
        use_adversarial_masker=False, masker_aux_loss_weight=0.1, masker_adversarial_weight=1.0,
        mask_keep_ratio=0.5, gumbel_temperature=1.0, gumbel_hard=True, masker_hidden_dim=256,
        masker_update_interval=1, log_masker_stats=True
    )
    base.update(overrides)
    return Namespace(**base)


def _build_hparams(**overrides):
    args = _build_args(**overrides)
    hparams = hparams_registry.default_hparams("DCCL", "DomainNet")
    hparams = setup_alg_hparams(hparams, args)
    hparams["pretrained"] = False
    hparams["freeze_bn"] = True
    return hparams


def test_cli_stage1_keys_are_recognized_by_sconf():
    Config = pytest.importorskip("sconf").Config
    hparams = _build_hparams()
    cfg = Config(open("DCCL/DCCL/config.yaml", encoding="utf8"), default=hparams)
    cfg.argv_update([
        "--use_fourier_intervention", "true",
        "--use_intervention_reliability", "true",
        "--fourier_mix_alpha", "0.5",
        "--fourier_mix_min", "0.1",
        "--fourier_mix_max", "0.9",
        "--fourier_donor_cross_domain_only", "true",
        "--intervention_mu", "0.0",
        "--intervention_temperature", "0.1",
        "--reliability_min_weight", "0.05",
        "--reliability_loss_weight", "1.0",
        "--detach_reliability_score", "true",
        "--log_intervention_stats", "true",
    ])
    assert cfg.use_fourier_intervention is True
    assert cfg.use_intervention_reliability is True


def test_cli_stage2_factorization_keys_are_recognized_by_sconf():
    Config = pytest.importorskip("sconf").Config
    hparams = _build_hparams()
    cfg = Config(open("DCCL/DCCL/config.yaml", encoding="utf8"), default=hparams)
    cfg.argv_update([
        "--use_factorization_loss", "true",
        "--factorization_loss_weight", "0.01",
        "--factorization_offdiag_weight", "0.005",
        "--factorization_eps", "0.0001",
        "--factorization_feature_source", "contrastive",
        "--log_factorization_stats", "true",
    ])
    assert cfg.use_factorization_loss is True
    assert abs(cfg.factorization_loss_weight - 0.01) < 1e-9


def test_invalid_reliability_without_fourier_raises_clear_error():
    hparams = _build_hparams(use_fourier_intervention=False, use_intervention_reliability=True)
    try:
        DCCL((3, 224, 224), 2, 3, hparams)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "use_intervention_reliability=True requires use_fourier_intervention=True" in str(e)


def test_dccl_init_sets_reliability_weight_and_alias():
    hparams = _build_hparams(use_fourier_intervention=True, use_intervention_reliability=True, reliability_loss_weight=1.23)
    alg = DCCL((3, 224, 224), 2, 3, hparams)
    assert alg.reliability_loss_weight == 1.23
    assert alg.re_w == alg.reliability_loss_weight


def test_invalid_factorization_without_fourier_raises_clear_error():
    hparams = _build_hparams(use_fourier_intervention=False, use_factorization_loss=True)
    with pytest.raises(ValueError, match="use_factorization_loss=True requires use_fourier_intervention=True"):
        DCCL((3, 224, 224), 2, 3, hparams)


def test_invalid_masker_without_factorization_raises():
    hparams = _build_hparams(use_fourier_intervention=True, use_factorization_loss=False, use_adversarial_masker=True)
    with pytest.raises(ValueError, match="use_adversarial_masker=True requires use_factorization_loss=True"):
        DCCL((3, 224, 224), 2, 3, hparams)


def test_invalid_masker_without_fourier_raises():
    hparams = _build_hparams(use_fourier_intervention=False, use_factorization_loss=True, use_adversarial_masker=True)
    with pytest.raises(ValueError, match="use_factorization_loss=True requires use_fourier_intervention=True"):
        DCCL((3, 224, 224), 2, 3, hparams)
