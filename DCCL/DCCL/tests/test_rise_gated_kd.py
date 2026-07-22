import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from domainbed.algorithms.algorithms import rise_gated_kd, rise_js_reliability


def _inputs():
    teacher = torch.tensor([[2.0, -1.0], [-.5, 1.0]])
    causal = torch.tensor([[.1, .2], [.3, -.2]], requires_grad=True)
    intervened = torch.tensor([[.1, .2], [.0, .4]], requires_grad=True)
    return teacher, causal, intervened, torch.tensor([0, 1])


def test_kd_direction_confidence_and_none_parity():
    teacher, causal, intervened, labels = _inputs()
    out = rise_gated_kd(teacher, causal, labels, 2.0, "none", intervened)
    expected = F.kl_div(F.log_softmax(causal / 2., -1), F.softmax(teacher / 2., -1), reduction="none").sum(-1).mean() * 4
    assert torch.allclose(out["loss_rise_kd"], expected)
    assert torch.allclose(out["teacher_confidence"], F.softmax(teacher, -1).gather(1, labels[:, None]).squeeze(1))


def test_js_and_gate_modes_stop_gradient_and_shapes():
    teacher, causal, intervened, labels = _inputs()
    js, reliability = rise_js_reliability(causal, causal)
    assert torch.allclose(js, torch.zeros_like(js), atol=1e-7)
    assert torch.allclose(reliability, torch.ones_like(reliability), atol=1e-7)
    js_far, reliability_far = rise_js_reliability(causal, -causal, beta=5.)
    assert js_far.mean() > js.mean() and reliability_far.mean() < reliability.mean()
    for mode in ("none", "confidence", "stability", "joint"):
        out = rise_gated_kd(teacher, causal, labels, 2., mode, intervened)
        assert out["gate_weight"].shape == labels.shape
        assert not out["gate_weight"].requires_grad
        out["loss_rise_kd"].backward(retain_graph=True)
    assert causal.grad is not None


def test_zero_weight_safety_and_missing_intervention_error():
    teacher, causal, _, labels = _inputs()
    out = rise_gated_kd(teacher, causal, labels, 2., "confidence")
    assert torch.isfinite(out["loss_rise_kd"])
    with pytest.raises(ValueError, match="intervened_causal_logits"):
        rise_gated_kd(teacher, causal, labels, 2., "joint")
