"""CPU-only smoke tests for the GD-FAS loss port.

Run from DCCL/DCCL:
    python -m pytest -q tests/test_gdfas_losses.py

Uses importlib to avoid importing the full DomainBed / CLIP package.
"""
import importlib.util
from pathlib import Path

import torch


path = Path(__file__).resolve().parents[1] / "domainbed/algorithms/gdfas_losses.py"
spec = importlib.util.spec_from_file_location("gdfas_losses_test", str(path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_orthogonal_projection_and_gradients():
    torch.manual_seed(0)
    image = torch.randn(18, 512, requires_grad=True)
    text = torch.randn(7, 512, requires_grad=True)
    invariant, specific, basis = mod.fod_decomposition(image, text)
    assert invariant.shape == specific.shape == image.shape
    assert 1 <= basis.size(0) <= 7
    assert torch.allclose(invariant.detach() + specific.detach(), image.detach(), atol=1e-5)
    assert (invariant.detach() * specific.detach()).sum(dim=1).abs().max() < 1e-4
    domains = torch.tensor([0, 1, 2] * 3 * 2)
    loss = mod.fod_domain_loss(specific, domains)
    assert torch.isfinite(loss)
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_groupwise_scaling_for_seven_classes_sparse_groups():
    torch.manual_seed(7)
    logits = torch.randn(18, 7, requires_grad=True)
    labels = torch.tensor([0, 2, 6] * 6)
    domains = torch.tensor([0, 1, 2] * 6)
    loss = mod.groupwise_scaling_loss(logits, labels, domains, alpha=0.5493061)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_two_view_instance_contrastive():
    torch.manual_seed(10)
    z = torch.randn(18, 256, requires_grad=True)
    loss = mod.paired_image_instance_loss(z)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_single_group_gs_has_unit_scale():
    logits = torch.randn(4, 7, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long)
    domains = torch.zeros(4, dtype=torch.long)
    result = mod.groupwise_scaling_loss(logits, labels, domains, alpha=0.5493)
    reference = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(result, reference, atol=1e-6)
