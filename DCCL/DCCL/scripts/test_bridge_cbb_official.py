"""Lightweight correctness checks for the official Bridge CBB port."""

import torch
import torch.nn as nn

from domainbed.lib import swa_utils
from domainbed.models.bridge_cbb_official import (
    MultiScaleBasisBlock,
    ResidualBridgeBlock,
)


class _ToySelectiveBNModel(nn.Module):
    """Model with a frozen-backbone BN and a Bridge BN for SWAD refresh tests."""

    def __init__(self):
        super().__init__()
        self.backbone_bn = nn.BatchNorm2d(4)
        self.bridge_adapter = nn.Sequential(nn.BatchNorm2d(4))

    def forward(self, x):
        x = self.backbone_bn(x)
        return self.bridge_adapter(x)


def _toy_iterator():
    while True:
        yield [{"x": torch.randn(8, 4, 3, 3) + 2.0}]


def main():
    torch.manual_seed(0)
    block = MultiScaleBasisBlock(in_channels=64, basis_reduction=2)
    x = torch.randn(2, 64, 7, 7, requires_grad=True)

    # Match the official mixed normalization design: outer mediator/fusion use
    # GN, while the two expectation-estimator refine convolutions use BN.
    assert isinstance(block.mediator_conv.norm, torch.nn.GroupNorm)
    assert isinstance(block.fusion_conv.norm, torch.nn.GroupNorm)
    assert isinstance(
        block.expected_input_estimator.refine_conv.norm,
        torch.nn.BatchNorm2d,
    )
    assert isinstance(
        block.expected_mediator_estimator.refine_conv.norm,
        torch.nn.BatchNorm2d,
    )

    block.train()
    y = block(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    assert torch.isfinite(y).all(), "CBB output contains NaN or Inf"

    loss = y.square().mean()
    loss.backward()

    required_grads = {
        "expected_input_basis": (
            block.expected_input_estimator.expectation_basis.grad
        ),
        "expected_mediator_basis": (
            block.expected_mediator_estimator.expectation_basis.grad
        ),
        "expected_input_query": (
            block.expected_input_estimator.sample_query_proj.conv.weight.grad
        ),
        "expected_mediator_query": (
            block.expected_mediator_estimator.sample_query_proj.conv.weight.grad
        ),
        "mediator_conv": block.mediator_conv.conv.weight.grad,
        "fusion_conv": block.fusion_conv.conv.weight.grad,
    }
    missing = [name for name, grad in required_grads.items() if grad is None]
    assert not missing, f"Missing gradients: {missing}"
    for name, grad in required_grads.items():
        assert torch.isfinite(grad).all(), f"Non-finite gradient: {name}"

    adapter_kwargs = {
        "bridge_channels": 32,
        "residual_scale": 0.1,
        "basis_reduction": 2,
        "basis_reduction_mode": "div",
        "with_ssp": True,
        "with_query": True,
        "with_input_subspace": False,
        "with_dropout": False,
        "basis_normalize": True,
        "conv_kernel_size": 3,
    }
    adapter = ResidualBridgeBlock(64, **adapter_kwargs)
    residual_input = torch.randn(2, 64, 7, 7, requires_grad=True)
    identity_output = adapter(residual_input)
    assert torch.equal(identity_output, residual_input), (
        "zero-initialized expand layer must preserve pretrained features exactly"
    )

    identity_output.square().mean().backward()
    assert adapter.expand.weight.grad is not None
    assert torch.isfinite(adapter.expand.weight.grad).all()
    assert adapter.expand.weight.grad.abs().sum() > 0

    # Once the expansion path has moved away from zero, gradients must reach
    # the entire CBB stack without relying on a learnable scalar gate.
    adapter.zero_grad(set_to_none=True)
    residual_input.grad = None
    with torch.no_grad():
        adapter.expand.weight.normal_(mean=0.0, std=1e-3)
    adapter(residual_input).square().mean().backward()
    missing_adapter_grads = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is None
    ]
    assert not missing_adapter_grads, (
        f"Missing residual-adapter gradients: {missing_adapter_grads}"
    )

    # Targeted SWAD BN refresh: Bridge BN changes, backbone BN remains intact.
    toy = _ToySelectiveBNModel()
    toy.eval()
    with torch.no_grad():
        toy.backbone_bn.running_mean.fill_(7.0)
        toy.backbone_bn.running_var.fill_(3.0)
        toy.bridge_adapter[0].running_mean.fill_(-5.0)
        toy.bridge_adapter[0].running_var.fill_(4.0)

    backbone_mean_before = toy.backbone_bn.running_mean.clone()
    backbone_var_before = toy.backbone_bn.running_var.clone()
    bridge_mean_before = toy.bridge_adapter[0].running_mean.clone()

    swa_utils.update_bn(
        _toy_iterator(),
        toy,
        n_steps=4,
        device="cpu",
    )

    assert torch.equal(toy.backbone_bn.running_mean, backbone_mean_before)
    assert torch.equal(toy.backbone_bn.running_var, backbone_var_before)
    assert not torch.equal(
        toy.bridge_adapter[0].running_mean,
        bridge_mean_before,
    )

    resnet50_adapter = ResidualBridgeBlock(
        2048, **{**adapter_kwargs, "bridge_channels": 256}
    )
    adapter_parameter_count = sum(
        parameter.numel() for parameter in resnet50_adapter.parameters()
    )
    assert adapter_parameter_count < 5_000_000, adapter_parameter_count

    print("Bridge CBB official-port test passed")
    print(f"input/output shape: {tuple(x.shape)}")
    print(f"num reduced basis: {block.num_reduced_basis}")
    print(f"resnet50 adapter parameters: {adapter_parameter_count}")


if __name__ == "__main__":
    main()
