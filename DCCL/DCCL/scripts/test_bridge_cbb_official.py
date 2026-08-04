"""Lightweight correctness checks for the official Bridge CBB port."""

import torch

from domainbed.models.bridge_cbb_official import MultiScaleBasisBlock


def main():
    torch.manual_seed(0)
    block = MultiScaleBasisBlock(in_channels=64, basis_reduction=2)
    x = torch.randn(2, 64, 7, 7, requires_grad=True)

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

    print("Bridge CBB official-port test passed")
    print(f"input/output shape: {tuple(x.shape)}")
    print(f"num reduced basis: {block.num_reduced_basis}")


if __name__ == "__main__":
    main()
