"""Lightweight correctness checks for the official Bridge CBB port."""

import torch

from domainbed.models.bridge_cbb_official import (
    MultiScaleBasisBlock,
    ResidualBridgeBlock,
)


def main():
    torch.manual_seed(0)
    block = MultiScaleBasisBlock(in_channels=64, basis_reduction=2)
    x = torch.randn(2, 64, 7, 7, requires_grad=True)

    bridge_batch_norms = [
        name
        for name, module in block.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert not bridge_batch_norms, (
        "CBB must not keep BatchNorm running buffers under SWAD: "
        f"{bridge_batch_norms}"
    )
    assert isinstance(
        block.expected_input_estimator.refine_conv.norm, torch.nn.GroupNorm
    )
    assert isinstance(
        block.expected_mediator_estimator.refine_conv.norm, torch.nn.GroupNorm
    )

    with torch.no_grad():
        block.train()
        train_output = block(x.detach())
        block.eval()
        eval_output = block(x.detach())
    assert torch.allclose(train_output, eval_output, atol=1e-6, rtol=1e-5), (
        "stateless CBB normalization must behave consistently in train/eval"
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
        "gate_init": 0.0,
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
        "zero-initialized Bridge gate must preserve pretrained features exactly"
    )

    identity_output.square().mean().backward()
    assert adapter.gate.grad is not None
    assert torch.isfinite(adapter.gate.grad).all()

    adapter.zero_grad(set_to_none=True)
    residual_input.grad = None
    with torch.no_grad():
        adapter.gate.fill_(0.1)
    adapter(residual_input).square().mean().backward()
    missing_adapter_grads = [
        name
        for name, parameter in adapter.named_parameters()
        if parameter.grad is None
    ]
    assert not missing_adapter_grads, (
        f"Missing residual-adapter gradients: {missing_adapter_grads}"
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
