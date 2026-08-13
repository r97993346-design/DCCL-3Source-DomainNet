"""CPU smoke/regression tests for the parameter-free PCL auxiliary loss."""

import torch

from domainbed.lib.pcl import PCLLoss


def main():
    torch.manual_seed(0)

    features = torch.randn(18, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2] * 3)
    domains = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)

    criterion = PCLLoss(
        transport_mass=0.8,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=100,
        uniform_temperature=2.0,
    )
    align, uniform, stats = criterion(features, labels, domains)
    total = align + 0.1 * uniform

    assert torch.isfinite(align), align
    assert torch.isfinite(uniform), uniform
    assert torch.isfinite(total), total
    assert stats["valid_domain_pairs"] == 3.0, stats
    assert stats["valid_class_pairs"] == 9.0, stats
    assert abs(stats["transport_mass"] - 0.8) < 5e-3, stats

    total.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert sum(p.numel() for p in criterion.parameters()) == 0

    # Regression: with epsilon=0.05 and cosine cost=2, exp(-cost/epsilon)
    # is ~4e-18. The old standard-space Sinkhorn implementation clamped its
    # scaling denominator to 1e-8 and returned only ~0.04 real-real mass.
    # Log-domain Sinkhorn must still honor the requested partial transport mass.
    extreme_cost = torch.tensor([[2.0]], dtype=torch.float64)
    extreme_plan = criterion._masked_partial_sinkhorn(extreme_cost)
    assert torch.isfinite(extreme_plan).all(), extreme_plan
    assert abs(float(extreme_plan.sum()) - 0.8) < 1e-6, extreme_plan

    print("PCL smoke/regression tests passed")
    print(
        {
            "align": float(align.detach()),
            "uniform": float(uniform.detach()),
            **stats,
            "extreme_transport_mass": float(extreme_plan.sum()),
        }
    )


if __name__ == "__main__":
    main()
