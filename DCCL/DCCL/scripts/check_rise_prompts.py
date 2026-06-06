#!/usr/bin/env python3
"""Minimal self-checks for RISE prompt modes and text prototype shapes."""

import torch

from domainbed import rise


class DummyClipModule:
    @staticmethod
    def tokenize(prompts):
        return torch.arange(len(prompts), dtype=torch.long).view(-1, 1)


class DummyClipModel:
    def __init__(self, clip_dim=8):
        self.clip_dim = clip_dim

    def encode_text(self, tokens):
        base = tokens.float() + 1.0
        dims = torch.arange(1, self.clip_dim + 1, dtype=torch.float32).view(1, -1)
        return base * dims


def main():
    expected_counts = {
        "simple": 1,
        "multi": 5,
        "domain_invariant": 7,
        "rise80": 80,
    }
    for mode, expected in expected_counts.items():
        count = rise.prompt_count(mode)
        assert count == expected, f"{mode}: expected {expected} prompts, got {count}"

    prompts = rise.build_prompts(["airplane"], "rise80")[0]
    assert len(prompts) == 80, f"rise80 expanded prompt count must be 80, got {len(prompts)}"
    assert prompts[:5] == [
        "a bad photo of a airplane.",
        "a photo of many airplane.",
        "a sculpture of a airplane.",
        "a photo of the hard to see airplane.",
        "a low resolution photo of the airplane.",
    ], prompts[:5]

    prototypes = rise.build_text_prototypes(
        DummyClipModel(clip_dim=8),
        DummyClipModule,
        ["airplane", "alarm_clock", "zebra"],
        "rise80",
        device="cpu",
    )
    assert prototypes.shape == (3, 8), f"Expected prototype shape (3, 8), got {tuple(prototypes.shape)}"
    norms = prototypes.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), norms
    print("RISE prompt self-check passed")


if __name__ == "__main__":
    main()
