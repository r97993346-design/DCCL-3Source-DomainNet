"""Check the upstream CE indexing and the four local DomainBed splits."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.functional import cross_entropy

from domainbed.algorithms import DPLCLIP
from domainbed.datasets import get_dataset
from domainbed.lib import dplclip_protocol as dpl


def audit_loss():
    # Two domains, two classes. Both class-1 samples match their own domain's
    # class-1 vector, and domain 1 has no similarity to the first prompt bank.
    images = [torch.eye(4)[[1]], torch.eye(4)[[3]]]
    texts = [torch.eye(4)[[0, 1]], torch.eye(4)[[2, 3]]]
    labels = torch.tensor([1, 1])
    upstream = 10 * torch.cat(images) @ torch.cat(texts).t()
    upstream_loss = cross_entropy(upstream, labels)
    corrected = torch.cat([10 * x @ t.t() for x, t in zip(images, texts)])
    corrected_loss = cross_entropy(corrected, labels)
    reversed_corrected = torch.cat([10 * x @ t.t() for x, t in zip(images[::-1], texts[::-1])])
    reversed_upstream = 10 * torch.cat(images[::-1]) @ torch.cat(texts[::-1]).t()
    torch.testing.assert_close(corrected.argmax(dim=1), labels)
    torch.testing.assert_close(cross_entropy(reversed_corrected, labels), corrected_loss)
    assert upstream[1].argmax().item() == 3 and labels[1] == 1
    assert reversed_upstream[1].argmax().item() == 3
    return {"upstream_logits": upstream.tolist(), "unchanged_labels": labels.tolist(),
            "upstream_ce": upstream_loss.item(), "paper_logits": corrected.tolist(),
            "paper_ce": corrected_loss.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    report = {"official_dpl_commit": dpl.OFFICIAL_COMMIT, "ce_indexing": audit_loss(), "datasets": {}}
    for name in dpl.DOMAINS:
        for target in range(4):
            options = SimpleNamespace(dataset=name, data_dir=args.data_dir, trial_seed=1,
                                      holdout_fraction=0.2)
            dataset, inside, outside = get_dataset([target], options, dpl.DEFAULTS, DPLCLIP)
            classes, domains = dpl.dataset_metadata(dataset, inside, outside, options)
            assert all(len(split[0]) > 0 for split in inside + outside)
            if target == 0:
                report["datasets"][name] = {"classes": classes, "domains": domains}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
