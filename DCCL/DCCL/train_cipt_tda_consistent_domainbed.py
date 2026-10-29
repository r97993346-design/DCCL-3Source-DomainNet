#!/usr/bin/env python3
"""Candidate-class-consistent TDA reproduction with original DomainBed splits.

The original entry point / original CIPT submodule are deliberately untouched.
Reuses exactly the source data, optimizer, 30-epoch loop, losses, checkpoints,
TDA evaluation, and CLI from train_cipt_official_domainbed.py.
"""
from __future__ import annotations

import sys
import train_cipt_official_domainbed as baseline

ORIGINAL_LOAD_API = baseline.load_official_api
DEFAULT_OUTPUT_ROOT = "train_output/cipt_tda_consistent_domainbed"


def consistent_api():
    _, cosine_scheduler, evaluate, make_optimizer, train_one_epoch = ORIGINAL_LOAD_API()
    from cipt_tda_consistent import build_consistent_cipt
    return (
        build_consistent_cipt,
        cosine_scheduler,
        evaluate,
        make_optimizer,
        train_one_epoch,
    )


def main():
    # Use a separate output root so the official reference runs are immutable.
    if not any(arg == "--output-root" or arg.startswith("--output-root=") for arg in sys.argv[1:]):
        sys.argv.extend(["--output-root", DEFAULT_OUTPUT_ROOT])
    baseline.load_official_api = consistent_api
    baseline.main()


if __name__ == "__main__":
    main()
