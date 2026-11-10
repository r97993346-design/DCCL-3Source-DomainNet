"""Reset each held-out-domain task before constructing any random state."""

import random

import numpy as np
import torch


SEED_POLICY = "per_target"


def seed_everything(seed, deterministic=False):
    """Use the same seed for every target, independent of prior task lengths.

    This controls RNGs and the existing cuDNN flags. It does not claim bitwise
    reproducibility across platforms or enable every deterministic CUDA kernel.
    DataLoaders must be recreated after this call, as trainer.train already does.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds CPU and CUDA RNGs.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
