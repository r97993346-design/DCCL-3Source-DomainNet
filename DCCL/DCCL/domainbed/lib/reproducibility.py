"""Reset each held-out-domain task before constructing any random state."""

import random
from contextlib import contextmanager

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


@contextmanager
def preserve_rng_state(enabled=True):
    """Optionally isolate evaluation, restoring RNGs even when it fails.

    Do not initialize CUDA for CPU evaluation. CUDA models have already
    initialized their device before entering this context.
    """
    if not enabled:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
