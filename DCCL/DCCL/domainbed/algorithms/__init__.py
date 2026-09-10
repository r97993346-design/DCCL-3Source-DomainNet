from .algorithms import *
# Route CIPTDCCL through the progressive paper-level ablation wrapper.
# The underlying Safe-Diverse prompt selector and TDA implementation remain
# unchanged in cipt_dccl_ablation.py.
from .cipt_dccl_progressive_ablation import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
