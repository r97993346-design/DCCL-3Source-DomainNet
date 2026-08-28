from .algorithms import *
# On the official-alignment branch, route CIPTDCCL through the aug-decomp
# wrapper plus two implementation details from the released CIPT code:
# normalized CLIP image features and identity-initialized causal/spurious adapters.
from .cipt_dccl_official_align import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
