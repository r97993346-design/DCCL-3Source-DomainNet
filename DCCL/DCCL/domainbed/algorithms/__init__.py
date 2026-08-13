from .algorithms import *
# Override the legacy feature/multiprompt CIPTDCCL class with the
# official-aligned implementation while leaving all other algorithms unchanged.
from .cipt_dccl import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
