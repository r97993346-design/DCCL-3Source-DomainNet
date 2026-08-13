from .algorithms import *
# Route CIPTDCCL to the official-aligned implementation while leaving all
# original DCCL/DomainBed algorithms unchanged.
from .cipt_dccl_official import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
