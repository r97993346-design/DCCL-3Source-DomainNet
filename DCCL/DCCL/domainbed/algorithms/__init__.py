from . import algorithms as _algorithms_module
from .algorithms import *

# Route CIPTDCCL to the official-aligned implementation while leaving all
# original DCCL/DomainBed algorithms unchanged.
from .cipt_dccl_official import CIPTDCCL

# algorithms.py still contains the historical prototype for compatibility with
# the monolithic file layout. Replace that module attribute too, so even
# `from domainbed.algorithms.algorithms import CIPTDCCL` resolves to the
# official-aligned implementation after package initialization instead of the
# stale prototype.
_algorithms_module.CIPTDCCL = CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
