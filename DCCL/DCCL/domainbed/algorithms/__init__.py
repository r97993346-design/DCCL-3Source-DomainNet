from .algorithms import *
# On this branch, route CIPTDCCL through the CRCC wrapper.  CRCC inherits the
# official no-augmentation component-ablation implementation, so L_de, L_ind,
# TDA and the contrastive on/off switch remain available while the contrastive
# objective itself can be switched between CRCC and the previous SupCon.
from .cipt_crcc import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
