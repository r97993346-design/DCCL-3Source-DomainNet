from .algorithms import *
# On this branch, route CIPTDCCL through the component-ablation wrapper.
# It preserves the no-augmentation causal-contrastive implementation while
# exposing independent L_de, L_ind, TDA, and Contrastive switches.
from .cipt_dccl_component_ablation import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
