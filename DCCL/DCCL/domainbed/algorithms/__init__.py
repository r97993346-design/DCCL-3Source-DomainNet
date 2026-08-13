from .algorithms import *
# On the ablation branch, route CIPTDCCL through the lightweight wrapper that
# keeps the known high-performance implementation and only switches B5 prompts.
from .cipt_dccl_ablation import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
