from .algorithms import *
# The wrapper delegates unchanged legacy runs, or enables training-only
# class-agnostic prompt KL with explicit decomposition terms and E-only inference.
from .cipt_env_kl import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
