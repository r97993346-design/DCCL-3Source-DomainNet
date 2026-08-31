from .algorithms import *
# Route CIPTDCCL through the official-normalize-identity implementation plus
# stochastic diversity-aware B5c intervention prompt sampling. Train and eval
# intentionally use the same stochastic prompt-selection rule.
from .cipt_stochastic_prompts import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
