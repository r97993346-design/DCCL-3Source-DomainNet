from .algorithms import *
# Route CIPTDCCL directly through the aug-decomp implementation. Official CIPT
# visual L2 normalization and identity adapter initialization are enforced in
# CausalDecomposition itself, avoiding an extra wrapper layer.
from .cipt_dccl_ablation import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
