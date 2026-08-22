from .algorithms import *
# Route CIPTDCCL through the minimal visual-prompt wrapper on this branch.
# The wrapper preserves the existing causal-contrastive implementation and only
# adds learnable visual prompt tokens on top of frozen CLIP-V.
from .cipt_visual_prompt_ablation import CIPTDCCL


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
