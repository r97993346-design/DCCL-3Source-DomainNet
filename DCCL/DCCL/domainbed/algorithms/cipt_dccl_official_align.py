"""Official-CIPT visual feature alignment on top of the aug-decomp branch.

This wrapper keeps every loss and augmentation choice from
``cipt_dccl_ablation.CIPTDCCL`` unchanged, and only aligns two implementation
details with the released CIPT code:

1) L2-normalize frozen CLIP image embeddings before causal decomposition.
2) Initialize both causal/spurious linear adapters as identity maps.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms.cipt_dccl_ablation import CIPTDCCL as _AugDecompCIPTDCCL


class CIPTDCCL(_AugDecompCIPTDCCL):
    """Aug-decomp CIPTDCCL with official CIPT visual normalization/init."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        # Match the released CIPT FeatureAdapter(identity_init=True): both
        # decomposition branches start as identity mappings of the normalized
        # CLIP image embedding. The optimizer has no accumulated state yet, so
        # resetting here is equivalent to constructing the adapters this way.
        with torch.no_grad():
            torch.nn.init.eye_(self.causal_decomposition.causal_adapter.weight)
            torch.nn.init.zeros_(self.causal_decomposition.causal_adapter.bias)
            torch.nn.init.eye_(self.causal_decomposition.spurious_adapter.weight)
            torch.nn.init.zeros_(self.causal_decomposition.spurious_adapter.bias)

        print(
            "CIPTDCCL official alignment: clip_visual_l2_norm=True, "
            "causal_adapter_init=identity, spurious_adapter_init=identity"
        )

    def _visual(self, images):
        """Frozen CLIP image encoding followed by official CIPT L2 normalize."""
        with torch.no_grad():
            visual = self.clip_model.encode_image(images).float()
        return F.normalize(visual, dim=-1)
