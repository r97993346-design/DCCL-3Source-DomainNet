from .algorithms import *
# Keep the high-performance ablation implementation for --algorithm CIPTDCCL.
from .cipt_dccl_ablation import CIPTDCCL

# Standalone official CIPT baseline for --algorithm CIPT.
from .cipt_official import CIPT as _OfficialCIPTBase
from torchvision import transforms as _T
try:
    from torchvision.transforms import InterpolationMode as _InterpolationMode
    _BICUBIC = _InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image as _PILImage
    _BICUBIC = _PILImage.BICUBIC

_CIPT_CLIP_PREPROCESS = _T.Compose([
    _T.Resize(224, interpolation=_BICUBIC),
    _T.CenterCrop(224),
    _T.Lambda(lambda image: image.convert("RGB")),
    _T.ToTensor(),
    _T.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])


class CIPT(_OfficialCIPTBase):
    """TPAMI CIPT baseline routed through the existing DomainBed trainer."""

    override_all_transforms = True
    official_cipt = True
    transforms = {"x": _CIPT_CLIP_PREPROCESS}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # Official CIPT does not use SWAD; mutate the shared config before the
        # trainer checks hparams["swad"] after model construction.
        hparams["swad"] = None
        super().__init__(input_shape, num_classes, num_domains, hparams)


# get_dataset() normally applies algorithm transforms only to the training
# split. CIPT needs the same deterministic OpenAI-CLIP preprocessing for train,
# validation and target evaluation, so patch only the opt-in official-CIPT path.
from domainbed import datasets as _datasets_pkg
_original_set_transforms = _datasets_pkg.set_transfroms


def _set_transforms_with_official_cipt(dset, data_type, hparams, algorithm_class=None):
    if algorithm_class is not None and getattr(algorithm_class, "override_all_transforms", False):
        dset.transforms = dict(algorithm_class.transforms)
        return
    return _original_set_transforms(dset, data_type, hparams, algorithm_class)


_datasets_pkg.set_transfroms = _set_transforms_with_official_cipt


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
