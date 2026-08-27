from torchvision import transforms as T
from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _convert_image_to_rgb(image):
    """Match OpenAI CLIP preprocessing by forcing three-channel RGB input."""
    return image.convert("RGB")


# ImageNet preprocessing retained for the original ResNet/DCCL code paths.
basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# OpenAI CLIP preprocessing constants. CIPTDCCL uses these transforms instead
# of the ImageNet/ResNet transforms above so the frozen CLIP image encoder sees
# CLIP-normalized inputs.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# CIPTDCCL basic view: preserve the complete image content by resizing the
# whole image to 224x224, then convert to RGB and apply CLIP normalization.
# This intentionally removes the official CLIP CenterCrop so the stable/basic
# view does not discard semantic content near image boundaries.
clip_basic = T.Compose([
    T.Resize(
        (224, 224),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.Lambda(_convert_image_to_rgb),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# CIPTDCCL augmented view: first establish the same complete 224x224 base view,
# then apply a controlled RandomResizedCrop and appearance perturbations.
# The crop keeps 80%-100% of the resized image area and limits aspect-ratio
# changes so the positive pair remains semantically consistent while still
# receiving spatial and appearance augmentation.
clip_aug = T.Compose([
    T.Resize(
        (224, 224),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.1),
    T.Lambda(_convert_image_to_rgb),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
