from torchvision import transforms as T
from PIL import Image, ImageFilter
import random
import numpy as np


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
# the input distribution it was trained with.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# Split the official CLIP preprocessing into spatial and post-processing stages
# so Fourier mixing can be inserted between CenterCrop and normalization.
# Keeping CenterCrop here means the Fourier ablation removes only the random
# crop from the augmented branch, not CLIP's deterministic center crop.
clip_spatial = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.Lambda(_convert_image_to_rgb),
])

clip_post = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# Original/non-augmented CIPTDCCL view. This mirrors OpenAI CLIP's official
# preprocessing: aspect-ratio-preserving bicubic resize, center crop, RGB
# conversion, tensor conversion, and CLIP normalization.
clip_basic = T.Compose([
    clip_spatial,
    clip_post,
])


# Existing augmented CIPTDCCL view retained as the "current" ablation mode.
clip_aug = T.Compose([
    T.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.1),
    T.Lambda(_convert_image_to_rgb),
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def colorful_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """CIRL-inspired Fourier amplitude mixing for one semantic anchor image.

    Both inputs must already have the same spatial size. The phase of ``img1``
    is kept while its amplitude spectrum is mixed with ``img2``. Therefore the
    returned image keeps ``img1``'s label and uses ``img2`` only as a style/domain
    donor.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if not 0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    if img1.shape != img2.shape:
        raise ValueError(
            f"Fourier inputs must have identical shapes, got {img1.shape} and {img2.shape}"
        )

    lam = np.random.uniform(0.0, alpha)
    h, w, _ = img1.shape
    side_scale = np.sqrt(ratio)
    h_crop = max(1, int(h * side_scale))
    w_crop = max(1, int(w * side_scale))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    fft1 = np.fft.fft2(img1, axes=(0, 1))
    fft2 = np.fft.fft2(img2, axes=(0, 1))
    amp1 = np.abs(fft1)
    phase1 = np.angle(fft1)
    amp2 = np.abs(fft2)

    amp1 = np.fft.fftshift(amp1, axes=(0, 1))
    amp2 = np.fft.fftshift(amp2, axes=(0, 1))
    mixed_amp = amp1.copy()

    hs = slice(h_start, h_start + h_crop)
    ws = slice(w_start, w_start + w_crop)
    mixed_amp[hs, ws] = (1.0 - lam) * amp1[hs, ws] + lam * amp2[hs, ws]

    mixed_amp = np.fft.ifftshift(mixed_amp, axes=(0, 1))
    mixed_fft = mixed_amp * np.exp(1j * phase1)
    mixed_img = np.real(np.fft.ifft2(mixed_fft, axes=(0, 1)))
    mixed_img = np.uint8(np.clip(mixed_img, 0, 255))
    return Image.fromarray(mixed_img)


class ClipFourierAugment:
    """Pair transform: CLIP spatial preprocessing -> Fourier mix -> CLIP norm."""

    def __init__(self, alpha=1.0, ratio=1.0):
        self.alpha = float(alpha)
        self.ratio = float(ratio)

    def __call__(self, image, donor):
        image = clip_spatial(image)
        donor = clip_spatial(donor)
        mixed = colorful_spectrum_mix(
            image,
            donor,
            alpha=self.alpha,
            ratio=self.ratio,
        )
        return clip_post(mixed)
