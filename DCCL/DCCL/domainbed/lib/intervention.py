import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize_imagenet(images: torch.Tensor):
    mean = IMAGENET_MEAN.to(images.device, images.dtype)
    std = IMAGENET_STD.to(images.device, images.dtype)
    return images * std + mean


def normalize_imagenet(images: torch.Tensor):
    mean = IMAGENET_MEAN.to(images.device, images.dtype)
    std = IMAGENET_STD.to(images.device, images.dtype)
    return (images - mean) / std


def select_fourier_donors(domains: torch.Tensor, cross_domain_only: bool = True):
    bsz = domains.shape[0]
    donor_indices = torch.empty(bsz, dtype=torch.long, device=domains.device)
    fallback = 0
    for i in range(bsz):
        if cross_domain_only:
            candidates = torch.where(domains != domains[i])[0]
        else:
            candidates = torch.where(torch.arange(bsz, device=domains.device) != i)[0]
        if candidates.numel() == 0:
            others = torch.where(torch.arange(bsz, device=domains.device) != i)[0]
            if others.numel() == 0:
                donor_indices[i] = i
            else:
                donor_indices[i] = others[torch.randint(0, others.numel(), (1,), device=domains.device)]
            fallback += 1
        else:
            donor_indices[i] = candidates[torch.randint(0, candidates.numel(), (1,), device=domains.device)]
    return donor_indices, fallback


def fourier_amplitude_intervention(images: torch.Tensor, domains: torch.Tensor, mix_alpha: float = 0.5,
                                   mix_min: float = 0.1, mix_max: float = 0.9,
                                   cross_domain_only: bool = True):
    assert images.ndim == 4, 'images must be [B,C,H,W]'
    donor_indices, fallback_count = select_fourier_donors(domains, cross_domain_only=cross_domain_only)
    donors = images[donor_indices]

    x_fft = torch.fft.fft2(images, dim=(-2, -1))
    d_fft = torch.fft.fft2(donors, dim=(-2, -1))

    amp_x = torch.abs(x_fft)
    amp_d = torch.abs(d_fft)
    phase_x = torch.angle(x_fft)

    lam = torch.empty(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype).uniform_(mix_min, mix_max)
    lam = lam * mix_alpha
    lam = torch.clamp(lam, 0.0, 1.0)

    amp_mix = (1.0 - lam) * amp_x + lam * amp_d
    mixed_fft = torch.polar(amp_mix, phase_x)
    intervened = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
    intervened = torch.nan_to_num(intervened, nan=0.0, posinf=1.0, neginf=0.0)
    intervened = intervened.clamp(0.0, 1.0)
    return intervened, donor_indices, fallback_count
