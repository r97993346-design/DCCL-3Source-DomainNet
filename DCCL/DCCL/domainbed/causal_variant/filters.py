import torch
import torch.nn.functional as F


@torch.no_grad()
def pretrained_anchor_filter(pre_model, x, candidates, source_indices, thresh=0.35, enabled=True):
    if not enabled or candidates.numel() == 0:
        keep = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
        return keep, {"causal/anchor_sim_mean": 0.0, "causal/anchor_keep_ratio": 1.0}, None
    pre_model.eval()
    src = pre_model(x[source_indices])
    cand = pre_model(candidates)
    sim = F.cosine_similarity(src.flatten(1), cand.flatten(1), dim=1)
    keep = sim >= float(thresh)
    return keep, {"causal/anchor_sim_mean": sim.mean().item(), "causal/anchor_keep_ratio": keep.float().mean().item()}, sim


@torch.no_grad()
def class_consistency_filter(featurizer, classifier, candidates, labels, source_indices, mode="confidence", conf_thresh=0.5, enabled=True):
    if not enabled or mode == "none" or candidates.numel() == 0:
        keep = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
        return keep, {"causal/cls_conf_mean": 0.0, "causal/cls_keep_ratio": 1.0}, None
    logits = classifier(featurizer(candidates))
    probs = F.softmax(logits, dim=1)
    target = labels[source_indices]
    conf = probs.gather(1, target.view(-1, 1)).squeeze(1)
    if mode == "argmax":
        keep = probs.argmax(1) == target
    elif mode == "confidence":
        keep = conf >= float(conf_thresh)
    else:
        raise ValueError("Unknown causal_cls_filter_mode: {}".format(mode))
    return keep, {"causal/cls_conf_mean": conf.mean().item(), "causal/cls_keep_ratio": keep.float().mean().item()}, conf


@torch.no_grad()
def diffusion_black_image_filter(candidates, kinds, enabled=True, mean_thresh=0.03, std_thresh=0.01):
    """Reject black images returned by diffusion safety checker.

    Diffusers safety checker can replace unsafe generations with an all-black
    image. These images can otherwise pass weak warm-up filters and become
    selected positives, so remove them before Top-M selection.
    """
    keep = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    if not enabled or candidates.numel() == 0:
        return keep, {"causal/diffusion_black_count": 0.0, "causal/diffusion_black_ratio": 0.0}
    diffusion_indices = [idx for idx, kind in enumerate(kinds) if kind == "diffusion"]
    if not diffusion_indices:
        return keep, {"causal/diffusion_black_count": 0.0, "causal/diffusion_black_ratio": 0.0}
    idx_tensor = torch.tensor(diffusion_indices, dtype=torch.long, device=candidates.device)
    imgs = candidates[idx_tensor]
    mean = torch.tensor((0.485, 0.456, 0.406), device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    imgs01 = (imgs * std + mean).clamp(0, 1)
    flat = imgs01.flatten(1)
    is_black = (flat.mean(1) <= float(mean_thresh)) & (flat.std(1) <= float(std_thresh))
    keep[idx_tensor[is_black]] = False
    return keep, {
        "causal/diffusion_black_count": float(is_black.sum().item()),
        "causal/diffusion_black_ratio": float(is_black.float().mean().item()),
    }
