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
