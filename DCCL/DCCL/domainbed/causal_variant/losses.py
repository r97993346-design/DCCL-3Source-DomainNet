import torch
import torch.nn.functional as F


def causal_semantic_loss(logits, labels):
    if logits is None or logits.shape[0] == 0:
        return None
    return F.cross_entropy(logits, labels)


def causal_kl_loss(orig_logits, cand_logits, temperature=1.0):
    if cand_logits is None or cand_logits.shape[0] == 0:
        return None
    p = F.softmax(orig_logits.detach() / float(temperature), dim=1)
    log_q = F.log_softmax(cand_logits / float(temperature), dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")


def causal_positive_contrastive_loss(anchor, positive, negative_pool=None, temperature=0.1):
    if positive is None or positive.shape[0] == 0:
        return None
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    if negative_pool is None:
        negative_pool = anchor
    negative_pool = F.normalize(negative_pool, dim=1)
    pos_logits = (anchor * positive).sum(1, keepdim=True) / float(temperature)
    neg_logits = anchor @ negative_pool.t() / float(temperature)
    logits = torch.cat([pos_logits, neg_logits], 1)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
