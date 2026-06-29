import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_causal_sensitivity(orig_logits, cand_logits, source_indices, temperature=1.0, metric="kl"):
    p = F.softmax(orig_logits[source_indices].detach() / float(temperature), dim=1)
    q = F.softmax(cand_logits / float(temperature), dim=1)
    metric = metric.lower()
    if metric == "kl":
        s = F.kl_div(q.clamp_min(1e-8).log(), p, reduction="none").sum(1)
    elif metric == "l1":
        s = (p - q).abs().sum(1)
    elif metric == "js":
        m = 0.5 * (p + q)
        s = 0.5 * F.kl_div(m.clamp_min(1e-8).log(), p, reduction="none").sum(1) + 0.5 * F.kl_div(m.clamp_min(1e-8).log(), q, reduction="none").sum(1)
    else:
        raise ValueError("Unknown causal_sensitivity_metric: {}".format(metric))
    return s.detach()
