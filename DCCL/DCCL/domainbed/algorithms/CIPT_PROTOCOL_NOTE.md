# Official CIPT protocol adapter

`--algorithm CIPT` is the standalone TPAMI/public-code CIPT baseline.

The DomainBed adapter uses the paper settings: 16 shots per class in each source domain, OpenAI CLIP deterministic preprocessing, global image batch size 64, 30 epochs, Adam with initial learning rate 2.5e-3 and zero weight decay, epoch-level cosine decay, beta=4, gamma=5, K=4, and 8-head TDA. Source-domain minibatches are balanced by DomainBed; the concatenated batch is randomly trimmed to the requested global batch size before the CIPT update. Target-domain `test_in` covers the full target domain for the CIPT path.

No DCCL second view, SupCon, projection head, PMA/GT, or SWAD is used by this algorithm.
