# Official CIPT protocol adapter

`--algorithm CIPT` is the standalone TPAMI/public-code CIPT baseline.

The core package under `DCCL/DCCL/cipt/` is copied **verbatim** from the author-linked repository `ckghostwj/CIPT`, commit `a805d878acc7d79778d1ec1c1e4d73ba6aff334b`. The Git blob SHAs of `__init__.py`, `engine.py`, `losses.py`, `model.py`, and `templates.py` are identical to that upstream commit. Do not edit those files in this branch; DomainBed compatibility belongs only in `domainbed/algorithms/cipt_official.py` and the dataset/training glue.

The DomainBed adapter uses the paper DG settings: 16 shots per class in each source domain, OpenAI CLIP deterministic preprocessing, global image batch size 64, 30 epochs, Adam with initial learning rate 2.5e-3 and zero weight decay, epoch-level cosine decay, beta=4, gamma=5, K=4, and 8-head TDA. Source-domain minibatches are balanced by DomainBed; the concatenated batch is randomly trimmed to the requested global batch size before the official CIPT update. Target-domain `test_in` covers the full target domain for the CIPT path.

No DCCL second view, SupCon, projection head, PMA/GT, or SWAD is used by this algorithm.
