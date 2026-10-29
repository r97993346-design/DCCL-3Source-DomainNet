# CIPT TDA candidate-consistency correction (DomainBed)

This branch is derived from `repro/cipt-official-domainbed`.
It makes exactly one **algorithmic** change: during training, class-conditioned
TDA scores **all candidate classes**, just like inference. The true label is
used **only by cross-entropy**, never to generate an augmented image feature.

## Why this is necessary

The pinned upstream `CIPTModel.forward(images, labels)` selects text contexts
using `labels` and computes CE on resulting `[B,K,C]` logits. Upstream
`predict(images)` instead constructs a different TDA feature **for every
candidate class** and scores each candidate against that class's text prototype.
Those are not the same decision problem.

In the original PACS seed=0 checkpoints, TDA target-in average was 37.502%;
the same checkpoint's E-only target-in average was 95.037%. This is a strong
reason to isolate the inconsistency, **not** proof that the paper's causal idea
is inherently invalid.

## Exactly what stays unchanged

- Original `third_party/CIPT` pinned at
  `a805d878acc7d79778d1ec1c1e4d73ba6aff334b`; **never edit submodule**
- Original CLIP ViT-B/16, frozen encoders, prompts, E/S adapters, TDA
  cross-attention and original `cipt_loss()`
- L = L_c + 4 L_de + 5 L_ind, K=4, context length 16
- Adam, 0.0025 learning rate, 30 epochs, cosine scheduler, global batch size 64
- Identical `make_splits()` using the existing DomainBed code, no few-shot
- Identical training/evaluation runner and **final-epoch checkpoint selection**.
  Standard DomainBed IID source-validation model selection remains **unimplemented**
  here so this experiment changes **only** TDA candidate scoring.
- Upstream random K sampled during training and deterministic first K at eval;
  this selection difference remains deliberately unchanged.

The correction is isolated in `cipt_tda_consistent.py`, as a subclass of the
unmodified original CIPT model. `train_cipt_tda_consistent_domainbed.py`
reuses the entire original reproduction runner and substitutes only its builder.
Separate default output root protects the existing original CIPT checkpoints.

## Training: PACS (GPU1)

```bash
git fetch origin
git switch repro/cipt-tda-consistent-domainbed
git submodule update --init --recursive
cd DCCL/DCCL

CUDA_VISIBLE_DEVICES=1 python -u train_cipt_tda_consistent_domainbed.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --test-envs 0 1 2 3 \
  --seed 0 --trial-seed 0 --holdout-fraction 0.2 \
  --epochs 30 --batch-size 64 --lr 0.0025 \
  --beta 4 --gamma 5 --k 4 --num-heads 8 --num-workers 4
```

Output: `train_output/cipt_tda_consistent_domainbed/PACS/`, including
per-target `metrics.json`, `history.json`, checkpoint and four-domain
`summary_targets_0-1-2-3_seed_0_trial_0.json`.

**Important:** Avoid starting the full sweep before running the smoke test:

```bash
python -m unittest discover -s tests -p 'test_cipt_tda_consistent.py' -v

CUDA_VISIBLE_DEVICES=1 python -u train_cipt_tda_consistent_domainbed.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --test-envs 0 --epochs 1 --num-workers 2 \
  --output-root train_output/cipt_tda_consistent_smoke
```

Candidate scores are computed in **8-class chunks** to control memory use
on larger datasets (e.g., OfficeHome's 65 classes). The batch still contributes
one CE over *all* classes. Diagnostic outputs from original checkpoints are
**not** comparable to re-trained consistent-model checkpoints as a controlled
method modification without full retraining.

`source_val` is currently a *pooled* source holdout accuracy, not the
domain-macro-averaged validation used in DomainBed IID model selection.
`target_in` is the standard DomainBed target test metric. This branch
deliberately preserves these existing bookkeeping conventions; evaluate
IID-selected checkpoints in a separate experiment.
