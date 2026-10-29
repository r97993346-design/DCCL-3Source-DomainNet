# Official CIPT reproduction using DomainBed splits

This is a separate reproduction branch based on `feature/cipt-b5b-entropy-mcc-rd`.
**The parent branch is unchanged.** Run `train_cipt_official_domainbed.py` instead
of `train_all.py` to avoid all inherited CIPTDCCL modifications.

## Upstream code, unchanged

The **entire original CIPT repo** is included as a pinned Git **submodule** at
`third_party/CIPT` (upstream commit `a805d878acc7d79778d1ec1c1e4d73ba6aff334b`).
This preserves the upstream implementation verbatim: adapters, prompt learner,
TDA module, ImageNet template bank, loss function, optimizer and evaluation.
It is not a rewritten approximation. The new entry point only adapts DomainBed
dataset samples to the upstream `(image, label)` loader interface.

```bash
git checkout repro/cipt-official-domainbed
git submodule update --init --recursive
cd DCCL/DCCL
pip install -r third_party/CIPT/requirements.txt
```

When the OpenAI CLIP package is already installed, its existing version can be
reused. On servers without internet, pass a local OpenAI ViT-B/16 checkpoint.

## PACS: all four leave-one-domain-out targets (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 python train_cipt_official_domainbed.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --test-envs 0 1 2 3 \
  --seed 0 --trial-seed 0 --holdout-fraction 0.2 \
  --epochs 30 --batch-size 64 --lr 0.0025 \
  --beta 4 --gamma 5 --k 4 --num-heads 8 \
  --num-workers 4
```

The script uses the same `domainbed.datasets.get_dataset()` as the parent
repo, with `label_ratio=1.0`, so **no few-shot subsampling** is performed.
The source domains have a 20% holdout, determined by
`misc.seed_hash(trial_seed, environment_id)`. Each target domain is untouched
during training. The split function already used by this repo labels the
partitions as `in` and `out`; the script reports **both** `target_in` and
`target_out` separately and never uses either to select model weights.

## Original training configuration

- CLIP ViT-B/16, frozen visual and text encoders; upstream prompt learner
  (16 context tokens; initialized using "a photo of a"); E/S linear adapters
- Upstream class-conditioned OpenAI ImageNet templates; random K=4 at train
  time; upstream TDA with its **original default 8 attention heads**
- Upstream loss: L = L_c + 4 * (CE(E) + KL(U || p_S)) + 5 * L_ind
- Upstream Adam optimizer, learning rate 0.0025, zero weight decay;
  cosine decay over 30 epochs; **global batch size 64**
- Final evaluation uses upstream `predict(..., use_tda=True)`, with the
  original K-template intervention aggregation; **not E-only**
- No RD, MCC, entropy selection, contrastive losses, SWAD or extra views.
  No early stopping, no target-domain hyperparameter search.

Caution: because the data partition, multi-source merged batch and checkpoint
conventions are tied to this DomainBed-based repository, this is an **original
CIPT method/training recipe on DomainBed**, not an assertion of bit-exact
reproduction of all unpublished author experiment details. The parent repo's
DomainBed loader is used, **not** the unrelated `ImageFolder` demo loader.

The output contains per-target config, epoch loss/history, source-domain
holdout accuracy, target-in and target-out accuracies, and a small checkpoint
of *trainable weights only*. To reload the checkpoint, rebuild the upstream
CIPT model with the same OpenAI CLIP weights/config and call:

```python
model.load_state_dict(torch.load(path, map_location="cpu")["trainable_state_dict"], strict=False)
```

Training cannot be fully validated without the dataset and OpenAI CLIP weights.
Use the quick checks:

```bash
git submodule status
python -m py_compile train_cipt_official_domainbed.py
python train_cipt_official_domainbed.py --help
```

For a no-write smoke run, pass `--test-envs 0 --epochs 1` and a unique
`--output-root` on a server with GPU/data; this is strictly a smoke test,
**not** an official result.
