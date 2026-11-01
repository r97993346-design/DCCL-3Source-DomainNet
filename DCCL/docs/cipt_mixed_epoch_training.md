# CIPT pooled-source epoch training

This optional training strategy is based on the public CIPT
[ImageFolder example](https://github.com/ckghostwj/CIPT/blob/main/examples/train_cipt_imagefolder.py)
and [Adam / epoch scheduler](https://github.com/ckghostwj/CIPT/blob/main/cipt/engine.py).
It does not claim to reproduce every experiment in the
original paper. It changes training orchestration, not the selected RD/MCC,
E/S/IND, text-bank, contrastive loss or inference implementation.

## Training behavior

Load the method YAML first, then `config_cipt_mixed_epoch30.yaml`:

| Setting | Mixed-epoch overlay |
| --- | --- |
| Training data | Union of source-domain training in-splits only |
| Sampling | Shuffle the union once per epoch, without replacement |
| Batch size | 64 images in total, across all source domains |
| Last batch | Kept, including a one-image batch |
| Optimizer | Adam, initial LR 0.0025, weight decay 0 |
| LR schedule | Cosine, one scheduler update after each completed epoch, minimum 0 |
| Training budget | 30 complete epochs, resolved separately for each target domain |
| Evaluation | Every requested checkpoint interval, every epoch end, and the last update |
| IID selection | Existing mean source validation accuracy (`train_out`) |
| Target metrics / SWAD | Existing evaluator, IID reporting and configured SWAD behavior |

For `N` source training examples and total batch size `B`, one epoch has
`ceil(N / B)` updates. A target-domain run performs exactly
`epochs * ceil(N / B)` optimizer updates. This budget is not rounded to a
checkpoint interval and has no extra `+1` update. Different target domains may
have different source totals and therefore different update counts.

The pooled loader preserves each image's original `d` domain label. After a
batch has been sampled, it groups its members by source domain to retain the
algorithm's existing list-of-source-batches interface. This grouping neither
balances nor resamples the batch. A domain can be absent, and group sizes can
vary. Cross-domain and neighborhood losses still identify distinct domains.

Training uses a private seeded loader generator, so evaluation's consumption
of global PyTorch RNG does not alter the next epoch's sample permutation.
The normal experiment seed, trial seed and held-out domain determine that
generator's initial seed. The loader has one worker pool: `--num_workers 4`
means four total training workers for mixed mode.

The existing DomainBed data splits, transforms, label ratio, loss settings,
`e_only` inference and checkpoint selection code are preserved. In this
repository IID selects `records.argmax("train_out")["test_in"]`; the new mode
does not rename or exchange `test_in` and `test_out`. SWAD remains enabled or
disabled according to the method configuration, and cannot stop the fixed
training budget. The overlay does not change contrastive warmup: for example,
an inherited 500-step warmup is still 500 optimizer updates, not 500 epochs.

## PACS example

From the repository's `DCCL/DCCL` directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_b5b_entropy_rd_mixed_epoch30 \
  config_cipt_b5b_entropy_rd.yaml config_cipt_mixed_epoch30.yaml \
  --dataset PACS \
  --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --seed 0 \
  --trial_seed 0 \
  --deterministic \
  --checkpoint_freq 100 \
  --num_workers 4 \
  --eval_num_workers 0 \
  --aug 0 \
  --output_root train_output/ablation/cipt_mixed_epoch30
```

Omitting `--test_envs` runs each target domain in the existing leave-one-out
loop. Use `--test_envs 0` to run just one target. Change `--dataset` to `VLCS`,
`OfficeHome` or `TerraIncognita` for the corresponding dataset. For DomainNet,
use explicit `--source_envs` and `--target_env` for one split; this repository's
automatic DomainNet sweep also schedules ERM comparisons with the existing
per-domain step-based trainer. Those comparisons inherit the shared optimizer
hyperparameters but do not adopt CIPTDCCL's mixed sampling or epoch budget.

The same training overlay can follow `config_cipt_b5b_entropy_mcc.yaml` or a
center-MSE method YAML. Pass `--epochs 12`, `--batch_size 32` or `--lr 1e-4`
to override training values explicitly. `--epochs` is an alias for
`--cipt_train_epochs`. Do not pass `--steps` with mixed-epoch training: this
combination is rejected, rather than silently truncating an epoch recipe.
Class-balanced replacement sampling is also rejected in this mode.

The startup log, result JSONL and any saved checkpoints contain the resolved
`training_plan`, including source sample counts, epoch length, total updates,
total batch size and LR stepping unit. The result rows also expose completed
`updates` and fractional `epoch`. Periodic checkpoints use the existing
`--model_save` switch; enabling this overlay does not itself turn on saving.

No existing YAML or default training mode is changed. Without the overlay,
the original per-domain infinite sampling and step-based budget/schedule remain
available. `--debug` caps mixed mode at 10 optimizer updates for inspection.

## CPU verification

```bash
python -m unittest discover -s tests -p 'test_cipt_mixed_training.py' -v
python -m unittest discover -s tests -p 'test_cipt_lr_schedule.py' -v
```

The tests use synthetic tensors and a tiny frozen CLIP substitute. They check
source-only coverage without replacement, last partial batches, domain identity,
per-target budgets, epoch LR timing, source-validation model selection and
legacy-mode behavior. Actual RD/MCC and cross-domain contrastive update code
is exercised on CPU; no dataset training or CLIP checkpoint download is run.
