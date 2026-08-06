# PICCL v2 hyperparameter search

This workflow keeps the PICCL v2 model and all DCCL hyperparameters fixed. Every
Optuna search, full-budget confirmation, and multi-seed confirmation ranks
configurations by the mean final target-domain `SWAD`. `SWAD (inD)` is retained
only as a diagnostic value.

This is explicitly a target-domain/Oracle hyperparameter-selection protocol.

Install Optuna once:

```bash
pip install optuna
```

## PACS

Run the core search and confirm its best six configurations at full budget:

```bash
python scripts/tune_piccl_v2_swad.py \
  --config configs/piccl_v2_pacs_hpo.json \
  --data-dir ../data \
  --output-root train_output/PACS/v2_hpo \
  --stage core --mode search \
  --gpus 0,1,2,3 --max-concurrent 4 --resume

python scripts/tune_piccl_v2_swad.py \
  --config configs/piccl_v2_pacs_hpo.json \
  --data-dir ../data \
  --output-root train_output/PACS/v2_hpo \
  --stage core --mode confirm --top-k 6 --confirm-seeds 0 \
  --gpus 0,1,2,3 --max-concurrent 4 --resume
```

Then search the schedule around the confirmed core winner and confirm the best
three settings with seeds 0, 1, and 2:

```bash
python scripts/tune_piccl_v2_swad.py \
  --config configs/piccl_v2_pacs_hpo.json \
  --data-dir ../data \
  --output-root train_output/PACS/v2_hpo \
  --stage schedule --mode search \
  --gpus 0,1,2,3 --max-concurrent 4 --resume

python scripts/tune_piccl_v2_swad.py \
  --config configs/piccl_v2_pacs_hpo.json \
  --data-dir ../data \
  --output-root train_output/PACS/v2_hpo \
  --stage schedule --mode confirm --top-k 3 --confirm-seeds 0,1,2 \
  --gpus 0,1,2,3 --max-concurrent 4 --resume
```

The final parameters are written to:

```text
train_output/PACS/v2_hpo/schedule/confirmation/best_config.json
```

## DomainNet (sources 1,2,3 -> target 4)

Use the same four commands with
`configs/piccl_v2_domainnet_hpo.json` and a separate output directory such as
`train_output/DomainNet/v2_hpo_s123_t4`. The DomainNet configuration uses a
5,000-step core budget, an 8,000-step schedule budget, and 15,000-step full
confirmation runs.

All generated `ranking.json`, `ranking.csv`, and `best_config.json` files are
ordered by mean target-domain `SWAD`.

## Preflight, scheduling, and parameter audit

Every invocation performs preflight before Optuna or a training subprocess is
created. It loads the real `PICCL` registry, rejects unknown fixed/searched
keys, invalid types/ranges, a non-PICCL algorithm, impossible ResNet-50 ranks,
and unsafe/duplicate GPU specifications. A command-only check is:

```bash
python scripts/tune_piccl_v2_swad.py \
  --config configs/piccl_v2_pacs_hpo.json --data-dir ../data \
  --output-root train_output/PACS/v2_hpo --stage core --dry-run \
  --gpus 0,1 --max-concurrent 2
```

With two GPUs, each worker owns one `CUDA_VISIBLE_DEVICES` value and processes
its assigned environments serially, so environments 0/2 use GPU 0 and 1/3 use
GPU 1 without overlap. Failed runs retain `command.json`, `params.json`,
`stdout.log`, `stderr.log`, and `failure.json`; Optuna marks that trial `FAIL`
and continues. `--resume` reuses only a `status=complete` metrics file whose
saved command exactly matches the command now requested. Old failed or
incompatible output therefore need not be deleted.

## Effective parameter chain and final ranges

`train_all.py` begins with `default_hparams(PICCL, dataset)` and sconf only
accepts overrides registered there. The historical ``key piccl_alpha_max do
not match`` failure consequently means the training checkout/process used a
registry predating that registration (or launched from the wrong checkout),
not that the value was merely absent from YAML. In this checkout the float
value reaches `PICCL.hparams`, `_alpha`, the mediator, and the logged
`piccl_alpha` diagnostic.

For progress `p = step / max(total_steps - 1, 1)`, warmup `w`, delay `d`, ramp
`r`, and maximum `A`, the implemented schedule is
`alpha=0` for `p <= d+w`, `alpha=A*(p-d-w)/r` in the ramp, and `alpha=A`
after `p >= d+w+r`. Delay and warmup are therefore additive offsets in this
implementation (not aliases), while ramp controls slope/duration. Alpha
reaches `A` iff the run reaches `p >= d+w+r`; every supplied combination has
`d+w+r <= 0.7`, so full runs reach it. The forward mediator uses
`z - alpha * sensitive` and logs the applied value.

The final core ranges are: alpha `[0.00, 0.05, ..., 0.50]` (a stepped float,
including no/weak-intervention controls), rank `{8,16,32}`, ISR weight
`{0.02,0.05,0.1,0.2}`, PICCL LR multiplier `{0.25,0.5,1}`, and orthogonality
weight `{0,1e-4,1e-3}`. PACS uses 36 trials and a 4,000-step coarse budget,
then 5,000-step confirmation. This narrows the coarse/full gap but still
requires full-budget confirmation because ranking reversal remains possible.
The schedule stage jointly revisits alpha and rank alongside delay
`{0,.05,.1,.2}`, ramp `{.1,.2,.4}`, and prototype momentum
`{.95,.99,.995}`; this preserves the important alpha/rank/ramp interaction
instead of freezing one core winner. PACS and DomainNet retain independent
configs, budgets, studies, and output roots.

Rank sets the learned `2048 x rank` basis and all choices are safely below the
ResNet-50 feature dimension used by QR/projection. ISR and orthogonality
weights multiply their respective losses; orthogonality weight zero gives
exactly zero contribution/gradient. The LR multiplier scales every sensitive
subspace and causal mediator optimizer group. Prototype momentum controls the
EMA prototype update. Alpha, rank, and ramp are strongly coupled through the
magnitude, dimensionality, and rate of removed projection; ISR/orthogonality
also shape that basis. `piccl_residual_scale` remains registered only for old
configuration compatibility but is not read by the fixed v2 forward path and
is deliberately rejected as a search parameter.

All rankings and confirmations are explicitly
`protocol=target_swad_oracle`: the objective is the arithmetic mean of the
four final PACS target-domain `SWAD` results. The parser takes the last exact
`SWAD` row from the final `=== Summary ===` PrettyTable (including its `Avg.`
column) and separately records the exact `SWAD (inD)` row, which never
contributes to selection. Legacy `SWAD =` records remain supported only when a
final table is absent. Thus a final table such as `SWAD ... Avg. 89.005%`
produces `swad_target=0.89005`, never the `SWAD (inD)` average beneath it.
