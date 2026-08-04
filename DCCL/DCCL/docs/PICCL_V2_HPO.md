# PICCL v2 hyperparameter search

This workflow keeps the PICCL v2 model and all DCCL hyperparameters fixed. It
ranks trials only with source-domain `SWAD (inD)` and stores target-domain
`SWAD` as report-only evidence.

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

Do not rank or manually choose trials using `target_swad_report_only`. It exists
only for the final locked-configuration report.
