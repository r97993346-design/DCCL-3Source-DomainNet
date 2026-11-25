# CondCov Optuna results

Snapshot captured 2026-11-25T19:12:49+08:00.

Training code commit `c8ac696f156b60e9d5d33679169fd8aacc973545`.
CondCov model commit `a55914e203d1b3e36706a681550d17e620eb4172`.

| Dataset | Complete trials | Complete target-domain folds |
| --- | ---: | ---: |
| PACS | 1 | 6 |
| VLCS | 0 | 1 |
| OfficeHome | 0 | 1 |

Each listed fold reached 5,000 optimizer updates. The four-domain summary uses the checkpoint with the highest source validation accuracy in each fold.
Trials with fewer than four published folds have blank aggregate metrics.

- [Trial results](trials_summary.csv)
- [Target-domain results](folds_summary.csv)
- [Training and search protocol](protocol.json)

Each fold directory contains its evaluation records, optimizer telemetry and console log. Paths to the training server and GPU identifiers were replaced with placeholders.
