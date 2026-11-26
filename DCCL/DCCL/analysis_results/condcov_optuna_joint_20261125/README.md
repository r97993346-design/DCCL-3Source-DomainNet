# CondCov Optuna results

Snapshot captured 2026-11-26T12:02:08+08:00.
Contrastive weights 0.1, 0.25, 0.5.

Training code commit c8ac696f156b60e9d5d33679169fd8aacc973545.
CondCov model commit a55914e203d1b3e36706a681550d17e620eb4172.

| Dataset | Complete trials | Imported trials | Complete folds | Reused folds |
| --- | ---: | ---: | ---: | ---: |
| PACS | 2 | 0 | 8 | 0 |
| VLCS | 0 | 0 | 2 | 0 |
| OfficeHome | 0 | 0 | 2 | 0 |

Each listed fold reached 5,000 optimizer updates. The four-domain summary uses each fold's highest source validation checkpoint. Partial trials have blank aggregate metrics.

- [Trial results](trials_summary.csv)
- [Target-domain results](folds_summary.csv)
- [Training and search protocol](protocol.json)
- [Related search](../condcov_optuna_joint_20261125_low_contrast)

Each fold includes evaluation records, optimizer telemetry, and console logs. Reused folds link to their original logs. Server paths and GPU identifiers have been replaced with placeholders.
