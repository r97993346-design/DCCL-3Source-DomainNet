# CondCov Optuna results

Snapshot captured 2026-11-26T12:02:08+08:00.
Contrastive weights 0.025, 0.05, 0.1.

Training code commit a5069e9199dd49de5a2ed199c076e40707661e1e.
CondCov model commit a55914e203d1b3e36706a681550d17e620eb4172.

| Dataset | Complete trials | Imported trials | Complete folds | Reused folds |
| --- | ---: | ---: | ---: | ---: |
| PACS | 7 | 1 | 30 | 4 |
| VLCS | 2 | 0 | 8 | 2 |
| OfficeHome | 2 | 0 | 9 | 2 |

Each listed fold reached 5,000 optimizer updates. The four-domain summary uses each fold's highest source validation checkpoint. Partial trials have blank aggregate metrics.

- [Trial results](trials_summary.csv)
- [Target-domain results](folds_summary.csv)
- [Training and search protocol](protocol.json)
- [Related search](../condcov_optuna_joint_20261125)

Each fold includes evaluation records, optimizer telemetry, and console logs. Reused folds link to their original logs. Server paths and GPU identifiers have been replaced with placeholders.
