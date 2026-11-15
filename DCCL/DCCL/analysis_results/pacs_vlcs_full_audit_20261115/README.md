# CIPT PACS / VLCS historical experiment audit

Snapshot date: 2026-11-15 (Asia/Shanghai). Source: hooasia, recursively under
`/home/hooasia/lgg/projects/`. This snapshot covers historical records throughout
the tree, including queued and resumed experiments from November 10 onward,
GPU0/GPU1 outputs, independent checkouts, and archived copies in analysis folders.
Date in the experiment table is the source results file modification time.
The presence of a results file does not imply that a queue entry finished.

## Files

| File | Content |
| --- | --- |
| all_experiments.csv | Run ID, provenance, effective args and hparams, four target folds, and aggregate metrics |
| all_checkpoints.csv | Every parseable checkpoint row, including per-environment fields, losses, and run/record identifiers |
| PACS_rank.csv / VLCS_rank.csv | All completed four-target runs with an available four-target IID test_in mean, ordered descending, plus rank within each protocol |
| duplicate_experiments.csv | Groups with identical recorded training configuration, after excluding location, naming, and logging-only fields |
| VLCS_check10_reselect_check50.csv | Offline IID selection on the 50-step subset of the original checkpoint records |
| audit_issues.csv | Missing, unverified, nonstandard, or excluded evidence |
| source_occurrences.csv | All standard CIPT source paths, including byte-identical archive copies |
| raw_manifest.csv | Original source paths and SHA-256 for archived files |
| raw/ | Original evaluation records, logs, commands, available YAMLs, and launch metadata |

## Selection and scope

For each target fold the standard runner selects the checkpoint with maximum
`train_out` (source validation accuracy). Equal values keep the first record
in log order, matching the runner's Python max behavior. Its `test_in`,
`test_out`, `train_out`, and step are reported together. Four-fold metrics are
unweighted means with accuracy represented as a fraction in CSVs. Original
logged summary percentages are retained in the raw logs. SWAD and SWAD fallback
are separately labeled per target, from the final log table when available.
A SWAD mean requires all four SWAD or fallback values.

A standard run is counted as complete when all four folds have usable checkpoint
evaluation and the final four-fold IID table was logged or all folds reached
the last scheduled checkpoint. A disagreement between the recomputed and logged
IID moves the run to review. When a log only contains some printed checkpoints,
its raw log is preserved and the run is excluded from checkpoint-based rankings.
Cutoff artifacts may include runs that were still active while the tree was read.

The older CIPT TDA runner is flagged as a separate protocol. It logged source
validation at training checkpoints and evaluated the target data for final selected
models only. Its per-target native IID checkpoint and evaluations are read from
metrics.json. Its native primary target split was target_out. Both splits are
preserved; ranking uses target_in across both protocols, and protocol_rank
separates the protocols. Their training and evaluation setups differ.

Standard experiment identity is the SHA-256 of the original results JSONL.
Byte-identical copies are attributed to a single run ID and their paths remain
in source_occurrences.csv. The duplicate configuration count is separate.
Config comparisons use full recorded args and hparams, omit run identity and
logging-only keys, and include the launch commit when available. A match
marked recorded_configuration_only leaves source code equivalence unverified.
Older per-target TDA runs use their dataset, source directory, and seed pair
as their grouping key and are excluded from the standard duplicate comparison.

Launch branch and commit are populated only when supported by recorded metadata.
A current checkout HEAD does not establish an older run's launch commit. For TDA,
upstream_commit is recorded separately from the runner's own revision.
YAML source labels describe provenance: run_local and command_workdir indicate
paths associated with the launch, code_snapshot_candidate is a nearby code
snapshot, and current_checkout_reference is the YAML available now. The last
two labels do not independently prove byte identity with the YAML at launch.
Effective settings come from the logged args and hparams. For settings with both
fields, the convenience columns give hparams precedence. Refer to checkpoint
fields such as contrastive_weight_eff for values that change during training.

## VLCS check=10 reselected at check=50

Select by source validation from recorded steps 0, 50, 100, … in each target.
full_four_target_grid is true when all four targets include all scheduled
50-step points through the logged budget. Rows with missing points are labeled
for review, and original IID scores are preserved in the experiment table.
This replay can reproduce an IID comparison on the same saved trajectory.
A real check=50 run has a different evaluation schedule. It can change SWAD
updates, early stopping, and any evaluation-related RNG use, so IID replay
cannot reproduce the training trajectory or estimate the resulting SWAD.

## Counts

- PACS: 322 complete. Status counts: {'complete': 322, 'incomplete': 43, 'log_only': 10}. Highest audited four-target IID: 96.962% (PACS_candidate2; pacs_3915161128106988).
- VLCS: 76 complete. Status counts: {'complete': 76, 'incomplete': 13, 'log_only': 3}. Highest audited four-target IID: 83.201% (04_es1_ind0p1_RD1_supcon_con0p1; vlcs_72127ec27bfe59ea).
- 56,734 evaluation table rows, including legacy final evaluation rows.
- 98 byte-identical additional source copies.
- 9 duplicate configuration groups, 19 member runs, 10 additional runs beyond one per group.
- 20 VLCS check=10 runs with eligible records, 19 with complete four-target grids.

## Named historical baselines

- PACS / PACS_candidate2: 96.962% IID, selected steps [300, 300, 300, 100]; see raw/PACS/pacs_3915161128106988.
- VLCS / 04_es1_ind0p1_RD1_supcon_con0p1: 83.201% IID, selected steps [330, 350, 330, 310]; see raw/VLCS/vlcs_72127ec27bfe59ea.

## Missing or irregular evidence

- command_sh_missing: 241 occurrences. See audit_issues.csv for paths and context.
- incomplete: 55 occurrences. See audit_issues.csv for paths and context.
- launch_branch_unverified: 377 occurrences. See audit_issues.csv for paths and context.
- launch_commit_unverified: 415 occurrences. See audit_issues.csv for paths and context.
- results_jsonl_missing: 13 occurrences. See audit_issues.csv for paths and context.
- target_checkpoint_metrics_unavailable: 3 occurrences. See audit_issues.csv for paths and context.
- yaml_not_found: 100 occurrences. See audit_issues.csv for paths and context.

Raw export uses an allowlist of experiment text files. Model weights, data, cache, binary artifacts and files matching credential patterns are excluded.

The VLCS baseline has a launch-queue code_commit.txt and a preserved YAML blob from that exact commit under its raw launch_commit_yaml directory. Logged overrides still determine the effective constraint and coefficients. The PACS baseline has no verified launch revision.
