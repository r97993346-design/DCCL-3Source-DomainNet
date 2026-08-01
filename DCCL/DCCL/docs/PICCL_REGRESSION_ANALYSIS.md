# PICCL regression contract

PICCL is a low-rank orthogonal projection: `m = z - beta * ((z @ Q) @ Q.T)`.
At `beta=0`, it returns the original tensor unchanged. `use_piccl=false` takes
the unmodified DCCL path, including optimizer groups and random initialization.

The only PICCL search parameters are `piccl_rank`, `piccl_beta_max`, and
`piccl_isr_weight`. Model selection must use SWAD; target-domain results are
report-only during source-validation HPO.

## Why the first reliability experiments did not improve accuracy

The original reliability implementation used the sensitive-energy fraction
`||Proj_Q(delta)||^2 / ||delta||^2` directly as a positive-pair reliability.
That ordering was reversed: domain-sensitive pairs received weights closest to
one, while invariant pairs were downweighted. Reliability is now defined as
one minus that fraction. Existing runs containing
`pair_reliability_raw_mean` were produced with the reversed definition and
must not be compared with corrected runs under the same experiment label.

There are two additional experimental pitfalls:

1. Before this correction, the registry default for `PICCL` set `use_piccl` to
   false. A command that selected `--algorithm PICCL` without also loading the
   PICCL YAML silently ran the DCCL bypass. PICCL is now active by default;
   `--use_piccl false` is reserved for the paired equivalence control.
2. The checked-in PACS evidence contains only one run per setting and changes
   `piccl_beta_max` between experiments. Differences of a few tenths of a point
   are not evidence of improvement without paired seeds and confidence
   intervals.

## Required next experiment

Run matched DCCL, causal-only PICCL, and corrected reliable-contrast PICCL for
at least three identical seeds. Keep every DCCL hyperparameter and data split
fixed, select checkpoints only by source-domain SWAD, and report the paired
per-target delta (mean and standard deviation). First verify in the logs that
`piccl_beta > 0`, `valid_domain_response_count > 0`, and that
`pair_weight_effective_mean < 1` only when reliable contrast is enabled. Tune
only `piccl_rank`, `piccl_beta_max`, and `piccl_isr_weight` on source validation;
do not choose a configuration from target accuracy.
