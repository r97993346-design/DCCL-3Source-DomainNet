# PICCL SWAD-oriented HPO

Use TPE only over `piccl_rank` in `{8, 16}`, `piccl_beta_max` in `[0.10, 0.35]`,
and log-sampled `piccl_isr_weight` in `[0.01, 0.08]`. The source-validation
objective is `mean(source_val_acc) - 0.2 * std(source_val_acc)`; target-domain
metrics are recorded but are never used for ranking, pruning, or selection.
