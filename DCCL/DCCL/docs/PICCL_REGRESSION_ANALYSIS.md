# PICCL regression contract

PICCL is a low-rank orthogonal projection: `m = z - beta * ((z @ Q) @ Q.T)`.
At `beta=0`, it returns the original tensor unchanged. `use_piccl=false` takes
the unmodified DCCL path, including optimizer groups and random initialization.

The only PICCL search parameters are `piccl_rank`, `piccl_beta_max`, and
`piccl_isr_weight`. Model selection must use SWAD; target-domain results are
report-only during source-validation HPO.
