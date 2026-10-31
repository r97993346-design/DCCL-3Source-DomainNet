# Saved CIPT checkpoint comparison

Coverage: FULL_DATASET_SPLITS; identical input tensor hashes verified.

Accuracies below are pooled over images. Source domain-macro accuracies are also saved in comparison.csv.

| Split | Model | E-only | LN(E) | Native TDA |
|---|---|---:|---:|---:|
| source_train | DomainBed | 99.70% | 99.70% | 88.44% |
| source_train | Upstream lr5e-5 | 98.91% | 98.88% | 87.56% |
| source_val | DomainBed | 97.73% | 97.67% | 87.08% |
| source_val | Upstream lr5e-5 | 97.86% | 97.86% | 86.77% |
| target_in | DomainBed | 97.25% | 97.32% | 90.48% |
| target_in | Upstream lr5e-5 | 97.38% | 97.44% | 87.13% |
| target_out | DomainBed | 97.80% | 98.04% | 90.71% |
| target_out | Upstream lr5e-5 | 97.80% | 97.56% | 87.53% |

Both target_in and target_out were unused for training. This audit does not select a target partition, model, or alpha.

Inspect report.json for per-class counts, confusion matrices, margins, dtype, and alpha controls.
