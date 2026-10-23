# CIPT and E-only RD comparison

Same image manifest, class order, held-out domains and CLIP checkpoint. High-dimensional metrics use unit features.

| Representation | Cross-domain neighbors within class | Class neighbor purity | Distance ratio |
|---|---:|---:|---:|
| cipt_e | 0.0597 | 0.9623 | 0.6500 |
| cipt_z | 0.0597 | 1.0000 | 0.4061 |
| ours_e | 0.2877 | 0.9900 | 0.1867 |

Interpretation:
- More cross-domain same-class neighbors is supportive only if class purity is retained; check target-domain accuracy separately.
- This is descriptive analysis, including source images that may have been used for training. It does not establish causal identification.
- Original CIPT Z uses ground-truth class-conditioned text and one predeclared template, not a label-free predictor or the complete inference procedure.
- The E-only model has no TDA representation here. Text-swap invariance of its E is not evidence of a learned improvement.
- Each figure shares one joint PCA/t-SNE fit; do not quantify inter-model displacement or 2D cluster distances.
- These comparisons do not isolate NR and RD contributions. For attribution, repeat the same analysis with matched E/S, +NR, +RD and +NR+RD checkpoints.
- The repository's TE*_5000.pth files store ordinary models; their filenames do not imply SWAD averaging. Compare checkpoint selection rules explicitly.
