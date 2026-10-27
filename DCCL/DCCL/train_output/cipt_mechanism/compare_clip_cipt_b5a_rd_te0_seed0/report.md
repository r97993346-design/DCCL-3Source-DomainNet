# Frozen CLIP versus trained E

| Representation | Same-class cross-domain 5NN | Domain silhouette | Class-neighbor purity |
|---|---:|---:|---:|
| clip_v | 6.0000% | 0.3171 | 92.8286% |
| cipt_e | 5.9714% | 0.2777 | 96.2286% |
| b5a_e | 5.9714% | 0.2814 | 96.2857% |
| ours_e | 28.7714% | 0.0281 | 99.0000% |

All metrics use original-dimensional unit features. t-SNE is illustrative.
Source images can include training samples. Neighbor purity is not classification accuracy.
Comparing trained E with frozen V describes the entire training path, not an isolated decomposition ablation.
Current RD+NR uses class-related text anchors and different loss settings; no individual module attribution.
