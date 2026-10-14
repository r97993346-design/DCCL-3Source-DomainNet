# CIPT paired text feature diagnostics

Frozen text-encoder pilot, using all 80 paired texts.
Quantitative metrics use the original feature dimension.
Class statistics use one feature per image; 80 texts are not 80 independent images.

| Class | Mean pk cosine | Mean pk cosine shift |
|---|---:|---:|
| dog | 0.885723 | 0.114277 |
| horse | 0.876885 | 0.123115 |
| guitar | 0.843173 | 0.156827 |

Unit pk norms are fixed by normalization; z norms are constrained by LayerNorm.
A shared subject bank has no class label: zero between-class pk distance is a construction property.
t-SNE map distances and cluster sizes are not quantitative evidence of feature separation.
This paired intervention test measures text replacement in one model, not performance of two trained models.
