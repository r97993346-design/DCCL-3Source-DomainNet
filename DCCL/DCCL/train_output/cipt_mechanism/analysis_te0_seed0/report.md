# CIPT mechanism diagnostics

No conclusion is selected automatically.

All quantitative metrics use unit features in the original embedding dimension.
Each image contributes one value to intervention summaries. All 80 templates are evaluated.

| Paired image metric | Mean | Image bootstrap 95% CI (class/domain macro) |
|---|---:|---|
| e_to_z_cosine_shift | 0.104442 | [0.10381399224955642, 0.10502255717426366] |
| ln_e_to_z_cosine_shift | 0.103484 | [0.1028353710444333, 0.10406413795486695] |
| constant_to_z_cosine_shift | 0.018499 | [0.01837307237513847, 0.018613083834593258] |
| template_pairwise_cosine_shift | 0.028793 | [0.028600963433434632, 0.02897041215768858] |
| class_swap_cosine_shift | 0.115442 | [0.11469498759825776, 0.11612690311628728] |
| neutral_swap_cosine_shift | 0.079172 | [0.07849588749248401, 0.0797421262736898] |
| class_minus_template_shift | 0.086649 | [0.08607698721824726, 0.08716926120486919] |

## Interpretation

- Positive within-class domain silhouette, cosine gap and neighbor excess support residual domain structure.
- Small template spread does not imply that E and Z are identical; check both paired shifts and topology.
- A positive class-minus-template contrast indicates larger class-word sensitivity under these swaps.
- Ground-truth-conditioned Z is a mechanism diagnostic, not a label-free test representation.
- Bootstrap intervals condition on one checkpoint and sampled domains; no training-seed uncertainty.
- Performance benefit or equivalence requires separately trained B5B, Bconst and TDA-off controls.

## Files

- class_domain_metrics.csv: E, LN(E), constant text, Z mean, and every fixed template; one row per class.
- within_class_topology.csv: E/Z and LN(E)/Z neighborhood overlap and distance rank correlation, all templates.
- per_image_metrics.csv: paired shifts, template spread, neutral-text and wrong-class swaps.
- domain_metrics.png/pdf and intervention_sensitivity.png/pdf: quantitative summaries.
- tsne/: one joint fit per class/seed/perplexity; arrows show correspondence only.
- tsne_templates/ (unless skipped): fixed images and class words, varying templates; one joint fit per class.
- tsne_overview/ (with --overview): all selected classes; colors = classes, markers = domains.

Z_mean80 averages unit Z_k and normalizes again. It is a descriptive feature, not the legacy inference rule.
Displayed domains include source/target designations from checkpoint metadata; dataset files may include training images.
