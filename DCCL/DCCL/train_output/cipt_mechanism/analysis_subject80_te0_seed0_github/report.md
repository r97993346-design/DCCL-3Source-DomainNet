# CIPT mechanism diagnostics

B5A: shared 'subject' texts; no label-conditioned fusion

No conclusion is selected automatically.

All quantitative metrics use unit features in the original embedding dimension.
Each image contributes one value to intervention summaries. All 80 templates are evaluated.

| Paired image metric | Mean | Image bootstrap 95% CI (class/domain macro) |
|---|---:|---|
| e_to_z_cosine_shift | 0.022804 | [0.022678092225634404, 0.02292702749818406] |
| ln_e_to_z_cosine_shift | 0.021403 | [0.021286883327018686, 0.021512280808793065] |
| constant_to_z_cosine_shift | 0.012582 | [0.01250928521873485, 0.0126492181819838] |
| template_pairwise_cosine_shift | 0.019997 | [0.01988214049327409, 0.020102308465571712] |
| class_insertion_cosine_shift | 0.026565 | [0.026415374271056082, 0.026696846267141937] |

## Interpretation

- Positive within-class domain silhouette, cosine gap and neighbor excess support residual domain structure.
- Small template spread does not imply that E and Z are identical; check both paired shifts and topology.
- Bootstrap intervals condition on one checkpoint and sampled domains; no training-seed uncertainty.
- Performance benefit or equivalence requires separately trained B5B, Bconst and TDA-off controls.
- Z_mean80 summarizes geometry; it is not the model's K-branch logit aggregation rule.
- Native B5A Z uses the saved shared neutral-subject bank; image labels only group the domain statistics.
- Class insertion is a post-training text probe, not a native class swap or evidence of training benefit.
- Compare separately trained B5B/B5A E on the same manifest to study learned visual representations.

## Files

- class_domain_metrics.csv: E, LN(E), constant text, Z mean, and every fixed template; one row per class.
- within_class_topology.csv: E/Z and LN(E)/Z neighborhood overlap and distance rank correlation, all templates.
- per_image_metrics.csv: native E/Z shifts, template spread, and explicitly named post-training probes.
- domain_metrics.png/pdf and intervention_sensitivity.png/pdf: quantitative summaries.
- tsne/: one joint fit per class/seed/perplexity; arrows show correspondence only.
- tsne_templates/ (unless skipped): fixed images and native subject words, varying templates; one joint fit per class.
- tsne_overview/ (with --overview): all selected classes; colors = classes, markers = domains.

Z_mean80 averages unit Z_k and normalizes again. It is a descriptive feature, not the legacy inference rule.
Displayed domains include source/target designations from checkpoint metadata; dataset files may include training images.
