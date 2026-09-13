# CIPT paired class-agnostic validation protocol

This branch supports the two experiments used to validate the class-agnostic TDA design while keeping the rest of the CIPTDCCL pipeline unchanged.

## 1. Neutral-subject robustness

Use the paired 80-template class-agnostic bank (`b5a`, i.e. S0) and change only `cipt_neutral_subject`:

| Variant | `cipt_template_mode` | `cipt_neutral_subject` |
| --- | --- | --- |
| S0-subject | `b5a` | `subject` |
| S0-thing | `b5a` | `thing` |
| S0-object | `b5a` | `object` |
| S0-entity | `b5a` | `entity` |

Example overrides:

```bash
--cipt_template_mode b5a --cipt_neutral_subject subject
--cipt_template_mode b5a --cipt_neutral_subject thing
--cipt_template_mode b5a --cipt_neutral_subject object
--cipt_template_mode b5a --cipt_neutral_subject entity
```

The 80 contexts remain paired one-to-one with B5b. Only the neutral subject word changes. Learned class prompts, image features, K, TDA, losses, and classifier remain unchanged.

## 2. 2x2 class identity x context diversity experiment

| Cell | Meaning | `cipt_template_mode` | `cipt_neutral_subject` |
| --- | --- | --- | --- |
| Bconst | class-conditioned + constant context | `bconst` | ignored |
| B0 | class-conditioned + diverse contexts | `b5b` | ignored |
| Sconst | class-agnostic + constant context | `sconst` | `subject` |
| S0 | class-agnostic + paired diverse contexts | `b5a` | `subject` |

Definitions:

- **Bconst** keeps 80 positions but every prompt for a class is `a photo of a {class}.`.
- **Sconst** keeps 80 positions but every prompt is `a photo of a subject.`.
- **B0** uses the original 80 class-conditioned B5b prompts.
- **S0** uses the exact 80 B5b contexts with the class placeholder replaced by `subject`.

All four cells keep the same K-selection logic. Training samples K prompts randomly; evaluation uses the deterministic first K. For the constant cells the selected positions are different but their text content is deliberately identical.

Recommended primary comparisons:

1. `S0 - B0`: effect of removing class identity under matched diverse contexts.
2. `S0 - Sconst`: contribution of context diversity in the class-agnostic setting.
3. `B0 - Bconst`: contribution of context diversity in the class-conditioned setting.
4. Optional interaction: `(S0 - Sconst) - (B0 - Bconst)`.

## Reproducibility rule

For a paper table, keep all non-TDA settings identical across cells: dataset split, seed, K, prompt length/init, beta/gamma, optimizer, learning rate, contrastive settings, preprocessing, evaluation protocol, and checkpoint/model-selection rule. Change only the two factors described above.
