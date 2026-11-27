# Fixed80 class-text classifier switch (env_anchor)

Before this change, the official frozen CLIP B5b 80-template bank was used
only for RD, MCC and center MSE. E-CE and S-uniformity used a trainable
CoOp/CIPT class PromptLearner, also used by E-only inference.

New switch `cipt_class_text_mode`:
- `learnable` (DEFAULT): retains the original trainable class prompts.
- `fixed80`: E-CE, S-uniformity and E-only prediction all use
  `W_c=normalize(mean_{j=1..80}(normalize(CLIPText(T_cj))))`.

For `fixed80`, the original PromptLearner module remains registered for
checkpoint compatibility, but is **frozen**, is excluded from the rebuilt
optimizer, and **is not evaluated** in training or inference. The visual CLIP
remains frozen as before; causal and spurious adapters still train.

The class-text switch is distinct from `cipt_env_selection`, which controls
how the environment loss selects templates. The old RD formula is unchanged:
`relative_distance + env_selection=all + topk=80 + rd_anchor_mode=full`
computes RD between E and 80 frozen official class texts relative to the
normalized all80 teacher center. The fixed80 class-text prototype equals
that teacher center but does not replace the RD loss.

Ready-to-run config:
`config_cipt_b5b_all80_rd_fixed_text80.yaml`

Equivalent CLI modification to an existing `env_anchor` run:
```
--cipt_class_text_mode fixed80
```
To revert, use `--cipt_class_text_mode learnable`. The default is backward
compatible. Compare against the original e1/s1/rd1 all80 baseline with the
same optimizer, seed, 5001 steps, augmentation, SWAD and DomainBed IID
model-selection protocol.

Unit test:
```bash
python -m unittest discover -s tests -p 'test_cipt_fixed80_class_text.py'
```
