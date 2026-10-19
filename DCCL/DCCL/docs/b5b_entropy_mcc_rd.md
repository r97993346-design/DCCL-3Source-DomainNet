# Official B5b entropy selection with direct-E MCC / RD

Source: `feature/cipt-class-agnostic-sigm-mcc`, commit
`49b50c6941e76f4a8f98c8e9856dd7a07e7cfee4`.
Implementation branch: `feature/cipt-b5b-entropy-mcc-rd`.
Run every command below from `DCCL/DCCL`.

## Files

| File | Change |
| --- | --- |
| `domainbed/algorithms/cipt_entropy_selector.py` | New label-free per-image `EntropyPromptSelector` and cosine logits for the complete bank. |
| `domainbed/algorithms/cipt_prompt.py` | Add `full_b5b_text_features()`: detached full `[C,80,D]` official bank; no random K lookup. |
| `domainbed/algorithms/cipt_losses.py` | Add `relative_distance_loss()`; reuse unchanged efficient `prompt_consistency_loss()`. |
| `domainbed/algorithms/cipt_env_kl.py` | Add independent `env_anchor` initialization/validation, direct-E losses and diagnostics; reuse existing single-E-CE update and `e_only` prediction. |
| `domainbed/hparams_registry.py` | Register the three new environment options for YAML / existing `Config.argv_update()` CLI support. |
| `config_cipt_b5b_entropy_mcc.yaml` | Complete MCC experiment preset. |
| `config_cipt_b5b_entropy_rd.yaml` | Identical preset except `cipt_env_constraint=relative_distance`. |
| `tests/test_cipt_env_anchor.py` | Numerical, gradient, real-constructor/update, CLI, compatibility and deterministic tests; executable training smoke test. |
| `docs/b5b_entropy_mcc_rd.md` | Formulas, settings, validation scope and PACS commands. |

The source SIGM preset, contrastive implementation/defaults/warmup, SWAD code,
CLIP preprocessing, PromptLearner, and legacy/env_kl/clip_ce prediction paths
are unchanged. TDA is inherited during construction but frozen and excluded
from the new mode's optimizer; SIGM is not created. Neither module is called
by `env_anchor` training or inference. The old SafeDiverse/random template
selection methods are bypassed.

## Parameters and validation

| Parameter | New mode / preset value |
| --- | --- |
| `cipt_objective` | `env_anchor` (explicit opt-in; global default remains `legacy`) |
| `cipt_env_constraint` | `mcc` by default; alternative `relative_distance`; `none` disables the environment term for ablation |
| `cipt_env_topk` | `40`, integer in `[2,80]` |
| `cipt_env_loss_weight` | `1.0`, finite and nonnegative; `0` bypasses selection/loss |
| `cipt_template_mode` / `cipt_selector_mode` | `b5b` / `random`; the latter only satisfies inherited initialization |
| `cipt_use_tda` / `cipt_env_inference` | `false` / `e_only` |
| `cipt_env_kl_weight` / `cipt_fused_cls_weight` | `0.0` / `0.0` |
| `cipt_e_cls_weight` / `cipt_s_uniform_weight` | Presets explicitly use `1.0` / `1.0` |
| `cipt_use_de` / `cipt_use_ind` / `cipt_gamma` | `true` / `true` / `5.0`; the commands below explicitly disable IND for the established E=1, S=1, no-IND comparison |
| `cipt_use_contrastive` / `cipt_contrastive_type` | `true` / `neighbor_retention` |
| `cipt_neighbor_k` / `cipt_neighbor_alpha` | `10` / `4.0`, matching the source SIGM preset |
| `cipt_contrastive_temperature` / `cipt_causal_contrastive_weight` | `0.07` / `1.5`, matching the source SIGM preset |
| `cipt_contrastive_warmup_steps` | `500`, matching the source SIGM preset |
| `lr` / `swad` | `5e-5` / `LossValley` |

All coefficients remain CLI-overridable. E/S/IND/environment coefficients must
be finite and nonnegative; inherited contrastive validation is unchanged.
Invalid combinations (TDA enabled, nonzero old fusion loss, fused inference,
wrong bank/selector, invalid top-k or multiple constraint names) raise errors.
`cipt_use_de=false` disables S uniformity; E classification remains once.
`cipt_use_ind=false` disables IND. Existing contrastive switches, including
`cipt_pure` and `cipt_use_contrastive`, preserve their previous behavior.

## Data flow and actual formulas

Let `T[c,j]` be the frozen official class-conditioned bank, `j=1,...,80`.
The read-only getter never samples templates. Every template contains its
class name. Learnable class text features are a separate, unchanged path.

Frozen CLIP encodes each original image once to V. Normalize V and each
official text vector, compute logits `[B,80,C]` with the frozen CLIP scale
`s = exp(clip.logit_scale)`, and apply class-axis softmax in float32:

\[
q_{ijc}=\operatorname{softmax}_c(s\cos(V_i,T_{cj})),\qquad
H_{ij}=-\sum_c q_{ijc}\log q_{ijc}.
\]

`torch.topk(H, k=n, dim=1, largest=False)` returns independent indices
`J_i` for every image. The indices use all classes and no training labels,
E or S; selection runs under `no_grad`. One index selects the same template
position across all classes.

The original causal adapter maps V to E/S. E and learnable class text yield
the supervised CE; S uses the existing uniformity loss; IND and the original
single-view contrastive objective retain their switches and definitions.
Only the environment branch changes:

**MCC**, in `CIPTDCCL._env_anchor_loss()` and
`cipt_losses.prompt_consistency_loss()`:

\[
p_{ijc}=\operatorname{softmax}_c(s\cos(E_i,T_{cj})),\quad j\in J_i,
\]
\[
L_{\rm MCC}=\frac{1}{Bn(n-1)}\sum_i\sum_{j\ne k\in J_i}
\operatorname{KL}(p_{ij}\Vert p_{ik}).
\]

Selected logits have shape `[B,n,C]`. Both KL arguments retain gradients to
E; neither labels, S, learnable class anchors nor fusion enter this branch.
The existing algebraic reduction uses O(B*n*C) loss work/storage and does not
construct `[B,n,n,C]`. The cosine helper computes `[B,80,C]` once and gathers
selected logits, avoiding a `[B,n,C,D]` bank expansion.

**RD**, with centers created in `CIPTDCCL._init_env_anchor()` and the loss in
`cipt_losses.relative_distance_loss()`:

\[
A_c=\operatorname{normalize}\left(\frac1{80}\sum_{j=1}^{80}T_{cj}\right),
\]
\[
L_{\rm RD}=\frac1{Bn}\sum_i\sum_{j\in J_i}
\left[\cos(E_i,T_{y_i j})-\cos(A_{y_i},T_{y_i j})\right]^2.
\]

The full-80 arithmetic mean precedes normalization. E, anchors and selected
texts are normalized for their cosine similarities. Anchors and texts are
detached and fixed. RD alone uses training labels after entropy selection.
The student retains its gradient; the objective preserves template-relative
relations rather than replacing them with `MSE(E,A[y])`.

For either mutually exclusive constraint:

\[
L=\lambda_e CE(E,\text{learnable text})+
\lambda_s KL(U\Vert p_S)+\gamma IND(E,S)+
\lambda_{con}(t)L_{con}(E)+\lambda_{env}L_{env}.
\]

CE appears once. There is no fused classification, TDA/SIGM input or added
projection/classification head. Inference is always
`image -> frozen CLIP -> E -> learnable class text -> logits`; it accepts no
labels and performs no environment selection. SWAD uses the existing trainer.

New logs include `env_loss`, `env_mcc_loss`, `env_rd_loss`, `env_loss_term`,
`env_loss_w`, `env_topk`, `env_anchor_active`, `entropy_selector_active`,
and mean/selected entropy statistics. `prompt_count` records n when active;
the old `prompt_selector_active` stays zero. Old fusion/KL loss fields stay
zero in the new mode, preventing confusion with direct-E regularization.

## Validation

Observed result: **84 tests passed** (68 existing + 16 new). Both methods
passed the five-step smoke test and five averaging updates with finite losses
and identical predictions after strict state reload. Validation used CPU
PyTorch `2.14.1+cpu`; syntax compilation and `git diff --check` also passed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_cipt_*.py'
python tests/test_cipt_env_anchor.py --smoke
```

The new tests cover the official `[C,80,D]` bank and class alignment,
independent `[B,n]` selection, label/E independence, `[B,n,C]` MCC logits,
explicit ordered-pair KL values and gradients, identical-prediction zero,
explicit RD and E=A[y] zero, effective E/adapter gradients, frozen official
text/anchors/CLIP, exact original PromptLearner CE gradients, excluded fusion
parameters/no fusion calls, zero disabled contrastive loss, real-constructor
legacy/env_kl/clip_ce updates/predictions, configuration/CLI overrides,
n=2/40/80, batch=1, float32/float16/bfloat16 inputs, and CPU determinism.

Smoke runs five optimizer steps per method, updates the production
`AveragedModel` five times, predicts without labels, and strictly reloads the
averaged state with identical predictions. Only pretrained CLIP weights and
tokenization are replaced with a tiny frozen CPU encoder; actual
PromptLearner/text bank, constructors, E/S, optimizer, contrastive loss, and
averaging execute. This does not measure PACS accuracy, CUDA determinism or
the full LossValley selection process. No GPU, PACS data, or pretrained CLIP
checkpoint is present in this validation environment.

## PACS commands

These use the existing AutoDL paths and the established E=1, S=1, IND off
comparison. Both keep source-preset contrastive settings, including warmup.
The default runner sweeps all four held-out PACS domains when `--test_envs`
is omitted. Ensure PACS and `ViT-B-16.pt` exist at the stated paths.

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_b5b_entropy_mcc_swad \
  config_cipt_b5b_entropy_mcc.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /root/autodl-tmp/data/repro_dccl_data \
  --output_root /root/autodl-tmp/train_output/ablation/b5b_entropy_mcc_rd \
  --cipt_clip_path /root/autodl-tmp/models/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0 \
  --cipt_use_ind false --cipt_env_topk 40 --cipt_env_loss_weight 1.0

CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_b5b_entropy_rd_swad \
  config_cipt_b5b_entropy_rd.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /root/autodl-tmp/data/repro_dccl_data \
  --output_root /root/autodl-tmp/train_output/ablation/b5b_entropy_mcc_rd \
  --cipt_clip_path /root/autodl-tmp/models/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 \
  --aug 0 --lr 5e-5 --batch_size 32 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_e_cls_weight 1.0 --cipt_s_uniform_weight 1.0 \
  --cipt_use_ind false --cipt_env_topk 40 --cipt_env_loss_weight 1.0
```

To enable the retained IND term, replace `--cipt_use_ind false` with
`--cipt_use_ind true --cipt_gamma 5`. To explicitly remove contrastive warmup
for a separate ablation, pass `--cipt_contrastive_warmup_steps 0`; neither the
presets nor the implementation changes the source's default behavior.
