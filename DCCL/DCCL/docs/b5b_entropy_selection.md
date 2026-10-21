# B5b low-entropy template selection

Load `config_cipt_b5b_entropy.yaml` **after** the existing objective configuration,
or use `--cipt_template_mode b5b --cipt_selector_mode entropy --cipt_entropy_topk 40`.
N can be 1..80 (MCC requires at least 2). `cipt_k` and selector candidate count do
not control this mode. Existing random/all/adaptive behavior is unchanged.

For each image, reuse its already-computed frozen CLIP visual feature V and the
cached official B5b bank `[C,80,D]`. Each template defines one full C-way classifier:

    p(c | V, template j) = softmax_c(exp(CLIP.logit_scale) * cos(V, t[c,j]))
    H[j] = -sum_c p[c,j] log(p[c,j])

Select the N lowest-entropy templates per image, with stable original-order tie
breaking. The same selected template indices are reused for ALL candidate classes.
Selection takes no ground-truth/predicted label, no E/S, and no gradients. It makes
no extra CLIP encoder calls. Low entropy expresses confidence, not correctness.

## Fusion and classification

This change only selects templates. It does not add MCC/RD or change existing
loss definitions, weights, E/S adapters, class prompts, contrastive learning, or SWAD.

- **Training:** after label-free template selection, preserve original B5b
  conditioning: gather those templates for the source image's true class and
  apply the existing TDA/SIGM. The training label conditions fusion, not selection.
- **Legacy inference with fusion:** target labels are unavailable. For each
  candidate class c, fuse E with that candidate's selected contexts and score the
  resulting features against the learned class-c text. Average N logits per class.
- **env_kl / e_only inference:** directly classify E with learned class texts;
  neither entropy selection nor fusion executes, even if fusion trained E.
- **env_kl / mean_logits or mean_probs:** reuse label-free candidate-class scores
  `[B,N,C]` and apply the selected aggregation. Candidate classes are streamed to
  bound peak fusion memory. No target labels or single pseudo-class are used.
- **cipt_use_tda=false:** fusion losses and training selection are skipped as
  before. Merely selecting `entropy` does not enable an environment objective.

For `env_kl`, B5b is newly allowed only with `entropy` selection. The existing
class-agnostic b5a/b5c paths retain their behavior. `cipt_use_tda` controls training
fusion; explicit fused inference remains controlled by `cipt_env_inference`.

Training logs include `entropy_selector_active`, `prompt_count`,
`env_entropy_mean`, and `env_selected_entropy_mean/min/max`. Entropy metrics are
emitted when training selection actually executes. N=80 uses the entire bank;
it still computes the entropy ranking but does not discard any template.

Example: append the selector overlay and override N without changing objectives:

```bash
python train_all.py pacs_b5b_entropy \
  config_cipt_sigm_mcc.yaml config_cipt_b5b_entropy.yaml \
  --cipt_entropy_topk 40 \
  --cipt_fusion_mode tda \
  --cipt_env_inference e_only
```

Keep the usual dataset/data/checkpoint paths and explicit loss/contrastive
switches for the intended experiment. This example does not establish optimal
weights or PACS/VLCS performance.
