# Faithful GD-FAS (ICCV 2025) transfer onto DomainBed

Implementation branch: `feature/gdfas-faithful-domainbed`; derived from
`feature/cipt-b5b-entropy-mcc-rd`, **without changing the existing CIPTDCCL**.

Paper / authors' code: https://github.com/SeungjinJung/GD-FAS
Core source: `models/networks.py::clip_encoder.compute_loss`.
No class-agnostic / CIPT / RD / MCC components are used in `GDFAS`.

## Preserved from official code

* End-to-end **trainable** CLIP ViT-B/16, not frozen CLIP;
  CLIP model loaded in fp32 for stable Adam fine tuning.
* Ensemble **six prompts per class**, first average raw CLIP text features,
  then normalize for image-text similarity.
* GS-RM on **domain x class** groups, official detached normalized group CE
  scaling (beta=1.5, alpha=log(source_domains)/2).
* Gram-Schmidt basis of class text ensemble features, with
  `invariant = proj(image)`, `specific = image - stopgrad(invariant)`.
* FOD supervised contrast of **same-domain** positives on raw `specific`,
  same unnormalized dot products and exact official unusual mean reduction.
* Additional 512 -> 4096 -> 4096 -> 256 BN/ReLU MLP, paired-instance
  contrast with temperature **0.8** (distinct from FOD temperature 0.1),
  and auxiliary C-way CE classifier.
* Original weights: `L = L_IT-GS + 0.8 L_FOD + 0.1 L_II + 1.0 L_aux`.
* **Inference uses raw CLIP image-text similarity**, not E-only nor the
  auxiliary classifier. No direct E-only inference is claimed.

## Necessary differences: FAS to DomainBed multiclass classification

1. The face-specific spoof/live prompts cannot describe dog/horse/guitar.
   Six generic but **class-conditioned** prompt sentences are used for each
   object category, without learning prompt context or using the CIPT bank.
2. Real vs spoof (two classes) becomes `num_classes` (PACS=7).
   Group index is `label + num_classes * domain`. Absent minibatch groups
   are skipped; the official two-class implementation assumes populated groups.
3. Original face-video sampling and MTCNN cropping do not apply to PACS.
   Each source-domain example supplies **two independently augmented views**:
   rotation, square random resized crop, horizontal flip, RGB and ImageNet
   normalization as the official GD-FAS training data transform.
   DomainBed eval keeps deterministic center-crop (the original FAS eval had
   stochastic random crop); original ImageNet normalization is retained.
4. Paper HTER/AUC face anti-spoofing metrics become DomainBed classification
   accuracy and `test_out` evaluation; no target-domain tuning.
5. DomainBed batches are **per source domain**, not the global batch size of
   the original FAS runner. `batch_size: 3` yields a total of nine source
   images per update for three domains, and eighteen view images.
6. The paper's standalone `StepLR(step_size=10, gamma=0.1)` is emulated
   with 10 iterations/epoch, hence 100 updates per LR drop. Do **not**
   reuse the original 5001-step DomainBed default with this schedule; use
   `--steps 400` as in the author's `max_iter=400` baseline.
7. Official source contains CUDA-specific tensor allocations and assumes
   a binary label set; the port uses the device of the minibatch tensors.

## PACS quick smoke (GPU 1)

```bash
cd /home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL
git fetch origin
git switch feature/gdfas-faithful-domainbed
CUDA_VISIBLE_DEVICES=1 python -u train_all.py gdfas_pacs_smoke \
  config_gdfas_pacs.yaml --dataset PACS --algorithm GDFAS \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --test_envs 0 --steps 2 --checkpoint_freq 1 \
  --num_workers 2 --eval_num_workers 2 \
  --gdfas_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt
```

## Four held-out domain protocol (single invocation)

```bash
CUDA_VISIBLE_DEVICES=1 python -u train_all.py gdfas_pacs_official_transfer \
  config_gdfas_pacs.yaml --dataset PACS --algorithm GDFAS \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --steps 400 --checkpoint_freq 100 --num_workers 2 --eval_num_workers 2 \
  --seed 0 --trial_seed 0 --deterministic \
  --gdfas_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt
```

Do not compare the original face anti-spoofing HTER directly with PACS
classification accuracy. This is a *method transfer*, not replication of the
original FAS dataset results. Best validation should be based on permissible
source-domain data, never on target `test_out`.

## Minimal ablations

* `--gdfas_gs false` disables GS-RM (same normal IT CE).
* `--gdfas_fod_weight 0` disables domain supervision loss.
* `--gdfas_ii_weight 0` disables paired-instance loss.
* `--gdfas_aux_weight 0` disables MLP classifier CE.

These keys are Config-based CLI overrides via `train_all.py` and are stored
in the run hparams; no new argparse entries were required.
