# DPLCLIP with DomainBed source IID selection

The reference is [DPLCLIP at `53b1ff4`](https://github.com/shogi880/DPLCLIP/tree/53b1ff4456a8c27fbe8f0e03d8d2136ac85ccace). The migration uses the existing DomainBed loaders, splits, trainer, evaluator and JSONL records. The algorithm implements [DPL equations (5)–(7)](https://arxiv.org/html/2111.12853#S3.SS3).

- OpenAI ViT-B/16 image and text parameters are frozen in FP32. Text autograd reaches the continuous context and the MLP. The 4,727,808 trainable parameters use Xavier uniform weights and biases initialized to 0.01.
- A separate replacement sampler draws 32 images per source domain. DPL averages the MLP outputs within each source batch. One 16-token context per batch goes between the start token and the class-name suffix. Classification uses normalized image and end-of-text features with the frozen CLIP logit scale.
- The upstream code concatenates the three text banks to form `3K` columns. Labels span `0...K−1`, so cross-entropy treats the first domain's text columns as the positive classes for every image. The port uses each domain's `K` scores and averages the domain losses, as specified by paper Eq. (6). This changes the released code's loss. `scripts/audit_dplclip.py` preserves the upstream expression and demonstrates the mismatch with two domains and two classes.
- Dataset transforms match the upstream `clip_transform` branch. All splits use bicubic resize, center crop, RGB and CLIP normalization. The target split retains DomainBed's seed-shuffled sequential order. CIPT groups source training keys by label at full label ratio, which preserves split membership and affects the exact training draw order and source-train evaluation batch composition.
- Evaluation is sequential and split-local, with 64 images per batch and the final short batch retained. Every evaluation batch provides one mean prompt. The target images and labels are used during evaluation, outside the optimizer path.
- The patched trainer performs exactly 5001 updates with evaluations at zero-based post-update steps 0, 300, …, 4800 and 5000. It selects the first maximum of `source_val`, the arithmetic mean of the three source holdout accuracies, which is also `train_out`. It reloads the best MLP and verifies its split metrics. `test_in` is the target 80% split and `test_out` is the target 20% split. The official repository's IID report uses the target 80% split. The requested target 20% comparison and the paper's three-seed, hyperparameter-selected averages therefore use different reporting protocols.
- SGD uses learning rate 0.001, momentum 0.1, zero weight decay and a constant rate. The three-layer MLP uses width 512 and dropout 0.1. The queue starts a fresh process for every target with model, sampler, worker and split seeds derived from the configured seed 1. SWAD and hyperparameter search are disabled.

Run from `DCCL/DCCL` in the project's PyTorch/OpenAI CLIP environment. Set `DATA_DIR` and `CLIP_PATH` to the existing data directory and official local weights.

```bash
PYTHONPATH=. python scripts/audit_dplclip.py "$DATA_DIR"
python train_all.py default_seed1 --algorithm DPLCLIP --dataset PACS --test_envs 0 \
  --seed 1 --trial_seed 1 --data_dir "$DATA_DIR" --dpl_clip_path "$CLIP_PATH" \
  --output_root train_output/baselines/DPLCLIP/default_seed1
```

The series runner is `scripts/run_dplclip_default_seed1.py`. It takes the audited data and CLIP paths, GPU UUID, output root and audit directory with `data_audit.json`, `smoke_console.log` and `smoke_best_summary.json`. Set `--archive-remote` to the private results repository for per-dataset publication. Local best MLP weights contain the fold configuration needed to reconstruct the frozen model and are stored alongside each run's logs.
