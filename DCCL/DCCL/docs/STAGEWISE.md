# DICM stage-wise training

The stagewise path is opt-in through --stage. Run one held-out domain per invocation with --test_envs. Each target has its own checkpoint lineage. The original DICM working tree had uncommitted changes relative to commit 6d1109e; the new branch first captures that state.

## Budgets and source selection

Use full data, --seed 0 --trial_seed 0 --deterministic, checkpoint frequency 100, and SWAD LossValley 3/6/0.3. Requested PACS budgets are 2500/1000/1000 updates for stages 0/1/2. VLCS budgets are 1500/500/500. Stage 0 early and middle checkpoints default to updates 500/1200 for PACS and 300/700 for VLCS. The final and source-validation-best checkpoints are also saved. Use --stage_save_steps to specify other update numbers aligned with evaluation checkpoints.

A stage continues to its fixed budget after the SWAD valley is dead and records the first SWAD stopping point. Best training checkpoints are selected using source train_out only. They are distinct from final SWAD inference models. The original multi-target process seeded once and ran folds serially. This stagewise interface runs each target as a separate process and reinitializes the run seed; it is not assumed bitwise equivalent to historical multi-target runs.

## Stage 0

For PACS target 0:

    python train_all.py pacs_s0 configs/dicm_pacs.yaml configs/stagewise/stage0.yaml \
      --algorithm DICM --dataset PACS --stage 0 --test_envs 0 \
      --steps 2500 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
      --deterministic --data_dir DATA --output_root OUTPUT

For VLCS use configs/dicm_vlcs.yaml, --dataset VLCS, and --steps 1500. Stage 0 uses A1 and disables GT, DICM projection, SupCon and PreCL. The frozen pretrained network is constructed and may still be evaluated by the existing algorithm implementation.

## Stage 1

Use the source-validation-best Stage 0 checkpoint from the same target. Substitute stage1_ce.yaml for the CE-continuation control:

    python train_all.py pacs_s1 configs/dicm_pacs.yaml configs/stagewise/stage1.yaml \
      --algorithm DICM --dataset PACS --stage 1 --stage_parent STAGE0_BEST \
      --test_envs 0 --steps 1000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
      --deterministic --data_dir DATA --output_root OUTPUT

Use --steps 500 for VLCS. Candidate beta, rank, basis learning rate and response weights may be specified in a further YAML file after the stage YAML. Confirm effective configuration using --show. Backbone, classifier and frozen reference are loaded from Stage 0; the basis and prototype bank are initialized for the candidate.

## Stage 2

Use the same source-selected Stage 1 DICM checkpoint for all four branches, holding configuration and budget fixed except for contrastive coefficients:

    python train_all.py pacs_s2a configs/dicm_pacs.yaml configs/stagewise/stage2_a.yaml \
      --algorithm DICM --dataset PACS --stage 2 --stage_parent STAGE1_BEST \
      --test_envs 0 --steps 1000 --checkpoint_freq 100 --seed 0 --trial_seed 0 \
      --deterministic --data_dir DATA --output_root OUTPUT

Use stage2_a.yaml for CE+ISR, stage2_b.yaml for SupCon, stage2_c.yaml for PreCL, and stage2_d.yaml for both. Enabled contrastive weights are one. Use --steps 500 for VLCS. The full Stage 1 model including heads, basis and EMA is inherited. The optimizer and SWAD are rebuilt in every branch. Preserve Stage 1 DICM settings in Stage 2.

## Checkpoints and limitations

Checkpoints are saved under the run's stage_checkpoints folder by target. They contain the full training-model state, optimizer, source-best score, RNG states, stage and cumulative updates, parent hash, dataset/split fingerprint, and SWAD state. --stage_resume restores these states in the same stage with identical configuration and split. It is mutually exclusive with --stage_parent.

The multi-worker DataLoader and prefetched batches are not serialized. Resumption does not promise an identical subsequent data or augmentation sequence and must not be described as exact continuation.

Checkpoints are atomically replaced and remain local. Experiment archives publish text summaries and metrics only. GPU0 is reserved for PACS and GPU1 for VLCS; free memory must exceed 30720 MiB before launching a GPU experiment.
