# PACS Center MSE: three selected experiments

01: NR=1, fixed LR=5e-5, 3800; 02: no contrastive, same LR, 3800; 03: no contrastive, same LR, 5000. Raw records are validated before upload.

Derived values are fractions (multiply by 100 for percent). IID selects each target
independently by maximum source train_out, retaining the first tie. Both target
test_in and test_out are reported at the same selected step. Cutoff tables reuse
recorded trajectories; they are not separately rerun short-budget experiments.

Prefix comparisons include loss values. Matching scalar records do not prove
bit-identical parameters; model weights are not included. NR-vs-no-NR prefix
differences are expected and are not a reproducibility test.

source_at_upload records the current worktree and may differ from training.
source_at_recorded_commit uses the saved commit, but cannot recover historical
uncommitted edits. Missing historical files are listed in upload_manifest.json.

Model weights and binary TensorBoard events are inventoried, not uploaded.
