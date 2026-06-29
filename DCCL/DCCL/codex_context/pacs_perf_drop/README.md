# PACS DCCL causal performance drop analysis

## Experiments

Baseline DCCL:
- train_output/PACS/260628_13-31-27_DCCL_PACS_0

Causal-enhanced DCCL:
- train_output/PACS/260628_17-29-15_DCCL_PACS_causal_kept_save

## Goal

Please analyze why the causal-enhanced DCCL version causes performance degradation compared with the baseline DCCL version.

Please compare the two experiment directories and inspect the current branch code.

Focus on:

1. Whether the two experiments are strictly comparable:
   - dataset
   - source domains
   - target domain
   - seed
   - steps
   - batch size
   - learning rate
   - checkpoint frequency
   - hyperparameters

2. Whether causal_kept_save changes the effective training data:
   - generated sample number
   - kept sample number
   - filtering ratio
   - class distribution
   - domain distribution

3. Whether anchor filtering is too strict or too noisy.

4. Whether generated causal samples introduce class inconsistency or noisy positives.

5. Whether causal loss / consistency loss / filtering suppresses the original DCCL SupCon objective.

6. Whether there are implementation bugs:
   - wrong label alignment
   - wrong batch concatenation
   - wrong positive pair construction
   - incorrect detach
   - unexpected gradient flow
   - wrong loss weight
   - logging or evaluation inconsistency

7. Whether the causal branch should be:
   - detached
   - delayed after warmup
   - assigned smaller loss weight
   - used only as auxiliary regularization
   - used only for samples passing stricter consistency checks

Please output:
1. The most likely 3-5 root causes.
2. Suspicious files and functions.
3. Code-level evidence.
4. Suggested fixes.
5. Ablation experiments to verify each hypothesis.
