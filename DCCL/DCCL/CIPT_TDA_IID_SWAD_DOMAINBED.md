# CIPT TDA + DomainBed IID / SWAD 模型选择

起点：`analysis/cipt-tda-results-20261029-212312`，提交
`07576d266a45c8f46ad0d81201a5ba12271dc8e1`。

新入口：`train_cipt_tda_iid_swad_domainbed.py`。
默认使用已修正的候选类别一致 TDA (`--model-variant consistent`)；
`--model-variant official` 可对原始上游评分方式使用同一套模型选择。
模型、损失、优化器和训练循环来自固定上游子模块；本功能通过优化器的
post-step hook 接入选择和平均，保留 30 epochs、Adam、lr=0.0025、
global batch=64、epoch-wise cosine、beta=4、gamma=5、K=4、8 heads、
CLIP 图像预处理和 TDA 推理。

## 模型如何选择

| Selection | 选择规则 |
| --- | --- |
| iid | 分别评估三个源域的 out 划分，按准确率的域平均取最大；并列取最早的检查点 |
| last | 完整训练预算结束时的模型 |
| SWAD | 复用本仓库的 LossValley，以源域验证 CE 的域平均检测平坦区间；对每个 optimizer step 的可训练参数做密集平均 |

SWAD 默认参数与 `config.yaml` 一致：`n_converge=3`、`n_tolerance=6`、
`tolerance_ratio=0.3`。这些窗口的单位是**验证检查点**。
默认每个 epoch 验证一次；可设 `--checkpoint-freq 200` 改为每 200 次参数更新验证。
无论使用哪个间隔，最终模型及最后一个不完整平均区间都会纳入处理。
`--eval-every N` 只在 `--checkpoint-freq 0` 时生效。

SWAD valley 结束后固定 SWAD 权重，基础模型继续训练到完整预算，IID 继续选择。
若预算内未检测到收敛，遵循本仓库原实现，返回最后一个密集平均区间；
`metrics.json` 中会明确记录 `swad.status=not_converged_last_segment` 并打印提示。

源域验证和最后的报告阶段保存并恢复 Python/NumPy/PyTorch/CUDA 随机状态。
SWAD 开关不会改变基础模型后续的随机模板选择。SWAD 队列仅保存 prompt、
E/S adapters 和 TDA 的可训练参数，冻结 CLIP 不进入平均队列。
原 ViT-B/16 CIPT 的这些模块使用 Linear/LayerNorm，无需重估 BN。

两个模型的选择和 checkpoint 保存全部完成后，才访问目标域进行最终评估。

## PACS：GPU1，四个目标域

从仓库根目录开始：

```bash
git fetch origin
git switch feature/cipt-tda-domainbed-iid-swad
git submodule update --init --recursive
cd DCCL/DCCL

CUDA_VISIBLE_DEVICES=1 python -u train_cipt_tda_iid_swad_domainbed.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --test-envs 0 1 2 3 \
  --seed 0 --trial-seed 0 --holdout-fraction 0.2 \
  --epochs 30 --batch-size 64 --lr 0.0025 \
  --beta 4 --gamma 5 --k 4 --num-heads 8 \
  --num-workers 4 --eval-num-workers 0 \
  --output-root train_output/ablation/cipt_tda_iid_swad
```

VLCS 使用相同命令，仅将 `--dataset PACS` 改为 `--dataset VLCS`。
两个数据集会存入不同子目录。每个已存在的目标域目录默认拒绝覆盖。
再次完整训练时可指定新的 `--output-root`。

## 输出表格和 checkpoint

终端会在每个目标域完成后更新以下结构的 `target_out` 表格：

| Selection | art_painting | cartoon | photo | sketch | Avg. |
| --- | --- | --- | --- | --- | --- |
| iid | 训练后填入 | 训练后填入 | 训练后填入 | 训练后填入 | 四域算术平均 |
| last | 训练后填入 | 训练后填入 | 训练后填入 | 训练后填入 | 四域算术平均 |
| SWAD | 训练后填入 | 训练后填入 | 训练后填入 | 训练后填入 | 四域算术平均 |

主表显式使用 `target_out`，同时生成 `target_in` 和源域验证表。
旧 `train_all.py` 的 iid/SWAD 行实际返回 `test_in`；跨版本比较时应使用
相同划分的表格。新结果的 JSON 保留全部分数，避免根据名称推测划分。

`train_output/ablation/cipt_tda_iid_swad/PACS/` 下：

- `summary_targets_0-1-2-3_seed_0_trial_0.md`：三张表（target_out、target_in、source_val）。
- 同名前缀的 `.json`：完整结果，准确率为 0 到 1 的数值。
- 同名前缀的 `_target_out.csv`、`_target_in.csv`、`_source_val.csv`：百分比表格。
- 每个 `target_*/seed_0_trial_0/`：`config.json`、`history.json`、
  `epoch_losses.json`、`results.jsonl`、`metrics.json`。
- 每个目标域保存 `iid_best_trainable.pth`、`final_trainable.pth`、
  `swad_trainable.pth`，均包含可训练权重及选择元数据。

重新生成已完成实验的表格：

```bash
python train_cipt_tda_iid_swad_domainbed.py \
  --dataset PACS --test-envs 0 1 2 3 \
  --output-root train_output/ablation/cipt_tda_iid_swad \
  --summarize-only
```

已有旧实验仅保存第30轮权重，无法从旧 `history.json` 恢复 IID 或 SWAD 权重；
需要使用新入口重新训练。此入口未在当前环境执行真实数据集训练，
新 IID/SWAD 准确率应以服务器运行结果为准。

## CPU 验证

```bash
python -m unittest discover -s tests -p 'test_cipt_iid_swad.py' -v
python -m unittest discover -s tests -p 'test_cipt_tda_consistent.py' -v
```

测试覆盖源域宏平均、IID 并列规则、目标域访问隔离、与原 LossValley
数值一致性、密集平均、valley 结束后继续基础训练、随机状态隔离、
checkpoint 重载、目标 in/out 分表，以及上游训练循环的微型端到端运行。
