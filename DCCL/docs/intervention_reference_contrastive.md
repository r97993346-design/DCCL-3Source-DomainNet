# 干预可靠性引导的 e 对比学习

分支：`feature/cipt-intervention-reliable-contrastive`

基础：`feature/cipt-official-norm-idinit-no-aug`，提交 `e306f871598e936792eb96d493611fbb9a92a782`。

本模块用已有 TDA 分类结果判断同类参照的可靠性，调整正样本对的权重。相似度只计算在干预前的 `e` 上。它是一个需要通过消融检验的训练机制，干预下分类正确不等于已识别出纯因果特征。

## 本分支实际接入的位置

基础分支使用原有模板采样，没有 Safe-Diverse 候选筛选器。因此直接复用本次训练已经计算的 `[B, K, C]` TDA logits，默认 `K=4`，不会另取 8 个候选或重新运行 TDA。

- B5a：原有四条通用文本，按原逻辑固定/循环取用。
- B5b：原有类别条件文本，训练时按标签索引；测试时仍按候选类别评分。
- B5c：原有类别无关文本，训练随机取 K 条，测试固定取用。

模板内容、采样、分类路径、因果分解、图像预处理、无增强视图设置和 SWAD 配置均继承基础分支。本次不合并其他分支的模板或选择器。

## 损失定义

对样本 j，可靠性 `r_j` 是本次 K 个 TDA 预测中，argmax 与真实类别相符的比例。始终预测错误时得分为 0，而非 1。分数停止梯度。

正样本参照权重：

```text
q_j = reference_floor + (1 - reference_floor) * r_j
w_ij = 1[y_i == y_j and i != j] * q_j
w_ij = w_ij / sum_j(w_ij)
```

对每个 anchor，使用这些正样本权重计算原有单视图 SupCon 的加权平均。分母仍包含所有非自身样本；相似度为归一化 e 之间的点积除以温度。anchor 之间统一平均，不乘 anchor 自身的可靠性。e/e 相似度两端都保留梯度，与基础 SupCon 相同；没有新增单向教师或 EMA 网络。

默认权重下限 0.1，所以不稳定样本仍参与训练。所有参照分数相等（包括全部为 0）时，数学上恢复原有 SupCon。只有一个正样本时，归一化后的权重必然为 1；此时分数变化不会改变该 anchor 的目标。

新增对比项直接更新 causal adapter；分数不向 TDA、分类文本或 s 回传梯度。原有 CIPT 分类和分解损失仍按原逻辑更新其相关参数。

## 模式和默认值

这些参数通过项目现有的 sconf 配置/命令行机制读取。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `cipt_contrastive_reference` | `intervention` | 使用本次 TDA 下的分类正确比例 |
| 同上 | 可设 `confidence` | 对照：干预前 e 的真实类别概率，停止梯度 |
| 同上 | 可设 `uniform` | 对照：原有统一权重 SupCon |
| `cipt_contrastive_reference_floor` | `0.1` | 参照权重下限，范围 `(0, 1]` |
| `cipt_causal_contrastive_weight` | `0.1` | 沿用基础 config.yaml 的总对比系数 |
| `cipt_contrastive_warmup_steps` | `500` | 从第一步开始线性增加系数，第 500 步达到设定值 |

设置总权重请用 `--cipt_causal_contrastive_weight`。基础配置中的这个参数优先于旧别名 `--cipt_contrastive_weight`；仅修改旧别名可能不会改变实际系数。

组件开关仍独立：

- `--cipt_use_contrastive false` 或 `--cipt_pure true`：关闭新增对比项。
- `--cipt_use_tda false`：`intervention` 模式明确退回统一参照，日志 `irc_tda_fallback=1`。不会把无干预的分类结果伪装成干预可靠性。
- `confidence` 模式在 L_de 关闭时仍可运行；必要时只计算已有 e 的分类 logits，不重新编码图像。

## PACS / VLCS 运行示例

在仓库的 `DCCL/DCCL` 目录执行。示例总对比系数显式设为 **1**；学习率等参数应与要比较的基线保持一致。省略 `--test_envs` 时，训练器按原逻辑遍历留一域实验。

PACS：

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_irc_b5c_seed0 \
  --dataset PACS \
  --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 \
  --checkpoint_freq 100 --aug 0 \
  --lr 0.001 \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c \
  --cipt_contrastive_reference intervention \
  --cipt_contrastive_reference_floor 0.1 \
  --cipt_causal_contrastive_weight 1 \
  --cipt_contrastive_warmup_steps 500 \
  --output_root train_output/ablation/irc/intervention
```

VLCS：

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py vlcs_irc_b5c_seed0 \
  --dataset VLCS \
  --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --deterministic --trial_seed 0 --seed 0 \
  --checkpoint_freq 100 --aug 0 \
  --lr 0.001 \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c \
  --cipt_contrastive_reference intervention \
  --cipt_contrastive_reference_floor 0.1 \
  --cipt_causal_contrastive_weight 1 \
  --cipt_contrastive_warmup_steps 500 \
  --output_root train_output/ablation/irc/intervention
```

对照实验保持相同数据划分、种子、模板、学习率、温度、总系数和训练步数，仅修改参照模式及输出目录。例如统一参照使用 `--cipt_contrastive_reference uniform --output_root train_output/ablation/irc/uniform`；置信度对照使用 `confidence`。无对比的基线使用 `--cipt_use_contrastive false` 和独立输出目录。继续按原有源域验证协议选择模型。

## 如何看日志

| 字段 | 含义 |
|---|---|
| `dccl_contrastive_loss` | 新的加权对比损失，保留原字段名 |
| `contrastive_weight_eff` | 当前实际总系数 |
| `contrastive_valid_anchor_fraction` | batch 中存在其他同类正样本的 anchor 比例 |
| `irc_reliability_mean` / `irc_reliability_std` | 当前参照分数均值 / 标准差 |
| `irc_positive_ess_fraction` | 每个有效 anchor 的正样本有效数量 / 实际数量；统一权重为 1 |
| `irc_weighted_anchor_fraction` | 正样本权重实际偏离统一权重的有效 anchor 比例 |
| `irc_intervention_active` | 是否实际使用 TDA 可靠性 |
| `irc_intervention_count` | 本次用于可靠性的 K 值；对照或关闭时为 0 |
| `irc_tda_fallback` | 是否因 TDA 关闭而退回统一权重 |

如果 `irc_weighted_anchor_fraction` 长期接近 0，说明参照加权很少实际改变损失；不能只凭整体分数标准差判断机制生效。可靠性变高也不等于泛化改善，应同时比较准确率与 `step_time`，并将评估耗时分开看。

## 验证范围

```bash
python -m unittest discover -s tests -v
```

CPU 测试覆盖损失/梯度与原定义的一致性、停止梯度、无正样本和全零分数、低精度输入，以及采用小型编码器的真实 TDA/模板采样/训练 wrapper 联合检查。检查所有参照模式、B5a/B5b/B5c、16 组组件开关和 warmup，并验证每步没有额外图像编码或 TDA 调用。

这些测试不包含预训练 CLIP 的真实 PACS/VLCS GPU 训练，不能据此声称提速比例或准确率提升。
