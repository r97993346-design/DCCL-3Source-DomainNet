# TDA 前后对比学习位置消融

分支：`feature/cipt-post-tda-mean-contrastive`。

基准：`feature/cipt-paired-class-agnostic-subject-validation-official-norm-idinit`，
基准提交：`48d80eba0d634146d13960d23338cbef94a7b9c4`。

## 本分支实现的内容

加入 `cipt_contrastive_position`，同时支持现有 `supcon` 和
`neighbor_retention`。本分支的配置和注册表默认使用 `post_tda_mean`。

| 位置 | 输入对比损失的表征 R | 邻域退化比较 |
|---|---|---|
| `e` | TDA 前的 E | V 与 E |
| `post_tda_mean` | 选中的 K 个 TDA 输出逐个归一化、求平均、再归一化 | V 与 R |

具体计算：

```text
Z = selected_TDA_outputs                # [B,K,D]，复用分类路径已算出的输出
M = mean(normalize(Z, dim=-1), dim=1)   # [B,D]
R = normalize(M, dim=-1)               # [B,D]
w_i = stop_gradient(1 + alpha * max(r_i(V) - r_i(R), 0))
loss = CIPT_base_loss + lambda_eff * mean_valid(w_i * SupCon_i(R))
```

普通 `supcon` 使用相同 R，但不应用邻域权重。正样本仍然是合并源域批次中
其他同类别图片，包含同域和跨域同类；异类作为负样本。没有把不同文本干预
展开成 B*K 个锚点，也没有新增图像视图、投影头或编码器前向。

V 和邻域权重停止梯度。新位置的损失经过 R、Z、TDA 回传到 E 适配器；
不为 S 添加新损失。自适应选择仅对索引停止梯度，选中的 Z 保留梯度。
原有视觉 L2 归一化、E/S identity init、文本库、选择器及分类推理路径继承基准。

本实现只是位置消融，保留原来的邻域纯度退化加权，未实现另一个“具体近邻边
软目标混合”的设计。这里也不声称严格保留预训练图的全局连通性。

## 开关行为

- `--cipt_contrastive_position e` 恢复原来的位置。
- `--cipt_contrastive_position post_tda_mean` 使用新位置。
- `--cipt_contrastive_type supcon` / `neighbor_retention` 与位置开关独立。
- `--cipt_use_tda false` 时明确回退到 E；分类仍是原有的直接 E 分类。
- `--cipt_use_contrastive false`、`--cipt_pure true` 或总系数为 0 时关闭对比。
- 关闭对比且没有要求邻域诊断时，跳过新增池化和邻域计算。
- `--cipt_neighbor_diagnostics true` 可以在普通 SupCon / 无对比组测量所配置
  位置的邻域，不增加损失；TDA-off 时诊断同样回退到 E。
- 原有 SWAD 开关及其训练/评估流程不变，位置开关不参与预测计算。

训练入口同时保护 sconf 的 `--key value` 参数，避免 argparse 将第一个未知
参数的值（例如 `b5c`）误当作位置参数中的配置文件。原有配置文件形参仍可使用。

## 日志

| 字段 | 含义 |
|---|---|
| `con_post_active` | 1：本步启用了对比损失，且使用真实 TDA 聚合输出 |
| `con_tda_fallback` | 1：当前需要对比/诊断，但 TDA 关闭，新位置回退到 E |
| `con_z_check` | 1：本步确实测量了 TDA 输出诊断；0 时其他零值是占位值 |
| `con_z_coherence` | 逐图片的 K 个单位方向平均后的长度，再对批次取平均，接近 1 表示这些输出方向接近 |
| `con_z_cos_e` | 聚合方向与 TDA 前 E 的平均余弦相似度 |
| `nbr_pur_e` | 保留历史列名；现在表示所选 R 空间的近邻同类比例，需结合位置配置解释 |

后两个表征诊断均分离梯度，不增加新的优化目标。`con_z_cos_e` 也受 LayerNorm
等映射影响，不等同于纯文本残差大小。高一致性本身不能判定 TDA 失效；需结合
分类效果、模板替换及关闭 TDA 的对照分析。聚合对比只直接约束整体方向，
不能保证每个干预输出都稳定。

## PACS 首组位置对照

在仓库的 `DCCL/DCCL` 目录运行。下面显式打开邻域对比、B5C、自适应选择、
SWAD；训练数据和 CLIP 路径沿用服务器设置。学习率为 5e-5，其他参数保持
基准常用值，作为位置对照起点，不表示已测得最优性能。

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_post_tda_mean_s0 \
  --dataset PACS \
  --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --output_root train_output/ablation/contrastive_position \
  --deterministic --trial_seed 0 --seed 0 \
  --steps 5000 --checkpoint_freq 100 --aug 0 \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c --cipt_tda_heads 1 \
  --cipt_selector_mode adaptive --cipt_selector_candidates 8 \
  --cipt_use_de true --cipt_use_ind true --cipt_use_tda true \
  --cipt_pure false --cipt_use_contrastive true \
  --cipt_contrastive_type neighbor_retention \
  --cipt_contrastive_position post_tda_mean \
  --cipt_neighbor_k 5 --cipt_neighbor_alpha 0.5 \
  --cipt_causal_contrastive_weight 1 \
  --cipt_contrastive_temperature 0.1 \
  --cipt_contrastive_warmup_steps 500 \
  --lr 5e-5 --batch_size 32 --weight_decay 0 \
  --swad LossValley
```

原 E 对照只改运行名 `pacs_e_s0` 和 `--cipt_contrastive_position e`。
普通 SupCon 对照只改运行名并使用 `--cipt_contrastive_type supcon`。
若测试随机选择，位置两组都改 `--cipt_selector_mode random`；若测试 lr=0.001，
两组都改 `--lr 0.001`。比较位置时先保持总对比系数一致，后续独立用源域验证
选择各位置的参数。不要用目标域测试表现选择参数。

VLCS 可复用同一命令，修改 `--dataset VLCS` 与运行名
`vlcs_post_tda_mean_s0` / `vlcs_e_s0`。runner 自动在上述输出目录下按数据集分开。

## 验证范围

```bash
python -m unittest discover -s tests -v
```

CPU 测试使用微型冻结编码器替代 CLIP，但运行实际 TDA、选择器和训练方法。
覆盖池化顺序、损失与梯度、V/R 邻域、两个训练入口、各模板和选择策略、16 组
组件开关、关闭/回退行为、无额外前向以及相同参数下推理一致。
真实 PACS/VLCS 准确率、训练时长及 SWAD 的 GPU 训练效果需在服务器验证。
