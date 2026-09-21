# V/E 跨域近邻退化加权 SupCon

分支：`feature/cipt-neighborhood-retention-contrastive`

基准：`feature/cipt-official-norm-idinit-no-aug`，提交
`e306f871598e936792eb96d493611fbb9a92a782`。

## 实现的方案

冻结 CLIP 提取 V，现有适配器产生 E/S。E 继续并行参与 SupCon 和 TDA。
在现有源域批次内，分别用 V、E 的余弦相似度查找其他源域的同类近邻比例：

1. 对每个锚点，在每个其他源域分别取最近的 `min(k, 该域批次大小)` 个样本。
2. 计算其中标签与锚点相同的比例。候选域必须同时含有该类和其他类样本；
   不满足条件的域不参与平均。锚点自己的域始终排除。
3. 各有效域等权平均，得到 `r_v` 和 `r_e`。
4. 先按域平均，再计算 `gap = max(r_v - r_e, 0)`。
5. 权重为 `w = stop_gradient(1 + alpha * gap)`。没有有效参照时 `w=1`。

最终目标为：

```text
loss = CIPT_base_loss + lambda_eff * mean_valid_anchors(w_i * supcon_i(E))
lambda_eff = lambda_max * min(1, update_count / warmup_steps)
```

正负样本、SupCon 分母、每个锚点的正样本平均均沿用基准实现。近邻只决定
锚点系数，不是新的正样本筛选。外层仍除以有效锚点数，不除以权重之和。
因此总体对比强度可能增加；验证机制时应对照调整全局系数后的普通 SupCon。

这里只使用已有的 V/E、训练类别标签和源域批次归属。不增加编码器前向、
投影头、增强视图或可学习参数。附加开销来自批内相似度矩阵和近邻统计。
推理过程不需要近邻、域标签或类别标签。

本方案检测的是局部类别邻域变化，不提供因果识别或图连通性保证。
CLIP 的邻域本身也可能存在偏差，实际收益需要实验验证。

## 参数与三组消融

在 `DCCL/DCCL` 目录运行，参数由现有 sconf 配置/命令行机制读取。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `cipt_contrastive_type` | `neighbor_retention` | 新方案；`supcon` 为原损失 |
| `cipt_neighbor_k` | `5` | 每个其他源域的近邻数量 |
| `cipt_neighbor_alpha` | `0.5` | 默认权重范围 `[1, 1.5]`；0 恢复 SupCon |
| `cipt_causal_contrastive_weight` | `1.0` | 沿用当前实验的全局对比系数 |
| `cipt_contrastive_warmup_steps` | `500` | 从第一步起线性增加，非第 500 步才启动 |
| `cipt_neighbor_diagnostics` | `false` | 在普通 SupCon/关闭对比时也计算分离梯度的诊断 |
| `cipt_use_contrastive` | `true` | 设为 false 关闭整个对比损失 |

| 实验 | 追加参数 |
|---|---|
| 无对比 | `--cipt_use_contrastive false` |
| 普通 SupCon | `--cipt_use_contrastive true --cipt_contrastive_type supcon` |
| 新方案 | `--cipt_use_contrastive true --cipt_contrastive_type neighbor_retention` |
| α=0 等价检查 | `--cipt_contrastive_type neighbor_retention --cipt_neighbor_alpha 0` |

`cipt_pure=true` 和全局对比系数为 0 也会关闭对比损失。原有
`cipt_use_de`、`cipt_use_ind`、`cipt_use_tda` 开关继续有效。

基准 config 的全局系数原为 0.1，本分支按当前实验要求默认设为 1.0。
公平比较三组时请显式使用相同系数；复现基准默认时三组均设为 0.1。
基准的实际预处理/初始化是：适配器前不做视觉 L2 归一化，适配器使用默认
初始化。历史分支名和旧 config 注释与此不一致；本分支仅更正注释。
近邻计算内部的余弦归一化不会改动送入适配器或 TDA 的特征。

## PACS / VLCS 运行示例

以下用 B5C、全局对比系数 1、seed/trial_seed 0，输出按消融项目和数据集组织。
需要实际收益时，用配对的多个随机种子重复三组实验，沿用源域验证选模型。

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_neighbor_retention_s0 \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c --cipt_tda_heads 1 \
  --cipt_pure false --cipt_use_contrastive true \
  --cipt_contrastive_type neighbor_retention \
  --cipt_neighbor_k 5 --cipt_neighbor_alpha 0.5 \
  --cipt_causal_contrastive_weight 1 --cipt_contrastive_warmup_steps 500 \
  --lr 0.001 --output_root train_output/neighbor_retention_ablation
```

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py vlcs_neighbor_retention_s0 \
  --dataset VLCS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5c --cipt_tda_heads 1 \
  --cipt_pure false --cipt_use_contrastive true \
  --cipt_contrastive_type neighbor_retention \
  --cipt_neighbor_k 5 --cipt_neighbor_alpha 0.5 \
  --cipt_causal_contrastive_weight 1 --cipt_contrastive_warmup_steps 500 \
  --lr 5e-5 --output_root train_output/neighbor_retention_ablation
```

普通 SupCon 和无对比实验分别替换模式/开关，并修改运行名称。
若要比较三组 V/E 邻域统计，请三组都追加 `--cipt_neighbor_diagnostics true`。
这只开启测量，不改变普通 SupCon 或无对比实验的目标函数。

## 日志含义与边界

| 字段 | 含义 |
|---|---|
| `nbr_check` | 1=本步计算了近邻；0=跳过，其他零统计不是实测结果 |
| `nbr_pur_v` / `nbr_pur_e` | 有有效跨域参照的锚点，其 V/E 近邻同类比例均值 |
| `nbr_gap` | 上述锚点的正向退化量均值 |
| `nbr_cover` | 有有效跨域参照的锚点占整个批次的比例 |
| `nbr_drop` | 有效参照锚点中发生退化的比例 |
| `nbr_wmean` | 实际用于有效 SupCon 锚点的平均权重；关闭/普通模式为 1 |
| `nbr_wmax` | 实际最大权重；关闭/普通模式为 1 |

当 `nbr_cover=0` 时，纯度等零值是无有效参照的占位值。若 k 覆盖整个候选域，
纯度只取决于该域类别组成，V/E 差值为零；大类别数、小批次也可能使有效参照
稀疏。上述指标有助于辨别方案是否实际生效。

这些是训练批次诊断，不能代替固定、类别均衡的验证集近邻测量，也不能单凭
训练统计宣称泛化改善。诊断不改变随机数状态，不额外采样训练图片。

## 验证

在 `DCCL/DCCL` 执行：

```bash
python tests/test_cipt_neighbor_contrastive.py -v
```

测试涵盖已知退化案例、各域等权平均、无参照回退、α=0 的损失/梯度等价、
原 SupCon 分母、分离梯度、冻结编码器、单次视觉前向、线性 warmup、
原有消融开关，以及诊断不改变更新。更新测试使用小型冻结编码器替代 CLIP；
不将其视为实际数据集训练或性能验证。
