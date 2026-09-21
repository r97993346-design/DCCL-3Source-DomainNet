# V/E 跨域邻域保持加权 SupCon

实现分支：`feature/cipt-paired-class-agnostic-subject-validation`。

该实现保持此分支已有的配对类别无关 subject 验证、四种 TDA 模式和 SWAD
训练逻辑不变，只替换/关闭附加在因果特征 `E` 上的对比目标。冻结视觉特征
`V` 仅用于估计 `V -> E` 后哪些锚点丢失了跨域同类邻域，不接收该损失的梯度，
也不进入推理。

## 三种可控模式

| 模式 | 参数 | 实际目标 |
|---|---|---|
| 无对比学习 | `--cipt_use_contrastive false` | 仅原 CIPT 目标 |
| 标准 SupCon | `--cipt_use_contrastive true --cipt_contrastive_type supcon` | 原始单视图 SupCon(`E`) |
| 邻域保持加权 | `--cipt_use_contrastive true --cipt_contrastive_type neighbor_retention` | 锚点加权 SupCon(`E`) |

`cipt_pure=true` 是兼容旧实验的总关闭开关；全局对比系数为 0 也会关闭对比
目标。`cipt_use_de`、`cipt_use_ind`、`cipt_use_tda` 仍可独立控制其他组件。

## 方法定义

对批次中锚点 `i` 和每个其他源域 `d`，分别在冻结视觉空间 `V` 和因果空间
`E` 中取该域的 top-k 余弦近邻，并计算同类比例。只有同时包含锚点同类和异类
样本的候选域才参与计算，避免纯类别组成直接决定近邻纯度。各有效候选域等权
平均：

```text
r_i^V = mean_d purity(top-k_d(V_i))
r_i^E = mean_d purity(top-k_d(E_i))
delta_i = max(r_i^V - r_i^E, 0)
w_i = stop_gradient(1 + alpha * delta_i)
```

没有有效跨域参照时 `w_i=1`。最终目标为：

```text
L = L_CIPT + lambda_eff * mean_{i in valid SupCon anchors}
                         [w_i * L_SupCon_i(E)]
lambda_eff = lambda_max * min(1, update_step / warmup_steps)
```

邻域模块只生成锚点系数：标准 SupCon 的同类正样本、异类/其余负样本、分母、
无正样本锚点过滤规则均不改变；外层仍除以有效锚点数，而不是权重和。因此：

- `cipt_contrastive_type=supcon` 保留标准实现；
- `neighbor_retention + alpha=0` 与标准 SupCon 在损失和梯度上严格等价；
- 邻域权重完全 detach，梯度只由 SupCon 回传到 `E`；
- `V`、域编号和标签只在训练期计算权重，不增加推理成本。

该分数描述的是“冻结 CLIP 的局部类别关系在 `E` 中是否退化”，不是因果识别
分数，也不证明 CLIP 邻域必然正确。它解决的是原单视图 SupCon 对所有锚点一视
同仁、无法重点修复分解后跨域同类连接被破坏的问题。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `cipt_use_contrastive` | `true` | 对比目标总开关 |
| `cipt_contrastive_type` | `neighbor_retention` | `supcon` 或新加权方法 |
| `cipt_contrastive_temperature` | `0.1` | 仅此处 SupCon 的温度；优先于兼容项 `t` |
| `cipt_causal_contrastive_weight` | `1.0` | `lambda_max`，附加对比目标的全局系数 |
| `cipt_contrastive_warmup_steps` | `500` | 全局系数线性 warmup 步数 |
| `cipt_neighbor_k` | `5` | 每个其他源域独立选取的近邻数 |
| `cipt_neighbor_alpha` | `0.5` | 退化权重强度；权重范围为 `[1, 1+alpha]` |
| `cipt_neighbor_diagnostics` | `false` | 其他模式下是否额外测量 V/E 邻域，仅用于日志 |

温度和全局系数不要求与标准 SupCon 使用相同数值。为了追求各方法自身最佳性能，
标准 SupCon 与新方法应分别通过源域验证选择
`cipt_contrastive_temperature` 和 `cipt_causal_contrastive_weight`；新方法再独立选择
`k/alpha/warmup`。保持数据划分、主干、训练预算、验证准则和随机种子协议一致，
且不得用目标域测试集选参。默认值只是起点，不表示已是数据集最优值。

建议先用以下小搜索空间，再围绕最优点细化：

```text
temperature: {0.05, 0.07, 0.10, 0.20}
lambda_max:  {0.05, 0.10, 0.20, 0.50, 1.00}
k:           {3, 5, 10}
alpha:       {0.25, 0.50, 1.00, 2.00}
warmup:      {0, 100, 500}
```

不必做全笛卡尔积：先固定 `k=5, alpha=0.5` 搜索温度和全局系数，再搜索
`k/alpha`，最后确认 warmup。标准 SupCon 只搜索前两项。

## 与配对类别无关验证对接

对比学习开关与 TDA 文本因素正交，原四格实验不变：

| 单元 | `cipt_template_mode` | `cipt_neutral_subject` |
|---|---|---|
| Bconst | `bconst` | 忽略 |
| B0 | `b5b` | 忽略 |
| Sconst | `sconst` | `subject/thing/object/entity` |
| S0 | `b5a` | `subject/thing/object/entity` |

例如在 S0-subject 上运行新方法：

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_s0_neighbor_retention_seed0 \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --cipt_beta 4 --cipt_gamma 5 --cipt_k 4 \
  --cipt_prompt_length 16 --cipt_prompt_init "a photo of a" \
  --cipt_template_mode b5a --cipt_neutral_subject subject \
  --cipt_tda_heads 1 --cipt_pure false \
  --cipt_use_contrastive true \
  --cipt_contrastive_type neighbor_retention \
  --cipt_contrastive_temperature 0.1 \
  --cipt_causal_contrastive_weight 1.0 \
  --cipt_neighbor_k 5 --cipt_neighbor_alpha 0.5 \
  --cipt_contrastive_warmup_steps 500 \
  --lr 0.001 --output_root train_output/neighbor_retention_ablation
```

切换到标准 SupCon 时只需使用 `--cipt_contrastive_type supcon`，同时填写该基线
自己经源域验证选出的温度和全局系数；关闭时使用
`--cipt_use_contrastive false`。B0 配对实验把 `b5a` 改为 `b5b`。

## 日志和复杂度

| 字段 | 含义 |
|---|---|
| `nbr_check` | 1 表示本步实际计算近邻；0 表示跳过 |
| `nbr_pur_v` / `nbr_pur_e` | 有有效参照锚点的 V/E 跨域近邻同类比例 |
| `nbr_gap` | 正向退化量均值 |
| `nbr_cover` | 有有效跨域参照的锚点比例 |
| `nbr_drop` | 有效锚点中发生退化的比例 |
| `nbr_wmean` / `nbr_wmax` | 实际锚点权重均值/最大值 |

训练期开销主要是批内 V/E 相似度矩阵，时间和显存为 `O(B^2)`；不增加编码器
前向、增强视图、投影头或可学习参数。普通 SupCon、`alpha=0` 或关闭对比学习
时，若 diagnostics 也关闭，则完全跳过近邻计算。

## 验证

在 `DCCL/DCCL` 下执行：

```bash
python tests/test_cipt_neighbor_contrastive.py -v
```

测试覆盖已知邻域退化、跨域等权平均、退化回退、标准 SupCon 分母、detach、
`alpha=0` 损失/梯度等价、三态开关、独立温度优先级、冻结编码器、单次视觉
前向、warmup、原组件开关以及 diagnostics 不改变参数更新。
