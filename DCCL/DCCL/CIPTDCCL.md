# CIPT + WBC-CL（仅作用于因果表示 e）

本分支在 `feature/cipt-ccl-safe-diverse-noaug` 的单视图实验基线上，
将原来的普通监督对比损失替换为 **Weakest-domain Bridge Causal
Contrastive Learning（WBC-CL）**。CLIP 图像编码器、因果/伪相关分解、
Safe-Diverse 提示选择、TDA、分类路径、优化器和 SWAD 均保持不变。

## 约束

WBC-CL 的相似度计算只接收原图对应的因果表示 `e`：

```text
x -> frozen CLIP -> v -> causal decomposition -> e -> WBC-CL
```

它不使用 `s`、干预后表示 `z_k`、文本特征、第二视图 `x_2` 或增强表示
`e_aug`，也不增加投影头。源域编号直接来自 DomainBed 传入的各源域
minibatch。

## 损失定义

设一个训练步合并后的源域 batch 为
`{(e_i, y_i, d_i)}_{i=1}^B`，先对因果表示做 L2 归一化：

$$
\bar e_i = \frac{e_i}{\lVert e_i\rVert_2}, \qquad
s_{ij}=\bar e_i^\top\bar e_j.
$$

对于锚点 $i$，在每个其他源域 $d\ne d_i$ 中寻找同类样本，取相似度
最大的 Top-K 个并求均值，得到该域的桥接分数：

$$
b_{i,d}=\operatorname{MeanTopK}
\left\{s_{ij}\mid y_j=y_i,\ d_j=d\right\}.
$$

如果某个“类别×域”分组在当前 batch 中为空，则只跳过该分组，不伪造
正样本。对所有可用桥接域做平滑最小值，使优化重点落在最弱的跨域同类
连接上：

$$
p_i=-\tau\log\left(
\frac{1}{|\mathcal D_i^+|}
\sum_{d\in\mathcal D_i^+}\exp(-b_{i,d}/\tau)
\right).
$$

再对 batch 内所有异类因果表示做平滑最大值，得到难负样本分数：

$$
n_i=\tau\log\left(
\frac{1}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}\exp(s_{ij}/\tau)
\right),\qquad
\mathcal N_i=\{j\mid y_j\ne y_i\}.
$$

单锚点损失为带间隔的 Softplus 排序损失：

$$
\ell_i=\operatorname{softplus}
\left(\frac{n_i-p_i+m}{\tau}\right).
$$

只有同时具有至少一个跨域同类桥接和至少一个异类负样本的锚点参与
平均。最终目标为：

$$
\mathcal L_{total}=\mathcal L_{CIPT}
+\lambda_{eff}\mathcal L_{WBC},
$$

其中

$$
\mathcal L_{CIPT}=\mathcal L_{cls}
+\beta\mathcal L_{de}+\gamma\mathcal L_{ind},
\qquad
\lambda_{eff}=\lambda_{max}\min(1,t/t_{warmup}).
$$

## 参数与监控项

默认参数：

- `cipt_causal_contrastive_weight: 0.1`：$\lambda_{max}$；
- `cipt_contrastive_warmup_steps: 500`：线性 warmup 步数；
- `cipt_wbc_topk: 2`：每个其他源域的同类 Top-K；
- `cipt_wbc_margin: 0.1`：排序间隔 $m$；
- `cipt_wbc_temperature: 0.1`：SoftMin、SoftMax 与 Softplus 温度 $\tau$。

训练日志新增：

- `wbc_contrastive_loss`：WBC-CL 损失；
- `contrastive_valid_anchor_fraction`：有效锚点比例；
- `wbc_domain_coverage_fraction`：锚点可找到同类桥接的其他源域比例；
- `wbc_weakest_positive_similarity`：平滑最弱正桥相似度；
- `wbc_hard_negative_similarity`：平滑难负样本相似度；
- `wbc_violation_fraction`：未满足间隔的有效锚点比例。

## DomainNet 三源域示例

以下示例用源域 `0 1 2`、目标域 `5`；可按实验表替换域编号：

```bash
cd DCCL/DCCL
CUDA_VISIBLE_DEVICES=0 python train_all.py domainnet_wbc \
  --dataset DomainNet --algorithm CIPTDCCL \
  --data_dir /path/to/data \
  --source_envs 0 1 2 --target_env 5 \
  --deterministic --trial_seed 0 --seed 0 \
  --cipt_clip_backbone ViT-B/16 \
  --cipt_clip_path /path/to/ViT-B-16.pt \
  --cipt_template_mode b5c \
  --cipt_selector_mode adaptive \
  --cipt_causal_contrastive_weight 0.1 \
  --cipt_contrastive_warmup_steps 500 \
  --cipt_wbc_topk 2 \
  --cipt_wbc_margin 0.1 \
  --cipt_wbc_temperature 0.1
```

建议先固定其他设置，仅扫描
`cipt_causal_contrastive_weight`（`0.05/0.1/0.2`）和
`cipt_wbc_margin`（`0.05/0.1/0.2`）。如果
`wbc_domain_coverage_fraction` 长期偏低，应优先采用类别均衡采样或增大
各源域 batch size，而不是把缺失域样本错误地当成正样本。
