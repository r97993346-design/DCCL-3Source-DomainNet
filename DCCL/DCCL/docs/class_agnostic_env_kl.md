# 类别无关环境扰动 + prompt 一致性

源分支：`feature/cipt-paired-class-agnostic-subject-validation-official-norm-idinit`。
实现基于提交 `02d02c4cfd62e9390b0d415d9d77a13c65c3dc92`。

## 训练与推理

冻结 CLIP 图像编码器，保留源分支的视觉 L2 归一化与 E/S adapter identity init。
一张图像仅编码一次，分解得到 E、S。E 接受原有类别监督和可切换的
SupCon / neighbor-retention 对比学习。b5c 类别无关环境库经过已有
Safe-Diverse 选择和 TDA 融合后，保留 `[B,K,D]`，使用同一类别文本锚点
得到 `[B,K,C]` 预测。环境分支仅接受 prompt 间 KL，不增加融合特征的标签 CE。

KL 的数学定义为：

```math
L_{env}=\frac{1}{BK(K-1)}\sum_i\sum_{k\ne l}KL(p_{ik}\|p_{il}).
```

这里包含两个方向，两侧均反向传播。没有 detach teacher，也不将原始 E 的预测
加入 KL 配对。已有选择器的索引计算仍然 detached、label-free；被选中的融合特征
保留梯度。KL 不额外 warmup，原有对比学习 warmup 独立保留。

推理直接用 E 对学习到的类别文本分类，不运行环境选择、TDA 或对比学习。

## 完整目标及权重

```math
L_E=CE(h(E),y),\quad L_U=KL(U\|p_S),\quad
L_{IND}=\frac{1}{2B}\sum_i\cos(E_i,S_i)^2.

L=\lambda_E L_E+\lambda_U L_U+\gamma L_{IND}
  +\lambda_{con}(t)L_{NR}(E)+\lambda_{env}L_{env}.
```

**没有单独的 TDA 分类项，也不把 DE 重复加回总损失。** 原 DE 为 `L_E+L_U`。
源分支关闭 TDA 时，额外分类项退化成同一 E logits 的 CE，因此其基础目标实际是
`(1+beta)*L_E + beta*L_U + gamma*L_IND`。新模式把重复的 E CE 合并成一个明确权重。

默认 `cipt_e_cls_weight: null` 自动取 `1+beta`（DE 关闭时取 1），
`cipt_s_uniform_weight: null` 取 `beta`。默认 beta=4、gamma=5 时为
`5*L_E + 4*L_U + 5*L_IND`，与源分支无 TDA 基线保持相同监督强度。
若希望只保留原 DE 的系数，可显式指定 `--cipt_e_cls_weight 4`；这是权重变化，
应与 KL 效果分开做消融。`cipt_use_de=false` 关闭 S 均匀项，并使自动 E 权重回到 1；
显式的 E 权重仍然有效。

环境 KL 默认权重为 0.1，是尚未验证的实验起点，不是已知最优参数。
保持现有邻域参数 k=10、alpha=4、temperature=0.07、最终权重=1.5、warmup=500。

## 运行

在仓库的 `DCCL/DCCL` 目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cipt_env_kl.sh PACS
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cipt_env_kl.sh VLCS
```

也可直接使用训练入口：

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_b5c_env_kl config_cipt_env_kl.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/ablation/env_kl \
  --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0
```

首轮只比较 KL 开关，其他设置一致：

```bash
CIPT_RUN_NAME=pacs_env_kl_off CUDA_VISIBLE_DEVICES=1 \
  bash scripts/run_cipt_env_kl.sh PACS --cipt_env_kl_weight 0
CIPT_RUN_NAME=pacs_env_kl_on CUDA_VISIBLE_DEVICES=1 \
  bash scripts/run_cipt_env_kl.sh PACS --cipt_env_kl_weight 0.1
```

权重为 0 或 `cipt_use_tda=false` 会跳过环境选择和融合，推理仍使用 E。
`--cipt_selector_mode random` 可关闭自适应选择；`--cipt_use_contrastive false`
关闭对比；`--cipt_contrastive_type supcon` 切换标准对比。
脚本支持 `CIPT_DATA_DIR`、`CIPT_CLIP_PATH`、`CIPT_RUN_NAME` 环境变量，以及追加训练参数。
输出在 `train_output/ablation/env_kl`。

`config.yaml` 默认仍为 `cipt_objective: legacy`，保留原训练/推理流程。
`config_cipt_env_kl.yaml` 显式开启新目标。配置优先级为 registry → YAML → 显式 CLI；
同时修复了已知 argparse 参数被实验 YAML 覆盖的问题，便于选择器开关消融。

## 日志与验证范围

分别记录 `e_cls_loss`、`s_uni_loss`、`cipt_ind_loss`、`dccl_contrastive_loss`、
`env_kl_loss`，以及显式权重和加权项。兼容字段 `cipt_de_loss` 仅作诊断，不能再加到总损失。
`env_shift` 为融合前后归一化特征的平均余弦距离，`env_spread` 为同一图像各 prompt
归一化融合特征的方差，用于检查融合是否忽略文本。KL 降低本身不能证明域泛化提升。

针对性 CPU 测试比较 KL 的逐对定义、数值和两侧梯度，检查 KL 经融合更新 E、
环境无标签 CE、E-only 推理、开关行为、legacy 回归和配置覆盖。
这些验证不等同于完整 PACS/VLCS 训练，性能需要服务器上的实际实验确认。
