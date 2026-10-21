# 冻结 CLIP 引导的跨域同类加权对比学习

基于 `feature/cipt-b5b-entropy-mcc-rd`，新分支为
`feature/cipt-clip-guided-crossdomain`。原实验 YAML 和旧损失计算保持原样。

## 数据流和正负样本

一次冻结 CLIP ViT-B/16 编码得到 V，再经原有、恒等初始化的 adapters 得到
E/S。E 与可学习类别文本做分类，并参与本页的对比损失；S 只参与原有
`KL(U || p_S)` 均匀化约束。新方法没有投影头、增强视图或额外 CLIP 调用。

训练器只向 `update(x, y)` 传入源域 minibatch。按 `torch.cat(y)` 的顺序，
给第 d 个源域的标签张量生成域 ID=d，允许各源域 batch 长度不同。
`batch_size=32` 是**每个源域 32 张**；PACS 留一域测试时，三个源域合并后
通常 B=96。目标测试域图像、标签及统计量均不参与本损失。

对锚点 i：

| 样本 j | 角色 | 进入分母 |
| --- | --- | --- |
| 同类、不同源域、非自身，且入选 | 正样本 | 是 |
| 不同类别，任意源域 | 负样本 | 是 |
| 同类、同源域 | 中性样本 | 否 |
| 同类、跨域，但未入选 Top-K | 中性样本 | 否 |
| 自身 | 排除 | 否 |

Top-K=0 保留全部跨域同类候选；K>0 时，在每个锚点的全部合格候选内按
冻结 V 的余弦相似度取前 min(K, 候选数) 个，不要求每个源域各取 K 个。
相同分数按 PyTorch `topk` 的并列选择规则处理；非并列选择及全候选模式
对同步打乱样本、标签、域 ID 保持不变。无正样本锚点跳过；全批无有效锚点
时返回保留 E 计算图的零损失，不用同域同类补充正样本。

## 公式

V 先 detach，再在 float32 下做 L2 normalize：

\[
s_{ij}=\langle\widehat V_i,\widehat V_j\rangle,\qquad
P_i=\{j:y_i=y_j,\ d_i\ne d_j,\ i\ne j\}.
\]

下式 P_i 表示经过可选 Top-K 后的集合，只在该集合内归一化：

\[
q_{ij}=\frac{\exp(s_{ij}/\tau_T)}{\sum_{p\in P_i}\exp(s_{ip}/\tau_T)},
\qquad
w_{ij}=\frac{1-\rho}{|P_i|}+\rho q_{ij}.
\]

权重均非负，每个有效锚点之和为 1，选择和权重均不反传。
令 \(N_i=\{j:y_i\ne y_j\}\)，对比损失为

\[
\mathcal L_{con}=-\frac1{|A|}\sum_{i\in A}\sum_{j\in P_i}w_{ij}
\log\frac{\exp(\langle\widehat E_i,\widehat E_j\rangle/\tau)}
{\sum_{a\in P_i\cup N_i}\exp(\langle\widehat E_i,\widehat E_a\rangle/\tau)},
\quad A=\{i:|P_i|>0\}.
\]

在 float32 下关闭 autocast 计算归一化、矩阵乘法与 `logsumexp`，只对有效行
计算分母，避免全遮挡行及 `0 * -inf` 造成 NaN。没有额外的正样本数量系数。

新配置使用 `cipt_objective=env_kl`，总损失严格为：

\[
\mathcal L=\underbrace{CE(p_E,y)}_{E\_CE}
+\underbrace{KL(U\Vert p_S)}_{S\_uniform}
+\lambda_{con}\mathcal L_{con}.
\]

E_CE 只计一次；IND、TDA/SIGM 环境融合、MCC/RD、融合 CE、预训练对比和
表示正则均不加入此配置的总损失。推理只用 E 与可学习类别文本的相似度。
SWAD 的训练、验证与模型平均流程不变。

## 参数和旧模式兼容

| 参数 | 新配置默认值 | 含义 |
| --- | --- | --- |
| `cipt_contrastive_type` | `clip_guided_cross_domain` | 新模式需显式启用 |
| `cipt_contrastive_temperature` | 0.1 | E 对比温度 tau，沿用原参数 |
| `cipt_clip_teacher_temperature` | 0.2 | 教师温度 tau_T，越小权重越集中 |
| `cipt_clip_teacher_mix` | 0.5 | rho，取值 [0,1]，0 为等权 |
| `cipt_clip_teacher_topk` | 0 | 每个锚点保留的跨域同类数；0 表示全部 |
| `cipt_causal_contrastive_weight` | 1.0 | 实际对比损失系数 lambda_con |
| `cipt_contrastive_warmup_steps` | 0 | 新配置从首步使用全部系数 |
| `cipt_use_contrastive` | true | false 时完全跳过对比损失 |

`cross_domain_supcon` 使用同一候选和可选 Top-K，强制 rho=0。
`clip_guided_cross_domain` 使用配置的 rho；rho=0、Top-K=0 时与前者数值和
梯度一致。旧 `supcon` 仍使用所有非自身同类正样本、其余非自身样本分母；
旧 `neighbor_retention` 仍对该 SupCon 的锚点按邻域退化加权，不更改候选。
原 warmup 行为保留；新 YAML 明确覆盖基础 YAML 的 500 为 0。
`_causal_contrastive_loss(visual, causal, labels, label_batches)` 仍返回
`(loss, valid_anchor_fraction, effective_weight, stats)` 四元组。

## 日志

| 字段 | 统计定义 |
| --- | --- |
| `cg_contrastive_loss` | 新模式的未乘系数损失；与 `dccl_contrastive_loss` 相等 |
| `cg_valid_anchor_fraction` | 有至少一个选中正样本的锚点占全批比例 |
| `cg_cross_domain_pos_count` | 有效锚点的选中正样本数均值 |
| `cg_teacher_pos_cos_mean` | 所有选中正样本对的 V 余弦均值 |
| `cg_teacher_weight_entropy` | 有效锚点权重分布熵的均值，使用自然对数 |
| `cg_teacher_weight_max` | 全批有效正样本权重最大值 |
| `cg_active` | 新模式启用、有效系数>0 且存在有效锚点时为 1 |
| `contrastive_weight_eff` | 本步实际加入总损失的系数；禁用时为 0 |

`cg_*` 只统计 detach 后的值。旧模式和禁用模式的 `cg_*` 为 0；旧 `nbr_*`、
`dccl_contrastive_loss` 和 `contrastive_valid_anchor_fraction` 字段继续保留。
启动日志打印组件开关、E/S/IND/环境系数、推理方式、对比类型、教师参数、
配置系数、初始有效系数和 warmup 步数。

## CPU 与 CUDA 验证

在 `DCCL/DCCL` 执行，使用已有的训练环境（PyTorch、NumPy、PyYAML、sconf）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_cipt_*.py' -v
```

新测试覆盖独立 CE 公式数值/梯度、正负/中性掩码、Top-K、rho=0 一致性、权重
归一化、异常批次、部分无效锚点、教师 detach、E 梯度、样本置换、CPU autocast、
实际构造器和 E/S 更新、仅一次 E_CE/CLIP 编码、关闭对比后的相同参数更新、
E-only 推理及 YAML/CLI 优先级。原 SupCon、neighbor_retention、环境约束测试保留。
构造器测试只替换 CLIP 权重/分词，使用真实 PromptLearner、E/S、优化器和更新代码。

CPU 通过后，CUDA 合成数据短程测试（五步、3x32，显式要求可用 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_cipt_clip_guided_contrastive.py --cuda-smoke
```

该测试仍使用微型冻结 CLIP，另检查 D=512 的 float16-autocast 损失与梯度；
没有 GPU 时明确失败，不将跳过当作通过。真实 CLIP+PACS 十步 smoke 命令：

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py pacs_cg_cuda_smoke config_cipt_clip_guided.yaml \
  --dataset PACS --algorithm CIPTDCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --output_root train_output/smoke/clip_guided \
  --deterministic --trial_seed 0 --seed 0 --aug 0 \
  --steps 10 --checkpoint_freq 5 --test_envs 0 --swad null
```

这条十步命令关闭 SWAD，仅验证真实训练可执行；完整实验均开启 LossValley。
本次交付环境与实际执行状态见 `clip_guided_validation.md`，不报告未运行的
CUDA/PACS 测试为通过，也不将短程结果作为泛化准确率。

## PACS 消融命令

在 `DCCL/DCCL` 依次执行。共用一个 Bash 函数以保证除指定消融因素外的设置
一致；按服务器实际位置修改数据和权重路径。省略 `--test_envs` 会按训练脚本
依次完成 PACS 的四个留一域任务。每域 batch=32，lr=5e-5，seed/trial_seed=0，
aug=0，SWAD=LossValley，E=1、S=1，其余辅助项全部关闭。

```bash
run_pacs_cg() {
  local run_name="$1"
  shift
  CUDA_VISIBLE_DEVICES=0 python train_all.py "$run_name" config_cipt_clip_guided.yaml \
    --dataset PACS --algorithm CIPTDCCL \
    --data_dir /home/hooasia/lgg/data/repro_dccl_data \
    --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
    --output_root train_output/ablation/clip_guided_crossdomain \
    --deterministic --trial_seed 0 --seed 0 --checkpoint_freq 100 --aug 0 \
    --lr 5e-5 --batch_size 32 --swad LossValley "$@"
}

# 1. E_CE + S_uniform，不开对比
run_pacs_cg pacs_es_no_con --cipt_use_contrastive false

# 2. 旧普通 SupCon，全部同类正样本
run_pacs_cg pacs_es_supcon --cipt_contrastive_type supcon

# 3. 跨域同类，全部候选等权
run_pacs_cg pacs_es_crossdomain --cipt_contrastive_type cross_domain_supcon

# 4. 相同候选，CLIP 引导权重，rho=0.5，全部候选
run_pacs_cg pacs_es_clip_guided --cipt_clip_teacher_mix 0.5 --cipt_clip_teacher_topk 0

# 5. 引导权重强度消融（与第 4 组比较）
run_pacs_cg pacs_es_cg_rho1 --cipt_clip_teacher_mix 1.0

# 6. 候选数消融（与第 4 组比较）
run_pacs_cg pacs_es_cg_k2 --cipt_clip_teacher_topk 2

# 可选：验证 rho=0 与第 3 组一致
run_pacs_cg pacs_es_cg_rho0 --cipt_clip_teacher_mix 0.0 --cipt_clip_teacher_topk 0

# 可选：保留旧邻域退化方式作为对照，固定与上述相同的 lambda=1 和 tau=0.1
run_pacs_cg pacs_es_neighbor --cipt_contrastive_type neighbor_retention \
  --cipt_neighbor_k 10 --cipt_neighbor_alpha 4.0
```

上述系数是控制变量的起始配置，不宣称最优或保证提升。若覆盖 YAML，日志中
`contrastive_weight_eff` 是判断实际对比强度的依据。
