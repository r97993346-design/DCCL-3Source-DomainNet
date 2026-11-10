# DICM：以差分干预分解替换 DCCL 的 GT

基础：远程 main 的 ea200b976dbd348ab8844e5d4377f90e24134c87。
当前分支：feature/dicm-per-target-seed-a0-20261110。
基于 feature/dicm-gt-replacement-20261110 的 921493d；父分支保留旧训练入口。

本实现新增独立的 `DICM(Algorithm)`，不继承会构造 GT 的 `DCCL.__init__`。
原始 `algorithms.py::DCCL`、数据增强、分割、采样与 SWAD LossValley 保留。
默认 A4 删除 GT；A5 是完整 A4 加回 GT 的补充对照。没有加入残差门控、
可靠性加权、额外 SupCon 或正交损失。

## 前向与损失

学生 fθ 可训练；参考 f₀ 冻结并始终 eval。两者处理完全相同的 x、x⁺。
x 和 x⁺ 沿用 DCCL 的两个独立随机增强视图，并非一个固定的未增强原图。

```
delta_aug = stopgrad((z_aug-z) - (z0_aug-z0))
residual  = stopgrad(z-z0)
mu_cd     = EMA(mean(residual | class=c, source=d))
delta_dom = mu_cd - mean_d(mu_cd)
Q         = qr(B).Q
S         = (z @ stopgrad(Q)) @ stopgrad(Q).T
E         = z - beta*S
```

默认 beta=1，得到 z=E+S、E 与 S 正交。beta=0 精确返回原输入；
0<beta<1 只是软抑制，不能声称删去了全部敏感信息。

有效响应先按最小范数过滤，再逐个单位化；增强 / 域响应分别取平均覆盖误差：

```
L_ISR = 0.25 * mean(||u_aug - Proj_Q(u_aug)||²)
      + 0.75 * mean(||u_dom - Proj_Q(u_dom)||²)

L_A4 = CE(classifier(E), y)
     + l * SupCon(normalize(proj_head(E)), normalize(proj_head(E_aug)), y)
     + l_layer * PreCL(normalize(pre_proj_head(E)), normalize(pre_proj_head(z0)), y)
     + 0.05 * L_ISR

L_A5 = L_A4 + dicm_gt_weight * L_GT
```

两组响应的有效权重会重新归一，只有一组可用时该组权重为 1；两组都无效时
B 没有梯度，Adam 也不会依靠已有动量移动 B。把某组权重设为 0 可做单响应对照。
占比是覆盖目标的配比，不是特征混合比例，也不是子空间维数的固定分配。

所有学生侧任务端点均使用 E；参考侧 z0 冻结，但共享 pre_proj_head 在两侧都有梯度。
任务损失不更新 B；ISR 不直接更新 fθ。B 仅经 ISR 的 QR 路径更新。
单纯缩放 ISR 系数会被 Adam 的归一化大体抵消，不应当作删除强度旋钮。

## A5：完整方案加 GT

A5 的 CE / SupCon / PreCL 与 A4 完全相同，学生端均使用 E。GT 复用
`domainbed/algorithms/algorithms.py::MeanEncoder`（恒等）和 `VarianceEncoder`
（通道方差），约束骨干的原始中间层，不能把这一项写成 GT(E)。

```
L_GT = sum_k mean((h_theta_k(x_aug) - stopgrad(h_0_k(x)))² / v_k + log(v_k)) / 2
```

这里显式保留已复现 main 的实际视图配对。原始 `DCCL.update` 首次取得的
`inter_feats` 是 `networks.ResNet.forward` 的共享 hook 列表；第二次学生前向
会清空并填入 x_aug 的特征，因此原实际 GT 是“学生第二视图 / 参考第一视图”。
新实现使用 tuple 固定两边容器，避免参考增强前向和 sample_d 额外视图覆盖。
这是对现有行为的保留，并非把同视图 GT 悄悄改成跨视图。若研究同视图 GT，
应另列实验，不能与当前 A5 混为一组。

GT 额外更新 fθ 和方差参数；参考模型仍冻结，B 仍只受 ISR 更新。
GT 不直接更新分类器或两个投影头，也不作为推理输入。GT 可能间接改变后续
差分观测和 B 的学习轨迹，因此 A5 不能描述成已删除 GT 的替换方法。
主论文的模块替换仍由 A4 对比 A0 / A1 验证；A5 检验 GT 的补充作用。

## 原型支持与源域编号

- EMA 动量 0.95；每类每域至少累计 8 个有效观测、3 次更新。
- 默认 `dicm_min_valid_domains=0` 表示该类需要全部源域有效；PACS 配置显式为 3。
- 不足支持或零响应不会被当成真实域方向。
- 记录 count、update_count、last_update_step 和年龄；没有宣称解决历史坐标漂移。
- 使用传入源域 minibatch 列表的局部编号 0..Dsource-1，不使用可能跳号的全局 d。
  trainer 打印局部到全局编号的映射，held-out 不会进入更新。
- predict / predict_embed / SWAD 前向均不写原型，不需要标签、域标签、参考模型或增强视图。

## 使用方式

从仓库的 `DCCL/DCCL` 目录运行。需要 PyTorch >=2.0；验证环境为
PyTorch 2.5.1、torchvision 0.20.1。保留现有已能运行 DCCL 的其余依赖。

先只审阅配置，此命令不加载数据、不创建训练输出，也不要求 CUDA：

```bash
python train_all.py dicm_review configs/dicm_pacs.yaml \
  --algorithm DICM --dataset PACS --show
```

以下是后续手动训练示例，本次实现过程没有执行此命令：

```bash
python -u train_all.py pacs_dicm_A4 configs/dicm_pacs.yaml \
  --algorithm DICM --dataset PACS --model resnet50 \
  --data_dir /path/to/data --steps 5000 --checkpoint_freq 100 \
  --seed 0 --trial_seed 0 --deterministic
```

不传 test_envs 时沿用原程序的 0→1→2→3 顺序。没有自动启动训练队列。
原程序将请求 5000 步换算为循环 step=0..5000，最多 5001 次更新；
step=0 在第一次优化后评估，原 SWAD 仍可能提前停止。
本分支改为逐目标域重置 seed；需要使用下述 A0 重新建立匹配基线。

## 逐目标域独立复现与 A0

`domainbed/trainer.py::train` 在每次任务的最开始调用
`domainbed/lib/reproducibility.py::seed_everything`，早于数据集、采样器、
DataLoader 与模型创建。Python、NumPy、PyTorch CPU / CUDA 均以 `args.seed`
重新开始；cuDNN deterministic / benchmark 沿用 `--deterministic` 的控制。
目标域编号和之前的任务长度不参与种子计算。一组内三源共同更新，正常推进 RNG。

因此 seed=0 时，PACS 的 A / C / P / S 四组都会在各自开始前重置为 0。
每组仍重新创建模型、优化器、B、EMA 和 SWAD，不加载上一组参数。
`trial_seed` 与原来的划分算法不变；每次 selection JSON 记录 seed、trial_seed、
seed_policy=per_target、deterministic、算法、消融编号和源域 split hash。

A0 直接使用未改动的 `domainbed/algorithms/algorithms.py::DCCL`，保留 CE(z)、
SupCon(z,z_aug)、PreCL(z,z0) 和原始 GT；没有 DICM、B 或原型模块。
新增 `configs/dccl_pacs_a0.yaml`，与 A4 共用 lr=5e-5、batch=32、任务权重和温度，
GT 使用原 PACS 有效权重 0.01。不要把 A0 配置与 dicm_pacs.yaml 叠加。
本分支 A0 / A1–A5 都用源域验证选 IID 与 SWAD，均不计算 Oracle 选模汇总。
这保持原 DCCL 的模型 / 损失，同时显式更换随机种子粒度和报告协议；旧 A0 日志
继续保留，不能充当新规则下已复现的基线。

先只检查 A0 配置：

```bash
python train_all.py a0_review configs/dccl_pacs_a0.yaml \
  --algorithm DCCL --dataset PACS --show
```

配套串行脚本会使用同一 seed、trial_seed、步数、评估频率和骨干依次准备 A0–A5。
先审阅全部六组（不会创建训练输出或加载数据）：

```bash
STEPS=300 CHECKPOINT_FREQ=100 bash scripts/run_pacs_a0_a5.sh --show
```

后续手动启动的示例，本次实现没有执行：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 TRIAL_SEED=0 STEPS=300 CHECKPOINT_FREQ=100 \
  bash scripts/run_pacs_a0_a5.sh --data_dir /path/to/data
```

不传 test_envs 时，每组运行四个目标域；可传 `--test_envs 3` 只补跑 Sketch。
`PYTHON=/path/to/python` 可指定解释器。新分支默认就按目标域重置，无需额外开关。
脚本仅在手动调用时启动；实现与检查过程不运行真实训练。

相同目标域仅改变总步数时，在学习率、checkpoint_freq、数据与环境相同的条件下，
公共更新前缀应一致。改变评估频率可能改变 RNG 消耗与 SWAD 行为；固定种子本身
不保证 GPU 所有算子逐位确定，也不保证跨软件 / 硬件一致或短长跑的最终 SWAD 一致。

配置 / sconf 可覆盖 dicm_* 参数，例如：

```bash
python train_all.py review configs/dicm_pacs.yaml --algorithm DICM --show \
  --dicm_aug_weight 0.5 --dicm_dom_weight 0.5 --dicm_basis_lr 0.0001
```

DICM 专门修复了已知 argparse 核心选项未回填的问题：显式 `--l`、`--l_layer`、
`--t`、`--t_pre`、`--n_layer` 在 YAML 后生效。未显式传入的 argparse 默认值不会
覆盖数据集 / YAML 配置。原始 DCCL 的配置解析行为不变。
A1–A4 的 l_d 必须为 0；非零值会报错。只有显式选择 A5 才恢复 GT。
A5 使用 `dicm_gt_weight` 配置 GT，日志中的 `l_d` 同步为有效值，也接受显式
`--l_d` 覆盖。默认继承原 DCCL 的数据集 / 骨干权重：PACS=0.01、VLCS=0.05、
TerraIncognita=0.05、OfficeHome ResNet=0.05 / CLIP=0.1、DomainNet=0.05。
PACS 不采用 argparse 中未生效的默认 0.05。A5 要求 GT、SupCon、PreCL 均启用。

只检查 A5 配置，不启动训练：

```bash
python train_all.py a5_review configs/dicm_pacs.yaml configs/dicm_pacs_a5.yaml \
  --algorithm DICM --dataset PACS --show
```

也可在基础配置后用 `--dicm_ablation A5` 替代 A5 overlay。调节 GT 时使用
`--dicm_gt_weight 0.02`；增强 / 域响应配比仍为 0.25 / 0.75，其他配置不变。

## 精确消融

通过 `--dicm_ablation A1/A2/A3/A4/A5` 选择：

| 模式 | 投影 / 原型 / ISR | CE | SupCon | PreCL | GT |
|---|---|---|---|---|---|
| A0，algorithm=DCCL | 原 DCCL，无 DICM | z | z、z⁺ | z、z0 | 原始中间层 |
| A1 | 不构造 | z | z、z⁺ | z、z0 | 无 |
| A2 | 构造 | E | 关闭 | 关闭 | 无 |
| A3 | 构造 | E | E、E⁺ | 关闭 | 无 |
| A4，默认 | 构造 | E | E、E⁺ | E、z0 | 无 |
| A5 | 构造 | E | E、E⁺ | E、z0 | 原始中间层 |

A1–A4 不构造 GT，A0 / A5 构造。A0 使用 `--algorithm DCCL` 配合 A0 专用配置，
不属于 `--dicm_ablation` 的可选值。
A2 / A3 强制关闭对应项，显式传入矛盾的 CLI 权重会报错。
A1 / A4 保留可配置的任务权重；若手动关掉某项，结果不再对应表中的标准消融定义。
A5 已重新定义为完整方案加 GT，不再是门控实验。门控和 A6 可靠性对比尚未实现。

支持原始 two_ce 与 sample_d：第二视图 CE 和额外正样本也先经过相同投影。
首版对 TN、CutMix aug、mix、re_w、pos_mask 的非零 / True 设置明确报错，
避免混合类别原型或静默改变特殊 mask。默认路径与已复现基线一致。

## 初始化、优化与平均

- 编码器 lr=5e-5，分类器 5e-4，两个投影头 5e-6，保留原始比例。
- B 默认继承编码器 lr（配置中显式 5e-5），weight_decay=0。
- B 用私有 CPU generator 正交初始化，不推进公共 RNG；seed=-1 使用当前全局初始 seed。
- A5 的 GT 参数沿用原始 10 倍编码器学习率（PACS 为 5e-4）、weight_decay=0。
- 原 DCCL 在两个头初始化之间生成 shape probe。A1–A4 只保留相同的 CPU 随机数消耗，
  不执行 probe forward、不构造 GT；A5 执行 probe 以构造同样的 GT。
  默认 dropout=0 路径的公共初始化、RNG 和 beta=0 基线更新等价性有单元验证。
- SWAD 只导出 fθ→projection→classifier；平均 raw B 后重新 QR，固定 beta 也保存在前向状态。
  不平均 / 更新原型，不导出参考编码器、训练用头或 A5 的 GT 模块。
- 普通预测、predict_embed、导出预测、状态恢复共用同一个投影实现。
- 参数平均不等于投影矩阵平均；需要结合 raw B 条件数和平均模型的源域诊断判断稳定性。

## 评估与日志

A0 / DICM 的 IID 仅依据 source train_out，平局保持第一个最大值。SWAD 使用原始源域
验证 LossValley。两者均不计算 / 返回 oracle 选模结果。

为兼容原复现，iid / SWAD / last 仍表示 test_in；另有明确的
`iid (test_out)`、`SWAD (test_out)`、`last (test_out)`。
它们都是选定模型的报告字段，不能再反向用于调参。
每目标额外保存 selection_te_*.json，包含源域 split hash、选中步、源域准则、
SWAD 范围及报告指标。原始 DCCL 的历史 summary 留在父分支，本分支不沿用 Oracle 行。

新增训练指标包含两类覆盖 / 有效数量 / 实际配比、原型支持与年龄、raw B 最小奇异值
和条件数、Q 正交误差、E 改变量、绝对增强距离、敏感能量和类间散度删除比例、
各参数梯度范数与实际学习率。诊断在 optimizer.step 之前计算，避免旧 z 配新 B。
`ce_loss` 保留原第一视图口径；`loss_cls` 是考虑 two_ce 后实际参与优化的 CE。
GT 另记录 `gt_enabled`、`gt_weight`、`reg_loss`、`weighted_gt_loss`、
`gt_layer_count`、`gt_grad_norm`；不能只根据含 log 方差的 GT 原始数值判断其作用。

## 验证与证据边界

```bash
python -m pytest tests/test_dicm.py -q
```

测试使用小张量 / 模拟编码器，以及随机权重 ResNet-50 的推理检查，不下载预训练
权重、不读取 PACS、不运行真实训练实验。覆盖：A1–A4 无 GT、A5 恢复 GT、任务端点、梯度隔离、
配比与空响应、EMA 支持与局部域编号、A1 基线等价性、状态保存、SWAD、CLI、
源域选模及 A0 / A4 / A5 trainer 集成。A5 还验证额外视图不覆盖 GT 快照、
beta=0 且 ISR 关闭时与原 DCCL 的 GT 损失及参数更新等价、公共初始化一致、
GT 权重解析和方差优化器状态恢复。

新增 A0–A5 的四域合成数据测试：比较同一目标独立运行、前置不同域 / 不同步数后
运行时的数据集随机起点、初始模型状态、批次、每步损失与梯度诊断、选模结果；
也比较短长预算的公共训练前缀。这些是 CPU 合约验证，不是 PACS 性能实验或
GPU 逐位复现的证据。

本分支实现了待验证方法，没有新 PACS 性能结果。Q 是差分响应子空间估计，
不能仅凭代码或覆盖率声称已识别真实因果因素，也不能保证超越原始 DCCL。
