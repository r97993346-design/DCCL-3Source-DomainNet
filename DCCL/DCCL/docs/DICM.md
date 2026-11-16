# DICM：以差分干预分解替换 DCCL 的 GT

基础：远程 main 的 ea200b976dbd348ab8844e5d4377f90e24134c87。
当前分支：feature/dicm-per-target-seed-a0-20261110。
训练协议沿用原始 DCCL。

本实现新增独立的 `DICM(Algorithm)`，不继承会构造 GT 的 `DCCL.__init__`。
原始 `algorithms.py::DCCL`、数据增强、分割、采样与 SWAD LossValley 保留。
默认 A4 删除 GT，A5 是完整 A4 加回 GT 的补充对照。两者默认使用等权 SupCon，
可选择启用跨域同类正样本对加权。

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
在完整 DCCL 上添加模块用 A5 对比 A0；A4 检验移除 GT 后的效果。

## 可选的跨域正样本对加权

用第一视图投影前的学生特征 `z` 计算同类、不同源域样本的差异
`delta_ij = z_i - z_j`。用当前的正交基 Q 衡量差异中的敏感能量：

```text
r_ij = clamp(1 - ||delta_ij @ Q||² / max(||delta_ij||², 1e-12), min_reliability, 1)
w_ij = 1 - gamma(step) * (1 - r_ij)
```

差异范数小于 1e-6 的样本对取权重 1。权重停止梯度。同域同类和同一样本的两视图
取权重 1。用同一张样本级权重矩阵加权现有两视图 SupCon 的正样本平均，每个锚点
除以自己的正样本权重之和。SupCon 的相似度仍在投影后的 E/E_aug 上计算，
softmax 分母仍使用所有非自身视图。CE、PreCL 和 ISR 的计算公式保持原样。

加载 `configs/dicm_reliable_contrast.yaml` 启用，默认在第 300 步开始渐增，
第 500 步达到 gamma=0.3，有效跨域正对权重介于 0.85 和 1 之间。
`--dicm_reliable_contrast_gamma_max 0` 使用原等权路径，可作为配对对照。
该选项要求启用子空间和 SupCon，使用 `sample_d=False`。训练日志记录有效
gamma、当前计算的跨域同类有向样本对数量及其平均权重；预热时数量为 0、均值为 1。

## 原型支持与源域编号

- EMA 动量 0.95；每类每域至少累计 8 个有效观测、3 次更新。
- 默认 `dicm_min_valid_domains=0` 表示该类需要全部源域有效；PACS 配置显式为 3。
- 不足支持或零响应不会被当成真实域方向。
- 记录 count、update_count、last_update_step 和年龄；没有宣称解决历史坐标漂移。
- 使用传入源域 minibatch 列表的局部编号 0..Dsource-1，不使用可能跳号的全局 d。
  trainer 打印局部到全局编号的映射，held-out 不会进入更新。
- predict / predict_embed / SWAD 前向均不写原型，不需要标签、域标签、参考模型或增强视图。

## 使用方式与训练协议

从仓库的 `DCCL/DCCL` 目录运行。A0 使用原 DCCL 默认配置，A5 继承相同的数据集超参数、任务损失和 GT 权重，只增加 DICM 投影与 ISR。

原始入口在每个进程启动时以 `--seed` 设置一次 Python、NumPy 和 PyTorch 随机状态，默认 0。目标域按顺序共享随机序列。每域仍重新创建数据加载器、模型和优化器。按作者公开示例做三次重复时，`--seed` 保持 0，分别使用 `--trial_seed 0/1/2`。该参数控制数据划分。

训练预算沿用 `floor(steps / checkpoint_freq) * checkpoint_freq + 1`，SWAD 可提前停止。评估与训练共享随机状态，加载器沿用原实现。可用 `--show` 预览实际配置。

```bash
python train_all.py pacs_A0 --dataset PACS --algorithm DCCL --deterministic --trial_seed 0 --checkpoint_freq 100 --show
python train_all.py pacs_A5 configs/dicm_pacs.yaml configs/dicm_pacs_a5.yaml --dataset PACS --algorithm DICM --deterministic --trial_seed 0 --checkpoint_freq 100 --show
```

实际训练时去掉 `--show` 并传入 `--data_dir`。VLCS A5 使用 `configs/dicm_vlcs.yaml` 并传入 `--dicm_ablation A5`。公共超参数沿用原入口的解析顺序，需要覆盖时对两组传入相同的 YAML。可通过 `--output_root` 指定独立结果目录。现有逐目标域重置种子的旧结果与本协议分别记录。

配套 PACS/VLCS 脚本每组启动一个独立进程，默认使用 seed=0、trial_seed=0、steps=5000、checkpoint_freq=100。可用 `TRIAL_SEED` 指定重复编号。脚本的 `--show` 仅预览配置。

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

A1–A4 不构造 GT，A0 / A5 构造。A0 使用 `--algorithm DCCL` 和原 DCCL 默认配置。
A2 / A3 强制关闭对应项，显式传入矛盾的 CLI 权重会报错。
A1 / A4 保留可配置的任务权重；若手动关掉某项，结果不再对应表中的标准消融定义。
A5 是完整方案加 GT。可靠性加权是独立选项，使用时沿用原有 A3/A4/A5 模式。

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

沿用原 DCCL 汇总。IID 按源域验证准确率选模，SWAD 按源域验证 LossValley 选平均区间，两项均报告目标域 test_in。Oracle 行按目标域 test_out 选 checkpoint，供对照原程序输出。模块比较采用源域选出的 IID 或 SWAD。

DICM 额外记录两类响应的覆盖率和有效数量、原型支持与年龄、基矩阵条件数、投影改变量、增强距离、类间散度投影比例和梯度范数。

## 验证

```bash
python -m pytest tests/test_dicm.py -q
```

测试涵盖 DICM 梯度、A5 与基线的身份投影等价性、SWAD 推理路径、配置预览，以及合成任务上的原训练协议。真实数据集的性能需在同协议下重新测量。
