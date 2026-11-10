# DICM：以差分干预分解替换 DCCL 的 GT

基础：远程 main 的 ea200b976dbd348ab8844e5d4377f90e24134c87。
分支：feature/dicm-gt-replacement-20261110。

本实现新增独立的 `DICM(Algorithm)`，不继承会构造 GT 的 `DCCL.__init__`。
原始 `algorithms.py::DCCL`、数据增强、分割、采样与 SWAD LossValley 保留。
没有加入残差门控、可靠性加权、额外 SupCon 或正交损失。

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
```

两组响应的有效权重会重新归一，只有一组可用时该组权重为 1；两组都无效时
B 没有梯度，Adam 也不会依靠已有动量移动 B。把某组权重设为 0 可做单响应对照。
占比是覆盖目标的配比，不是特征混合比例，也不是子空间维数的固定分配。

所有学生侧任务端点均使用 E；参考侧 z0 冻结，但共享 pre_proj_head 在两侧都有梯度。
任务损失不更新 B；ISR 不直接更新 fθ。B 仅经 ISR 的 QR 路径更新。
单纯缩放 ISR 系数会被 Adam 的归一化大体抵消，不应当作删除强度旋钮。

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
保留原始“目标域循环外设置 seed”的规则；若另改逐目标重置 seed，须匹配重跑 A0。

配置 / sconf 可覆盖 dicm_* 参数，例如：

```bash
python train_all.py review configs/dicm_pacs.yaml --algorithm DICM --show \
  --dicm_aug_weight 0.5 --dicm_dom_weight 0.5 --dicm_basis_lr 0.0001
```

DICM 专门修复了已知 argparse 核心选项未回填的问题：显式 `--l`、`--l_layer`、
`--t`、`--t_pre`、`--n_layer` 在 YAML 后生效。未显式传入的 argparse 默认值不会
覆盖数据集 / YAML 配置。原始 DCCL 的配置解析行为不变。
GT 的 l_d 必须为 0；DICM 对非零 l_d 报错，不会偷偷保留 GT。

## 精确消融

通过 `--dicm_ablation A1/A2/A3/A4` 选择：

| 模式 | 投影 / 原型 / ISR | CE | SupCon | PreCL |
|---|---|---|---|---|
| A1 | 不构造 | z | z、z⁺ | z、z0 |
| A2 | 构造 | E | 关闭 | 关闭 |
| A3 | 构造 | E | E、E⁺ | 关闭 |
| A4，默认 | 构造 | E | E、E⁺ | E、z0 |

所有行均不构造 GT。A0 仍使用原来的 `--algorithm DCCL`。
A2 / A3 强制关闭对应项，显式传入矛盾的 CLI 权重会报错。
A1 / A4 保留可配置的任务权重；若手动关掉某项，结果不再对应表中的标准消融定义。
A5 门控与 A6 可靠性对比未实现，避免首版同时引入多个新机制。

支持原始 two_ce 与 sample_d：第二视图 CE 和额外正样本也先经过相同投影。
首版对 TN、CutMix aug、mix、re_w、pos_mask 的非零 / True 设置明确报错，
避免混合类别原型或静默改变特殊 mask。默认路径与已复现基线一致。

## 初始化、优化与平均

- 编码器 lr=5e-5，分类器 5e-4，两个投影头 5e-6，保留原始比例。
- B 默认继承编码器 lr（配置中显式 5e-5），weight_decay=0。
- B 用私有 CPU generator 正交初始化，不推进公共 RNG；seed=-1 使用当前全局初始 seed。
- 原 DCCL 在两个头初始化之间生成 shape probe。新模型只保留相同的 CPU 随机数消耗，
  不执行 probe forward、不构造 GT。默认 dropout=0 路径的公共初始化与 A1 等价性有单元验证。
- SWAD 只导出 fθ→projection→classifier；平均 raw B 后重新 QR，固定 beta 也保存在前向状态。
  不平均 / 更新原型，不导出参考编码器或训练用头。
- 普通预测、predict_embed、导出预测、状态恢复共用同一个投影实现。
- 参数平均不等于投影矩阵平均；需要结合 raw B 条件数和平均模型的源域诊断判断稳定性。

## 评估与日志

DICM 的 IID 仅依据 source train_out，平局保持第一个最大值。SWAD 使用原始源域
验证 LossValley。DICM 不计算 / 返回 oracle 选模结果。

为兼容原复现，iid / SWAD / last 仍表示 test_in；另有明确的
`iid (test_out)`、`SWAD (test_out)`、`last (test_out)`。
它们都是选定模型的报告字段，不能再反向用于调参。
每目标额外保存 selection_te_*.json，包含源域 split hash、选中步、源域准则、
SWAD 范围及报告指标。原始 DCCL 的历史 summary 保留不变。

新增训练指标包含两类覆盖 / 有效数量 / 实际配比、原型支持与年龄、raw B 最小奇异值
和条件数、Q 正交误差、E 改变量、绝对增强距离、敏感能量和类间散度删除比例、
各参数梯度范数与实际学习率。诊断在 optimizer.step 之前计算，避免旧 z 配新 B。
`ce_loss` 保留原第一视图口径；`loss_cls` 是考虑 two_ce 后实际参与优化的 CE。

## 验证与证据边界

```bash
python -m pytest tests/test_dicm.py -q
```

测试使用小张量 / 模拟编码器，以及随机权重 ResNet-50 的推理检查，不下载预训练
权重、不读取 PACS、不运行真实训练实验。覆盖：无 GT 结构、任务端点、梯度隔离、
配比与空响应、EMA 支持与局部域编号、A1 基线等价性、状态保存、SWAD、CLI、
源域选模及 trainer 集成。

本分支实现了待验证方法，没有新 PACS 性能结果。Q 是差分响应子空间估计，
不能仅凭代码或覆盖率声称已识别真实因果因素，也不能保证超越原始 DCCL。
