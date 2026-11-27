# 融合特征与可学习类别文本的分类监督

在 `cipt_objective=env_kl` 下新增独立参数 `cipt_fused_cls_weight`。
环境文本用于融合，类别文本仍由原有 PromptLearner 学习；CLIP 编码器保持冻结。

## 计算方式

```math
Z_{ik}=Fusion(E_i,t_k^{env}),\qquad
p_{ik}=softmax(s\,cos(Z_{ik},T^{learn})).

L_{fused}=\frac{1}{BK}\sum_i\sum_k CE(p_{ik},y_i).

L=\lambda_E L_E+\lambda_{fused}L_{fused}+\lambda_S L_S
  +\gamma L_{IND}+\lambda_{con}(t)L_{con}+\lambda_{env}L_{MCC}.
```

每个环境分别计算标签 CE，然后平均；不先平均融合特征或 logits。
融合 CE 和 MCC 共用一次环境选择、融合与类别打分。选择仍不读取真实标签；
真实标签只在选择完成后用于 CE。梯度经相似度、融合模块回传到 E 适配器和
可学习类别文本。支持已有 TDA 与 SIGM，使用 b5a/b5c 类别无关干预库。

## 独立开关

以下表格假设 `cipt_use_tda=true`：

| `cipt_fused_cls_weight` | `cipt_env_kl_weight` | 环境训练目标 |
| --- | --- | --- |
| 0 | 0 | 跳过环境选择与融合 |
| >0 | 0 | 融合后标签 CE，可使用 K=1 |
| 0 | >0 | 原有 MCC，要求 K>=2 |
| >0 | >0 | 融合后 CE + MCC，要求 K>=2 |

`cipt_use_tda=false` 是环境训练总开关，同时关闭融合 CE 与 MCC。
`cipt_fusion_mode=tda/sigm` 选择融合算子，`cipt_selector_mode` 选择文本策略。
使用 `all` 时 K 指整个实际文本库的大小，而非 `cipt_k`。

原有配置与 registry 将融合 CE 权重默认设为 0，维持旧实验目标。
新增的 `config_cipt_fused_ce.yaml` 是叠加配置，需放在现有实验配置之后：

```bash
python train_all.py RUN_NAME config_cipt_sigm_mcc.yaml config_cipt_fused_ce.yaml [原有运行参数]
```

叠加配置开启环境训练，将 E CE、融合 CE 权重分别设为 1，MCC 设为 0。
融合方式、文本选择、S/IND/对比学习和推理方式继承原实验配置。
CLI 优先级更高；如果原命令含 `--cipt_use_tda false`，需要改成 `true`。

也可以在原有运行命令中直接设置：

```bash
--cipt_use_tda true --cipt_fused_cls_weight 1 --cipt_env_kl_weight 0
```

保留 MCC 时将 `cipt_env_kl_weight` 设为希望比较的正权重。
只保留融合后的分类监督时，可额外设置 `--cipt_e_cls_weight 0`；
S/IND/对比学习仍由其各自开关控制。

## 推理与诊断

`e_only` 推理仍直接用 E 分类，叠加配置保留 E CE 以监督这条路径。
`mean_logits`、`mean_probs` 仍聚合融合后预测；只训练融合分类时可显式选择
这两种推理方式。推理模式和训练中的环境总开关独立。

新增日志：`fused_cls_loss`、`fused_cls_w`、`fused_cls_term`、`fused_cls_active`。
其中权重日志记录有效权重，总开关关闭时为 0；融合分类项计入 `cipt_base_loss`，
并且只加一次。原有 `cipt_cls_loss` 保留为 E CE 的兼容诊断。
`env_active` 表示融合 CE 或 MCC 至少一项启用，`env_kl_active` 单独表示 MCC 启用。
融合仅用于可选推理时，不会使训练日志的 `env_active` 变为 1。

CPU 测试覆盖两种融合算子的 CE 梯度、冻结编码器、逐环境 CE 数值、
CE/MCC 共用融合、独立开关、K=1、参数检查、配置覆盖和 E-only 推理：

```bash
python -m unittest discover -s tests -p 'test_cipt_fused_ce.py' -v
```

这些检查不代表 PACS/VLCS 精度，实际性能需要重新训练验证。
