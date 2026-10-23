# 原 CIPT：E、文本干预与同类域子簇的证据链

此流程使用本分支已有的 `CIPTDCCL + cipt_objective=legacy`，复用真实的
`CausalDecomposition` 和 `TextDiversityAugmentation`，关闭新增的对比、RD、MCC、熵选择目标。
它固定了本仓库的原 CIPT 复现路径；不等于已经复现论文的全部实验协议或报告精度。

## 1. 三个问题分别检验什么

| 问题 | 图 | 原始特征空间的证据 | 能支持的表述 |
|---|---|---|---|
| E 内同类样本仍形成域相关子簇 | 每个类别单独作图，颜色表示域，查看 E 面板 | 类内域 silhouette、同域近邻比例超过随机基线的量、同域/跨域余弦差 | 因果分解输出仍保留域相关结构 |
| 文本类别信息占主导，描述多样性贡献有限 | intervention_sensitivity.png；E→Z 配对面板 | 固定图像及模板只改类别词的位移，与固定类别词只改模板的位移；E/Z、LN(E)/Z 位移及类内邻域保持率 | 类别词敏感性较强、不同描述产生的额外变化较小；具体以结果为准 |
| Z 中同类域子簇仍存在 | 同一图的固定模板 Z_k 与 Z_mean80 面板 | 全部 80 个模板分别计算类内域指标，与 E、LN(E) 对照 | 文本融合后同类域结构仍存在，或其改善有限 |

**“类别词影响很大”“不同描述高度冗余”“E 与 Z 几乎相同”是不同命题。**
例如，文本可以显著平移/旋转特征并提高类别可分性，同时保留原来的类内域子簇。
这种结果应描述为“未充分改善类内跨域结构”，不能描述为“融合前后没有变化”。
指标可能否定上述假设，报告不会自动选择“无效”的结论。

## 2. 固定检查点内的五种特征

- `E`：L2(CLIP 图像特征) → 训练好的因果适配器 → L2。
- `LN(E)`：同一 TDA 中训练好的 LayerNorm 直接作用于 E，再 L2；这是零文本残差对照。
- `Z_constant`：E 与 `a photo of a {真实类别}.` 融合，再 L2。
- `Z_template_00`：所有图像使用同一个编号的 B5B 描述，类别词按真实类别填入，再 L2。
- `Z_mean80`：先对每个 Z_k 做 L2，平均全部 80 个，再 L2；只用来概括几何结构。

本实现是 80 次独立的单文本 token 注意力，每次只有一个 key/value。
在 eval 模式下注意力权重为 1，可写为 `Z_k = LN(E + g(p_k))`。
因此比较 E 和 Z 时要控制 LayerNorm；直接比较原始范数会混入归一化效应。
`LN(E)` 不是另行训练的 TDA-off 模型，也不是给 TDA 输入零向量文本。

另外，固定每个图像/模板，把类别词逐一替换为所有其他类别，测量 Z 的变化；
同时将类别词替换为 `subject`，作为辅助对照。前者不会把所有结果归因于某个中性词。
这些替换是训练后机制诊断，可能偏离训练输入分布，不能直接解释成泛化收益。

所有真实类别条件化 Z 都用于机制分析，**不能当作无需标签的测试表征来报告精度**。
legacy 推理仍按候选类别分别融合并平均 K 个 logits；不是 `Z_mean80` 的真标签条件化分类。

## 3. 先得到原 CIPT 检查点

在包含 `train_all.py` 的 `DCCL/DCCL` 目录、已有训练环境中执行。
将下面两个路径改为服务器实际路径。不要同时载入 RD/MCC 的 YAML。

```bash
export CIPT_DATA_DIR='/path/to/repro_dccl_data'
export CIPT_CLIP_PATH='/path/to/ViT-B-16.pt'

python train_all.py cipt_original_te0_seed0 config_cipt_original_visualization.yaml \
  --algorithm CIPTDCCL --dataset PACS \
  --data_dir "$CIPT_DATA_DIR" --cipt_clip_path "$CIPT_CLIP_PATH" \
  --test_envs 0 --seed 0 --trial_seed 0 \
  --steps 5001 --checkpoint_freq 100 --model_save 5000 \
  --output_root train_output/cipt_mechanism
```

PACS 环境顺序是 art_painting、cartoon、photo、sketch；上例目标域为 art_painting。
循环步号从 0 开始，`5001` 次更新可产生 `checkpoints/TE0_5000.pth`。
示例固定最后一个检查点用于描述性分析，不根据目标域图形或目标域精度挑选检查点。
若需要按源域验证集选择，预先固定规则并保存对应检查点，所有对照使用同一规则。

预设：`legacy + b5b + pure=true + DE/IND/TDA=true + contrastive=false`，
`beta=4, gamma=5, K=4, lr=5e-5`，总损失为 `L_cls(Z) + 4 L_de + 5 L_ind`。
SWAD 在这个预设中显式关闭，避免把普通步号快照当成 SWAD 模型。
若已有匹配检查点，可直接进入下一步；旧检查点还需确认训练时也使用 L2 视觉输入及 identity 初始化。
分析器会拒绝 RD/MCC、TDA 关闭、B5A 或额外对比训练的检查点，但无法从权重反推旧代码的全部训练历史。

## 4. 用同一批图片导出并画图

建议初次使用 PACS 全部 7 类、全部 4 域、每个类×域 25 张，共 700 张。
脚本默认该平衡抽样；数量不足会报错，不静默删去某个域或类别。
全部 80 个模板用于定量指标；每张图片在统计上仍只算一个样本。

```bash
export CIPT_CHECKPOINT='/path/to/checkpoints/TE0_5000.pth'

python analyze_cipt_mechanism.py \
  --dataset PACS --data-dir "$CIPT_DATA_DIR" \
  --checkpoint "$CIPT_CHECKPOINT" --clip-path "$CIPT_CLIP_PATH" \
  --samples-per-class-domain 25 --sample-seed 0 \
  --tsne-seeds 0 1 2 --perplexities 10 30 \
  --template-id 0 --batch-size 8 \
  --output-dir train_output/cipt_mechanism/analysis_te0_seed0
```

t-SNE 默认每个类别分别联合拟合 E、LN(E)、固定文本 Z、Z_mean80 和固定模板 Z。
一个类别图内的六个面板共享坐标；不同类别、不同 seed/perplexity 的坐标不可直接比较。
配对箭头只表达同一图片的对应关系，不能用其长度推断特征变化大小。
同一类别的两种特征不能分别拟合 t-SNE 后用坐标位移作比较。

输出目录必须为空，以防覆盖原始提取结果。需要只导出可加 `--extract-only`；
需要跳过 t-SNE 可加 `--no-tsne`。后续重复画图无需再次运行 CLIP：

```bash
python analyze_cipt_mechanism.py --report-only \
  --output-dir train_output/cipt_mechanism/analysis_te0_seed0 \
  --template-id 20 --tsne-seeds 0 1 2 --perplexities 10 30
```

建议预先固定展示模板 0、20、40、60，分别运行 `--report-only`。
定量报告每次均覆盖全部 80 个模板；不要只展示“最能证明结论”的模板或 t-SNE seed。
同一张图重复 80 个文本产生的 80 个点不能当作 80 张独立图片。

换训练 seed、检查点时用 `--sample-manifest .../samples.csv` 重用同一批图像。
若只选部分域，抽取和重放都显式传入相同 `--domains`。该抽样来自数据文件目录，
源域中可能包含训练图片，所以是描述性机制证据，不是独立测试精度。
研究结论应在训练 seed 0/1/2、各留一目标域上复核，不只依赖一个类别或一个检查点。

## 5. 查看哪些文件

| 输出 | 用途 |
|---|---|
| `tsne/class*_k*_seed*_p*.png/.pdf` | 每类六面板图：E、LN(E)、固定文本、80 模板均值、固定 Z_k、E→Z 配对 |
| `domain_metrics.png/.pdf` | 各类别的域 silhouette、同域近邻超额比例、同域/跨域余弦差 |
| `intervention_sensitivity.png/.pdf` | E→Z、LN(E)→Z、固定文本→Z、改模板、改类别词、中性词替换的位移分布 |
| `class_domain_metrics.csv` | E、对照及全部 80 个固定模板逐类的域结构指标 |
| `within_class_topology.csv` | E/Z、LN(E)/Z 逐类逐模板的近邻重合率和距离秩相关 |
| `per_image_metrics.csv` | 每张图像的模板冗余、类别词敏感性和融合位移 |
| `summary.json`、`report.md` | 图像级汇总与解释限制 |
| `samples.csv`、`texts.csv` | 精确图片和模板清单 |
| `metadata.json`、`plot_settings.json` | 检查点/CLIP/source 文件哈希、训练参数、目标域及绘图设置 |
| `*_unit.npy`、`text_banks.npz` | 可重画和复核的高维单位特征；完整 Z 是 `[N,80,D]` |

指标定义：

- 融合位移：每张图片先对全部模板平均 `1-cos(E_i,Z_ik)`，同时报告 `LN(E_i)` 对照。
- 模板变化：同一图像、同一真实类别词下，全部不同模板对的平均 `1-cos(Z_ik,Z_il)`。
- 类别词变化：固定图像和模板，遍历所有错误类别词的平均 `1-cos(Z_i,y,k,Z_i,c,k)`。
- 类别减模板变化：以上两种位移的图像级配对差。大于 0 表示在此替换协议下类别词变化更大，
  不代表类别词本身有害，也不是严格正交的语义因素分解。
- 域 silhouette：每个真实类别内部、原始单位特征的余弦距离、域作为组标签。
- 同域近邻超额比例：每张图像 k 近邻中的同域比例减去排除自身后的随机同域概率；边界距离并列时分配权重。
- 域余弦差：同类同域平均余弦相似度减去同类跨域平均余弦相似度。
- 类内结构保持：E 与 Z 的图像距离秩相关及 k 近邻重合率；即使整体坐标改变，结构仍可能高度保持。

bootstrap 在固定类×域内重采样**图像**，不重采样 80 个文本当作新图像。
区间只表示这个检查点、这些域中的图像采样不确定性，不涵盖训练随机性或新域不确定性。
域指标是描述性指标，没有自动的显著性判定或预设“无效”阈值。

## 6. 要进一步声称“文本多样性没有带来收益”

仅凭特征图不能证明训练增强无收益；还需同预算的训练对照：

| 组 | 在同一原 CIPT 预设上追加参数 | 检验范围 |
|---|---|---|
| 完整 B5B | 不追加 | 基线 |
| Bconst | `--cipt_template_mode bconst` | 保留类别词、TDA、K 与损失平均方式，仅移除描述多样性 |
| TDA-off | `--cipt_use_tda false` | 去除 TDA 的整体训练/推理贡献；该组直接使用 E 推理 |

这三组需要分别重新训练。训练 seed、数据 split、预算和源域选模规则保持一致。
它们不共享同一个 checkpoint 改开关来冒充重新训练的消融。
B5B 与 Bconst 是检验“描述多样性是否必要”的主要性能对照；TDA-off 也改变了推理结构，解释范围更宽。
若另用统一 E-only 评估隔离训练作用，需要额外实现该评估；legacy 下 `cipt_env_inference=e_only` 不生效。

准确率差异小或未达显著不能单独证明“等价”。若要声称收益小于实用阈值，
应事先定义可接受差异，报告跨 seed/目标域的配对差和区间，检验区间是否落在阈值内。

如果实验支持三项假设，合适的机制结论是：
“分解后的 E 仍保留同类域相关结构；融合输出对类别词的敏感性高于对描述模板的敏感性，
且描述多样性未充分改变类内域结构。其独立泛化收益还需训练消融确认。”
若 E/Z 位移明显或域结构改善，则应如实报告这些结果，缩小或修改原假设。

## 7. 代码验证

```bash
python -m unittest discover -s tests -p 'test_cipt_mechanism.py'
python tests/test_cipt_mechanism.py --smoke-output /tmp/cipt_mechanism_synthetic
```

验证涵盖：域子簇/域混合两种方向、重复模板、LayerNorm 对照、并列近邻、旋转不变结构指标、
错误检查点拒绝、平衡清单、图像级区间、全部模板覆盖。smoke 图完全来自合成数据，不能作为实验结果。
真实训练与特征提取仍需训练服务器上的 PyTorch、OpenAI CLIP、本地数据和检查点；
离线报告只需 NumPy、SciPy、scikit-learn、Matplotlib。
