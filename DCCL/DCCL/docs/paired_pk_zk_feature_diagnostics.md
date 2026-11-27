# 类别相关 / 类别无关文本的少样本特征诊断

对应基础分支：`feature/cipt-paired-class-agnostic-subject-validation-official-norm-idinit`。

第一轮回答两个问题：去掉类别词后，冻结环境文本特征 `p_k` 改变多少？这些变化经过同一个已训练 TDA 后，是否明显改变图像干预特征 `z_k` 的方向、分布和图像类别关系？当前脚本是分析工具，不进行训练。

## 1. 对照与特征位置

| 对照 | 文本 | 数量 | 代码位置 |
|---|---|---:|---|
| 类别相关 B5B | `a bad photo of a dog.` 等 | 每个选定类别 80 条 | `B5B_IMAGENET_TEMPLATES` |
| 类别无关 B5A | 同一条模板中的类别换成 `subject` | 全类别共享 80 条 | `build_paired_generic_templates("subject")` |

这里使用逐条配对的 B5B/B5A。B5C 的 42 条文本还改变了模板内容，适合后续单独分析文本库设计。

| 名称 | 定义 | 用途 |
|---|---|---|
| `pk_raw_*` | `CLIP.encode_text(text)`，归一化前 | 看文本编码器输出的原始 L2 范数 |
| `pk_*` | `F.normalize(pk_raw, dim=-1)` | 实际进入 TDA 的环境文本特征；核心 p_k 对照 |
| `visual` / `e` | 归一化 CLIP 图像特征 / 因果 adapter 输出 | 相同图像输入；E 作为类关系的参考 |
| `tda_update_*` | 注意力输出，即 `preLN - E` | 检查文本干预量 |
| `zk_preln_*` | `E + attention(E, pk, pk)` | LayerNorm 前的尺度 |
| `zk_*` | 分支中 `tda(E, pk)` 的真实输出 | 核心 z_k 对照：LayerNorm 后、L2 归一化前 |
| `zk_unit_*` | `F.normalize(zk, dim=-1)` 的等价 NumPy 结果 | 分类余弦空间中的方向 |

环境文本银行使用冻结 CLIP 编码，不是可学习类别分类提示。全程固定同一模型权重、E、图像与模板顺序，只替换 TDA 输入文本；绕过随机 K 抽样和自适应选择，完整分析 80 条。当前 TDA 对每个文本独立执行单 token 干预，因此同时导出 80 条不会把它改成 80 个 token 竞争的注意力。

类别相关 z_k 按图像已知类别选取文本，这是带标签的特征诊断，不能当作使用真实标签进行 B5B 测试推理。原分支 B5B 的候选类推理流程不受影响。

## 2. 第一轮样本

推荐 PACS：`dog / horse / guitar`，`photo / sketch` 两个域，每类每域 5 张，共 30 张；两组共 4,800 个 z_k。dog/horse 便于观察近邻类别，guitar 提供外观差异较大的参照。需要更快时可先每类每域 3 张，共 18 张。固定 seed=0，并保存 `samples.csv`，避免换样本造成观察差异。

使用已经开启 TDA 训练的 checkpoint。不要用随机初始化的 TDA 来评价训练后 z_k 的性质。可以首先选 B5B 模型的 checkpoint，再在 B5A 模型上用同一 `samples.csv` 重复这项配对替换，检查现象是否依赖训练模板。

同一模型上的替换测量“输入文本的直接影响”。分别训练的 B5B/B5A 模型之间的比较还包含参数适配差异，两种比较需要分开解释。

## 3. 服务器运行

进入有 `train_all.py` 的目录（仓库相对路径 `DCCL/DCCL`）。绘图依赖：`numpy matplotlib scikit-learn`，特征提取复用训练环境中的 torch、torchvision 和仓库 CLIP。

查看已有 checkpoint：

```bash
rg --files train_output | rg '/checkpoints/TE.*\.pth$'
```

把下面 `CIPT_FEATURE_CKPT` 的值替换成已有、开启 TDA 训练的 `.pth` 路径：

```bash
CIPT_FEATURE_CKPT=/实际路径/checkpoints/TE0_5000.pth
CUDA_VISIBLE_DEVICES=1 python analyze_cipt_text_features.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --checkpoint "$CIPT_FEATURE_CKPT" \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --classes dog horse guitar \
  --domains photo sketch \
  --samples-per-class-domain 5 \
  --seed 0 \
  --output-dir train_output/ablation/pk_zk_paired_80
```

如果实际域文件夹名称不同，按报错列出的目录修改 `--domains`。不指定该参数时，优先取 photo/sketch，否则取实际目录排序后的首尾两个域，并将实际选择记入结果。

没有保存模型时，可以先只看 p_k：

```bash
CUDA_VISIBLE_DEVICES=1 python analyze_cipt_text_features.py \
  --dataset PACS \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --classes dog horse guitar \
  --pk-only --seed 0 \
  --output-dir train_output/ablation/pk_only_paired_80
```

此分支训练器在设置 `--model_save` 后保存 `checkpoints/TE*_*.pth`；未设置时可能只有日志。该格式保存的是当时的 `algorithm.state_dict()`，不是自动保存的最终 SWAD 平均模型。图像分析使用的是所提供 checkpoint 的实际权重，需要在汇报中注明。已有 SWAD wrapper state 使用 `module.` 等前缀时，脚本根据 causal adapter 键移除包装前缀。

在第二个 checkpoint 上固定同一批图片，增加：

```bash
  --sample-manifest train_output/ablation/pk_zk_paired_80/samples.csv
```

并指定新的 `--output-dir`。Manifest 的相对路径用于复用样本，脚本检查类别顺序和图像是否存在。

VLCS 可使用 `--dataset VLCS --classes bird car person`，域名按实际目录指定。

## 4. 交给老师的结果

| 文件 | 看什么 |
|---|---|
| `report.md` / `summary.json` | 总体配对差异、类关系、跨域同类相似度、范数及完整配置 |
| `pk_joint_tsne.png/.pdf` | B5B 各类 80 条文本 + B5A 共享 80 条文本，联合降维一次 |
| `zk_joint_tsne.png/.pdf` | 每图 80 个 z_k；两组联合降维，两个面板共享坐标范围 |
| `zk_image_mean_joint_tsne.png/.pdf` | 每图的 80 个归一化 z_k 平均后再归一化，配对连线观察方向变化 |
| `pk_class_distance.png/.pdf` | 类别相关文本的类中心余弦距离 |
| `zk_class_distance.png/.pdf` | E、B5B z 均值、B5A z 均值的图像类别中心距离 |
| `feature_norms.png/.pdf` | 原始文本 / 实际 p_k / LayerNorm 前后 z_k 的范数对照 |
| `pk_paired.csv` | 每类别、每模板的配对余弦、余弦变化、原始及单位向量 L2 距离与范数 |
| `zk_paired.csv` | 每图、每模板的配对变化、与 E 的余弦、文本与冻结图像的余弦、范数 |
| `zk_image_means.csv` | 每图的平均 z 对照与文本干预分散程度 |
| `samples.csv` / `texts.csv` | 使用的照片和全部逐条配对文本，可回查来源 |
| `features.npz` / `*_tsne_coordinates.npz` | 原始特征及降维坐标，支持后续重画 |

所有数值距离、类内/类间相似度都在原始特征维度计算。类关系以每张图的平均 z 为一个观测单位，不把同一张图的 80 条干预当成 80 张独立图像；p_k 不按照片复制，B5A 也不按类别复制。

默认 t-SNE 使用全部 80 条。绘图较慢时可加 `--tsne-templates-per-image 16`；只减少图上点数，两组使用相同模板索引，所有指标仍使用 80 条。`--no-tsne` 可先导出数值与其他图。每个位置两组一起降维，p_k 与 z_k 分开绘图。自动降低不适合少样本的 perplexity，固定种子；图上类簇的间距、面积或两张独立 t-SNE 图的位置不能当作高维距离证据。

## 5. 怎样解释观察

1. **p_k 变化大，z_k 的配对方向变化也大**：说明在该 checkpoint、该样本上，文本替换明显影响 TDA 输出；再看这是否改变图像类关系或跨域同类相似度。
2. **p_k 变化大，z_k 变化小**：说明该模型对文本差异的传递较弱，可以结合 E 残差和 `tda_update_*` 检查，不预设结果。
3. **z_k 变化大，但类关系相近**：可能主要改变共享方向或尺度；配对变化大不等于分类结构被破坏。
4. **B5A 的跨域同类相似度更高**：可以作为后续较大样本实验的线索，还需排除整体特征塌缩；一起看异类相似度、类内/类间距离及 prompt dispersion。

两条解释限制尤其重要：实际 p_k 已归一化，L2 范数接近 1 是代码设定；z_k 的范数受 LayerNorm 约束，初始仿射参数下常接近 sqrt(D)，训练后仿射参数可以改变它。应将原始范数与方向指标一起看。单位向量之间 `L2² = 2(1−cosine)`，它们不是两项独立证据。

B5A 是同一套共享文本，“不同类别的 B5A p_k 类间距离为 0”只是人为复制同一银行的必然结果，不能用它论证方法优势，因此脚本不制造这种重复类标签。B5B 的文本按类别聚簇也可能只是类别词存在的结果；第一轮是描述性观察，不预先宣称类别无关更好、不据此直接证明前门假设或泛化提升。

## 6. 验证范围

```bash
python -m unittest discover -s tests -p test_cipt_feature_diagnostics.py
python tests/test_cipt_feature_diagnostics.py --smoke-output /tmp/cipt_plot_smoke
```

测试覆盖配对距离、单位 L2 与余弦的关系、按图像统计、共享文本去重、checkpoint 前缀、固定样本与 manifest 重放，以及 CSV/JSON/图像输出。Smoke 图使用合成数据验证布局，不能作为真实实验结果。实际 CLIP/图像/checkpoint 特征提取必须在训练环境用真实数据运行。
