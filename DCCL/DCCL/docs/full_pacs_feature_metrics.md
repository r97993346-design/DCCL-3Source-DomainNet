# PACS 全量特征分析：每个点、每个指标是什么

## 要回答的问题

固定一个已训练 checkpoint，对相同 PACS 图片分别输入类别相关 B5B 文本与逐条配对的类别无关 B5A/subject 文本。观察两个位置：文本编码器输出 p_k，以及文本干预后的图像特征 z_k。本工具进行特征提取和描述性分析，不改变 checkpoint 权重。

此前的 pilot 只用了 3 类、2 域、每类每域 5 张，共 30 张。`--all-images` 使用本地 PACS 目录中 **7 类、4 域的全部图像文件**，实际总数和每个类别/域的数量写入 `samples.csv` 与 `summary.json`，不会抽成一个固定的小子集。全部图片都使用相同 80 个配对模板。

## 1. p_k 和 z_k 各是什么

| 位置 | 输入 | 输出 | 全量版的样本单位 |
|---|---|---|---|
| p_k | 一条环境文字 | CLIP 文本向量，经 L2 归一化 | 一条文字一个点 |
| z_k | 一张图片的 E 和第 k 条环境文字的 p_k | 实际 TDA 输出，LayerNorm 后、分类 L2 归一化前 | 一张图片与一条文字的一次干预 |

对 p_k，7 类各有 80 条 B5B 文本，共 560 个向量；B5A 的 80 条 subject 文本由所有类别共享，总共 **640 个文本点**。它们不随照片数量变化，不按域或照片复制。

对 z_k，若共有 N 张照片，每组产生 N×80 个向量。全量指标逐一计算这 N×80 次配对干预。

主 t-SNE 图用一张照片的 80 个向量汇总为一个点：

```text
先分别归一化每个 z_k
→ 将该图的 80 个单位向量平均
→ 再归一化平均向量，得到 q_i
```

因此每组图有 N 个图像点；每张照片都在图中，80 条文本也都参与了这个点的计算。E/B5B/B5A 三组共同拟合一次 t-SNE 后分面展示，每个面板 N 个点。原始逐文本的变化保存在 `zk_paired.csv`，单图内部 80 个干预的分散程度另外统计，不靠把近似重复的点堆到主图中判断。

## 2. t-SNE：把高维邻居关系画出来

一个 CLIP 向量有许多维度，不能直接在纸上画。t-SNE 将这些向量映射到二维，尽量保留局部近邻关系。

具体步骤：收集 E/B5B/B5A 的全量图像表示 → 向量归一化 → 拼接三组 → 联合 PCA 到 50 维 → 用固定种子和 perplexity 拟合一次 t-SNE → 用原始类别/域给二维点着色。

PCA 和 t-SNE 不接收类别或域标签，标签只用于画图；B5B 的 TDA 输入本身按真实类别选文本，这个事实在协议中注明。坐标不是分类分数，坐标轴没有类别含义。

提供两张主图：

- `image_joint_tsne_by_class.png`：颜色为 7 个类别，点形状为 4 个域。
- `image_joint_tsne_by_domain.png`：颜色为 4 个域，使用完全相同的二维坐标。

它们用于看同类图像的局部聚集、域之间是否混合、是否有孤立区域。t-SNE 上的距离、簇面积与密度不能直接替代原始空间的数值指标。默认 PCA 是为了全量图形计算；保留的方差比例记录在结果中。`--tsne-pca-dim 0` 可取消这一步，所有数值指标始终在原始维度计算。

文本 t-SNE 单独拟合全部 640 个 p_k，输出在 `pk/pk_joint_tsne.png`。

## 3. 相似度：两个向量的方向接近吗

余弦相似度为：

```text
cos(a,b) = (a·b) / (||a||₂ ||b||₂)
```

1 表示同方向，0 表示正交，−1 表示反方向。必须说明比较的对象：

| 相似度名字 | 实际比较 | 回答的问题 |
|---|---|---|
| p_k 配对相似度 | 第 k 条含类别词的文本 vs 同一模板的 subject 文本 | 只替换类别词，文本向量改变多少？ |
| z_k 配对相似度 | 同一张图、同一模板索引下的两组 z_k | 只替换文本，该次图像干预输出改变多少？ |
| 图像均值配对相似度 | 同一张图的 q_i(B5B) vs q_i(B5A) | 汇总全部 80 条后，方向改变多少？ |
| 类内相似度 | 同类别、不同照片的 q_i 与 q_j | 同类照片的表示方向是否接近？ |
| 类间相似度 | 不同类别照片的 q_i 与 q_j | 不同类别是否容易混在一起？ |
| 跨域同类相似度 | 类别相同、域不同的照片的 q_i 与 q_j | 同类的照片/绘画/卡通/素描能否靠近？ |

类内/类间/跨域的均值包含 **全部符合条件的图像对**，排除一张图与自己比较。程序通过向量和计算精确均值，无需存储整个 N×N 相似度矩阵，未抽取少量图像对来近似。全量版不报告这些海量图像对的中位数。

PACS 类别数量不均衡，因此同时提供：micro（每个图像对等权）与 macro（每个类别或类别对等权），以及每类、每域的细分结果。

## 4. 类间距离：两个类别的中心相隔多远

对于每个类别，把该类别所有照片的单位表示平均为类中心 μ_c。

主距离热图使用：

```text
d_cos(c,d) = 1 − cos(μ_c, μ_d)
```

对角线约为 0，两个类中心越接近，距离越小。程序对 V、E、B5B 图像均值、B5A 图像均值分别形成 7×7 距离矩阵；7 类有 21 个不同类别对。

还输出每个类别对的所有跨类图像对平均余弦距离。这与“两个类中心的距离”是不同统计对象，文件和字段分别注明。类中心距离不能代表类内分散程度，所以与类内相似度一起观察。

对 p_k，B5B 可按 7 个类别形成文本类中心；共享的 B5A p_k 没有 7 个不同类别的银行，不能把同一批 subject 向量复制七遍后把零类间距离当实验结论。对 z_k，两组都可以按图片的真实类别分组。

## 5. L2 范数：一个向量有多长

```text
L2 范数：||f||₂ = sqrt(sum_j f_j²)
L2 距离：||a−b||₂
```

范数是一个向量的长度；距离是两个向量之间的差，不能混称。

例如 (1,0) 和 (0,1) 的范数都为 1，但余弦相似度为 0，说明范数相同并不代表特征相同。对于单位向量，L2 距离平方等于 2×(1−余弦相似度)，因此两者不是两项独立证据。

本工具分位置报告范数：

- 文本编码器原始输出：看看归一化前的尺度。
- 实际输入 TDA 的 p_k：已经归一化，范数约为 1。
- E：adapter 输出的实际长度。
- z_k 的 LayerNorm 前后：检查尺度变化。
- 分类方向中的单位 z_k：范数约为 1。

z_k 的原始范数受 LayerNorm 约束，接近不说明其方向相同。范数图用全量均值和标准差，并将单位 p_k 的纵轴固定在 1 附近，避免把数值精度误差放大成看似明显的差异。

## 6. 全量运行命令

在 `train_all.py` 所在目录拉取原整合分支的最新代码，然后执行：

```bash
git switch feature/cipt-paired-class-agnostic-subject-validation-official-norm-idinit
git pull --ff-only

CUDA_VISIBLE_DEVICES=1 python analyze_cipt_text_features.py \
  --dataset PACS \
  --data-dir /home/hooasia/lgg/data/repro_dccl_data \
  --checkpoint train_output/ablation/main6/PACS/261014_17-06-02_pacs_b5b_pk_zk_ckpt/checkpoints/TE0_5000.pth \
  --clip-path /home/hooasia/.cache/clip/ViT-B-16.pt \
  --all-images \
  --batch-size 8 \
  --seed 0 \
  --tsne-perplexity 30 \
  --output-dir train_output/ablation/pacs_all_pk_zk
```

全量模式不再传 `--classes`、`--domains` 或旧的 30 张 `--sample-manifest`。程序检查 PACS 有 7 类、4 域，并保存准确的全量清单。它扫描目录中所有支持的图像文件，读取失败时停止并报告，不默默跳过。

使用现有 B5B checkpoint；所有域图像用于描述性特征分析，包含被该 checkpoint 留出的域，不用于训练、更新权重或选择 checkpoint。这一结果测量固定模型的文本替换影响。

## 7. 保存和重新绘图

逐图、逐文本标量指标流式写入 CSV；图像级向量写入磁盘映射的 `.npy`，不会在 RAM 中堆积所有 N×80×D 中间向量。默认保留全部数值指标和每图均值向量；增加 `--save-all-z` 可另外保存两组全部原始 z_k，需额外数 GiB 磁盘空间，启动时会显示实际估算。

全量 t-SNE 的 CPU 时间可能比提取更长。需要先得到数值时可加 `--no-tsne`。以后使用保存好的向量重画，无需重新执行 CLIP：

```bash
python tools/cipt_full_feature_report.py \
  --output-dir train_output/ablation/pacs_all_pk_zk
```

已有结果目录不被覆盖；重新提取时指定新目录。

| 输出 | 内容 |
|---|---|
| `samples.csv` / `texts.csv` | 全部照片和配对文本清单 |
| `pk/` | 所有 7 类的 p_k 配对指标、文本 t-SNE、文本范数与类中心距离 |
| `zk_paired.csv` | 每张图×每条文本的配对变化，全部 N×80 行 |
| `zk_image_means.csv` | 每图一行，80 条干预汇总后的配对变化 |
| `summary.json` / `report.md` | 全量均值、原始维度和覆盖数量、checkpoint 与绘图协议 |
| `class_metrics.csv` | 每类的类内和跨域同类相似度 |
| `class_pair_metrics.csv` | 21 个不同类别对的全部图像对平均相似度/距离，分条件报告 |
| `class_geometry_by_domain.json` | 每个域单独的类别关系 |
| `image_class_centroid_distance.png/.pdf` | V/E/B5B/B5A 的 7×7 类中心距离矩阵 |
| `same_class_domain_similarity.png/.pdf` | 4 个域间同类图像相似度矩阵 |
| `feature_norm_statistics.csv` / `feature_norms.png/.pdf` | 各位置全量范数统计 |
| `image_joint_tsne_by_class.png/.pdf` | 全部图像点，按类别着色 |
| `image_joint_tsne_by_domain.png/.pdf` | 相同二维坐标，按域着色 |
| `vectors/` / `image_tsne_coordinates.npz` | 可复用的原始图像表示和二维坐标 |

## 8. 验证

```bash
python -m unittest discover -s tests -p 'test_cipt*feature*.py'
```

测试检查：全量枚举无遗漏；精确向量和统计等价于显式全部图像对矩阵；流式指标等价于一次性计算；每图全部 80 条均写出；图像、类别、域统计与小样本路径一致。合成数据只用于验证程序与布局，不作为真实 PACS 实验结果。
