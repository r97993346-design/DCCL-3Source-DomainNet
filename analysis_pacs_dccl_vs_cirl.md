# PACS DCCL baseline vs DCCL+CIRL+Adaptive-KL 日志分析报告

> 生成时间：2026-06-18  
> 分析分支：当前工作区  
> 重点日志目录：
>
> 1. `train_output/PACS/260325_17-19-44_DCCL_OH_0`
> 2. `train_output/PACS/260618_11-50-49_pacs_dccl_cirl_akl`

## 0. 数据可用性结论

本工作区中未找到用户指定的两个 PACS 日志目录。实际检查命令如下：

```bash
find train_output/PACS/260325_17-19-44_DCCL_OH_0 train_output/PACS/260618_11-50-49_pacs_dccl_cirl_akl -maxdepth 2 -type f | sort
find .. -path '*260325_17-19-44_DCCL_OH_0*' -o -path '*260618_11-50-49_pacs_dccl_cirl_akl*'
find . -maxdepth 5 -type d | grep train_output | sort
find . -maxdepth 6 -type f -name 'log.txt'
```

结果显示当前仓库只有：

```text
./DCCL/DCCL/train_output/DomainNet/260502_12-58-14_exp-domainnet-3source-v1/log.txt
```

因此，本报告无法从本地文件中真实提取 PACS 四目标域的 oracle / iid / last / SWAD / inD SWAD 数值，也无法计算真实 Avg 差值。下文将严格区分：

- **日志直接证明**：本工作区没有可用 PACS 日志，无法直接证明。
- **代码层面可证明**：从当前实现可确认的实现路径、潜在风险和下一步实验设计。

---

## 1. 实验配置对比

### 1.1 期望提取项

| 项目 | DCCL baseline | DCCL+CIRL+AKL | 当前状态 |
|---|---:|---:|---|
| 启动命令 | 待从 `log.txt` 提取 | 待从 `log.txt` 提取 | 日志缺失 |
| Python / PyTorch / CUDA / CUDNN / NumPy / PIL | 待从 `log.txt` 提取 | 待从 `log.txt` 提取 | 日志缺失 |
| deterministic | 待从 Args 提取 | 待从 Args 提取 | 日志缺失 |
| 模型参数量 | 待从 `# of params` 提取 | 待从 `# of params` 提取 | 日志缺失 |
| DCCL loss 组成 | 代码可确认 | 代码可确认 | 可做代码检查 |
| CIRL loss 组成 | 不适用 | 代码可确认 | 可做代码检查 |

### 1.2 日志写入机制检查

当前训练入口会将启动命令写入日志：

```python
logger.info(f"Command :: {cmd}")
```

也会写入 Python、PyTorch、Torchvision、CUDA、CUDNN、NumPy、PIL 等环境版本：

```python
logger.nofmt("Environment:")
...
logger.nofmt("\tPyTorch: {}".format(torch.__version__))
logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
```

`deterministic` 来自 CLI：

```python
parser.add_argument("--deterministic", action="store_true")
torch.backends.cudnn.deterministic = args.deterministic
torch.backends.cudnn.benchmark = not args.deterministic
```

模型参数量由 trainer 统计：

```python
n_params = sum([p.numel() for p in algorithm.parameters()])
logger.info("# of params = %d" % n_params)
```

因此，一旦提供两个 PACS 目录中的 `log.txt` / `results.jsonl`，上述配置可以直接自动提取。

---

## 2. 性能结果对比

### 2.1 PACS 目标域结果表

> 注意：由于指定日志目录不存在，以下表格保留为待填充模板；不能伪造结果。

| Target env | DCCL baseline oracle | DCCL baseline iid | DCCL baseline last | DCCL baseline SWAD | DCCL baseline inD SWAD | DCCL+CIRL+AKL oracle | DCCL+CIRL+AKL iid | DCCL+CIRL+AKL last | DCCL+CIRL+AKL SWAD | DCCL+CIRL+AKL inD SWAD | SWAD 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| P | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| S | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Avg | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### 2.2 iid Avg / SWAD Avg 差值

| 指标 | DCCL baseline Avg | DCCL+CIRL+AKL Avg | 差值 |
|---|---:|---:|---:|
| iid Avg | N/A | N/A | N/A |
| SWAD Avg | N/A | N/A | N/A |

### 2.3 trainer 中结果选择定义

当前 trainer 最终返回：

- `oracle`: `records.argmax("test_out")["test_in"]`
- `iid`: `records.argmax("train_out")["test_in"]`
- `last`: `records[-1]["test_in"]`
- `SWAD`: SWAD averaged model 的 `test_in`
- `SWAD (inD)`: SWAD averaged model 的 in-domain key

因此，真实日志存在时，应优先从 final summary 或 `results.jsonl` 中提取这些字段，而不是人工从中间 checkpoint 猜测。

---

## 3. 日志异常点

当前无法从 PACS 日志直接验证以下异常点，因为日志目录缺失：

1. baseline 是否 `deterministic=True`，CIRL 实验是否 `deterministic=False`。
2. 两次实验 Python / PyTorch / CUDA / NumPy 是否不同。
3. baseline 参数量是否约 49M，CIRL 实验是否约 150M。
4. `alpha_d_mean` / `alpha_c_mean` 是否长期接近 0.5。
5. `L_cirl_official`、`L_adaptive_kl` 是否从训练开始数值过大。
6. SWAD valley 起止区间是否不同。

但代码层面可以确认：

- 训练日志会写入环境版本和 args。
- 每个 checkpoint 会聚合 `algorithm.update()` 返回的 loss 字典。
- 当前 CIRL 日志项包括 `L_cirl_official`、`L_adaptive_kl`、`alpha_d_mean`、`alpha_c_mean`、`cirl_loss_sup`、`cirl_loss_inf`、`cirl_loss_fac`、`cirl_masker_loss`。

---

## 4. 可能下降原因排序

### P1. 对比实验可能不公平：deterministic / 环境 / run setting 不一致

**证据类型：需要日志验证。**

用户提示中提到 baseline 是 `deterministic=True`，CIRL 实验是 `deterministic=False`。如果属实，这不是严格公平对比。当前训练入口会直接用 `args.deterministic` 控制 CUDNN deterministic 和 benchmark：

```python
torch.backends.cudnn.deterministic = args.deterministic
torch.backends.cudnn.benchmark = not args.deterministic
```

这会改变卷积选择、吞吐和可复现性，尤其 PACS 数据量不大时，seed / deterministic 差异可能带来可见波动。

**优先级：最高。** 先跑同环境、同 deterministic、同 seed 的新 baseline，再比较 CIRL。

### P2. CIRL 权重过大，训练一开始就强干扰 DCCL 主目标

**证据类型：代码可证明；数值大小需日志验证。**

当前总损失为：

```python
loss = loss + self.cirl_weight * cirl_official_loss + self.cirl_kl_weight * cirl_adaptive_kl_loss
```

默认 `cirl_weight=1.0`、`cirl_kl_weight=0.1`。如果 `L_cirl_official` 和 DCCL CE/contrastive loss 同量级或更大，那么 CIRL 从 step 0 开始就会显著改变 backbone 梯度。当前实现没有 CIRL warm-up，也没有 KL warm-up。

**优先级：高。** 建议先试 `cirl_weight=0.05/0.1`，`cirl_kl_weight=0.01/0.03`。

### P3. adaptive KL 可能退化为固定双向 KL

**证据类型：代码可证明公式；是否退化需日志验证 alpha 分布。**

当前 `alpha_d` 和 `alpha_c` 仅记录 mean：

```python
alpha_d_mean = alpha_d.mean()
alpha_c_mean = alpha_c.mean()
```

如果训练中二者长期接近 `0.5`，则 adaptive KL 实际接近固定对称 KL，不能实现“哪个分支更可靠，就让另一个学习”。当前没有记录 std/min/max，也没有记录 `ce_d_mean`、`ce_c_mean`、`kl_d2c`、`kl_c2d`，所以单靠 mean 很难判断是否真的动态。

**优先级：高。** 需要补充 alpha 分布和 KL 分项日志。

### P4. DCCL contrastive 空间与 CIRL factorization 空间共享 backbone 主特征，存在梯度冲突

**证据类型：代码可证明共享；冲突强度需实验验证。**

DCCL 原 contrastive loss 使用 `feature_x` 经过 `proj_head` 后的 projection features：

```python
embed_1 = self.proj_head(feature_x)
embed_2 = self.proj_head(feature_x_2)
loss_sup_cl = self.supcon_loss(features, all_y)
```

CIRL factorization 直接使用 `feature_x` 和 `feature_x_a`：

```python
cirl_loss_fac = self._safe_cirl_factorization_loss(feature_x, feature_x_a)
```

虽然 contrastive loss 在 projection head 上计算，但二者梯度最终都回到同一个 `featurizer` 主特征。DCCL 希望增强跨域判别/聚合，而 CIRL factorization/mask 希望进行因果/非因果分解，两者可能对主特征施加不同目标。

**优先级：中高。** 最小改法是给 CIRL 增加独立轻量 adapter/head，而不是直接在 `feature_x` 上施加强约束。

### P5. CIRL adversarial masker 的 optimizer / detach 可能与官方实现不完全等价

**证据类型：代码可证明存在适配差异；影响需实验验证。**

官方 CIRL 训练顺序是先更新 encoder/classifier/classifier_ad，再更新 masker。当前实现也大体遵循这个顺序。但存在适配差异：

- step1 中 mask 来自 `cirl_features.detach()`，masker 本身不随主 loss 更新。
- step2 中重新计算 detached feature，再只 step `cirl_masker_optimizer`。
- classifier 在 masker step 中也参与 forward，但不被 optimizer step；不过 classifier 参数会累积梯度，直到下一轮主 optimizer zero_grad。

这通常不致命，但不是完全严格的官方多 optimizer 训练 loop。建议显式冻结 classifier/featurizer 或在 masker step 后清理非 masker 梯度，减少副作用。

**优先级：中。**

### P6. Fourier intervention 未真正复用官方 dataloader，实现退化为 identity augmentation

**证据类型：代码可证明。**

当前适配函数写明：vendored CIRL tree 引用了 `get_fourier_train_dataloader`，但本仓库未包含官方 `data` package，因此 fallback 为 identity augmentation：

```python
return x.detach().clone()
```

这意味着当前 `feature_x_a` 与 `feature_x` 来自相同图像副本，而不是官方 Fourier intervention 图像。这样 CIRL factorization 分支不能复现官方 Fourier causal intervention，可能只是在相同样本上施加额外约束。

**优先级：中。** 若要严肃评估 CIRL，应补齐官方 Fourier data transform，而不是 identity fallback。

---

## 5. 代码实现检查点

### 5.1 DCCL 原始 loss 计算位置

DCCL CE：

```python
feature_x, inter_feats = self.featurizer(all_x, ret_feats=True)
pred_x = self.classifier(feature_x)
loss = F.cross_entropy(pred_x, all_y)
```

DCCL contrastive loss：

```python
embed_1 = self.proj_head(feature_x)
embed_2 = self.proj_head(feature_x_2)
loss_sup_cl = self.supcon_loss(features, all_y)
loss += self.l * loss_sup_cl
```

layer-wise pre contrastive loss：

```python
pre_cl_loss += self.supcon_loss_pre(features, all_y_pre)
loss += self.l_layer * pre_cl_loss
```

### 5.2 CIRL official 模块接入位置

官方 CIRL components：

```python
OfficialCIRLClassifier
OfficialCIRLMasker
official_cirl_factorization_loss
```

启用 `use_cirl_official` 后初始化：

```python
self.cirl_classifier = OfficialCIRLClassifier(...)
self.cirl_classifier_ad = OfficialCIRLClassifier(...)
self.cirl_masker = OfficialCIRLMasker(...)
```

### 5.3 adaptive KL 计算位置

```python
def _adaptive_kl_loss(self, logits_d, logits_c, labels):
    ...
```

实现中：

- `ce_d` / `ce_c` 用 `reduction="none"`。
- confidence 来自 softmax max prob。
- `score_d` / `score_c` detach CE 和 confidence。
- teacher probabilities `p_d_t` / `p_c_t` detach。
- alpha 在最终 loss 中 detach。
- 最终乘 `T*T`。

### 5.4 CIRL logits 与 DCCL logits 是否独立

DCCL logits：

```python
pred_x = self.classifier(feature_x)
```

CIRL logits：

```python
logits_c = self.cirl_classifier(feature_x)
```

两者是不同 classifier，但共享同一个 backbone feature `feature_x`。所以它们是 **classifier head 独立，backbone 不独立**。

### 5.5 alpha / teacher detach / KL 方向

当前实现：

```python
p_d_t = p_d.detach()
p_c_t = p_c.detach()
kl_d_to_c = p_d_t * (log p_d_t - log p_c)
kl_c_to_d = p_c_t * (log p_c_t - log p_d)
loss = alpha_d.detach() * kl_d_to_c + alpha_c.detach() * kl_c_to_d
```

方向解释：

- `kl_d_to_c`：DCCL teacher detached，CIRL student 学 DCCL。
- `kl_c_to_d`：CIRL teacher detached，DCCL student 学 CIRL。

这与设计目标一致。

### 5.6 CIRL loss 是否乘权重

是。当前总 loss：

```python
loss = loss + self.cirl_weight * cirl_official_loss + self.cirl_kl_weight * cirl_adaptive_kl_loss
```

### 5.7 是否支持 CIRL warm-up

当前不支持。`cirl_weight` 和 `cirl_kl_weight` 从 step 0 开始直接生效。建议新增：

- `--cirl_warmup_steps`
- `--cirl_kl_warmup_steps`
- `--cirl_start_step`

示例策略：

```python
warm = min(1.0, step / max(1, cirl_warmup_steps))
loss = loss + warm * cirl_weight * L_cirl + warm_kl * cirl_kl_weight * L_akl
```

---

## 6. 推荐修改方案

### 6.1 最小代码修改

1. **增加 warm-up 参数**
   - `--cirl_warmup_steps 500`
   - `--cirl_kl_warmup_steps 1000`
   - `--cirl_start_step 0`

2. **降低默认实验权重**
   - `cirl_weight`: 从 `1.0` 降到 `0.05` 或 `0.1`
   - `cirl_kl_weight`: 从 `0.1` 降到 `0.01` 或 `0.03`

3. **记录更多 adaptive KL 诊断项**
   - `alpha_d_mean/std/min/max`
   - `alpha_c_mean/std/min/max`
   - `ce_d_mean`
   - `ce_c_mean`
   - `kl_d2c_mean`
   - `kl_c2d_mean`

4. **记录梯度冲突诊断项**
   - `grad_norm_featurizer_dccl`
   - `grad_norm_featurizer_cirl`
   - 或至少分阶段记录总 grad norm。

5. **分离 CIRL factorization adapter**
   - 保留 DCCL backbone 主特征 `r`。
   - 增加轻量 `cirl_adapter = Linear/BN/ReLU/Linear`。
   - CIRL factorization 和 CIRL classifier 使用 `r_c = cirl_adapter(r)`。
   - DCCL CE / contrastive loss 继续使用原路径。

6. **masker step 梯度清理**
   - masker step 后清理 classifier/featurizer 梯度，避免下一轮前存在非预期梯度残留。

### 6.2 不建议的大改

- 不建议替换 DCCL CE。
- 不建议把 Fourier augmented images 加入 DCCL positives。
- 不建议默认推理切换到 CIRL classifier。
- 不建议一开始就大幅改 backbone 或训练器框架。

---

## 7. 下一轮实验命令

以下命令假设从 `DCCL/DCCL` 目录运行。

### 7.1 同环境 DCCL baseline，确认可比性

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_dccl_base_same_env \
  --dataset PACS \
  --algorithm DCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --seed 0 \
  --trial_seed 0 \
  --checkpoint_freq 100 \
  --deterministic
```

### 7.2 CIRL 低权重，无 KL

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_dccl_cirl_w005 \
  --dataset PACS \
  --algorithm DCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --seed 0 \
  --trial_seed 0 \
  --checkpoint_freq 100 \
  --deterministic \
  --use_cirl_official \
  --cirl_weight 0.05
```

### 7.3 CIRL 低权重 + 低 KL

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_dccl_cirl_akl_w005_kl001 \
  --dataset PACS \
  --algorithm DCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --seed 0 \
  --trial_seed 0 \
  --checkpoint_freq 100 \
  --deterministic \
  --use_cirl_official \
  --cirl_use_adaptive_kl \
  --cirl_weight 0.05 \
  --cirl_kl_weight 0.01 \
  --cirl_kl_temperature 2.0
```

### 7.4 CIRL 中低权重 + 低 KL

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_dccl_cirl_akl_w01_kl003 \
  --dataset PACS \
  --algorithm DCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --seed 0 \
  --trial_seed 0 \
  --checkpoint_freq 100 \
  --deterministic \
  --use_cirl_official \
  --cirl_use_adaptive_kl \
  --cirl_weight 0.1 \
  --cirl_kl_weight 0.03 \
  --cirl_kl_temperature 2.0
```

### 7.5 warm-up 版本，需先实现 warm-up 参数

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_dccl_cirl_akl_warmup \
  --dataset PACS \
  --algorithm DCCL \
  --data_dir /home/hooasia/lgg/data/repro_dccl_data \
  --seed 0 \
  --trial_seed 0 \
  --checkpoint_freq 100 \
  --deterministic \
  --use_cirl_official \
  --cirl_use_adaptive_kl \
  --cirl_weight 0.1 \
  --cirl_kl_weight 0.01 \
  --cirl_kl_temperature 2.0 \
  --cirl_warmup_steps 500 \
  --cirl_kl_warmup_steps 1000
```

---

## 8. 总结

当前无法完成真实 PACS 数值对比，因为两个指定日志目录不在工作区中。基于代码检查，最可能导致 DCCL+CIRL+AKL 无提升或下降的原因按优先级为：

1. **实验不公平**：deterministic / 环境版本 / seed / SWAD 区间可能不同。需要日志验证。
2. **CIRL loss 权重过大且无 warm-up**：`cirl_weight=1.0` 从 step 0 直接加入主 loss。代码可证明。
3. **KL 权重偏大且无 warm-up**：`cirl_kl_weight=0.1` 可能在 CIRL classifier 尚不可靠时强行互教。代码可证明。
4. **adaptive KL 可能退化为近似固定双向 KL**：只记录 mean，不记录分布，若 alpha 长期接近 0.5 则动态性不足。需日志验证。
5. **共享 backbone 特征导致梯度冲突**：DCCL contrastive 和 CIRL factorization/mask 都作用回同一 `feature_x`。代码可证明。
6. **Fourier intervention 未完整接入官方 dataloader**：当前 fallback 为 identity augmentation，不能复现官方 CIRL intervention。代码可证明。

下一步应先补齐/提供 PACS 两个日志目录；若日志确认 CIRL 下降，再优先跑同环境 baseline、低权重 CIRL、低权重 KL 和 warm-up 组合。
