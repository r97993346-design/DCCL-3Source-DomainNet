# 可选的跨域对比学习（RD/MCC 分支）

直接修改 `feature/cipt-b5b-entropy-mcc-rd`，不合并另一个分支，因此 RD、MCC、熵筛选、selected-center RD、原始 CIPT 机制可视化以及已有训练脚本保持原状。

## 模式选择

单独设置 `--cipt_use_contrastive true` 后，使用下面四选一开关：
- `--cipt_contrastive_type supcon`：旧监督对比，同类所有非自身样本为正样本
- `--cipt_contrastive_type neighbor_retention`：旧邻域退化感知 SupCon
- `--cipt_contrastive_type cross_domain_supcon`：跨源域同类为正样本，所有候选等权
- `--cipt_contrastive_type clip_guided_cross_domain`：同样的跨域候选，CLIP 视觉相似度负责选择及加权

新分支选项单独启动，不和原邻域损失同时累加。同类同域和 Top-K 未选中同类跨域样本不进入分母。不同类别任何源域样本作为负样本；CLIP 特征只是已冻结的教师，损失梯度只进入 E 表示路径。无跨域同类正样本时安全跳过对应锚点。

`--cipt_clip_teacher_mix` 范围 [0,1]，0 为等权；`--cipt_clip_teacher_temperature` 默认 0.2，只控制教师权重；`--cipt_contrastive_temperature` 控制 E 对比学习温度。`--cipt_clip_teacher_topk 0` 表示全选跨域同类正样本。所有源域 batch 拼接后选样，目标测试域不参与。

## 运行（老师服务器，GPU1，PACS 全部目标域）

在 `DCCL/DCCL` 目录，使用已存在的数据集与 ViT-B/16：

```bash
CUDA_VISIBLE_DEVICES=1 python train_all.py pacs_rd2_clip_guided \
    config_cipt_b5b_entropy_rd_clip_guided.yaml \
    --algorithm CIPTDCCL --dataset PACS \
    --data_dir /home/hooasia/lgg/data/repro_dccl_data \
    --cipt_clip_path /home/hooasia/.cache/clip/ViT-B-16.pt \
    --output_root train_output/ablation/rd2_clip_guided_crossdomain \
    --deterministic --seed 0 --trial_seed 0 --aug 0 \
    --checkpoint_freq 100
```

不指定 `--test_envs`，按 `train_all.py` 的 PACS 留一域流程依次执行 4 个目标域。

控制变量消融：只替换 `--cipt_contrastive_type cross_domain_supcon` 或 `neighbor_retention`，其它参数（尤其 tau、warmup、权重、RD）保持一致。当前新预设为温度 0.1、warmup 0、contrast weight 1.5、RD weight 2、selected-center / entropy40、E/S=1/1、IND 关闭、TDA 关闭、SWAD LossValley。若与旧 RD 脚本比较，旧脚本温度 0.07、warmup 500；需要显式统一这两个变量才能作为严格对照。

## 回归测试

```bash
python -m unittest discover -s tests -p 'test_cipt_*.py' -v
```

其中 `test_cipt_clip_guided_contrastive.py` 验证数学/梯度/教师分离/正负中性样本；`test_cipt_rd_crossdomain_integration.py` 验证新损失与 selected-anchor RD 在同一次真实更新中同时生效，及禁用/等权兼容。

以上是仓库内新增的测试，不代表在老师的 GPU 服务器上已完成训练验证。
