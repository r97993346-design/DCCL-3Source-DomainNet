# CLIP SupCon Optuna 0.001 版已完成结果

采集时间 2026-11-27T12:16:26.349425+08:00。每个已完成 Trial 包含四个目标域的配置、验证结果和训练日志。仍在运行的 Trial 只记录最近一次评估进度。

| 数据集 | 已完成 Trial | 已完成折 | 状态 |
| --- | ---: | ---: | --- |
| PACS | 7 | 28 | Trial 7 · Cartoon · Step 3300 |
| VLCS | 2 | 8 | Trial 2 · Caltech · Step 3000 |
| OfficeHome | 2 | 8 | Trial 2 · Product |

- [已完成 Trial 的分折结果](folds_summary.csv)
- [Trial 汇总](trials_summary.csv)
- [运行状态](snapshot.json)
- [训练协议](protocol.json) · [评估缓存版本](eval_visual_cache_protocol.json)

分折指标取 Source Val 最高的 checkpoint。只有完整完成四折的 Trial 才报告四域平均分。
