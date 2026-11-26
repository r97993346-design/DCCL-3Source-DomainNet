# CLIP SupCon Optuna 运行快照

采集时间 2026-11-26T21:10:30.868985+08:00。已完成的折包含完整验证结果和训练日志。仍在运行的折只记录最近一次评估进度。

| 数据集 | 已完成 Trial | 已完成折 | 状态 |
| --- | ---: | ---: | --- |
| PACS | 0 | 3 | Trial 0 · Sketch · Step 4200 |
| VLCS | 0 | 0 | 等待旧版 VLCS 结束 |
| OfficeHome | 0 | 0 | Trial 0 · Art · Step 3300 |

- [分折结果](folds_summary.csv)
- [Trial 汇总](trials_summary.csv)
- [运行状态](snapshot.json)
- [训练协议](protocol.json) · [评估缓存版本](eval_visual_cache_protocol.json)

分折指标取 Source Val 最高的 checkpoint。只有完整完成四折的 Trial 才报告四域平均分。
