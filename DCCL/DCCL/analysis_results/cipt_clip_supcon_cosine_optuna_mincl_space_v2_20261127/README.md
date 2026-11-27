# cipt_clip_supcon_cosine_optuna_mincl_space_v2_20261127

| 数据集 | 已完成 Trial | 已完成目标域 | 运行状态 |
| --- | ---: | ---: | --- |
| OfficeHome | 5 | 24 | running |
| PACS | 15 | 62 | running |
| VLCS | 1 | 7 | running |

- [同步时间和运行进度](snapshot.json)
- [Trial 汇总](trials_summary.csv) · [目标域汇总](folds_summary.csv)
- [Source Val 最优](best_source_val.json) · [IID 最优](best_iid.json)
- [训练协议](protocol.json) · [日志指标覆盖](metrics_coverage.json)

正式超参数和 checkpoint 选择依据 Source Val。IID 单独报告。
稳定的训练日志和已写完的评估记录在各 Trial 目录保存。缺少的指标留空。
