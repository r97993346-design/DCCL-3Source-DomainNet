**本次上传核对结果（2026-10-31）**

②为原始 CIPT 的 DomainBed legacy/B5B 路径，③为 legacy/B5A subject80。两组原始文件 SHA256 均与既有可视化元数据一致；上传副本保留全部非 CLIP state。原始权重各约 609 MB，导出副本各约 9.94 MB。

两组共同配置：PACS→Art，test_env=0，seed=0，trial_seed=0，holdout=0.2，lr=5e-5，Adam，beta=4，gamma=5，K=4，heads=1，TDA/DE/IND 开启，额外对比学习关闭，SWAD 关闭，普通 TE0_5000 checkpoint。

有效文本配置的区别为 b5b 与 b5a。③另含三个 clip_teacher 默认字段；当前 legacy、pure、contrastive-off 路径不启用这些对比学习功能。上传时的 checkout commit 不是训练 commit，不能用它证明两次训练的所有历史代码与随机初始化都完全相同。

**参数事实**

| 检查项 | ②原始 | ③subject80 |
|---|---:|---:|
| E 权重与 identity 参考的 Frobenius 距离 | 4.326283 | 4.314600 |
| E bias 范数 | 0.210106 | 0.202347 |
| S 权重与 identity 参考的 Frobenius 距离 | 3.972930 | 4.026515 |

当前实现以 identity/zero 初始化 E/S。相对于这个参考，两组都不是未更新的 adapter；两组全部11项可学习参数都不完全相同。

| 两组参数差异（以②该参数范数为分母） | 相对差异 |
|---|---:|
| E 权重 | 1.69% |
| S 权重 | 1.69% |
| Prompt context | 3.50% |
| TDA attention in_proj 权重 | 7.37% |
| TDA attention out_proj 权重 | 12.61% |

两组 E 权重差的范数为0.387266。它约占②E偏离identity参考的8.95%。由三个范数推得，两组 E 更新方向的余弦约0.99599；该推导仅以共同identity参考为基准，并未读取实际初始化快照。

这说明两组 E 参数非常接近但有区别。较高的原始权重余弦0.999872也会受到共同identity成分的影响。TDA out_proj的相对参数差更明显，但不同模块参数差百分比不能直接解释输出重要性或准确率变化。

**本轮尚未得到的结果**

此次上传执行的是CPU导出和参数比较，未执行源域/目标域前向评估。没有新的完整source_train/source_val/target_in/target_out准确率，也没有新的旧权重alpha消融。此前700张可视化抽样的准确率不是完整target_out准确率。

这份参数比较回答的是②与③的相似程度；尚不能回答①（0.0025上游坏权重）与②为何坍塌程度不同，或证明学习率为根因。

**完整复评已准备好**

脚本：`DCCL/DCCL/scripts/audit_cipt_checkpoint_pair.py`
启动器：`DCCL/DCCL/scripts/run_cipt_checkpoint_pair_full_eval.sh`

启动器读取服务器上已确认的两个原始checkpoint，在独立worktree中使用固定提交1e97e3d31077711015c6a2ebd0874a7d8f05652e的native DomainBed代码。它顺序完整评估两组，报告原生TDA、E-only、LN(E)、固定alpha控制以及各域/各类混淆矩阵。真实类别上下文控制仅用于②的源域诊断，不作为有效测试结果；③使用共享subject文本。

运行时自动核对alpha=1与native predict的logits、alpha=0与LN(E)、两组类别顺序与完整数据划分、两个原始checkpoint哈希。评估结束后生成comparison.csv并上传到新的analysis分支。目标数据不选择checkpoint或alpha。

目前只完成了语法、CLI、模式保护检查；本环境未执行真实checkpoint推理。需要等所选GPU空闲后在训练服务器运行。环境变量CIPT_GPU选择物理GPU，默认1。
