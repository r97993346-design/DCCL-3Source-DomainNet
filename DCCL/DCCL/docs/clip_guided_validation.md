# CLIP 引导跨域对比学习：本次验证记录

- 验证日期：2026-10-21（Asia/Shanghai）。
- 仓库：`r97993346-design/DCCL-3Source-DomainNet`。
- 基分支：`feature/cipt-b5b-entropy-mcc-rd`。
- 基提交：`74e7ad622593318961fb108788f782efa2bee4d2`。
- 新分支：`feature/cipt-clip-guided-crossdomain`。
- 环境：Python 3.12.14、PyTorch `2.14.1+cpu`，CUDA 设备数 0。

## 已执行并通过

实际 CPU 命令（前缀为此次执行环境的现成 CPU PyTorch 和临时测试依赖目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/swad_audit_deps:/tmp/cipt_cg_test_deps \
python -m unittest discover -s tests -p 'test_cipt_*.py' -v
```

结果：`Ran 104 tests in 17.218s`，`OK`。其中新测试 20 项，原测试 84 项，
覆盖旧 SupCon、neighbor_retention、env_kl、MCC/RD、SIGM、融合 CE、CLIP CE
以及原诊断工具，最终完整测试无失败或错误。

重点验证：

- 独立逐正样本 CE 公式与新损失的数值和 E 梯度相符。
- 同域同类及未入选同类不进入分母；所有异类作为负样本。
- rho=0、Top-K=0 与等权跨域 SupCon 的损失、梯度和优化器更新完全一致。
- 权重非负、有限、逐有效锚点归一化；教师无梯度，E 有正常梯度。
- B=0/1、单源域、全批无跨域同类、部分锚点无正样本均安全处理。
- 同步置换 V、E、标签、域 ID 后损失不变（含非并列 Top-K）。
- float32/float16/bfloat16 输入及 CPU autocast 下损失/梯度有限。
- 实际生产构造器及更新链中，E_CE 仅调用一次，图像编码仅一次；未调用
  IND、环境融合、MCC/RD、第二视图、投影头或表示正则。
- 关闭对比时，新旧四种对比模式产生相同的 E/S 损失和参数更新。
- 对比项只改变 E 梯度，S 和类别文本提示的梯度与禁用对比时一致。
- E-only 推理不调用教师正样本选择或环境融合。
- 新参数注册、新 YAML、CLI 覆盖以及初始实际生效系数的打印均正常。

集成测试使用微型冻结 CLIP 权重与分词替身，实际 PromptLearner、E/S adapters、
优化器、构造器和 update/predict 代码均执行。这些结果证明代码执行和梯度约束，
不代表真实 ViT-B/16 或 PACS 的精度验证。

额外检查通过：`git diff --check`；新增/修改 Python 文件编译检查；文档 Bash
命令语法检查。与基提交逐一核对，旧 `_single_view_supcon` AST 完全相同，
`cipt_neighbor_contrastive.py`、`cipt_env_kl.py`、`cipt_modules.py`、
`cipt_prompt.py`、`trainer.py` 和原实验 YAML 内容未变。

## CUDA 验证未完成

CPU 完整测试通过后，已尝试：

```bash
PYTHONPATH=/tmp/swad_audit_deps:/tmp/cipt_cg_test_deps \
python tests/test_cipt_clip_guided_contrastive.py --cuda-smoke
```

进程以状态 1 明确退出：

```text
RuntimeError: CUDA smoke test NOT RUN: no CUDA device available
```

此项由执行环境阻塞，未报告为通过。真实 CUDA/ViT-B/16/PACS 十步 smoke 和
完整 PACS 消融也未在此环境运行。服务器上可按 `clip_guided_crossdomain.md`
中的命令依次完成 CUDA 合成短程测试与真实数据 smoke；CUDA 验收仍待完成。

## 本次修改范围

新增：

- `domainbed/algorithms/cipt_clip_guided_contrastive.py`
- `tests/test_cipt_clip_guided_contrastive.py`
- `config_cipt_clip_guided.yaml`
- `docs/clip_guided_crossdomain.md`
- `docs/clip_guided_validation.md`

修改：

- `domainbed/algorithms/cipt_dccl_ablation.py`：接入两种新模式，保留四元组入口，
  加入 detach 统计和实际系数启动日志。
- `domainbed/hparams_registry.py`：注册三个教师参数，旧模式默认值不变。
- `tests/test_cipt_neighbor_contrastive.py`：给原 AST 测试加载器提供新模块依赖，
  原测试断言不变。
- `tests/test_cipt_env_anchor.py`：给真实构造器 AST 测试加载器提供相同依赖，
  原测试断言不变。
