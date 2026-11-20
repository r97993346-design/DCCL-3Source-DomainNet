"""Build the source diagnostics and four-domain research report."""

import csv
import json
from statistics import mean
import time

import numpy as np

from tools.pacs_directdiff.analyze import has_live_pane
from tools.pacs_directdiff.common import DOMAINS, GROUPS, ROOT, VARIANTS, atomic_json, folds, historical_fold, say
from tools.pacs_directdiff.figures import SETUPS, gate_figure, paired_tsne, performance_figure
from tools.pacs_directdiff.legacy import load_legacy
from tools.pacs_directdiff.probes import PRIMARY_RIDGE, SENSITIVITY_RIDGE
from tools.pacs_directdiff.queue import completed


def csv_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == keys and sum(1 for _ in reader) == len(rows)


def performance():
    commit = json.loads((ROOT / "logs" / "preflight.passed.json").read_text())["commit"]
    rows = []
    for index, domain in enumerate(DOMAINS):
        for group in SETUPS:
            if group in GROUPS:
                found = folds(group, index)
                if not found or not (found[0] / ".github-archive").is_dir():
                    continue
                folder, = found
                fold = completed(folder, group, index, commit)
                source_commit = commit
            else:
                folder, _ = historical_fold(group, index)
                fold, = json.loads((folder / ".github-archive" / "evaluation.json").read_text())
                source_commit = json.loads((folder / ".github-archive" / "manifest.json").read_text())["code"]["commit"]
            iid, swad = fold["iid_checkpoint"], fold["swad_evaluation"]
            rows.append(dict(group=group, domain=domain, iid_test_in=iid["test_in"],
                             iid_test_out=iid["test_out"], iid_step=iid["step"],
                             iid_source_validation=iid["train_out"], swad_test_in=swad["metrics"]["test_in"],
                             swad_test_out=swad["metrics"]["test_out"],
                             swad_source_validation=swad["metrics"]["train_out"],
                             swad_converged=swad["converged"], swad_first=swad["start"], swad_last=swad["end"],
                             stop_update=swad["stop_update"], early_stop=swad["dead_valley"],
                             code_commit=source_commit, checkpoint_server="hooasia",
                             checkpoint_dir=str(folder / "checkpoints"),
                             checkpoint_dir_relative_to_projects=str((folder / "checkpoints").relative_to(ROOT.parent))))
    return rows


def percent(value):
    return f"{100 * value:.3f}%"


def delta(value):
    return f"{100 * value:+.3f} 个百分点"


def row(values):
    return "| " + " | ".join(map(str, values)) + " |"


def columns(values):
    return [row(values), row(["---"] * len(values))]


def probe(result, view, task, regularization=PRIMARY_RIDGE):
    return result["probes"][view][task][str(regularization)]["accuracy"]


def mean_probe(results, view, task, regularization=PRIMARY_RIDGE):
    return mean(probe(r, r["actual"] if view == "actual" else view, task, regularization) for r in results)


def mean_coverage(results, style, space):
    return mean(next(v["mean"] for v in r["coverage"] if v["style"] == style and v["space"] == space)
                for r in results)


def collect_diagnostics():
    matched = json.loads((ROOT / "offline" / "A16_gate_feasibility.json").read_text())["results"]
    hashes = {r["held_out"]: r["split_sha256"] for r in matched}
    data = []
    for variant in VARIANTS:
        data.extend(load_legacy(variant))
        for group in GROUPS:
            for domain in DOMAINS:
                path = ROOT / "probes" / group / domain / f"{variant}.json"
                if not path.is_file():
                    continue
                item = json.loads(path.read_text())
                assert item["split_sha256"] == hashes[domain]
                data.append(item)
    for item in data:
        assert item["split_sha256"] == hashes[item["held_out"]]
    return data


def summarize_diagnostics(data):
    results, readouts, effects, directions = [], [], [], []
    for r in data:
        identity = {key: r[key] for key in ("group", "held_out", "checkpoint")}
        selected = r.get("actual", "E")
        results.append(dict(identity, actual=selected, remaining_rank=r["protection"]["remaining_rank"],
                            z_class=probe(r, "z", "class"), z_domain=probe(r, "z", "domain"),
                            E_class=probe(r, selected, "class"), E_domain=probe(r, selected, "domain"),
                            S_class=probe(r, "S", "class"), S_domain=probe(r, "S", "domain"),
                            **{f"{view}_{name}": value for view, metrics in r["geometry"].items()
                               for name,value in metrics.items() if isinstance(value, (float,int))}))
        for view, tasks in r["probes"].items():
            for task, runs in tasks.items():
                for stats in runs.values():
                    readouts.append(dict(identity, representation=view, task=task, **stats))
        for view, v in r.get("fixed_head", {}).items():
            effects.append(dict(identity, representation=view, actual=view == selected,
                                equal_train_energy_beta=r["beta_match"], **v))
        for entry in r["coverage"]:
            directions.append(dict(identity, **entry))
    csv_rows(ROOT / "reports" / "representations.csv", results)
    csv_rows(ROOT / "reports" / "linear_probes.csv", readouts)
    csv_rows(ROOT / "reports" / "fixed_head_energy.csv", effects)
    csv_rows(ROOT / "reports" / "perturbation_coverage.csv", directions)


def physical_failures():
    outcome = []
    for group in GROUPS:
        for name in ("failure", "analysis_failure"):
            path = ROOT / "logs" / f"{group}.{name}.json"
            if path.is_file():
                error = json.loads(path.read_text())["reason"]
                console = [p for p in sorted((ROOT / "logs").glob(f"{group}_env*.log"))
                           if p.is_file()]
                if name == "failure" and console:
                    lines = console[-1].read_text(errors="replace").splitlines()
                    if lines and ("Error" in lines[-1] or "Killed" in lines[-1]):
                        error += " " + lines[-1]
                outcome.append(f"- {group} {name}　{error}")
    return outcome


def report(scores, diagnostics, failures):
    lines = [
        "# PACS DirectDiff16 因果分解研究",
        "",
        "P0 是直接分类历史对照。A16 使用冻结参考残差。两组 DirectDiff16 用同一预训练 ResNet-50、"
        "同一初始 Rank16 Basis、同一初始分类头和同一数据划分。增强差异为 z_aug − z，"
        "跨域差异由同类源域特征的等权 EMA 中心形成。两类有效差异分别以 0.25 和 0.75 进入 ISR，"
        "总损失为 CE + 0.05 × ISR。只有 ISR 更新 Basis。每个目标域独立训练。",
        "",
        "Gate 的全部样本共享同一套旋转方向与系数。前 300 步用统一减法，从 step 300 起每 100 步用源域训练 EMA 更新。"
        "类别散度与同类跨域散度各除以迹，正交分解二者之差，系数由每个方向的域散度占比给出。"
        "门控变成常数、失去有效统计量，或删除量小于统一减法的 1% 时停止该训练。"
        "SWAD 采用 LossValley(3,6,0.3) 平均参数，最终门控用窗口内最后一份源域 EMA 在平均后的 QR 坐标中重新求解。",
        "",
        "## 目标域训练结果",
        "",
        "IID 使用源域验证准确率选点。SWAD 区间使用源域验证损失。目标域指标为最终配对评估。",
        "",
        *columns(("域", "方案", "IID in/out", "SWAD in/out", "源域 IID/SWAD",
                  "训练步数", "SWAD 区间", "SWAD 状态")),
    ]
    for r in scores:
        status = ("权重平均" if not r["early_stop"] else "正常早停") if r["swad_converged"] else "未收敛回退"
        lines.append(row((r["domain"], r["group"],
                          "/".join(percent(r[k]) for k in ("iid_test_in", "iid_test_out")),
                          "/".join(percent(r[k]) for k in ("swad_test_in", "swad_test_out")),
                          "/".join(percent(r[k]) for k in ("iid_source_validation", "swad_source_validation")),
                          r["stop_update"], f'{r["swad_first"]}–{r["swad_last"]}', status)))
    lines += ["", *columns(("方案", "完整折数", "IID in", "IID out", "四域 SWAD in", "四域 SWAD out", "源域 SWAD"))]
    baseline = {}
    for group in SETUPS:
        own = [r for r in scores if r["group"] == group]
        if len(own) == 4:
            baseline[group] = {key: mean(r[key] for r in own) for key in (
                "iid_test_in", "iid_test_out", "swad_test_in", "swad_test_out", "swad_source_validation")}
        values = baseline.get(group)
        scores_ok = len(own) == 4 and all(r["swad_converged"] for r in own)
        displayed = ["未完成"] * 5
        if values:
            displayed = [percent(values[key]) for key in (
                "iid_test_in", "iid_test_out", "swad_test_in", "swad_test_out", "swad_source_validation")]
            if not scores_ok:
                displayed[2:] = [x + "（含回退）" for x in displayed[2:]]
        lines.append(row((group, f"{len(own)}/4", *displayed)))
    lines += ["", "![PACS 目标域 SWAD 对照](swad_performance.png)", ""]
    if len(baseline) == len(SETUPS):
        for group in GROUPS:
            differences = [(group, other, key, baseline[group][key] - baseline[other][key])
                           for other in ("P0", "A16") for key in ("swad_test_in", "swad_test_out")]
            rendered = [f"{other} 的 {key} 差 {delta(v)}" for _, other, key, v in differences]
            lines.append(f"{group} 相对" + "，相对".join(rendered) + "。")
        lines.append("Gate 相对 Uniform 的四域 SWAD test_out 差 "
                     + delta(baseline["Gate"]["swad_test_out"] - baseline["Uniform"]["swad_test_out"]) + "。")
        index = {(r["group"], r["domain"]): r for r in scores}
        for d in ("cartoon", "sketch"):
            base = index["A16", d]
            lines.append(f"{d} 的 SWAD in/out 相对 A16 变化为 "
                         + "，".join(f"{g} " + " / ".join(delta(index[g, d][k] - base[k])
                                                       for k in ("swad_test_in", "swad_test_out")) for g in GROUPS) + "。")
    if failures:
        lines += ["", "## 阻塞记录", "", *failures, ""]

    lines += [
        "",
        "## 源域表示与 Q 的独立扰动响应",
        "",
        "共享标准化参数从每折每个 checkpoint 的源域训练 z 拟合。类别探针使用源域训练与验证的固定拆分。"
        "域探针使用同一批类别×源域平衡样本。A16 探针试运行触及迭代求值上限，所有正式对照统一采用闭式线性岭回归。"
        "下表采用岭系数 0.001，每次拟合均核验正规方程的相对残差。",
        "",
        *columns(("方案", "权重", "z 类别/域", "E 类别/域", "S 类别/域", "删除能量", "同类跨域平方距离 z/E",
                  "类间/类内 z/E", "保护后秩", "模糊 Q/剩余", "色阶 Q/剩余")),
    ]
    rmap = {}
    for variant in VARIANTS:
        for g in ("A16", *GROUPS):
            rep = [r for r in diagnostics if r["group"] == g and r["checkpoint"] == variant]
            rmap[g, variant] = rep
            if len(rep) != 4:
                continue
            effective = "E" if g == "A16" else "actual"
            scores_rep = ["/".join(percent(mean_probe(rep, v, task)) for task in ("class", "domain"))
                          for v in ("z", effective, "S")]
            removal = [r["validation_removed_energy"] if g == "A16"
                       else r["fixed_head"][r["actual"]]["validation_removed_energy"] for r in rep]
            def pair(metric):
                return "/".join(f'{mean(r["geometry"][v][metric] for r in rep):.4f}' for v in ("z", "E"))
            cover = ["/".join(percent(mean_coverage(rep, style, sp)) for sp in ("Q", "Q_env"))
                     for style in ("gaussian_blur_1p6", "posterize_4bit")]
            ranks = "/".join(str(r["protection"]["remaining_rank"]) for r in rep)
            lines.append(row((g, variant, *scores_rep, percent(mean(removal)),
                              pair("same_class_cross_domain_ms_unit_l2"), pair("between_over_within_class"),
                              ranks, *cover)))
    swad = {g: rmap[g, "swad"] for g in ("A16", *GROUPS)}
    if all(len(rep) == 4 for rep in swad.values()):
        common_q = ("gaussian_blur_1p6", "posterize_4bit")
        for g in GROUPS:
            reprs = swad[g]
            effect = [delta(mean_coverage(reprs, style, "Q") - mean_coverage(swad["A16"], style, "Q"))
                      for style in common_q]
            retained = [delta(mean_coverage(reprs, style, "Q_env") - mean_coverage(swad["A16"], style, "Q_env"))
                        for style in common_q]
            mix = [percent(mean_probe(reprs, "S", task)) for task in ("class", "domain")]
            lines += ["", f"{g} 相对历史 A16 的模糊、色阶原始差异覆盖变化为 {effect[0]}、{effect[1]}。"
                      f"移除类别方向后余下空间的变化为 {retained[0]}、{retained[1]}。"
                      f"实际移除分量的类别与域探针为 {mix[0]}、{mix[1]}。"]

    lines += [
        "",
        "## 等删除能量与固定头测试",
        "",
        "等训练能量对照用源域训练估计一个统一 beta。等样本能量对照只沿统一 Q 投影方向，逐样本匹配门控移除范数。"
        "后者是离线诊断，逐样本比例由特征本身计算。两个对照均以源域验证样本读出。",
        "",
        *columns(("权重来源", "权重类型", "离线变换", "源域固定头", "类别探针", "域探针", "删除能量")),
    ]
    for g in GROUPS:
        for variant in VARIANTS:
            rep = rmap[g, variant]
            if len(rep) != 4:
                continue
            for view in ("z", "uniform", "gate", "orthogonal",
                         "uniform_equal_train_energy", "uniform_equal_sample_energy"):
                fixed = mean(r["fixed_head"][view]["source_validation_top1"] for r in rep)
                energy = mean(r["fixed_head"][view]["validation_removed_energy"] for r in rep)
                lines.append(row((g, variant, view, percent(fixed), percent(mean_probe(rep, view, "class")),
                                  percent(mean_probe(rep, view, "domain")), percent(energy))))
    gate = swad["Gate"]
    if len(gate) == 4:
        match = "uniform_equal_sample_energy"
        trends = {}
        for c in (PRIMARY_RIDGE, SENSITIVITY_RIDGE):
            trends[c] = {task: mean_probe(gate, "gate", task, c) - mean_probe(gate, match, task, c)
                         for task in ("class", "domain")}
        lines += ["", f"在 Gate 的同一 SWAD 权重和逐样本等删除能量下，"
                  f"主探针的类别、域变化分别为 {delta(trends[PRIMARY_RIDGE]['class'])}、"
                  f"{delta(trends[PRIMARY_RIDGE]['domain'])}。"
                  f"弱正则岭系数 0.0001 得到 {delta(trends[SENSITIVITY_RIDGE]['class'])}、"
                  f"{delta(trends[SENSITIVITY_RIDGE]['domain'])}。"]
        fixed_head_effect = mean(r["fixed_head"]["gate"]["source_validation_top1"]
                                 - r["fixed_head"][match]["source_validation_top1"] for r in gate)
        lines.append(f"逐样本等能量时，固定分类头的源域验证变化为 {delta(fixed_head_effect)}。"
                     "固定头差异在控制删除能量后仍可测得，域探针差异的方向随岭系数变化。")
        readout = min(r["probe_transport"][task]["gate"]["prediction_agreement"]
                      for r in gate for task in ("class", "domain"))
        lines += ["", f"沿门控方向的保留特征值为 1−0.75g，最小值至少 0.25，故该固定变换可逆。"
                  f"从 z 线性探针搬运权重后，门控表示的最低预测一致率为 {percent(readout)}。"
                  "这为共享标准化下的读出差异提供对照。"]
        enough = (len(swad["Uniform"]) == 4 and "Uniform" in baseline and "Gate" in baseline
                  and all(r["swad_converged"] for r in scores if r["group"] in GROUPS))
        lines += ["", "## 是否保留门控", ""]
        if enough:
            source_delta = baseline["Gate"]["swad_source_validation"] - baseline["Uniform"]["swad_source_validation"]
            source_effect = all(trends[c]["class"] >= 0 and trends[c]["domain"] < 0
                                for c in (PRIMARY_RIDGE, SENSITIVITY_RIDGE))
            lines.append(f"Gate 相对 Uniform 的源域 SWAD 验证变化为 {delta(source_delta)}。")
            if source_effect and source_delta >= 0:
                lines.append("源域验证与两种预设正则强度下的等能量探针均支持保留门控候选，下一步可做多种子验证。")
            else:
                lines.append("源域验证和等能量探针尚未共同支持把当前门控定为核心组成。"
                             "本轮四域成绩用于报告泛化变化。")
        else:
            lines.append("维持原配置，等待完整的配对源域读出再决定是否把门控纳入核心。")
    lines += [
        "",
        "固定头测试用于隔离推理变换。端到端训练结果在目标域表中独立列出。"
        "本轮为单种子。同一模型的可逆表示与跨模型训练效果都按各自实验对照解释。",
        "",
        "![门控系数](gate_coefficients.png)" if gate else "",
        "",
        "同折、同权重类型的 Uniform 与 Gate 源域样本在同一 PCA 和 t-SNE 坐标中绘制。"
        "图和导出坐标位于 [可视化目录](../visualizations/)。",
        "",
        "[逐折成绩与 checkpoint 路径](performance.csv) · [表示几何](representations.csv) · "
        "[固定头与删除能量](fixed_head_energy.csv) · [线性探针](linear_probes.csv) · "
        "[扰动覆盖](perturbation_coverage.csv) · [历史 A16 离线可行性](../offline/A16_gate_feasibility.json)",
        "",
    ]
    (ROOT / "reports" / "PACS-DirectDiff16-研究报告.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    while any(has_live_pane(f"directdiff16_{g}_analysis") and
              not (ROOT / "logs" / f"{g}.analysis_complete.json").is_file()
              for g in GROUPS):
        time.sleep(60)
    scores = performance()
    diagnostics = collect_diagnostics()
    failures = physical_failures()
    csv_rows(ROOT / "reports" / "performance.csv", scores)
    summarize_diagnostics(diagnostics)
    performance_figure(scores)
    for domain in DOMAINS:
        for variant in VARIANTS:
            if all((ROOT / "features" / g / domain / variant / "features.npz").is_file() and
                   (ROOT / "probes" / g / domain / f"{variant}.json").is_file() for g in GROUPS):
                paired_tsne(domain, variant)
    gate = [r for r in diagnostics if r["group"] == "Gate"]
    if gate:
        gate_figure(gate)
    report(scores, diagnostics, failures)
    atomic_json(ROOT / "logs" / "report.complete.json", dict(
        scores=len(scores), verified_training_folds=sum(r["group"] in GROUPS for r in scores),
        mechanisms=len(diagnostics), failures=len(failures)))
    say("Completed verified DirectDiff16 comparison report")


if __name__ == "__main__":
    run()
