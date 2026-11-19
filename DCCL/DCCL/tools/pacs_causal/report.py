"""Produce the paired PACS result tables and a Chinese research report."""
import csv
import re
import statistics

from tools.pacs_causal.queue import DOMAINS, GROUPS, ROOT, completed, folds


def csv_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validation_pair(index, commit):
    runs, scores, signatures = {}, {}, {}
    for group in GROUPS:
        run, = folds(group, index)
        scores[group] = completed(run, group, index, commit)
        log = (run / "log.txt").read_text(encoding="utf-8")
        signatures[group] = re.search(r"PACS paired split (.+), initial weights (\{.+\})", log).groups()
        runs[group] = run
    if signatures["P0"] != signatures["P1"]:
        raise ValueError(f"P0/P1 training split or initialization mismatch on {DOMAINS[index]}")
    return runs, scores, signatures["P1"][0]


def train_metrics(index, group, fold, split_hash, commit):
    iid, swad = fold["iid_checkpoint"], fold["swad_evaluation"]
    result = dict(held_out=DOMAINS[index], group=group, iid=iid["test_in"],
                  iid_test_out=iid["test_out"], iid_source_validation=iid["train_out"],
                  iid_step=iid["step"], swad=swad["metrics"]["test_in"],
                  swad_test_out=swad["metrics"]["test_out"], swad_source_validation=swad["metrics"]["train_out"],
                  swad_converged=swad["converged"], fallback=not swad["converged"],
                  dead_valley=swad["dead_valley"], swad_converge_step=swad["converge_step"],
                  swad_start_step=swad["start"], swad_end_step=swad["end"],
                  swad_average_count=swad["count"], stop_update=swad["stop_update"], train_commit=commit,
                  split_sha256=split_hash)
    return result


def write_report(output, train_rows, diagnostics):
    mean = statistics.mean
    pc = lambda x: f"{100*x:.3f}%"
    pp = lambda x: f"{100*x:+.3f} 个百分点"
    accum = {group: [row for row in train_rows if row["group"] == group] for group in GROUPS}
    avg = {g: {key: mean(v[key] for v in rows) for key in ("iid", "swad", "iid_test_out", "swad_test_out")}
           for g, rows in accum.items()}
    normal = all(r["swad_converged"] for r in train_rows)
    features = list(diagnostics[0]["probes"])
    probe_means = {rep: {k: mean(row["probes"][rep][k]["acc"] for row in diagnostics)
                         for k in ("category", "domain")} for rep in features}
    dist_keys = ("z", "E_beta", "E_full_res", "E_orth")
    dist_means = {rep: {k: mean(row["geometry"][rep][k] for row in diagnostics)
                        for k in diagnostics[0]["geometry"][rep]} for rep in dist_keys}
    energy_means = {k: mean(row["energies"][k] for row in diagnostics) for k in diagnostics[0]["energies"]}
    styles = sorted({item["style"] for item in diagnostics[0]["perturbations"]})
    iid_delta = avg["P1"]["iid"] - avg["P0"]["iid"]
    swad_delta = avg["P1"]["swad"] - avg["P0"]["swad"]
    e_cls = probe_means["E_beta"]["category"] - probe_means["z"]["category"]
    e_dom = probe_means["E_beta"]["domain"] - probe_means["z"]["domain"]
    hard_cls = probe_means["E_full_res"]["category"] - probe_means["E_beta"]["category"]
    hard_dom = probe_means["E_full_res"]["domain"] - probe_means["E_beta"]["domain"]

    parts = [
        "# PACS 上 DICM 因果分解验证",
        "",
        "## 四域目标域分类",
        "",
        "P0 采用 A1 恒等映射，以预训练 ResNet-50 的特征 z 直接分类，只优化 CE。P1 采用 A2，冻结参考 ResNet-50，"
        "利用 PIRE 和 CDRM 响应通过 ISR 学习 rank=8 的 Q，使用 E_beta 分类并优化 CE + 0.05 ISR。"
        "残差投影 β=0.75，增广/域差分 0.25/0.75，Basis LR 为 1e-4。两组均关闭 GT 和全部对比学习。",
        "",
        "每折独立初始化，Adam，Backbone LR 为 5e-5，三个源域各 batch 32。seed=trial_seed=0，"
        "holdout=0.2，check=100，最大 5001 次更新，LossValley 3/6/0.3 由源域验证损失确定平均区间和早停。"
        "每个配对折的数据划分、预训练编码器和分类器初值哈希一致。IID 按源域验证准确率选择，单独报告目标域 test_in 和 test_out。",
        "",
        "| 目标域 | 模型 | IID test_in | IID test_out | 源域最佳 | SWAD test_in | SWAD test_out | 停止更新数 | SWAD 状态 | 平均区间 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in train_rows:
        status = "收敛/早停" if r["dead_valley"] and r["swad_converged"] else (
            "收敛/达到预算" if r["swad_converged"] else "未收敛/回退")
        fields = [r["held_out"], r["group"], pc(r["iid"]), pc(r["iid_test_out"]),
                  pc(r["iid_source_validation"]), pc(r["swad"]), pc(r["swad_test_out"]),
                  r["stop_update"], status, f"{r['swad_start_step']}–{r['swad_end_step']}"]
        parts.append("| " + " | ".join(map(str, fields)) + " |")
    parts += ["", f"P0 四域 IID 均值 **{pc(avg['P0']['iid'])}**，P1 为 **{pc(avg['P1']['iid'])}**，差值 {pp(iid_delta)}。"
              f"目标域 IID test_out 四域均值分别为 {pc(avg['P0']['iid_test_out'])} 和 {pc(avg['P1']['iid_test_out'])}。"]
    if normal:
        parts += [f"P0 四域 SWAD 均值 **{pc(avg['P0']['swad'])}**，P1 为 **{pc(avg['P1']['swad'])}**，差值 {pp(swad_delta)}。"
                  f"目标域 SWAD test_out 均值分别为 {pc(avg['P0']['swad_test_out'])} 和 {pc(avg['P1']['swad_test_out'])}。"]
    else:
        parts += [f"存在未收敛回退折。包含回退的四域最终输出均值为 P0 {pc(avg['P0']['swad'])}、"
                  f"P1 {pc(avg['P1']['swad'])}。正常收敛折请查表；缺少可覆盖两组四域的正常 SWAD 比较。"]
    parts += [
        "", "## 冻结特征的类别和域探针", "",
        "四折各自使用 P1 源域验证最佳 IID 权重。两个探针均为固定 L2 正则的逻辑回归，仅在源域训练部分拟合。"
        "类别探针使用完整源域拆分，域探针在训练和验证内分别对所有类别、源域组合等量抽样，三源域随机水平约 33.3%。"
        "所有表示共享样本 ID，标准化从各自训练特征拟合，未使用目标域标签。",
        "",
        "| 表示 | 类别识别四折均值 | 源域识别四折均值 |",
        "|---|---:|---:|",
    ]
    for rep, scores in probe_means.items():
        parts.append(f"| {rep} | {pc(scores['category'])} | {pc(scores['domain'])} |")
    parts += ["", f"E_beta 对应的类别识别变化为 {pp(e_cls)}，源域识别变化为 {pp(e_dom)}。"
              f"完整残差去除 E_full_res 相对 E_beta 的类别变化为 {pp(hard_cls)}，域变化为 {pp(hard_dom)}。"
              f"实际移除分量 S_removed 单独识别类别为 {pc(probe_means['S_removed']['category'])}，"
              f"识别域为 {pc(probe_means['S_removed']['domain'])}。", ""]
    for rec in diagnostics:
        z, e = rec["probes"]["z"], rec["probes"]["E_beta"]
        counts = rec["counts"]
        parts.append(f"- {rec['held_out']}：类别 {pc(z['category']['acc'])} → {pc(e['category']['acc'])}，"
                     f"域 {pc(z['domain']['acc'])} → {pc(e['domain']['acc'])}。"
                     f"每个类别×源域单元用于域探针拟合/评分的样本数为 "
                     f"{counts['domain_train_per_cell']}/{counts['domain_validation_per_cell']}。")
    parts += ["", "## 学习 Q、随机 Q 与留出干预", "",
              "每折冻结同一 P1 Backbone，按相同 rank 和 β 比较 12 个固定种子的随机正交基。"
              "对平衡后的源域验证样本施加固定高斯模糊和 4-bit 色阶量化，并计算 Q 对微调残差差分 Δr "
              "及原始特征差分 Δz 的投影能量比例。两种干预均在模型训练外构造。", "",
              "| 干预 | Δr 学习 Q | Δr 随机 Q 均值 | 超过该折全部 12 个随机 Q 的折数 |",
              "|---|---:|---:|---:|"]
    favorable = []
    for style in styles:
        learned, random_means, n_win = [], [], 0
        for record in diagnostics:
            contrasts = [v for v in record["perturbations"]
                         if v["style"] == style and v["response"] == "delta_r"]
            actual, = [v["mean"] for v in contrasts if v["q"] == "learned"]
            ref = [v["mean"] for v in contrasts if v["q"] != "learned"]
            learned.append(actual)
            random_means.append(mean(ref))
            n_win += int(actual > max(ref))
        favorable.append(mean(learned) > mean(random_means))
        parts.append(f"| {style} | {pc(mean(learned))} | {pc(mean(random_means))} | {n_win}/4 |")
    learned_domain = mean(row["q_controls"][0]["stable_domain"] for row in diagnostics)
    random_domain = mean(mean(v["stable_domain"] for v in row["q_controls"][1:]) for row in diagnostics)
    learned_category = mean(row["q_controls"][0]["stable_category"] for row in diagnostics)
    random_category = mean(mean(v["stable_category"] for v in row["q_controls"][1:]) for row in diagnostics)
    parts += ["", f"学习 Q 与随机 Q 的保留表示域识别均值分别为 {pc(learned_domain)} 和 {pc(random_domain)}，"
              f"类别识别均值分别为 {pc(learned_category)} 和 {pc(random_category)}。"
              "各折的随机基结果、分位数和 Δz 覆盖率见 CSV。",
              "", "## 同类跨域一致性与实际移除量", "",
              "类条件统计在源域验证集上按类别和源域配对平均。距离比较同时列出原始平方欧氏距离和逐样本单位化距离。"
              "协方差差异采用训练 z 拟合的共同 48 维 PCA 后的相对 Frobenius 距离。类间/类内比用于追踪类分离。"
              "该组诊断参考 XDomainMix 的表示不变性与图 3、图 4 的移除、可视化思路。"
              " [Cross-Domain Feature Augmentation for Domain Generalization](https://arxiv.org/html/2405.08586v1#S4.SS4)",
              "", "| 表示 | 同类跨域平方 L2 | 单位化平方 L2 | 相对协方差差异 | 类间/类内 |",
              "|---|---:|---:|---:|---:|"]
    for rep, values in dist_means.items():
        parts.append(f"| {rep} | {values['same_class_cross_domain_ms_l2']:.4f} | "
                     f"{values['same_class_cross_domain_ms_unit_l2']:.4f} | "
                     f"{values['same_class_covariance_relative_frobenius_pca48']:.4f} | "
                     f"{values['between_over_within_class']:.4f} |")
    parts += ["", f"S_raw=QQᵀ(z−z_ref)，实际削弱量 S_removed=0.75·S_raw，其能量占 z 的 "
              f"{pc(energy_means['removed_energy_over_z'])}。"
              f"E_beta 与 S_removed 的平均绝对余弦为 {energy_means['mean_abs_cos_E_beta_removed']:.4f}。"
              "E_full_res=(I−QQᵀ)z+QQᵀz_ref，E_orth=(I−QQᵀ)z。"
              f"两者在 Q 内的能量与 z 总能量之比分别为 "
              f"{pc(energy_means['e_full_res_projected_energy_fraction'])} 和 "
              f"{pc(energy_means['e_orth_projected_energy_fraction'])}。",
              "", "固定 P1 分类头的源域验证准确率随残差系数变化。以下四折均值使用同一权重，属于离线诊断。",
              "", "| β | 0 | 0.25 | 0.5 | 0.75 | 1 |", "|---|---:|---:|---:|---:|---:|"]
    parts.append("| 源域验证准确率 | " + " | ".join(
        pc(mean(d["source_validation_alpha"][a] for d in diagnostics)) for a in ("0", "0.25", "0.5", "0.75", "1")) + " |")
    parts += ["", "## 四折共享坐标图", "",
              "每个目标域单独合并五种表示，使用一次共同预处理、PCA 和 t-SNE 拟合。"
              "该折的三张图复用同一投影和轴范围。横纵轴仅用于观察同一折的局部邻域。",
              "", "| 目标域 | 类别分布 | 源域分面 | 去除强度 | 原始坐标 |", "|---|---|---|---|---|"]
    for name in DOMAINS:
        folder = f"../visualizations/{name}"
        parts.append(f"| {name} | [PNG]({folder}/category_views.png) | [PNG]({folder}/domain_views.png) | "
                     f"[PNG]({folder}/removal_views.png) | [CSV]({folder}/tsne_coordinates.csv) |")
    conclusion = ("两种预设干预的平均投影覆盖率均高于随机基，支持 Q 对这些环境变化的响应具有特异性。"
                  if all(favorable) else "有预设干预的平均覆盖率未高于随机基。当前结果只支持有限范围的环境敏感性。")
    trial_judgement = ("建议保留当前分解进入多随机种子复验。"
                       if normal and swad_delta > 0 and e_cls >= -.01 and e_dom < 0
                       else "暂不把分解作为已取得稳定泛化增益的模块。应先复验退化的域与类别，再决定保留。")
    parts += ["", "## 结论", "",
              f"1. **Q 是否捕获环境敏感方向。** {conclusion} 证据范围是这组固定干预和三个源域的可辨识性。",
              f"2. **域信息变化。** E_beta 的平衡域探针较 z 变化 {pp(e_dom)}。"
              f"相对于随机 Q 的域识别变化为 {pp(learned_domain-random_domain)}。",
              f"3. **类别信息保留。** E_beta 的类别探针较 z 变化 {pp(e_cls)}。"
              f"移除分量仍可达到 {pc(probe_means['S_removed']['category'])} 的类别识别。",
              f"4. **完全去除。** E_full_res 相对 E_beta 的类别/域变化为 {pp(hard_cls)} / {pp(hard_dom)}。"
              "是否更合适取决于类别损失和域降幅，离线结果只适用于当前冻结模型。",
              f"5. **是否保留。** {trial_judgement} 当前为单随机种子探索。"
              "P0/P1 衡量分解整套机制的增量。随机基对照固定 Backbone，因此没有估计随机 Q 重训练后的泛化效果。"
              "这些实验检验表征和域敏感性，无法单独建立严格因果识别。", "",
              "完整表格见 [目标域性能](pacs_performance.csv)、[线性探针](linear_probes.csv)、"
              "[随机基对照](random_q.csv)、[干预覆盖](intervention_coverage.csv) 和 [跨域几何](geometry.csv)。", ""]
    output.write_text("\n".join(parts), encoding="utf-8")


def reports(train_rows, diagnostics):
    directory = ROOT / "reports"
    csv_rows(directory / "pacs_performance.csv", train_rows)
    probe, q, coverage, geometry, energies, alpha = [], [], [], [], [], []
    for row in diagnostics:
        target = row["held_out"]
        for key, probes in row["probes"].items():
            for name, score in probes.items():
                probe.append(dict(held_out=target, representation=key, probe=name, **score))
        for value in row["q_controls"]:
            q.append(dict(held_out=target, **value))
        for value in row["perturbations"]:
            coverage.append(dict(held_out=target, **value))
        for key, stats in row["geometry"].items():
            geometry.append(dict(held_out=target, representation=key, **stats))
        energies.append(dict(held_out=target, **row["energies"]))
        for beta, accuracy in row["source_validation_alpha"].items():
            alpha.append(dict(held_out=target, beta=beta, source_validation_accuracy=accuracy))
    for name, values in (("linear_probes", probe), ("random_q", q), ("intervention_coverage", coverage),
                         ("geometry", geometry), ("energies", energies), ("alpha_scan", alpha)):
        csv_rows(directory / (name + ".csv"), values)
    write_report(directory / "PACS-DICM-研究报告.md", train_rows, diagnostics)
