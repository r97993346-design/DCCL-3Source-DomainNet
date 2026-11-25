"""Summaries of source-validation curves and sampled optimizer telemetry."""

from collections import Counter
from pathlib import Path
from statistics import mean, median

from protocol import LOSS_KEYS, read_rows, write_text


LOSS_LABELS = {
    "loss_total": "Total loss", "loss_e_cls": "E classification", "loss_s_uniform": "S uniform",
    "loss_condcov": "CondCov", "loss_rd": "RD", "loss_clip_contrast": "CLIP contrastive",
    "tr_outloss": "Source validation cross-entropy (tr_outloss)",
}
COSINES = {
    "ind_grad_e_cls_cosine": "CondCov vs E classification on E",
    "ind_grad_e_rd_cosine": "CondCov vs RD on E",
    "ind_grad_s_uniform_cosine": "CondCov vs S uniform on S",
}


def learning_rate_report(dataset, path, records):
    five = sorted(records, key=lambda record: record["trial_id"])[:5]
    lines = [f"# {dataset} · Adam 1e-5 learning-rate diagnosis", "",
             f"Complete observations available {len(records)}."]
    if len(five) < 5:
        lines.append(f"Diagnosis starts when five valid observations are available ({len(five)}/5).")
        write_text(path, "\n".join(lines) + "\n")
        return
    folds = [f for trial in five for f in trial["folds"]]
    traces = [read_rows(Path(f["log"])) for f in folds]
    # Historical logs may contain four folds in one file.
    traces = [[r for r in rows if r["args"]["real_test_envs"] == [fold["fold"]]]
              for rows, fold in zip(traces, folds)]
    steps = Counter(f["selected_step"] for f in folds)
    lines.extend(["", "First five trial IDs " + ", ".join(str(t["trial_id"]) for t in five) + ".",
                  "Positive late slope means the metric rose between steps 3000 and 4999.",
                  "", "| Metric | Step 0 | Step 1500 | Step 3000 | Step 4999 | Late slope per 1000 updates |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    means = {}
    for key in (*LOSS_KEYS, "train_out"):
        stats = [mean(next(r[key] for r in rows if r["step"] == step) for rows in traces)
                 for step in (0, 1500, 3000, 4999)]
        means[key] = stats
        scores = " | ".join(f"{x:.6g}" for x in [*stats, (stats[-1] - stats[-2]) * 1000 / 1999])
        lines.append(f"| {LOSS_LABELS.get(key, 'Source validation accuracy')} | {scores} |")
    lines.extend(["", "Selected checkpoint histogram " +
                  ", ".join(f"{k} ({v} folds)" for k, v in sorted(steps.items())) + "."])
    telemetry = [s for f in folds if f.get("telemetry") for s in read_rows(Path(f["telemetry"]))]
    gradients = [sample for rows in traces for r in rows for sample in r.get("ind_grad_by_step", {}).values()]
    if telemetry:
        lines += ["", "| Sample | Gradient norm | Actual Adam update norm | Relative update |",
                  "| --- | ---: | ---: | ---: |"]
        for step in sorted({r["step"] for r in telemetry}):
            values = [r for r in telemetry if r["step"] == step]
            norms = " | ".join(f"{median(r[x] for r in values):.5g}"
                               for x in ("grad_norm", "update_norm", "relative_update"))
            lines.append(f"| {step} | {norms} |")
    else:
        lines += ["", "Optimizer telemetry availability 0 of the first five observations."]
    if gradients:
        lines += ["", "| Feature-gradient pair | Median cosine | Fraction below −0.1 | Samples |",
                  "| --- | ---: | ---: | ---: |"]
        for key, label in COSINES.items():
            values = [r[key] for r in gradients if key in r]
            if values:
                lines.append(f"| {label} | {median(values):.3f} | "
                             f"{sum(v < -.1 for v in values) / len(values):.1%} | {len(values)} |")
    late = sum(n for step, n in steps.items() if step >= 4200) / len(folds)
    source_gain = means["train_out"][-1] - means["train_out"][-2]
    ce_drop = 1 - means["tr_outloss"][-1] / max(means["tr_outloss"][-2], 1e-12)
    flips = []
    for rows in traces:
        vals = [r["train_out"] for r in rows if r["step"] >= 3000]
        changes = [y - x for x, y in zip(vals, vals[1:]) if abs(y - x) >= 0.002]
        flips.append(sum(a * b < 0 for a, b in zip(changes, changes[1:])))
    lines += ["", f"Late checkpoint selections {late:.0%}. Mean late source validation gain "
              f"{source_gain * 100:+.3f} pp. Mean late source cross-entropy decrease {ce_drop:.2%}.",
              f"Median late source-validation reversals of at least 0.2 pp {median(flips):.1f}."]
    if late >= .5 and source_gain > .002 and ce_drop > .02:
        lines.append("Late improvement suggests 5,000 updates may be short at 1e-5. "
                     "A later controlled comparison at 2e-5 could test this.")
    elif median(flips) >= 2 and source_gain <= 0:
        lines.append("Late validation reversals warrant a controlled comparison at 5e-6 in a later run.")
    else:
        lines.append("These five observations support keeping 1e-5 as the reference for a later comparison.")
    if gradients:
        conflict = []
        for key, name in COSINES.items():
            vals = [r[key] for r in gradients if key in r]
            if vals and sum(v < -.1 for v in vals) / len(vals) >= .25:
                conflict.append(name)
        if conflict:
            lines.append("Recurring negative feature-gradient cosines were observed for " +
                         ", ".join(conflict) + ". Check the per-step series before linking this to learning rate.")
    write_text(path, "\n".join(lines) + "\n")
