"""Prepare a bounded loss-weight search from source validation only."""
import argparse
import json
from pathlib import Path
from statistics import mean
import time

DATASETS = ("PACS", "VLCS")
LETTERS = "abcdef"
CLOSE_PP = 0.2


def status(root, dataset):
    path = root / f"{dataset}_ef_status.json"
    if not path.exists():
        return None
    snapshot = json.loads(path.read_text(encoding="utf8"))
    if snapshot["state"] == "failed":
        raise RuntimeError(f"{dataset} Stage 2 stopped: {snapshot.get('error')}")
    return snapshot


def analyse(dataset, snapshot):
    rows = snapshot["results"]
    if len(rows) != 36 or len({(row["kind"], row["target"]) for row in rows}) != 36:
        raise ValueError(f"{dataset}: expected 36 unique verified experiment records")
    gt = 0.01 if dataset == "PACS" else 0.05
    weights = dict(zip(LETTERS, ((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (1, 1, 0), (0, 0, gt), (1, 1, gt))))
    parents = {row["target"]: row["best_checkpoint"] for row in rows if row["kind"] == "stage1"}
    if set(parents) != set(range(4)) or not all(Path(path).is_file() for path in parents.values()):
        raise ValueError(f"{dataset}: missing verified Stage 1 parents")
    evidence = {}
    for letter in LETTERS:
        group = sorted((r for r in rows if r["kind"] == f"stage2_{letter}"), key=lambda r: r["target"])
        if [r["target"] for r in group] != list(range(4)):
            raise ValueError(f"{dataset}: incomplete branch {letter}")
        if any(r["parent"] != parents[r["target"]] for r in group):
            raise ValueError(f"{dataset}: different Stage 1 parent in branch {letter}")
        scores = [100 * r["iid_source_val"] for r in group]
        evidence[letter] = {"weights": weights[letter], "domain_scores": scores,
                            "mean": mean(scores), "swad_converged": sum(not r["swad_fallback"] for r in group)}
    top = max(r["mean"] for r in evidence.values())
    incumbent = min((k for k, r in evidence.items() if top - r["mean"] <= CLOSE_PP),
                    key=lambda k: (sum(w > 0 for w in weights[k]), -evidence[k]["mean"],
                                   -min(evidence[k]["domain_scores"])))

    def paired(left, right):
        return [a - b for a, b in zip(evidence[left]["domain_scores"], evidence[right]["domain_scores"])]

    def has_signal(delta):
        return mean(delta) > 0 and sum(d > 0 for d in delta) >= 2

    standalones = [k for k in "bc" if has_signal(paired(k, "a"))]
    if standalones:
        axis = max(standalones, key=lambda k: evidence[k]["mean"])
        shortlist = [(0.25, 0, 0), (0.5, 0, 0)] if axis == "b" else [(0, 0.5, 0), (0, 2, 0)]
        reason = ["Test neighboring weights on the stronger single contrastive module by paired source validation."] * 2
    else:
        shortlist = [(0.25, 0, 0), (0, 0.5, 0)]
        reason = ["Check a lower single-module weight where the original weight had weak source evidence."] * 2

    gt_signals = [(left, right) for left, right in (("e", "a"), ("f", "d")) if has_signal(paired(left, right))]
    if gt_signals:
        left, right = max(gt_signals, key=lambda pair: mean(paired(*pair)))
        sup, pre, _ = weights[right]
        shortlist.append((sup, pre, gt / 2))
        reason.append(f"Test a lower GT weight on the matched {right.upper()} background after {left.upper()}−{right.upper()} showed a source gain.")
    elif len(standalones) == 2:
        secondary = min(standalones, key=lambda k: evidence[k]["mean"])
        shortlist.append((0.25, 0, 0) if secondary == "b" else (0, 0.5, 0))
        reason.append("Check whether the other single-module gain survives at lower weight.")
    proposed = []
    for candidate, explanation in zip(shortlist, reason):
        if candidate not in weights.values() and all(job["weights"] != candidate for job in proposed):
            proposed.append({"weights": candidate, "reason": explanation})
    return {"domains": [r["domain"] for r in sorted((r for r in rows if r["kind"] == "stage1"), key=lambda r:r["target"])],
            "evidence": evidence, "incumbent": incumbent, "paired_effects": {
                f"{a.upper()}−{b.upper()}": paired(a, b) for a, b in
                (("b", "a"), ("c", "a"), ("e", "a"), ("f", "d"), ("f", "e"), ("d", "a"))},
            "proposed": proposed, "budget": 1000 if dataset == "PACS" else 500}


def markdown(plans):
    lines = ["# DICM next-round loss-weight search", "",
             "Only source validation accuracy selects candidates. Each score is the best source validation accuracy"
             " within its fold at the existing budget. Use the four-fold mean. Within 0.20 percentage points of"
             " the leader, prefer fewer enabled losses, then higher mean, then higher minimum source score."
             " The 0.20-point band is an operational preference, not a significance threshold.",
             "", "Weights are SupCon / PreCL / GT. A–F results are reused. SWAD coverage is recorded separately."
             " Target-domain metrics are excluded from the search input. The proposals below are pending approval.",
             "", "Keep seed and trial_seed 0, ResNet-50, existing data splits and DICM settings, batch size 32,"
             " check 100 and SWAD (3, 6, 0.3). Each candidate starts from the same per-target Stage 1"
             " source-validation-best checkpoint as A–F. GT=0 uses A4, positive GT uses A5."
             " The classifier always consumes the DICM E representation.",
             "", "In round 2, evaluate up to two previously unseen combinations per dataset, using the first"
             " round's source evidence. Change one weight or add/drop one module at a time. Use an explicit"
             " same-budget control if an extended VLCS budget is later approved.", ""]
    for dataset, plan in plans.items():
        lines += [f"## {dataset}", "", f"Current source-selected incumbent: **{plan['incumbent'].upper()}**.",
                  f"Proposed full-data budget: {plan['budget']} updates per fold.", "",
                  "| Stage 2 | Weights | " + " | ".join(plan["domains"]) + " | Source mean | Converged SWAD / 4 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k, row in plan["evidence"].items():
            scores = " | ".join(f"{v:.3f}%" for v in row["domain_scores"])
            lines.append(f"| {k.upper()} | {row['weights']} | {scores} | {row['mean']:.3f}% | {row['swad_converged']} |")
        lines += ["", "Paired changes in source validation (percentage points)", ""]
        lines += [f"- {k}: {mean(delta):+.3f} across four folds"
                  for k, delta in plan["paired_effects"].items()]
        lines += ["", "Proposed round 1", ""]
        lines += [f"- {row['weights']}. {row['reason']}" for row in plan["proposed"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        records = {d: status(args.formal_root, d) for d in DATASETS}
        if all(s and s["state"] == "completed_all_stages" for s in records.values()):
            break
        if not args.wait:
            parser.error("Both 36-fold A–F chains must be completed and archived.")
        time.sleep(60)
    plans = {d: analyse(d, s) for d, s in records.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "next_search_plan.md").write_text(markdown(plans), encoding="utf8")
    (args.output / "next_search_plan.json").write_text(
        json.dumps({"approval": "pending", "datasets": plans}, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(f"Source-validation planning finished. {args.output / 'next_search_plan.md'}", flush=True)


if __name__ == "__main__":
    main()
