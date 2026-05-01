import os
import re
import csv

LOG_DIR = "logs_3source_one_target"
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

def extract_acc(text):
    patterns = [
        r"SWAD.*?Avg:\s*([0-9.]+)%",
        r"Avg:\s*([0-9.]+)%"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return float(matches[-1])

    return None

def parse_filename(filename):
    # example: dccl_t5_s012.log
    name = filename.replace(".log", "")
    method, tag = name.split("_", 1)

    target_part, source_part = tag.split("_")
    target = int(target_part.replace("t", ""))
    sources = [int(x) for x in source_part.replace("s", "")]

    return method, target, sources

records = {}

for filename in os.listdir(LOG_DIR):
    if not filename.endswith(".log"):
        continue

    path = os.path.join(LOG_DIR, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    acc = extract_acc(text)

    if acc is None:
        continue

    method, target, sources = parse_filename(filename)
    key = (target, tuple(sources))

    if key not in records:
        records[key] = {}

    records[key][method] = acc

rows = []

for (target, sources), values in sorted(records.items()):
    dccl_acc = values.get("dccl")
    erm_acc = values.get("erm")

    if dccl_acc is None or erm_acc is None:
        continue

    erm_error = 100.0 - erm_acc
    dccl_error = 100.0 - dccl_acc
    relative_error_drop = (erm_error - dccl_error) / erm_error * 100.0

    rows.append({
        "target_env": target,
        "target_domain": DOMAINS[target],
        "source_envs": " ".join(map(str, sources)),
        "source_domains": "+".join(DOMAINS[i] for i in sources),
        "erm_acc": erm_acc,
        "dccl_acc": dccl_acc,
        "erm_error": erm_error,
        "dccl_error": dccl_error,
        "relative_error_drop": relative_error_drop
    })

csv_path = "one_target_3source_results.csv"
md_path = "one_target_3source_results.md"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "target_env",
        "target_domain",
        "source_envs",
        "source_domains",
        "erm_acc",
        "dccl_acc",
        "erm_error",
        "dccl_error",
        "relative_error_drop"
    ])
    writer.writeheader()
    writer.writerows(rows)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("| Target | Source Domains | ERM Acc | DCCL Acc | ERM Error | DCCL Error | Relative Error Drop |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")

    for r in rows:
        f.write(
            f"| {r['target_domain']} | {r['source_domains']} | "
            f"{r['erm_acc']:.3f} | {r['dccl_acc']:.3f} | "
            f"{r['erm_error']:.3f} | {r['dccl_error']:.3f} | "
            f"{r['relative_error_drop']:.2f}% |\n"
        )

print("\n=== One Target 3-Source Results ===\n")

for r in rows:
    print(
        f"Target: {r['target_domain']:10s} | "
        f"Sources: {r['source_domains']:40s} | "
        f"ERM Acc: {r['erm_acc']:.3f} | "
        f"DCCL Acc: {r['dccl_acc']:.3f} | "
        f"Relative Error Drop: {r['relative_error_drop']:.2f}%"
    )

if rows:
    avg_erm = sum(r["erm_acc"] for r in rows) / len(rows)
    avg_dccl = sum(r["dccl_acc"] for r in rows) / len(rows)
    avg_drop = sum(r["relative_error_drop"] for r in rows) / len(rows)

    print("\n=== Average ===")
    print(f"Avg ERM Acc: {avg_erm:.3f}")
    print(f"Avg DCCL Acc: {avg_dccl:.3f}")
    print(f"Avg Relative Error Drop: {avg_drop:.2f}%")

print(f"\nSaved CSV: {csv_path}")
print(f"Saved Markdown table: {md_path}")
