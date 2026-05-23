import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from torchvision.datasets import ImageFolder


DOMAINNET_DIRNAME = "domain_net"
DOMAIN_ALIASES = {
    "clip": ["clip", "clipart"],
    "info": ["info", "infograph"],
    "paint": ["paint", "painting"],
    "quick": ["quick", "quickdraw"],
    "real": ["real", "real"],
    "sketch": ["sketch", "sketch"],
}
CANONICAL_DOMAINS = ["clip", "info", "paint", "quick", "real", "sketch"]
EXPECTED_NUM_CLASSES = 345


def _resolve_domain_path(base: Path, aliases: List[str]) -> Path:
    for a in aliases:
        p = base / a
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Cannot find domain folder under {base} for aliases={aliases}")


def _scan_domain_classes(domain_path: Path):
    ds = ImageFolder(str(domain_path))
    return ds.classes, ds.class_to_idx


def build_class_order(data_dir: Path):
    root = data_dir / DOMAINNET_DIRNAME
    if not root.is_dir():
        raise FileNotFoundError(f"Expected DomainNet root at: {root}")

    domain_paths: Dict[str, Path] = {}
    domain_classes: Dict[str, List[str]] = {}
    domain_sets: Dict[str, Set[str]] = {}
    for d in CANONICAL_DOMAINS:
        p = _resolve_domain_path(root, DOMAIN_ALIASES[d])
        classes, _class_to_idx = _scan_domain_classes(p)
        domain_paths[d] = p
        domain_classes[d] = classes
        domain_sets[d] = set(classes)

    ref_domain = "clip"
    ref_order = domain_classes[ref_domain]
    ref_set = domain_sets[ref_domain]

    mismatches = {}
    for d in CANONICAL_DOMAINS:
        missing = sorted(ref_set - domain_sets[d])
        extra = sorted(domain_sets[d] - ref_set)
        if missing or extra:
            mismatches[d] = {"missing_vs_ref": missing, "extra_vs_ref": extra}
    if mismatches:
        raise ValueError(
            "Domain class set mismatch across domains. "
            f"reference={ref_domain}, mismatches={json.dumps(mismatches)[:2000]}"
        )

    if len(ref_order) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} classes from {ref_domain}, got {len(ref_order)}."
        )

    return {
        "dataset": "DomainNet",
        "num_classes": EXPECTED_NUM_CLASSES,
        "class_names": ref_order,
        "source": {
            "data_dir": str(data_dir),
            "domain_root": str(root),
            "reference_domain": ref_domain,
            "domain_paths": {k: str(v) for k, v in domain_paths.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build DomainNet class order JSON from real ImageFolder classes.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data dir that contains domain_net/")
    parser.add_argument(
        "--output_json",
        type=str,
        default="DCCL/DCCL/assets/prompts/domainnet_class_order.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    obj = build_class_order(Path(args.data_dir))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    print(f"Num classes: {obj['num_classes']}")


if __name__ == "__main__":
    main()
