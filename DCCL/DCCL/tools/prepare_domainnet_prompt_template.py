import argparse
import json
from pathlib import Path


SPURIOUS_LIBRARY = [
    "The class identity should not be determined by photographic background, lighting variation, or natural image texture.",
    "The class identity should not be determined by sketch stroke patterns, line thickness, or monochrome contour style.",
    "The class identity should not be determined by painting brush texture, pigment blending, or artistic color style.",
    "The class identity should not be determined by clipart-like simplified graphics, flat fills, or icon-style rendering.",
    "The class identity should not be determined by infographic layout, symbols, labels, or decorative text elements.",
    "The class identity should not be determined by quickdraw-like rough strokes, incomplete outlines, or noisy doodle style.",
]


def load_class_order(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    class_names = obj.get("class_names")
    if not isinstance(class_names, list):
        raise ValueError("class_order_json must contain key `class_names` as a list.")
    if len(class_names) != 345:
        raise ValueError(f"DomainNet expects 345 classes, got {len(class_names)}")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class names found in class_order_json.")
    return class_names


def build_template(class_names):
    prompts = {
        cname: {
            "causal_prompts": [],
            "spurious_prompts": SPURIOUS_LIBRARY.copy(),
        }
        for cname in class_names
    }
    return {
        "dataset": "DomainNet",
        "num_classes": 345,
        "class_names": class_names,
        "prompt_protocol": {
            "name": "csr-domainnet-v1",
            "version": "1.0.0",
            "causal_policy": "Leave causal_prompts empty in draft; fill later in batches with class-defining semantics.",
            "spurious_policy": "Use shared DomainNet style/domain nuisances that should not determine class identity.",
            "notes": [
                "Do not use domain style cues as class evidence.",
                "Keep class_names order fixed for embedding index consistency.",
            ],
        },
        "prompts": prompts,
    }


def dump_batches(class_names, out_dir: Path, batch_size: int = 25):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(class_names)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_obj = {
            "dataset": "DomainNet",
            "batch_index": start // batch_size,
            "start": start,
            "end_exclusive": end,
            "class_names": class_names[start:end],
        }
        batch_path = out_dir / f"domainnet_class_batch_{start:03d}_{end-1:03d}.json"
        batch_path.write_text(json.dumps(batch_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare DomainNet CSR prompt draft template and class batches.")
    parser.add_argument("--class_order_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--batch_output_dir", type=str, default="")
    args = parser.parse_args()

    class_order_json = Path(args.class_order_json)
    output_json = Path(args.output_json)
    batch_output_dir = Path(args.batch_output_dir) if args.batch_output_dir else output_json.parent / "class_batches"

    class_names = load_class_order(class_order_json)
    template = build_template(class_names)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

    dump_batches(class_names, batch_output_dir, batch_size=args.batch_size)
    print(f"Saved draft prompt json: {output_json}")
    print(f"Saved class batches dir: {batch_output_dir}")


if __name__ == "__main__":
    main()
