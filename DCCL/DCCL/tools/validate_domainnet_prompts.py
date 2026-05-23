import argparse
import json
from pathlib import Path


def _non_empty_str_list(values):
    if not isinstance(values, list):
        return []
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Validate DomainNet CSR prompt JSON.")
    parser.add_argument("--class_order_json", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--strict_causal", action="store_true", help="Require >=2 non-empty causal prompts per class.")
    args = parser.parse_args()

    class_order = json.loads(Path(args.class_order_json).read_text(encoding="utf-8"))
    prompt_obj = json.loads(Path(args.prompt_json).read_text(encoding="utf-8"))

    expected = class_order.get("class_names")
    got = prompt_obj.get("class_names")
    if not isinstance(expected, list):
        raise ValueError("class_order_json must contain `class_names` list.")
    if not isinstance(got, list):
        raise ValueError("prompt_json must contain `class_names` list.")
    if len(expected) != 345:
        raise ValueError(f"Expected class_order size 345, got {len(expected)}.")
    if len(got) != 345:
        raise ValueError(f"Expected prompt_json class_names size 345, got {len(got)}.")
    if got != expected:
        raise ValueError("class_names order mismatch between class_order_json and prompt_json.")

    prompts = prompt_obj.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("prompt_json must contain dict `prompts`.")

    expected_set = set(expected)
    prompt_set = set(prompts.keys())
    missing = sorted(expected_set - prompt_set)
    extra = sorted(prompt_set - expected_set)
    if missing:
        raise ValueError(f"Missing prompt entries for classes: {missing[:10]} (total={len(missing)})")
    if extra:
        raise ValueError(f"Unexpected extra prompt entries: {extra[:10]} (total={len(extra)})")

    for cname in expected:
        item = prompts[cname]
        if not isinstance(item, dict):
            raise ValueError(f"prompts[{cname}] must be a dict.")
        c_list = _non_empty_str_list(item.get("causal_prompts", []))
        s_list = _non_empty_str_list(item.get("spurious_prompts", []))
        if args.strict_causal and len(c_list) < 2:
            raise ValueError(f"{cname}: requires >=2 non-empty causal_prompts in strict mode.")
        if len(s_list) < 1:
            raise ValueError(f"{cname}: requires >=1 non-empty spurious_prompts.")

    print("Validation passed.")


if __name__ == "__main__":
    main()
