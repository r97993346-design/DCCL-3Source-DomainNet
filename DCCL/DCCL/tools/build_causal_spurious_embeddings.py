import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _resolve_clip_model_name(name: str) -> str:
    # Accept common user spellings, e.g. ViT-B-32 -> ViT-B/32.
    return name.replace("-32", "/32").replace("-16", "/16")


def load_prompt_json(prompt_json: Path) -> Tuple[List[str], Dict]:
    with prompt_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    class_names = obj.get("class_names")
    prompts = obj.get("prompts")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("`class_names` must be a non-empty list.")
    if not isinstance(prompts, dict):
        raise ValueError("`prompts` must be a dict keyed by class name.")
    return class_names, prompts


def validate_prompts(class_names: List[str], prompts: Dict) -> None:
    for cname in class_names:
        if cname not in prompts:
            raise ValueError(f"Missing prompts for class `{cname}`.")
        item = prompts[cname]
        causal_list = item.get("causal_prompts", [])
        spurious_list = item.get("spurious_prompts", [])
        if not isinstance(causal_list, list) or len(causal_list) == 0:
            raise ValueError(f"`causal_prompts` for class `{cname}` must be non-empty list.")
        if not isinstance(spurious_list, list) or len(spurious_list) == 0:
            raise ValueError(f"`spurious_prompts` for class `{cname}` must be non-empty list.")


@torch.no_grad()
def encode_prompt_list(model, tokenize_fn, prompt_list: List[str], device: str) -> torch.Tensor:
    """
    Encode prompts and return a single L2-normalized class embedding: [D]
    """
    tokens = tokenize_fn(prompt_list).to(device)
    text_feats = model.encode_text(tokens).float()  # [N, D]
    text_feats = F.normalize(text_feats, dim=1)
    class_emb = text_feats.mean(dim=0, keepdim=True)  # [1, D]
    class_emb = F.normalize(class_emb, dim=1)
    return class_emb.squeeze(0).cpu()


@torch.no_grad()
def build_embeddings(model, tokenize_fn, class_names: List[str], prompts: Dict, device: str):
    causal_embs = []
    spurious_embs = []
    for cname in class_names:
        item = prompts[cname]
        causal_embs.append(encode_prompt_list(model, tokenize_fn, item["causal_prompts"], device))
        spurious_embs.append(encode_prompt_list(model, tokenize_fn, item["spurious_prompts"], device))

    causal_tensor = torch.stack(causal_embs, dim=0)      # [C, D]
    spurious_tensor = torch.stack(spurious_embs, dim=0)  # [C, D]

    if not torch.isfinite(causal_tensor).all():
        raise ValueError("Non-finite values found in causal embeddings.")
    if not torch.isfinite(spurious_tensor).all():
        raise ValueError("Non-finite values found in spurious embeddings.")
    if causal_tensor.shape != spurious_tensor.shape:
        raise ValueError(
            f"Causal/spurious embedding shape mismatch: {tuple(causal_tensor.shape)} vs {tuple(spurious_tensor.shape)}"
        )
    return causal_tensor, spurious_tensor


def save_embedding_file(path: Path, embeddings: torch.Tensor, class_names: List[str], encoder_name: str, prompt_type: str, source_json: str):
    obj = {
        "embeddings": embeddings,
        "class_names": class_names,
        "encoder_name": encoder_name,
        "prompt_type": prompt_type,
        "source_json": source_json,
    }
    torch.save(obj, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Build causal/spurious class text embeddings via CLIP text encoder.")
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai", help="Kept for compatibility; clip package uses model name only.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_json = Path(args.prompt_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, prompts = load_prompt_json(prompt_json)
    validate_prompts(class_names, prompts)

    # Reuse project CLIP dependency (package name: clip).
    import clip  # noqa: WPS433

    model_name = _resolve_clip_model_name(args.clip_model)
    model, _ = clip.load(model_name, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    causal_tensor, spurious_tensor = build_embeddings(model, clip.tokenize, class_names, prompts, args.device)

    causal_out = output_dir / "causal_embeddings.pt"
    spurious_out = output_dir / "spurious_embeddings.pt"
    save_embedding_file(causal_out, causal_tensor, class_names, model_name, "causal", str(prompt_json))
    save_embedding_file(spurious_out, spurious_tensor, class_names, model_name, "spurious", str(prompt_json))

    print(f"Saved: {causal_out}")
    print(f"Saved: {spurious_out}")
    print(f"Shape: {tuple(causal_tensor.shape)}")


if __name__ == "__main__":
    main()
