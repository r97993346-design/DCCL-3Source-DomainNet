import json
from pathlib import Path

import torch

from tools.build_causal_spurious_embeddings import (
    build_embeddings,
    load_prompt_json,
    save_embedding_file,
    validate_prompts,
)


class DummyTextEncoder(torch.nn.Module):
    def encode_text(self, tokens):
        # tokens: [N, L] -> [N, 8]
        x = tokens.float().mean(dim=1, keepdim=True)
        base = torch.arange(1, 9, device=tokens.device).float().unsqueeze(0)
        return x * base


def dummy_tokenize(texts):
    rows = []
    for t in texts:
        vals = [ord(c) % 97 for c in t[:12]]
        vals += [0] * (12 - len(vals))
        rows.append(vals)
    return torch.tensor(rows, dtype=torch.long)


def test_build_and_save_embeddings_from_example_json(tmp_path):
    src = Path("DCCL/DCCL/assets/prompts/example_causal_spurious_prompts.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    in_json = tmp_path / "in.json"
    in_json.write_text(json.dumps(data), encoding="utf-8")

    class_names, prompts = load_prompt_json(in_json)
    validate_prompts(class_names, prompts)

    model = DummyTextEncoder().eval()
    causal, spurious = build_embeddings(model, dummy_tokenize, class_names, prompts, device="cpu")
    assert causal.shape == spurious.shape
    assert causal.shape[0] == len(class_names)
    assert torch.isfinite(causal).all()
    assert torch.isfinite(spurious).all()

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / "causal_embeddings.pt"
    s_path = out_dir / "spurious_embeddings.pt"
    save_embedding_file(c_path, causal, class_names, "Dummy", "causal", str(in_json))
    save_embedding_file(s_path, spurious, class_names, "Dummy", "spurious", str(in_json))

    c_obj = torch.load(c_path, map_location="cpu")
    s_obj = torch.load(s_path, map_location="cpu")
    assert c_obj["class_names"] == class_names
    assert s_obj["class_names"] == class_names
    assert c_obj["embeddings"].shape == s_obj["embeddings"].shape
    assert torch.isfinite(c_obj["embeddings"]).all()
    assert torch.isfinite(s_obj["embeddings"]).all()
