import json
from pathlib import Path

from tools.prepare_domainnet_prompt_template import build_template, dump_batches


def test_prepare_template_and_batches(tmp_path):
    class_names = [f"class_{i}" for i in range(345)]
    template = build_template(class_names)
    assert template["dataset"] == "DomainNet"
    assert template["num_classes"] == 345
    assert template["class_names"] == class_names
    assert len(template["prompts"]) == 345
    assert template["prompts"]["class_0"]["causal_prompts"] == []
    assert len(template["prompts"]["class_0"]["spurious_prompts"]) >= 6

    batch_dir = tmp_path / "class_batches"
    dump_batches(class_names, batch_dir, batch_size=25)
    files = sorted(batch_dir.glob("*.json"))
    assert len(files) == 14
    first = json.loads(files[0].read_text(encoding="utf-8"))
    last = json.loads(files[-1].read_text(encoding="utf-8"))
    assert len(first["class_names"]) == 25
    assert len(last["class_names"]) == 20
