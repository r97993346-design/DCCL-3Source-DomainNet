"""Compare all four deployed image trees with the known complete backup."""
import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/datasets/office_home")
reference = json.loads(Path(__file__).with_name("officehome_reference.json").read_text())
for domain, expected in reference.items():
    folder = root / domain
    assert folder.is_dir(), f"Missing OfficeHome domain: {folder}"
    classes = [directory for directory in folder.iterdir() if directory.is_dir()]
    assert len(classes) == 65, f"{domain}: expected 65 class directories"
    rows = sorted((str(path.relative_to(root)),path) for path in folder.rglob("*") if path.is_file())
    assert len(rows) == expected["files"], f"{domain}: expected {expected['files']} files, found {len(rows)}"
    combined = hashlib.sha256()
    for relative, path in rows:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        combined.update(f"{relative} {digest}\n".encode())
    assert combined.hexdigest() == expected["sha256"], f"{domain}: data checksum mismatch"
    print(f"{domain}: 65 classes, {len(rows)} files verified", flush=True)
