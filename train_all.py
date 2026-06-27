#!/usr/bin/env python
"""Repository-root entrypoint for the DCCL training script."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DCCL_DIR = ROOT / "DCCL" / "DCCL"
for path in (ROOT, DCCL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

spec = importlib.util.spec_from_file_location("dccl_train_all", DCCL_DIR / "train_all.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main()
