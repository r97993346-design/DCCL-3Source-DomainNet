"""Offline static checks of the original CIPT + DomainBed adapter."""
from pathlib import Path
import ast
import subprocess
import unittest

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "train_cipt_official_domainbed.py"
UPSTREAM = HERE / "third_party" / "CIPT"
SHA = "a805d878acc7d79778d1ec1c1e4d73ba6aff334b"


class OfficialReproChecks(unittest.TestCase):
    def test_script_compiles(self):
        compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")

    def test_original_entrypoint_only(self):
        src = SCRIPT.read_text(encoding="utf-8")
        syntax = ast.parse(src)
        self.assertIn("from cipt import build_cipt, cosine_scheduler, evaluate, make_optimizer, train_one_epoch", src)
        self.assertIn("get_dataset(", src)
        self.assertIn("label_ratio=1.0", src)
        self.assertIn("if i != target", src)
        self.assertIn("make_optimizer(model", src)
        self.assertIn("train_one_epoch(", src)
        self.assertIn("use_tda=True", src)
        self.assertIn("cosine_scheduler(", src)
        self.assertNotIn("from domainbed.algorithms", src)
        self.assertTrue(any(isinstance(x, ast.FunctionDef) and x.name == "run_target" for x in syntax.body))

    def test_official_default_recipe(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("epochs=30, batch_size=64, lr=0.0025, beta=4.0, gamma=5.0", src)
        self.assertIn("k=4, n_ctx=16, num_heads=8, weight_decay=0.0", src)

    def test_original_submodule_pinned(self):
        gitmodules = HERE.parents[1] / ".gitmodules"
        self.assertTrue(gitmodules.exists())
        self.assertIn("https://github.com/ckghostwj/CIPT.git", gitmodules.read_text())
        self.assertTrue((UPSTREAM / "cipt" / "model.py").exists())
        head = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(head, SHA)


if __name__ == "__main__":
    unittest.main()
