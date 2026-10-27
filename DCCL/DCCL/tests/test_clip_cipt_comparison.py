"""Paired-control checks using synthetic fixtures, never research results."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import analyze_clip_cipt_comparison as cli
from test_cipt_mechanism_extraction import ExtractionTests, image_tensor, torch


@unittest.skipIf(torch is None, "CPU PyTorch required")
class FrozenControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ExtractionTests.setUpClass()

    def setUp(self):
        self.fixture = ExtractionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fx = self.fixture
        self.root = fx.root
        self.cmp, self.b5a = self.root / "comparison", self.root / "b5a"
        self.cmp.mkdir(); self.b5a.mkdir()
        ckpt, decomposition, _, _, _ = fx.checkpoint("b5b")
        projection = torch.diag(torch.linspace(.8, 1.2, 16))
        ckpt["model_dict"]["clip_model.image_projection"] = projection
        self.ckpt = self.root / "baseline.pth"
        torch.save(ckpt, self.ckpt)
        self.samples = cli.select_samples(fx.data, fx.names, [0, 1], ["photo", "sketch"], 2, 0)
        from PIL import Image
        images = []
        for row in self.samples:
            with Image.open(row["path"]) as image:
                images.append(image_tensor(image))
        with torch.no_grad():
            v = torch.nn.functional.normalize(torch.stack(images) @ projection, dim=-1)
            e, _ = decomposition(v)
            e = torch.nn.functional.normalize(e, dim=-1).numpy()
        self.expected_v = v.numpy()
        self.expected_e = e
        for folder, name in [(self.cmp, "paired_samples.csv"), (self.b5a, "samples.csv")]:
            cli.write_csv(folder / name, self.samples)
        meta = {"dataset": "PACS", "class_names": fx.names, "domains": ["photo", "sketch"],
                "target_domains": ["photo"], "checkpoint_sha256": cli.sha256(self.ckpt),
                "clip_sha256": cli.sha256(fx.clip_path)}
        cli.dump_json(self.cmp / "provenance.json", {"baseline_metadata": meta, "ours_metadata": {"fixture": True}})
        cli.dump_json(self.b5a / "metadata.json", {**meta, "template_mode": "b5a", "z_uses_ground_truth_class_word": False})
        np.save(self.b5a / "e_unit.npy", e)
        np.savez_compressed(self.cmp / "paired_features.npz", cipt_e=e, ours_e=e,
                            classes=[r["class_id"] for r in self.samples], domains=[r["domain"] for r in self.samples],
                            relative_paths=[r["relative_path"] for r in self.samples])

    def args(self):
        return cli.parse_args(["--checkpoint", str(self.ckpt), "--clip-path", str(self.fixture.clip_path),
            "--data-dir", str(self.fixture.data.parent), "--comparison-dir", str(self.cmp),
            "--b5a-dir", str(self.b5a), "--output-dir", str(self.root / "output"), "--device", "cpu"])

    def loader(self, name, path):
        return {"_clip_comparison_prompt": self.fixture.prompt,
                "_clip_comparison_modules": self.fixture.modules,
                "_clip_comparison_transforms": SimpleNamespace(clip_basic=image_tensor)}[name]

    def test_frozen_v_restores_saved_clip_and_reproduces_archived_e(self):
        args = self.args()
        with patch.object(cli, "load_module", side_effect=self.loader):
            cli.extract(args)
        out = Path(args.output_dir)
        np.testing.assert_allclose(np.load(out / "v_unit.npy"), self.expected_v, atol=1e-7)
        meta = json.loads((out / "metadata.json").read_text())
        self.assertTrue(meta["clip_weights_restored_from_checkpoint"])
        self.assertLess(meta["baseline_e_max_abs_reconstruction_error"], 1e-6)
        pack, meta = cli.report(out, 3)
        self.assertEqual(len(cli.read_csv(out / "metrics_summary.csv")), 4)
        cli.plots(out, pack, meta, [0], [3], 250)
        self.assertEqual(len(list((out / "tsne").glob("*.npz"))), 3)

    def test_rejects_reordered_b5a_manifest(self):
        cli.write_csv(self.b5a / "samples.csv", self.samples[::-1])
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            cli.load_inputs(self.cmp, self.b5a)

    def test_rejects_unmatched_clip_file(self):
        args = self.args()
        self.fixture.clip_path.write_bytes(b"wrong frozen weights")
        with self.assertRaisesRegex(ValueError, "clip-path hash"):
            cli.extract(args)
        self.assertFalse(Path(args.output_dir).exists())

    def test_rejects_incompatible_baseline_features_before_saving(self):
        args = self.args()
        with np.load(self.cmp / "paired_features.npz") as archive:
            pack = {k: archive[k] for k in archive.files}
        pack["cipt_e"] = np.roll(pack["cipt_e"], 1, axis=1)
        np.savez_compressed(self.cmp / "paired_features.npz", **pack)
        with patch.object(cli, "load_module", side_effect=self.loader):
            with self.assertRaises(AssertionError):
                cli.extract(args)
        self.assertFalse(Path(args.output_dir).exists())


if __name__ == "__main__":
    unittest.main()
