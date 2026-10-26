"""CPU extraction tests with real E/TDA modules and a synthetic CLIP stand-in.

These checkpoint/image fixtures validate implementation, never model quality.
PyTorch is optional for report-only environments; it is required to run these tests.
"""

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze_cipt_mechanism as cli

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def tokenize(texts):
    return torch.tensor([[b / 127.5 - 1 for b in hashlib.sha256(t.encode()).digest()[:16]]
                         for t in texts], dtype=torch.float32)


def image_tensor(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255
    return torch.tensor(np.concatenate([rgb, rgb ** 2, np.sin(rgb), np.cos(rgb),
                                       [1., .2, .4, .6]]), dtype=torch.float32)


if torch is not None:
    class FakeFrozenClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = SimpleNamespace(output_dim=16)
            self.register_buffer("image_projection", torch.eye(16))

        def encode_image(self, x):
            return x @ self.image_projection

        def encode_text(self, tokens):
            return tokens


@unittest.skipIf(torch is None, "PyTorch required for CPU extraction tests")
class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = cli.load_module("_test_cipt_real_modules", ROOT / "domainbed/algorithms/cipt_modules.py")
        tree = ast.parse((ROOT / "domainbed/algorithms/cipt_prompt.py").read_text())
        cls.templates = next(ast.literal_eval(n.value) for n in tree.body
                             if isinstance(n, ast.Assign) and any(
                                 isinstance(t, ast.Name) and t.id == "B5B_IMAGENET_TEMPLATES" for t in n.targets))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data" / "PACS"
        self.names = ["dog", "horse"]
        # Identical pixels in different class folders test label dependence.
        for d, domain in enumerate(["photo", "sketch"]):
            for name in self.names:
                folder = self.data / domain / name
                folder.mkdir(parents=True)
                for i in range(2):
                    Image.new("RGB", (4, 4), (40 + 70 * d, 50 + 30 * i, 130 - 20 * i)).save(folder / (str(i) + ".png"))
        self.clip_path = self.root / "clip.pt"
        self.clip_path.write_bytes(b"synthetic clip file for hash/provenance validation")
        self.prompt = SimpleNamespace(
            B5B_IMAGENET_TEMPLATES=self.templates,
            build_paired_generic_templates=lambda subject: [t.format(subject) for t in self.templates],
            load_frozen_clip=lambda *args: (FakeFrozenClip(), tokenize),
        )

    def checkpoint(self, mode, subject="subject"):
        torch.manual_seed(9)
        decomposition = self.modules.CausalDecomposition(16)
        with torch.no_grad():
            decomposition.causal_adapter.weight.mul_(1.3)
            decomposition.causal_adapter.bias.copy_(torch.linspace(-.1, .1, 16))
        tda = self.modules.TextDiversityAugmentation(16, 1).eval()
        pc = F.normalize(tokenize([t.format(n) for n in self.names for t in self.templates]), dim=-1).reshape(2, 80, 16)
        pn = F.normalize(tokenize([t.format(subject) for t in self.templates]), dim=-1)
        # A tiny verified drift catches accidentally re-encoding instead of
        # consuming the exact checkpoint buffer in native B5A forward passes.
        pn = F.normalize(pn + .001 * torch.roll(pn, 1, -1), dim=-1)
        state = {"clip_model." + k: v for k, v in FakeFrozenClip().state_dict().items()}
        state.update({"causal_decomposition." + k: v for k, v in decomposition.state_dict().items()})
        state.update({"tda." + k: v for k, v in tda.state_dict().items()})
        state.update({"text_features.b5b_text_bank": pc, "text_features.b5a_text_bank": pn})
        ckpt = {"args": {"dataset": "PACS", "seed": 0, "trial_seed": 0}, "test_envs": [0],
                "model_hparams": {"cipt_objective": "legacy", "cipt_template_mode": mode,
                                  "cipt_neutral_subject": subject, "cipt_pure": True,
                                  "cipt_class_names": self.names, "cipt_tda_heads": 1},
                "model_dict": state}
        return ckpt, decomposition.eval(), tda, pc, pn

    def run_extract(self, checkpoint, name, manifest=None):
        path = self.root / (name + ".pth")
        torch.save(checkpoint, path)
        out = self.root / name
        argv = ["--dataset", "PACS", "--data-dir", str(self.data.parent), "--clip-path", str(self.clip_path),
                "--checkpoint", str(path), "--output-dir", str(out), "--device", "cpu",
                "--batch-size", "3", "--samples-per-class-domain", "2", "--extract-only"]
        if manifest is not None:
            argv += ["--sample-manifest", str(manifest), "--sample-seed", "99"]
        def loader(name, path):
            if name == "_mechanism_prompt":
                return self.prompt
            if name == "_mechanism_modules":
                return self.modules
            if name == "_mechanism_transforms":
                return SimpleNamespace(clip_basic=image_tensor)
            raise AssertionError("Unexpected module: " + name)
        with patch.object(cli, "load_module", side_effect=loader):
            cli.extract(cli.parse_args(argv))
        with (out / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
            samples = list(csv.DictReader(stream))
        return out, samples, json.loads((out / "metadata.json").read_text())

    def expected_z(self, samples, decomposition, tda, contexts):
        tensors = []
        for row in samples:
            with Image.open(row["path"]) as image:
                tensors.append(image_tensor(image))
        with torch.inference_mode():
            e, _ = decomposition(F.normalize(torch.stack(tensors), dim=-1))
            return F.normalize(tda(e, contexts), dim=-1).numpy()

    def test_b5a_uses_exact_saved_shared_bank_without_class_labels(self):
        ckpt, decomposition, tda, pc, pn = self.checkpoint("b5a")
        out, samples, meta = self.run_extract(ckpt, "b5a")
        z = np.load(out / meta["z_file"])
        np.testing.assert_allclose(z, self.expected_z(samples, decomposition, tda, pn), atol=3e-7)
        self.assertEqual(meta["z_file"], "z_neutral_unit.npy")
        self.assertFalse(meta["z_uses_ground_truth_class_word"])
        self.assertTrue(meta["native_bank_loaded_from_checkpoint"])
        self.assertFalse((out / "z_class_unit.npy").exists())
        probes = np.load(out / "intervention_metrics.npz")
        self.assertEqual(probes.files, ["class_insertion_cosine_shift"])
        # Equal image pixels, different true classes: native B5A Z is equal.
        for i, row in enumerate(samples):
            other = next(j for j, r in enumerate(samples) if r["class"] != row["class"]
                         and r["domain"] == row["domain"] and Path(r["path"]).name == Path(row["path"]).name)
            np.testing.assert_array_equal(z[i], z[other])
        neutral_const = F.normalize(tokenize(["a photo of a subject."]), dim=-1)
        np.testing.assert_allclose(np.load(out / "z_constant_unit.npy"),
                                   self.expected_z(samples, decomposition, tda, neutral_const)[:, 0], atol=3e-7)

    def test_b5b_keeps_class_conditioned_z_and_replays_same_images_for_b5a(self):
        ckpt, decomposition, tda, pc, pn = self.checkpoint("b5b")
        out, samples, meta = self.run_extract(ckpt, "b5b")
        labels = torch.tensor([int(r["class_id"]) for r in samples])
        z = np.load(out / "z_class_unit.npy")
        np.testing.assert_allclose(z, self.expected_z(samples, decomposition, tda, pc[labels]), atol=3e-7)
        self.assertTrue(meta["z_uses_ground_truth_class_word"])
        self.assertEqual(set(np.load(out / "intervention_metrics.npz").files),
                         {"class_swap_cosine_shift", "neutral_swap_cosine_shift"})
        other, _, _, _, _ = self.checkpoint("b5a")
        _, replay, other_meta = self.run_extract(other, "b5a_replay", out / "samples.csv")
        self.assertEqual(samples, replay)
        self.assertEqual(meta["samples_identity_sha256"], other_meta["samples_identity_sha256"])
        self.assertEqual(meta["samples_sha256"], other_meta["samples_sha256"])

    def test_checkpoint_neutral_subject_is_respected_and_mismatch_is_rejected(self):
        ckpt, decomposition, tda, _, pn = self.checkpoint("b5a", "object")
        out, samples, meta = self.run_extract(ckpt, "object")
        self.assertEqual(meta["neutral_subject"], "object")
        np.testing.assert_allclose(np.load(out / "z_neutral_unit.npy"),
                                   self.expected_z(samples, decomposition, tda, pn), atol=3e-7)
        with (out / "texts.csv").open(encoding="utf-8-sig", newline="") as stream:
            native_texts = [r for r in csv.DictReader(stream) if r["native_z_bank"] == "True"]
        self.assertEqual(len(native_texts), 80)
        self.assertTrue(all("object" in r["text"] for r in native_texts))
        ckpt["model_hparams"]["cipt_neutral_subject"] = "subject"
        with self.assertRaisesRegex(ValueError, "subject/CLIP mapping"):
            self.run_extract(ckpt, "wrong_subject")

    def test_saved_bank_validation_rejects_missing_old_misordered_or_invalid_bank(self):
        bank = F.normalize(torch.arange(48, dtype=torch.float32).reshape(3, 16) + 1, dim=-1)
        self.assertIs(cli.restore_text_bank({"key": bank}, "key", bank, "cpu"), bank)
        self.assertIs(cli.restore_text_bank({}, "key", bank, "cpu"), bank)
        for bad in [None, bank[:2], bank.flip(0), torch.zeros_like(bank), bank * float("nan")]:
            with self.subTest(bad="missing" if bad is None else "invalid"):
                with self.assertRaises(ValueError):
                    cli.restore_text_bank({} if bad is None else {"key": bad}, "key", bank, "cpu", required=True)


if __name__ == "__main__":
    unittest.main()
