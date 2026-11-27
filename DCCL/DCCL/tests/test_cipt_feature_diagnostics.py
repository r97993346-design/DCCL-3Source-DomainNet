"""CPU/NumPy tests for paired metrics, independent image units and manifests.

python -m unittest discover -s tests -p test_cipt_feature_diagnostics.py
python tests/test_cipt_feature_diagnostics.py --smoke-output /tmp/cipt_plot_smoke
The smoke output uses SYNTHETIC features solely to verify plotting.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cipt_feature_report import (class_geometry, image_mean, paired_metrics,
                                 save_report, summarize, unit)

spec = importlib.util.spec_from_file_location("cipt_diagnostic_cli", ROOT / "analyze_cipt_text_features.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def synthetic_features():
    rng = np.random.default_rng(7)
    d = 32
    e = unit(rng.normal(size=(6, d))).astype(np.float32)
    pc_raw = rng.normal(size=(2, 80, d)).astype(np.float32)
    pa_raw = rng.normal(size=(80, d)).astype(np.float32)
    pc, pa = unit(pc_raw), unit(pa_raw)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    pre_c, pre_a = e[:, None] + pc[labels], e[:, None] + pa[None]
    zc, za = unit(pre_c) * np.sqrt(d), unit(pre_a) * np.sqrt(d)
    return {"pk_raw_class": pc_raw, "pk_raw_agnostic": pa_raw,
            "pk_class": pc, "pk_agnostic": pa, "visual": e, "e": e,
            "zk_class": zc, "zk_agnostic": za,
            "zk_preln_class": pre_c, "zk_preln_agnostic": pre_a}, labels


class FeatureDiagnosticsTests(unittest.TestCase):
    def test_identical_pairs_have_zero_shift(self):
        x = np.random.default_rng(2).normal(size=(3, 80, 8))
        result = paired_metrics(x, x.copy())
        np.testing.assert_allclose(result["cosine"], 1, atol=1e-12)
        np.testing.assert_allclose(result["cosine_shift"], 0, atol=1e-12)
        np.testing.assert_allclose(result["l2_raw"], 0)

    def test_unit_l2_and_cosine_are_consistent(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(80, 8)), rng.normal(size=(80, 8))
        result = paired_metrics(a, b)
        np.testing.assert_allclose(result["l2_unit"] ** 2, 2 * result["cosine_shift"], atol=1e-12)
        scaled = paired_metrics(a * 7, b * 2)
        np.testing.assert_allclose(scaled["cosine"], result["cosine"], atol=1e-12)

    def test_repeating_texts_does_not_multiply_image_samples(self):
        e = np.asarray([[1., 0], [1, 0], [0, 1], [0, 1]])
        z = np.repeat(e[:, None], 80, axis=1)
        labels, domains = [0, 0, 1, 1], ["photo", "sketch", "photo", "sketch"]
        np.testing.assert_allclose(image_mean(z), e)
        stats = class_geometry(image_mean(z), labels, ["dog", "horse"], domains)
        self.assertEqual(stats["images_per_class"], [2, 2])
        self.assertEqual(stats["within_class_cosine"]["count"], 2)
        self.assertEqual(stats["between_class_cosine"]["count"], 4)
        self.assertEqual(stats["within_class_cosine"]["mean"], 1)
        self.assertEqual(stats["between_class_cosine"]["mean"], 0)
        self.assertEqual(stats["same_class_cross_domain_cosine"]["mean"], 1)

    def test_shared_pk_is_not_duplicated_by_class(self):
        features, labels = synthetic_features()
        stats = summarize(features, ["dog", "horse"], [0, 1], labels,
                          ["photo", "photo", "sketch", "photo", "photo", "sketch"])
        self.assertEqual(stats["pk"]["class_agnostic"]["norm"]["count"], 80)
        self.assertEqual(stats["pk"]["class_agnostic"]["pairwise_cosine_off_diagonal"]["count"], 3160)
        self.assertEqual(stats["zk"]["class_conditioned"]["images_per_class"], [3, 3])

    def test_checkpoint_wrapper_prefix_is_removed(self):
        state = {"module.causal_decomposition.causal_adapter.weight": "E",
                 "module.tda.layer_norm.weight": "LN", "n_averaged": 10}
        result = cli.checkpoint_state({"model_dict": state})
        self.assertEqual(result["causal_decomposition.causal_adapter.weight"], "E")
        self.assertEqual(cli.substate(result, "tda."), {"layer_norm.weight": "LN"})

    def test_selection_is_balanced_deterministic_and_replayable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for domain in ["photo", "sketch"]:
                for name in ["dog", "horse"]:
                    class_root = root / domain / name
                    class_root.mkdir(parents=True)
                    for k in range(6):
                        (class_root / (str(k) + ".jpg")).touch()
            first = cli.select_samples(root, ["dog", "horse"], [0, 1], ["photo", "sketch"], 2, 0)
            second = cli.select_samples(root, ["dog", "horse"], [0, 1], ["photo", "sketch"], 2, 0)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 8)
            cli.write_csv(root / "samples.csv", first)
            replay = cli.select_samples(root, ["dog", "horse"], [0, 1], None, 2, 99, root / "samples.csv")
            self.assertEqual(first, replay)

    def test_reports_are_finite_and_have_expected_rows(self):
        features, labels = synthetic_features()
        metadata = {"seed": 0, "tsne_perplexity": 5, "tsne_templates_per_image": 80,
                    "synthetic_validation_only": True}
        with tempfile.TemporaryDirectory() as folder:
            stats = save_report(folder, features, ["dog", "horse"], [0, 1], metadata,
                                labels, ["photo", "photo", "sketch"] * 2, make_tsne=False)
            saved = json.loads((Path(folder) / "summary.json").read_text())
            self.assertEqual(saved["protocol"], metadata)
            self.assertEqual(stats["zk"]["paired_all_interventions"]["cosine"]["count"], 480)
            self.assertEqual(len((Path(folder) / "zk_paired.csv").read_text().splitlines()), 481)
            self.assertEqual(len((Path(folder) / "zk_image_means.csv").read_text().splitlines()), 7)
            self.assertTrue((Path(folder) / "feature_norms.png").is_file())


if __name__ == "__main__":
    if "--smoke-output" in sys.argv:
        out = sys.argv[sys.argv.index("--smoke-output") + 1]
        features, labels = synthetic_features()
        save_report(out, features, ["dog", "horse"], [0, 1],
                    {"seed": 0, "tsne_perplexity": 5, "tsne_templates_per_image": 80,
                     "synthetic_validation_only": True}, labels,
                    ["photo", "photo", "sketch"] * 2, make_tsne=True)
        print("Synthetic rendering check complete; this is not an experiment result.")
    else:
        unittest.main()
