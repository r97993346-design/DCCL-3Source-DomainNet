"""CPU tests for evidence direction, normalization controls and image units.

python -m unittest discover -s tests -p test_cipt_mechanism.py
python tests/test_cipt_mechanism.py --smoke-output /tmp/cipt_mechanism_smoke
All generated features are SYNTHETIC validation data, never research results.
"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from analyze_cipt_mechanism import validate_original_checkpoint, validate_samples
from cipt_mechanism_report import (cosine_shift, domain_metrics, dump_json, neighbor_weights,
                                    plot_metrics, plot_tsne, stratified_bootstrap, summarize,
                                    template_spread, topology_metrics, unit, write_csv)


def synthetic_run(folder):
    """Create a diagnostic fixture with domain structure and nonzero shifts."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(17)
    samples = [{"image_id": i, "class_id": i // 8, "class": ["dog", "horse"][i // 8],
                "domain": ["photo", "sketch"][(i // 4) % 2], "relative_path": "synthetic/{}.jpg".format(i)}
               for i in range(16)]
    labels = np.array([row["class_id"] for row in samples])
    domain = np.array([0 if row["domain"] == "photo" else 1 for row in samples])
    basis = np.eye(16)
    e = unit(basis[labels] + .7 * basis[domain + 2] + rng.normal(0, .08, (16, 16)))
    ln_e = unit(e - e.mean(axis=1, keepdims=True))
    constant = unit(e + .5 * basis[labels + 4])
    z = unit(constant[:, None, :] + rng.normal(0, .01, (16, 80, 16)))
    for name, data in [("e_unit", e), ("ln_e_unit", ln_e), ("z_constant_unit", constant), ("z_class_unit", z)]:
        np.save(folder / (name + ".npy"), data.astype(np.float32))
    np.savez_compressed(folder / "intervention_metrics.npz",
                        class_swap_cosine_shift=np.full(16, .15), neutral_swap_cosine_shift=np.full(16, .1))
    write_csv(folder / "samples.csv", samples)
    dump_json(folder / "metadata.json", {"class_names": ["dog", "horse"], "target_domains": ["sketch"],
                                        "sample_seed": 0, "synthetic_validation_only": True})


class MechanismTests(unittest.TestCase):
    def test_identical_templates_have_zero_spread(self):
        x = np.random.default_rng(0).normal(size=(4, 1, 16))
        np.testing.assert_allclose(template_spread(np.repeat(x, 80, axis=1)), 0, atol=1e-12)

    def test_spread_matches_explicit_distinct_pairs(self):
        z = unit(np.random.default_rng(4).normal(size=(3, 5, 16)))
        expected = np.mean([cosine_shift(z[:, i], z[:, j]) for i in range(5) for j in range(i + 1, 5)], axis=0)
        np.testing.assert_allclose(template_spread(z), expected, atol=1e-12)

    def test_ln_can_move_e_without_any_text_residual(self):
        e = np.array([[1., 2, 4, 9], [9., 2, 3, 4]])
        ln = (e - e.mean(axis=1, keepdims=True)) / e.std(axis=1, keepdims=True)
        z = np.repeat(ln[:, None], 80, axis=1)
        self.assertTrue(np.all(cosine_shift(e[:, None], z).mean(axis=1) > .1))
        np.testing.assert_allclose(cosine_shift(ln[:, None], z), 0, atol=1e-12)

    def test_metrics_distinguish_domain_clusters_from_interleaved_domains(self):
        rng = np.random.default_rng(10)
        domains = np.array(["photo"] * 8 + ["sketch"] * 8)
        separated = np.eye(16)[np.repeat([0, 1], 8)] + rng.normal(0, .01, (16, 16))
        # Same image geometry repeated across domains: nearest partners cross domain.
        base = rng.normal(size=(8, 16))
        mixed = np.concatenate([base, base])
        a, b = domain_metrics(separated, domains, 3), domain_metrics(mixed, domains, 3)
        self.assertGreater(a["domain_silhouette"], .9)
        self.assertGreater(a["neighbor_excess"], .4)
        self.assertLess(b["domain_silhouette"], 0)
        self.assertLess(b["neighbor_excess"], 0)

    def test_ties_do_not_fabricate_domain_clustering(self):
        result = domain_metrics(np.ones((8, 4)), ["a"] * 4 + ["b"] * 4, 2)
        self.assertAlmostEqual(result["neighbor_excess"], 0)
        self.assertAlmostEqual(result["domain_silhouette"], 0)
        w = neighbor_weights(np.zeros((8, 8)), 2)
        np.testing.assert_allclose(w.sum(1), 2)
        np.testing.assert_allclose(np.diag(w), 0)

    def test_topology_ignores_feature_scale_and_global_rotation(self):
        rng = np.random.default_rng(1)
        e = rng.normal(size=(12, 16))
        q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        result = topology_metrics(e, 7 * e @ q)
        self.assertAlmostEqual(result["neighbor_overlap"], 1)
        self.assertAlmostEqual(result["distance_spearman"], 1)

    def test_wrong_checkpoint_rejected(self):
        original = {"cipt_objective": "legacy", "cipt_template_mode": "b5b", "cipt_pure": True}
        validate_original_checkpoint(original)
        validate_original_checkpoint({**original, "cipt_pure": "false", "cipt_use_contrastive": "false"})
        for override in [{"cipt_objective": "env_anchor"}, {"cipt_use_tda": False},
                         {"cipt_template_mode": "b5a"}, {"cipt_pure": False}, {"cipt_use_de": False}]:
            with self.assertRaises(ValueError):
                validate_original_checkpoint({**original, **override})

    def test_duplicate_or_imbalanced_manifest_rejected(self):
        samples = [{"class_id": c, "class": str(c), "domain": d, "relative_path": "{}/{}/{}".format(d, c, i)}
                   for c in [0, 1] for d in ["a", "b"] for i in range(2)]
        validate_samples(samples, [0, 1], ["a", "b"])
        with self.assertRaises(ValueError):
            validate_samples(samples + samples[:1], [0, 1], ["a", "b"])
        with self.assertRaises(ValueError):
            validate_samples(samples[:-1], [0, 1], ["a", "b"])
        corrupt = [dict(row) for row in samples]
        corrupt[0]["relative_path"] = "wrong_domain/0/0"
        with self.assertRaises(ValueError):
            validate_samples(corrupt, [0, 1], ["a", "b"])

    def test_bootstrap_resamples_images_within_cells(self):
        result = stratified_bootstrap([1, 1, 3, 3], ["a", "a", "b", "b"])
        self.assertEqual(result["macro_class_domain_mean"], 2)
        self.assertEqual(result["image_bootstrap_95ci"], [2, 2])

    def test_full_report_includes_every_template_without_pseudoreplication(self):
        with tempfile.TemporaryDirectory() as folder:
            synthetic_run(folder)
            _, _, _, _, rows, per_image = summarize(folder)
            self.assertEqual(len(rows), 2 * (80 + 4))
            self.assertEqual(len(per_image["template_pairwise_cosine_shift"]), 16)
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertEqual(summary["per_image"]["template_pairwise_cosine_shift"]["images"], 16)
            self.assertGreater(summary["per_image"]["class_minus_template_shift"]["mean"], 0)
            with (Path(folder) / "within_class_topology.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2 * 2 * 80)


if __name__ == "__main__":
    if "--smoke-output" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--smoke-output") + 1])
        synthetic_run(out)
        meta, samples, representations, z, rows, per_image = summarize(out)
        plot_metrics(out, rows, per_image)
        plot_tsne(out, meta, samples, representations, z, [0], [5], iterations=300)
        print("SYNTHETIC plotting validation only; not a CIPT experiment.")
    else:
        unittest.main()
