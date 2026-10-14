"""Full coverage, exact all-pair means and streamed/report equivalence."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cipt_feature_report import class_geometry, image_mean, paired_metrics, unit
from cipt_full_feature_report import FullFeatureReport, OnlineStats, exact_class_geometry

spec = importlib.util.spec_from_file_location("full_cipt_cli", ROOT / "analyze_cipt_text_features.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(class_names=None, domain_names=None):
    rng = np.random.default_rng(5)
    class_names = class_names or ["dog", "horse"]
    domain_names = domain_names or ["photo", "sketch"]
    dim, samples = 12, []
    for c,name in enumerate(class_names):
        for domain in domain_names:
            for _ in range(2):
                i=len(samples)
                samples.append({"image_id": i,"class_id": c,"class": name,"domain": domain,
                                "relative_path": str(i)+".jpg","path": str(i)+".jpg"})
    pk_raw_c, pk_raw_a = rng.normal(size=(len(class_names),80,dim)), rng.normal(size=(80,dim))
    pk = {"pk_raw_class": pk_raw_c, "pk_raw_agnostic": pk_raw_a,
          "pk_class": unit(pk_raw_c), "pk_agnostic": unit(pk_raw_a)}
    e = unit(rng.normal(size=(len(samples),dim)))
    labels = np.asarray([s["class_id"] for s in samples])
    pre_c, pre_a = e[:,None]+pk["pk_class"][labels]*.2, e[:,None]+pk["pk_agnostic"][None]*.2
    return samples,class_names,pk,{"e": e, "visual": e, "visual_raw": e*11,
        "zk_class": unit(pre_c)*np.sqrt(dim), "zk_agnostic": unit(pre_a)*np.sqrt(dim),
        "zk_preln_class": pre_c, "zk_preln_agnostic": pre_a,
        "tda_update_class": pre_c-e[:,None], "tda_update_agnostic": pre_a-e[:,None]}


class FullReportTests(unittest.TestCase):
    def test_exact_all_pair_means_equal_explicit_matrix(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(17,9))
        labels = np.asarray([0]*8+[1]*6+[2]*3)
        domains = np.asarray(["A","B","C","D"]*4+["A"])
        expected = class_geometry(features, labels, ["dog","horse","guitar"], domains)
        actual = exact_class_geometry(features, labels, ["dog","horse","guitar"], domains)
        for key in ["within_class_cosine","between_class_cosine","same_class_cross_domain_cosine","same_class_same_domain_cosine"]:
            self.assertEqual(actual[key]["count"],expected[key]["count"])
            self.assertAlmostEqual(actual[key]["mean"],expected[key]["mean"],places=12)
        np.testing.assert_allclose(actual["centroid_cosine_distance"],expected["centroid_cosine_distance"],atol=1e-12)
        self.assertEqual(len(actual["class_pairs"]),3)

    def test_singleton_class_groups_report_no_within_pairs(self):
        result = exact_class_geometry(np.eye(3),[0,1,2],["a","b","c"],["A"]*3)
        self.assertIsNone(result["within_class_cosine"])
        self.assertIsNone(result["macro_within_class_cosine"])
        self.assertEqual(result["between_class_cosine"]["count"],3)
        json.dumps(result,allow_nan=False)

    def test_streamed_norm_statistics_match_all_values(self):
        x = np.random.default_rng(8).normal(size=83)
        stats = OnlineStats()
        for chunk in np.array_split(x,7):
            stats.add(chunk)
        self.assertAlmostEqual(stats.result()["mean"],x.mean(),places=12)
        self.assertAlmostEqual(stats.result()["std"],x.std(),places=12)
        self.assertEqual(stats.result()["count"],len(x))

    def test_full_selection_includes_every_domain_class_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = 0
            for i,domain in enumerate(["art_painting","cartoon","photo","sketch"]):
                for c in ["dog","horse"]:
                    folder = root/domain/c
                    folder.mkdir(parents=True)
                    for k in range(i+2):
                        (folder/(str(k)+".jpg")).touch()
                        expected += 1
            samples = cli.select_samples(root,["dog","horse"],[0,1],None,5,0,all_images=True)
            self.assertEqual(len(samples),expected)
            self.assertEqual(len(set(s["path"] for s in samples)),expected)
            self.assertEqual(len(set(s["domain"] for s in samples)),4)

    def test_streamed_results_match_unsplit_features(self):
        samples,names,pk,batch = fixture()
        metadata = {"seed":0,"tsne_perplexity":5,"tsne_pca_dim":5,"tsne_iterations":300,
                    "synthetic_validation_only":True,"all_images":True}
        with tempfile.TemporaryDirectory() as directory:
            report = FullFeatureReport(directory,samples,names,[0,1],pk,metadata,True)
            for start,end in [(0,3),(3,7),(7,8)]:
                report.add_batch(start,{k:v[start:end] for k,v in batch.items()})
            actual = report.finish(make_tsne=False)
            expected = paired_metrics(batch["zk_class"],batch["zk_agnostic"])
            for key,values in expected.items():
                self.assertAlmostEqual(actual["zk"]["paired_all_interventions"][key]["mean"],values.mean(),places=11)
            labels = [s["class_id"] for s in samples]
            domains = [s["domain"] for s in samples]
            expected_geometry = class_geometry(image_mean(batch["zk_class"]),labels,names,domains)
            self.assertAlmostEqual(actual["zk"]["class_conditioned"]["within_class_cosine"]["mean"],expected_geometry["within_class_cosine"]["mean"],places=7)
            with (Path(directory)/"zk_paired.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)),8*80)
            stored = np.load(Path(directory)/"vectors/zk_class.npy",mmap_mode="r")
            np.testing.assert_allclose(stored,batch["zk_class"],rtol=1e-7)
            self.assertTrue((Path(directory)/"class_pair_metrics.csv").is_file())
            self.assertTrue(actual["protocol"]["extraction_complete"])
            self.assertFalse(actual["protocol"]["tsne_complete"])


if __name__ == "__main__":
    if "--smoke-output" in sys.argv:
        samples,names,pk,batch = fixture(["dog","elephant","giraffe","guitar","horse","house","person"],
                                        ["art_painting","cartoon","photo","sketch"])
        directory = sys.argv[sys.argv.index("--smoke-output")+1]
        report = FullFeatureReport(directory,samples,names,list(range(len(names))),pk,
            {"seed":0,"tsne_perplexity":5,"tsne_pca_dim":5,"tsne_iterations":300,
             "synthetic_validation_only":True,"all_images":True})
        report.add_batch(0,batch)
        report.finish(make_tsne=True)
        print("Synthetic full-report rendering check complete; not real PACS results.")
    else:
        unittest.main()
