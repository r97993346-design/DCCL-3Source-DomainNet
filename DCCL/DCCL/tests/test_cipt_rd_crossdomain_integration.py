"""Regression: RD/MCC environment constraints coexist with cross-domain E contrast."""
import unittest
import torch
import yaml

import test_cipt_env_anchor as fixture


class RDCrossDomainIntegrationTests(unittest.TestCase):
    def params(self):
        path = fixture.ROOT / "config_cipt_b5b_entropy_rd_clip_guided.yaml"
        return yaml.safe_load(path.read_text())

    def test_rd_and_clip_guided_both_contribute_to_one_update(self):
        options = self.params()
        model = fixture.build("relative_distance", **options)
        metrics = model.update(*fixture.batches())
        self.assertEqual(model.objective_mode, "env_anchor")
        self.assertEqual(model.contrastive_type, "clip_guided_cross_domain")
        self.assertEqual(model.rd_anchor_mode, "selected")
        self.assertGreater(metrics["env_rd_loss"], 0.0)
        self.assertGreater(metrics["dccl_contrastive_loss"], 0.0)
        self.assertGreater(metrics["cg_valid_anchor_fraction"], 0.0)
        self.assertEqual(metrics["cg_active"], 1.0)
        self.assertGreater(metrics["env_loss_term"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])).item())

    def test_equal_weight_cross_domain_and_clip_mix_zero_match_with_rd(self):
        options = self.params()
        options.update(cipt_clip_teacher_mix=0.0, cipt_clip_teacher_topk=0)
        a = fixture.build("relative_distance", **dict(options, cipt_contrastive_type="cross_domain_supcon"))
        b = fixture.build("relative_distance", **dict(options, cipt_contrastive_type="clip_guided_cross_domain"))
        ra = a.update(*fixture.batches())
        rb = b.update(*fixture.batches())
        self.assertAlmostEqual(ra["dccl_contrastive_loss"], rb["dccl_contrastive_loss"], places=6)
        self.assertAlmostEqual(ra["env_rd_loss"], rb["env_rd_loss"], places=6)
        self.assertAlmostEqual(ra["total_loss"], rb["total_loss"], places=6)

    def test_disable_new_contrastive_preserves_rd(self):
        options = self.params()
        options["cipt_use_contrastive"] = False
        model = fixture.build("relative_distance", **options)
        results = model.update(*fixture.batches())
        self.assertEqual(results["contrastive_weight_eff"], 0.0)
        self.assertEqual(results["dccl_contrastive_loss"], 0.0)
        self.assertEqual(results["cg_active"], 0.0)
        self.assertGreater(results["env_rd_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
