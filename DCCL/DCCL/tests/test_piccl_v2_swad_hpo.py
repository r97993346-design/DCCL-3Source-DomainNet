import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tune_piccl_v2_swad.py"
SPEC = importlib.util.spec_from_file_location("piccl_v2_swad_hpo", SCRIPT)
HPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HPO)


class HPOTests(unittest.TestCase):
    def test_parse_swad_metrics_uses_final_values_and_scales_percentages(self):
        text = "\n".join(
            [
                "INFO 00:00 | SWAD = 80.000%",
                "INFO 00:00 | SWAD (inD) = 90.000%",
                "\x1b[36mINFO\x1b[0m 00:01 | SWAD = 82.500%",
                "\x1b[36mINFO\x1b[0m 00:01 | SWAD (inD) = 91.250%",
            ]
        )
        self.assertEqual(
            HPO.parse_swad_metrics(text),
            {"swad_indomain": 0.9125, "swad_target_report_only": 0.825},
        )

    def test_source_objective_uses_only_values_passed_as_source_scores(self):
        self.assertAlmostEqual(HPO.source_objective([0.9, 0.9], 0.1), 0.9)
        self.assertAlmostEqual(HPO.source_objective([0.8, 1.0], 0.1), 0.89)

    def test_search_space_rejects_dead_and_dccl_parameters(self):
        with self.assertRaisesRegex(ValueError, "dead v2"):
            HPO.validate_search_space({"piccl_residual_scale": {}})
        with self.assertRaisesRegex(ValueError, "DCCL baseline"):
            HPO.validate_search_space({"lr": {}})
        with self.assertRaisesRegex(ValueError, "uses their sum"):
            HPO.validate_search_space(
                {"piccl_warmup_ratio": {}, "piccl_delayed_start_ratio": {}}
            )

    def test_configs_are_valid_and_only_search_live_piccl_parameters(self):
        for name in ["piccl_v2_pacs_hpo.json", "piccl_v2_domainnet_hpo.json"]:
            cfg = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            for stage in cfg["stages"].values():
                HPO.validate_search_space(stage["search_space"])

    def test_build_command_supports_pacs_and_domainnet_tasks(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            cfg = {
                "dataset": "PACS",
                "algorithm": "PICCL",
                "model": "resnet50",
                "checkpoint_freq": 100,
                "deterministic": True,
            }
            command = HPO.build_command(
                cfg,
                {"piccl_rank": 16, "use_piccl": True},
                {"name": "env3", "test_envs": [3]},
                tmp_path / "out",
                tmp_path / "data",
                3000,
                0,
            )
            self.assertEqual(command[command.index("--test_envs") + 1], "3")
            self.assertEqual(command[command.index("--use_piccl") + 1], "true")
            self.assertEqual(command[command.index("--trial_seed") + 1], "0")

            cfg["dataset"] = "DomainNet"
            command = HPO.build_command(
                cfg,
                {},
                {"name": "s123_t4", "source_envs": [1, 2, 3], "target_env": 4},
                tmp_path / "out",
                tmp_path / "data",
                5000,
                0,
            )
            source_at = command.index("--source_envs")
            self.assertEqual(command[source_at + 1 : source_at + 4], ["1", "2", "3"])
            self.assertEqual(command[command.index("--target_env") + 1], "4")


if __name__ == "__main__":
    unittest.main()
