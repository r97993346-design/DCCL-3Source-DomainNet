import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tune_piccl_v2_swad.py"
SPEC = importlib.util.spec_from_file_location("piccl_v2_swad_hpo", SCRIPT)
HPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HPO)


class HPOTests(unittest.TestCase):
    def load_config(self, name="piccl_v2_pacs_hpo.json"):
        return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))

    def fake_registry(self, cfg):
        registry = dict(cfg["fixed_params"])
        for stage in cfg["stages"].values():
            for name, spec in stage["search_space"].items():
                registry[name] = HPO._representative_value(spec)
        return registry

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
            {"swad_target": 0.825, "swad_indomain_report_only": 0.9125},
        )

    def test_parse_final_prettytable_swad_and_not_swad_indomain(self):
        text = """
        INFO | SWAD = 12.000%
        +------------+--------------+---------+---------+---------+---------+
        | Selection  | art_painting | cartoon |  photo  |  sketch |  Avg.   |
        +------------+--------------+---------+---------+---------+---------+
        | SWAD       | 90.055%      | 84.328% | 97.605% | 84.033% | 89.005% |
        | SWAD (inD) | 97.944%      | 97.478% | 97.479% | 98.371% | 97.818% |
        +------------+--------------+---------+---------+---------+---------+
        """
        metrics = HPO.parse_swad_metrics(text)
        self.assertAlmostEqual(metrics["swad_target"], 0.89005)
        self.assertAlmostEqual(metrics["swad_indomain_report_only"], 0.97818)

    def test_selection_objective_is_mean_target_swad(self):
        self.assertAlmostEqual(HPO.selection_objective([0.8, 0.9, 1.0, 0.7]), 0.85)

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
            with mock.patch.object(HPO, "registered_hparams", return_value=self.fake_registry(cfg)):
                registry = HPO.preflight(cfg, ["0", "1"], 2)
            for stage in cfg["stages"].values():
                self.assertLessEqual(set(stage["search_space"]), set(registry))
            self.assertLessEqual(set(cfg["fixed_params"]), set(registry))

    def test_preflight_rejects_unknown_parameter_and_unsafe_gpu_list(self):
        cfg = self.load_config()
        cfg["fixed_params"]["piccl_typo"] = 1
        registry = self.fake_registry(cfg)
        registry.pop("piccl_typo")
        with mock.patch.object(HPO, "registered_hparams", return_value=registry):
            with self.assertRaisesRegex(ValueError, "piccl_typo"):
                HPO.preflight(cfg, ["0", "1"], 2)
        cfg["fixed_params"].pop("piccl_typo")
        with mock.patch.object(HPO, "registered_hparams", return_value=self.fake_registry(cfg)):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                HPO.preflight(cfg, ["0", "0"], 2)

    def test_alpha_max_is_read_bounded_and_changes_effective_alpha(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from domainbed.algorithms.piccl import PICCL
        def alpha(maximum, step):
            obj = object.__new__(PICCL)
            obj.hparams = {"piccl_total_steps": 101, "piccl_warmup_ratio": .1,
                           "piccl_delayed_start_ratio": .1, "piccl_ramp_ratio": .2,
                           "piccl_alpha_max": maximum}
            obj.piccl_alpha = torch.tensor(0.0)
            return float(PICCL._alpha(obj, step))
        self.assertEqual(alpha(.2, 10), 0.0)
        self.assertAlmostEqual(alpha(.2, 30), .1)
        self.assertAlmostEqual(alpha(.5, 30), .25)
        self.assertLessEqual(max(alpha(.5, step) for step in range(101)), .5)

    def test_failed_subprocess_persists_evidence_and_is_trial_failure(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            completed = mock.Mock(returncode=7, stdout="out", stderr="bad")
            cfg = {"dataset": "PACS", "algorithm": "PICCL", "tasks": []}
            with mock.patch.object(HPO.subprocess, "run", return_value=completed):
                with self.assertRaises(HPO.TrialExecutionError):
                    HPO.run_task(cfg, {}, {"name": "env0", "test_envs": [0]}, root,
                                 root, ROOT, 1, 0, "0", False)
            failure = json.loads((root / "failure.json").read_text())
            self.assertEqual(failure["returncode"], 7)
            self.assertTrue((root / "command.json").exists())
            self.assertTrue((root / "stdout.log").exists())

    def test_two_gpu_assignment_serializes_each_gpu(self):
        cfg = {"tasks": [{"name": f"env{i}"} for i in range(4)]}
        calls = []
        def fake_run(_cfg, _params, task, *_args):
            gpu = _args[-2]
            calls.append((task["name"], gpu))
            return {"swad_target": .8, "swad_indomain_report_only": .9}
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, mock.patch.object(HPO, "run_task", side_effect=fake_run):
            HPO.evaluate_params(cfg, {}, Path(directory), Path(directory), ROOT,
                                1, 0, ["0", "1"], 2, False)
        self.assertEqual(dict(calls), {"env0": "0", "env2": "0",
                                       "env1": "1", "env3": "1"})

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
