"""Regression checks for the position switch and existing sconf CLI overrides."""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "domainbed/lib/cli.py"
spec = importlib.util.spec_from_file_location("hparam_cli", path)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class HparamCLITests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("name")
        parser.add_argument("configs", nargs="*")
        parser.add_argument("--dataset", default="PACS")
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--test_envs", nargs="+", type=int)
        return parser

    def test_first_override_value_is_not_a_config_filename(self):
        args, overrides = cli.parse_known_with_overrides(self.parser(), [
            "run", "--dataset", "PACS", "--cipt_template_mode", "b5c",
            "--cipt_contrastive_position", "post_tda_mean", "--lr", "5e-5",
        ])
        self.assertEqual(args.configs, [])
        self.assertEqual(overrides, [
            "--cipt_template_mode", "b5c", "--cipt_contrastive_position",
            "post_tda_mean", "--lr", "5e-5",
        ])

    def test_real_config_files_and_known_options_are_preserved(self):
        args, overrides = cli.parse_known_with_overrides(self.parser(), [
            "run", "experiment.yaml", "extra.yaml", "--deterministic",
            "--dataset", "VLCS", "--test_envs", "0", "2",
            "--cipt_contrastive_position", "e", "--cipt_pure", "false",
        ])
        self.assertEqual(args.configs, ["experiment.yaml", "extra.yaml"])
        self.assertEqual(args.dataset, "VLCS")
        self.assertTrue(args.deterministic)
        self.assertEqual(args.test_envs, [0, 2])
        self.assertEqual(overrides, ["--cipt_contrastive_position", "e", "--cipt_pure", "false"])

    def test_attached_values_and_nested_overrides_keep_exact_content(self):
        _, overrides = cli.parse_known_with_overrides(self.parser(), [
            "run", "--cipt_contrastive_position=e", "--lr=-1e-3",
            "---swad_kwargs.n_converge", "5", "--example", "a=b c",
        ])
        self.assertEqual(overrides, [
            "--cipt_contrastive_position", "e", "--lr", "-1e-3",
            "---swad_kwargs.n_converge", "5", "--example", "a=b c",
        ])

    def test_missing_override_value_fails_without_consuming_next_flag(self):
        for argv in (["run", "--cipt_contrastive_position"],
                     ["run", "--cipt_contrastive_position", "--deterministic"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parse_known_with_overrides(self.parser(), argv)

    def test_end_of_options_and_plain_command_still_work(self):
        args, remaining = cli.parse_known_with_overrides(self.parser(), ["run"])
        self.assertEqual((args.name, args.configs, remaining), ("run", [], []))
        args, remaining = cli.parse_known_with_overrides(self.parser(), ["run", "--", "--config.yaml"])
        self.assertEqual(args.configs, ["--config.yaml"])
        self.assertEqual(remaining, [])

    def test_existing_known_flag_abbreviations_still_work(self):
        args, overrides = cli.parse_known_with_overrides(self.parser(), [
            "run", "--det", "--dat", "VLCS", "--cipt_contrastive_position", "e",
        ])
        self.assertTrue(args.deterministic)
        self.assertEqual(args.dataset, "VLCS")
        self.assertEqual(overrides, ["--cipt_contrastive_position", "e"])


if __name__ == "__main__":
    unittest.main()
