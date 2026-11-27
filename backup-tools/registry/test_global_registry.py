import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from subprocess import run

from global_registry import Registry, Unavailable, configuration, digest, import_observation


def config(lr=2e-5, mix=.75, prompt=4, trial_seed=1):
    return configuration("PACS", "1"*64, "2"*64, "3"*64,
        dict(algorithm="CIPTDCCL", steps=5001, checkpoint_freq=300, batch_size=32,
             source_sampling="domain_balanced", holdout_fraction=.2, trial_seed=trial_seed,
             hparams_seed=0, deterministic=True, source_only_selection=True, label_ratio=1,
             evalmode="all", aug=0, cipt_cache_eval_visual=True),
        dict(cipt_causal_contrastive_weight=0, cipt_clip_teacher_mix=mix, cipt_prompt_length=prompt, lr=lr),
        [1, 2, 3, 4], {"type": "cosine", "T_max": 5001, "eta_min": 0})


class RegistryTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(digest(config()), digest(config(mix=0)))
        for variant in (config(lr=1e-5), config(prompt=8), config(trial_seed=2)):
            self.assertNotEqual(digest(config()), digest(variant))

    def test_race_recovery_and_no_network_bypass(self):
        with tempfile.TemporaryDirectory() as folder:
            remote = str(Path(folder) / "remote.git")
            run(["git", "init", "--bare", "-q", remote], check=True)
            registry = Registry(remote)
            registry.initialize()
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                claims = list(pool.map(lambda host: registry.claim(config(), server_id=host, study="s",
                    trial_id=0, commit="test", owner_token=host), ("school", "autodl")))
            self.assertEqual(sum(won for won, _ in claims), 1)
            row, = registry.records()
            key = registry.key("PACS", row["config_hash"])
            token = row["owner_token"]
            self.assertFalse(registry.retry(key, previous_token=token, new_token="blind", server_id="other",
                                            study="s", trial_id=1, commit="test")[0])
            self.assertTrue(registry.progress(key, token, status="FAILED", error="stopped worker")[0])
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                retries = list(pool.map(lambda host: registry.retry(key, previous_token=token, new_token=f"new-{host}",
                    server_id=host, study="s", trial_id=1, commit="test"), ("school", "autodl")))
            self.assertEqual(sum(won for won, _ in retries), 1)
            with self.assertRaises(Unavailable):
                Registry(str(Path(folder) / "missing")).claim(config(), server_id="school", study="s",
                    trial_id=0, commit="test", owner_token="test")

    def test_complete_is_shared_once(self):
        import optuna
        from global_registry import protocol_fingerprint
        cfg = config()
        metrics = {"source_val": .82, "iid": .7, "test_out": .69}
        record = dict(status="COMPLETE", completed_folds=list(range(4)), effective_config=cfg, config_hash=digest(cfg),
                      protocol_fingerprint=protocol_fingerprint(cfg), server_id="school", result_branch="results/test",
                      result_path="test.json", **metrics)
        study = optuna.create_study(direction="maximize")
        params = {"prompt_length": 4}
        distributions = {"prompt_length": optuna.distributions.CategoricalDistribution([4, 8, 16])}
        self.assertTrue(import_observation(study, record, params, distributions, cfg, metrics))
        self.assertFalse(import_observation(study, record, params, distributions, cfg, metrics))
        self.assertEqual(study.best_value, .82)


if __name__ == "__main__":
    unittest.main()
