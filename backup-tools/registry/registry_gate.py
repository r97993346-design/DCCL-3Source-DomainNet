"""Optuna controller hooks. Workers may start only inside an owned context."""
import contextlib
import threading
import uuid
from statistics import mean

from global_registry import Unavailable, digest, import_observation


def metrics(manifest):
    return {key: mean(fold[key] for fold in manifest["folds"])
            for key in ("source_val", "iid", "test_out")}


class Occupied(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"configuration already {record['status']}")


class TrialGate:
    def __init__(self, registry, server_id, study_name, training_commit):
        self.registry = registry
        self.server_id = server_id
        self.study_name = study_name
        self.training_commit = training_commit

    @contextlib.contextmanager
    def eligible(self, trial, config, *, heartbeat_seconds=120, warning=print):
        # The controller must hold its existing local process lock for the full context.
        token = trial.user_attrs.get("registry_owner")
        if not token:
            token = uuid.uuid4().hex
            trial.set_user_attr("registry_owner", token)
        fingerprint = digest(config)
        trial.set_user_attr("config_hash", fingerprint)
        won, row = self.registry.claim(config, server_id=self.server_id, study=self.study_name,
              trial_id=trial.number, commit=self.training_commit, owner_token=token)
        if not won:
            raise Occupied(row)
        key = self.registry.key(config["dataset"], fingerprint)
        stopped = threading.Event()

        def pulse():
            while not stopped.wait(heartbeat_seconds):
                try:
                    self.registry.progress(key, token)
                except Unavailable:
                    warning("registry heartbeat unavailable; current worker may finish")

        thread = threading.Thread(target=pulse, daemon=True)
        thread.start()
        try:
            yield key, token
        finally:
            stopped.set()
            thread.join()

    def fold_finished(self, reservation, folds):
        key, token = reservation
        return self.registry.progress(key, token, completed_folds=folds)

    def publish_complete(self, reservation, branch, manifest_path):
        key, token = reservation
        return self.registry.complete(key, token, branch, manifest_path, metrics)

    def share(self, study, record, params, distributions, config):
        manifest = self.registry.published(record["result_branch"], record["result_path"], record["config_hash"])
        return import_observation(study, record, params, distributions, config, metrics(manifest))
