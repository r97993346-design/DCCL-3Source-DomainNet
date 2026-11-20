"""Pair both new initializations and source splits to the real A16 and P0 checkpoints."""

from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sconf import Config
import torch
import yaml

from domainbed import trainer
from domainbed.algorithms.dicm import DICM, networks
from domainbed.datasets import get_dataset
from domainbed.lib.causal_checkpoints import shared_initialization
from domainbed.lib.stagewise import split_identity
from tools.pacs_directdiff.common import DATA, DOMAINS, GROUPS, PROJECT, ROOT, atomic_json, historical_fold, say
from tools.pacs_directdiff.queue import check_hparams, command, protocol


class InitialStateCaptured(Exception):
    pass


def effective(group, original):
    shown = subprocess.check_output([*command(group, 0), "--show"], cwd=PROJECT, text=True,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    parameters = yaml.safe_load(shown.split("checkpoint_freq=100\n", 1)[1])
    parameters.setdefault("total_num_domains", original["total_num_domains"])
    for key, old in original.items():
        if isinstance(old, float) and isinstance(parameters.get(key), str):
            parameters[key] = float(parameters[key])
    check_hparams(group, original, parameters)
    (ROOT / "configs" / f"{group}_effective.txt").write_text(shown, encoding="utf-8")
    return parameters


def capture(index, fold, paired, candidates):
    args = SimpleNamespace(**fold["args"])
    args.data_dir = DATA
    args.pacs_matched_protocol = None
    args.seed = args.trial_seed = 0
    args.algorithm = "DICM"
    hparams = Config(default=fold["hparams"])
    constructor = networks.Featurizer
    observation = {}

    def featurizer(*positional, **keywords):
        if keywords.get("freeze") != "all":
            return constructor(*positional, **keywords)
        before = torch.get_rng_state()
        ref = constructor(*positional, **keywords)
        observation["paired_rng"] = dict(before=sha256(before.numpy().tobytes()).hexdigest(),
                                         after=torch.get_rng_state().tolist())
        return ref

    def capture_algorithm(shape, classes, domains, hp):
        start = torch.get_rng_state()
        with patch.object(networks, "Featurizer", side_effect=featurizer):
            baseline = DICM(shape, classes, domains, hp)
        after = torch.get_rng_state()
        assert paired == shared_initialization(baseline), "A16 initializer differs from the saved run"
        initial_basis = baseline.projection.subspace.basis.detach().clone()
        assert initial_basis.shape == (2048, 16)
        del baseline
        for name, parameters in candidates.items():
            torch.set_rng_state(start)
            model = DICM(shape, classes, domains, parameters, paired_rng=observation["paired_rng"])
            assert model.pre_featurizer is None and model.get_forward_model().reference is None
            assert model.residual_bank.num_domains == 3 and model.mode == "A2"
            assert model.use_direction_gate == (name == "Gate")
            assert shared_initialization(model) == paired and torch.equal(torch.get_rng_state(), after)
            torch.testing.assert_close(model.projection.subspace.basis, initial_basis, atol=0, rtol=0)
        raise InitialStateCaptured()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # train_all performs the same startup dataset construction before the actual fold.
    get_dataset([0], args, hparams)
    dataset, train, val = get_dataset([index], args, hparams)
    actual_split = split_identity(dataset, train, val, args, [index])
    capture_algorithm.transforms = DICM.transforms
    with patch.object(trainer.algorithms, "get_algorithm_class", return_value=capture_algorithm):
        try:
            trainer.train([index], args=args, hparams=hparams, n_steps=5001, checkpoint_freq=100,
                          logger=logging.getLogger("directdiff_prepare"), writer=None)
        except InitialStateCaptured:
            pass
    return actual_split, observation["paired_rng"]


def run():
    torch.set_num_threads(2)
    offline = json.loads((ROOT / "offline" / "A16_gate_feasibility.json").read_text())
    assert offline["status"] == "passed" and len(offline["results"]) == len(DOMAINS)
    original = None
    for index, domain in enumerate(DOMAINS):
        folder, checkpoint = historical_fold("A16", index)
        old, = json.loads((folder / ".github-archive" / "evaluation.json").read_text())
        assert old["target"] == [domain] and old["steps"] == 5001
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        p0_folder, p0_path = historical_fold("P0", index)
        p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
        assert old["hparams"] == state["hparams"]
        assert p0["identity"] == state["identity"] and p0["initial_sha256"] == state["initial_sha256"]
        paired = state["initial_sha256"]
        identity = state["identity"]
        commit = state["source"]["commit"]
        del state, p0
        if original is None:
            original = old["hparams"]
            candidates = {group: effective(group, original) for group in GROUPS}
        assert original == old["hparams"]
        actual, rng = capture(index, old, paired, candidates)
        assert actual == identity
        atomic_json(protocol(index), dict(reference_group="A16", target=domain, identity=identity,
                                           initial_sha256=paired, hparams=original, paired_rng=rng,
                                           historical_commit=commit))
        say(domain, "verified same sampler RNG, splits, Backbone and Classifier as P0 and A16; paired fresh Rank16 basis")
    print("PASS: all four genuine pretrained initialization and protocol pairs", flush=True)


if __name__ == "__main__":
    run()
