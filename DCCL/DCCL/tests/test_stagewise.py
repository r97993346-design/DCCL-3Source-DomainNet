import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from domainbed.lib import stagewise


class ToyModel(nn.Module):
    def __init__(self, rank=2):
        super().__init__()
        self.featurizer = nn.Linear(3, 3)
        self.pre_featurizer = nn.Linear(3, 3)
        self.classifier = nn.Linear(3, 2)
        self.proj_head = nn.Linear(3, 2)
        self.pre_proj_head = nn.Linear(3, 2)
        self.projection = nn.Linear(rank, 3, bias=False)
        self.register_buffer("prototypes", torch.zeros(2, 3))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


def params(mode="A4"):
    return {"dicm_ablation": mode, "l": 0, "l_layer": 0, "model": "resnet50",
            "n_layer": 1, "resnet18": False, "pretrained": True, "freeze_bn": True,
            "resnet_dropout": 0, "dicm_rank": 2, "dicm_beta": 0.75,
            "dicm_residual_projection": True, "dicm_proto_momentum": 0.95,
            "dicm_min_domain_samples": 8, "dicm_min_domain_updates": 3,
            "dicm_min_valid_domains": 3, "dicm_min_delta_norm": 1e-4}


def save_parent(path, model, stage, identity, hparams):
    stagewise.save(path, model, hparams, None, identity, stage, 10, 10, None, None, None, 0.8)


def test_partial_transfer_and_domain_validation(tmp_path):
    identity = {"dataset": "PACS", "test_envs": [0], "split_sha256": "correct"}
    torch.manual_seed(1)
    parent = ToyModel()
    path = tmp_path / "stage0.pth"
    save_parent(path, parent, 0, identity, params("A1"))
    torch.manual_seed(2)
    child = ToyModel()
    basis_before = child.projection.weight.detach().clone()
    stagewise.load_parent(path, 1, child, identity, params())
    assert torch.equal(parent.featurizer.weight, child.featurizer.weight)
    assert torch.equal(parent.classifier.weight, child.classifier.weight)
    assert torch.equal(child.projection.weight, basis_before)
    with pytest.raises(ValueError, match="does not match"):
        stagewise.load_parent(path, 1, child, {**identity, "test_envs": [1]}, params())


def test_full_stage_two_transfer_and_metadata(tmp_path):
    identity = {"dataset": "VLCS", "test_envs": [2], "split_sha256": "fixed"}
    parent = ToyModel()
    parent.prototypes.fill_(3)
    path = tmp_path / "stage1.pth"
    save_parent(path, parent, 1, identity, params())
    value = torch.load(path, weights_only=False)
    assert not value["data_loader_state_saved"]
    assert {"python", "numpy", "torch", "cuda"} <= set(value["rng"])
    assert "state" in value["optimizer"]
    child = ToyModel()
    updates, fingerprint = stagewise.load_parent(path, 2, child, identity, params())
    assert updates == 10 and len(fingerprint) == 64
    for key, tensor in parent.state_dict().items():
        assert torch.equal(tensor, child.state_dict()[key])
    changed = {**params(), "dicm_beta": 0.9}
    with pytest.raises(ValueError, match="dicm_beta"):
        stagewise.load_parent(path, 2, child, identity, changed)


def test_stage_validation():
    base = params("A1")
    stagewise.validate_stage(0, base, None)
    stagewise.validate_stage(1, params(), "parent")
    stagewise.validate_stage(2, {**params(), "l": 1}, "parent")
    with pytest.raises(ValueError):
        stagewise.validate_stage(0, {**base, "l": 1}, None)
    with pytest.raises(ValueError):
        stagewise.validate_stage(2, {**params(), "l": 2}, "parent")


def test_split_fingerprint_changes_with_indices():
    class Part:
        def __init__(self, keys):
            self.keys = keys
            self.underlying_dataset = list(range(4))
    args = SimpleNamespace(dataset="PACS", trial_seed=0, seed=0, holdout_fraction=0.2, label_ratio=1.0)
    dataset = SimpleNamespace(environments=["A", "B"])
    first = stagewise.split_identity(dataset, [(Part([0, 1]), None)], [(Part([2]), None)], args, [1])
    second = stagewise.split_identity(dataset, [(Part([0, 2]), None)], [(Part([1]), None)], args, [1])
    assert first["split_sha256"] != second["split_sha256"]


def test_partial_transfer_preserves_aliased_network_weights(tmp_path):
    identity = {"dataset": "PACS", "test_envs": [0], "split_sha256": "correct"}
    parent, child = ToyModel(), ToyModel()
    parent.network = nn.Sequential(parent.featurizer, nn.Identity(), parent.classifier)
    child.network = nn.Sequential(child.featurizer, nn.Identity(), child.classifier)
    parent.featurizer.weight.data.fill_(1.5)
    parent.classifier.weight.data.fill_(2.5)
    path = tmp_path / "stage0.pth"
    save_parent(path, parent, 0, identity, params("A1"))
    stagewise.load_parent(path, 1, child, identity, params())
    assert torch.equal(parent.featurizer.weight, child.network[0].weight)
    assert torch.equal(parent.classifier.weight, child.network[2].weight)


def test_resume_restores_swad_optimizer_and_rng(tmp_path):
    from domainbed.swad import LossValley
    from domainbed.lib.swa_utils import AveragedModel

    identity = {"dataset": "VLCS", "test_envs": [0], "split_sha256": "fixed"}
    parent = ToyModel()
    loss = parent.featurizer(torch.ones(1, 3)).sum()
    loss.backward()
    parent.optimizer.step()
    swad = LossValley(None, n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    segment = AveragedModel(parent)
    segment.update_parameters(parent, step=0)
    swad.update_and_evaluate(segment, 0.5, 0.9, lambda *args: None)
    next_segment = AveragedModel(parent)
    next_segment.update_parameters(parent, step=1)
    path = tmp_path / "resume.pth"
    stagewise.save(path, parent, params(), None, identity, 1, 2, 12, "parent",
                   swad, next_segment, 0.8, [{"train_out": 0.8}])
    expected_random = torch.rand(3)
    resumed = ToyModel()
    resumed_swad = LossValley(None, n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    value, restored_segment = stagewise.resume(path, 1, identity, params(), resumed, resumed_swad)
    assert torch.equal(torch.rand(3), expected_random)
    assert len(resumed_swad.converge_Q) == len(resumed_swad.smooth_Q) == 1
    assert restored_segment.start_step == restored_segment.end_step == 1
    assert value["records"] == [{"train_out": 0.8}]
    assert len(resumed.optimizer.state) == len(parent.optimizer.state)
    for name in parent.state_dict():
        assert torch.equal(parent.state_dict()[name], resumed.state_dict()[name])
    with pytest.raises(ValueError, match="hyperparameters"):
        stagewise.resume(path, 1, identity, {**params(), "l": 1}, resumed, resumed_swad)
