import inspect

import pytest

torch = pytest.importorskip("torch")

from domainbed.algorithms.piccl import PICCL, InterventionSensitiveSubspace


def _dummy_piccl():
    obj = object.__new__(PICCL)
    torch.nn.Module.__init__(obj)
    obj.hparams = {"lr": 1e-3, "weight_decay": 0., "use_piccl": True,
                   "piccl_min_delta_norm": 1e-4, "piccl_isr_aug_weight": 1.,
                   "piccl_isr_dom_weight": .25}
    obj.use_piccl = True
    obj.featurizer = torch.nn.Linear(8, 8)
    obj.classifier = torch.nn.Linear(8, 2)
    obj.proj_head = torch.nn.Linear(8, 4)
    obj.mean_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.var_encoders = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
    obj.pre_proj_head = torch.nn.Linear(8, 4)
    obj.sensitive_subspace = InterventionSensitiveSubspace(8, 2)
    return obj


def test_only_basis_is_added_to_piccl_optimizer():
    obj = _dummy_piccl()
    groups = PICCL._optimizer_groups(obj)
    assert len(groups) == 7
    assert groups[-1]["lr"] == .00025


def test_isr_keeps_aug_and_domain_losses_separate_with_explicit_weights():
    obj = _dummy_piccl()
    obj.sensitive_subspace.coverage_loss = lambda values, *args, **kwargs: torch.tensor(float(values.shape[0]))
    total, aug, dom = PICCL._isr_loss(obj, torch.ones(96, 8), torch.ones(2, 8))
    assert aug.item() == 96 and dom.item() == 2 and total.item() == pytest.approx(96.5)


def test_no_gate_or_legacy_losses_remain():
    source = inspect.getsource(PICCL)
    for forbidden in ("ResidualGateFusion", "piccl_fusion_mode", "_nt_xent", "_cross_domain_supcon"):
        assert forbidden not in source
