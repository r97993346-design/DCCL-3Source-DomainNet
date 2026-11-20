"""Regression checks for direct observations, dual gradient paths, and recovery."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn
import yaml

from domainbed.algorithms.dicm import DICM
from domainbed.algorithms.dicm_gate import GateError
from domainbed.lib import causal_checkpoints as cp
from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley
from tools.pacs_directdiff.common import PROJECT


SHAPE = (3, 4, 4), 7, 3
SWAD = dict(n_converge=3, n_tolerance=6, tolerance_ratio=.3)


def config(gate):
    standard = yaml.safe_load((PROJECT / "tools/pacs_projection/A16.yaml").read_text())
    settings = yaml.safe_load(Path(__file__).with_name("direct.yaml").read_text())
    return dict(standard, **{**settings, "dicm_direction_gate": gate,
                              "dicm_gate_warmup": 2, "dicm_gate_interval": 1})


class Features(nn.Module):
    n_outputs = 32

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, self.n_outputs)

    def forward(self, x):
        return self.lin(x.flatten(1))


def tiny(shape, classes, domains, settings):
    def feature_factory(*args, **kwargs):
        assert kwargs["freeze"] == 0, "DirectDiff instantiated a frozen second encoder"
        return Features()

    with patch("domainbed.algorithms.dicm.networks.Featurizer", side_effect=feature_factory) as f:
        value = DICM(shape, classes, domains, settings)
        assert f.call_count == 1 and value.pre_featurizer is None
        return value


def new(settings, device):
    torch.manual_seed(12)
    return tiny(*SHAPE, settings).to(device)


def synthetic(device):
    gen = torch.Generator().manual_seed(91)
    y = torch.arange(7).repeat_interleave(4)
    shifts = torch.randn(3, 3, 4, 4, generator=gen)
    classes = torch.randn(7, 3, 4, 4, generator=gen)
    x = [classes[y] + shifts[i] + .25 * torch.randn(28, 3, 4, 4, generator=gen)
         for i in range(3)]
    aug = [z + .2 * torch.randn(z.shape, generator=gen) for z in x]
    return [t.to(device) for t in x], [y.to(device) for _ in x], [t.to(device) for t in aug]


def gradient_and_invalid(model, x, y, aug):
    first = model.featurizer(x[0])
    second = model.featurizer(aug[0])
    basis = model.projection.subspace.basis
    q = model.projection.subspace.orthonormal_basis()
    ce = model.classifier(model._stable(first, q)).square().mean()
    head = model.classifier[0].weight
    backbone = model.featurizer.lin.weight
    bgrad, hgrad, qgrad = torch.autograd.grad(ce, (backbone, head, basis), allow_unused=True)
    assert bgrad.norm() > 0 and hgrad.norm() > 0 and qgrad is None
    values = model._isr_loss(second - first, first - first.mean(0), q)
    loss, aug_loss, dom_loss, n_aug, n_dom, aw, dw = values
    assert n_aug and n_dom and (aw, dw) == (.25, .75)
    torch.testing.assert_close(loss, .25 * aug_loss + .75 * dom_loss)
    qb, bb = torch.autograd.grad(loss, (basis, backbone), allow_unused=True)
    assert qb.norm() > 0 and bb is None
    empty = first.detach().new_zeros((3, first.shape[1]))
    void, _, _, na, nd, _, _ = model._isr_loss(empty, empty, q)
    assert not void.requires_grad and na == nd == 0

    forward_refs = []
    capture = model.projection.subspace.coverage_loss
    all_x = torch.cat(x)
    expected_z = model.featurizer(all_x).detach()
    expected_aug = model.featurizer(torch.cat(aug)).detach() - expected_z

    def check_loss(data, *args):
        forward_refs.append(data)
        return capture(data, *args)

    with patch.object(model.projection.subspace, "coverage_loss", side_effect=check_loss):
        metrics = model.update(x, y, x_2=aug, step=0)
    torch.testing.assert_close(forward_refs[0], expected_aug)
    assert not forward_refs[0].requires_grad and not forward_refs[1].requires_grad
    assert metrics["valid_dom_response_count"] == 0 and metrics["isr_aug_weight_effective"] == .25
    expected_mean = expected_z[:len(x[0])][y[0] == 0].mean(0)
    torch.testing.assert_close(model.residual_bank.prototypes[0, 0], expected_mean)
    before = basis.detach().clone()
    metrics = model.update(x, y, x_2=x, step=1)
    assert metrics["valid_aug_response_count"] == metrics["valid_dom_response_count"] == 0
    torch.testing.assert_close(basis, before, atol=0, rtol=0)
    assert basis.grad is None
    # Environment 2 is unsupported for class 0; its invalid full-domain difference is omitted.
    counts = model.residual_bank.counts
    counts.fill_(8)
    model.residual_bank.update_counts.fill_(3)
    counts[0, 2] = 7
    responses = model.residual_bank.domain_responses()
    assert len(responses) == 18
    torch.testing.assert_close(responses.reshape(6, 3, 32).mean(1), torch.zeros(6, 32, device=before.device),
                               atol=1e-6, rtol=0)
    print("PASS: direct detached augmented/center differences, strict 0.25/0.75, CE gradients to head/backbone, ISR to Basis only")
    print("PASS: unsupported domains and invalid differences skipped with no stale Adam update")


def recovery(model, device, x, y, aug):
    sample = x[0][:2]
    averaged = AveragedModel(model)
    valley = LossValley(None, **SWAD)
    for step in range(4):
        model.update(x, y, x_2=aug, step=step)
        averaged.update_parameters(model, step=step)
    stable = model.predict_embed(sample)
    pooled = model.featurizer(sample)
    q = model.projection.subspace.orthonormal_basis(detach=True)
    if model.use_direction_gate:
        state = model.projection.gate
        v = q @ state.rotation
        torch.testing.assert_close(stable, pooled - .75 * ((pooled @ v) * state.gains) @ v.T)
        assert state.last_step.item() == 3 and state.gains.max() - state.gains.min() > .01
        assert bool((state.gains >= 0).all() and (state.gains <= 1).all())
        assert not any(p.requires_grad for p in state.parameters())
        assert torch.isfinite(state.operator_drift)
        # A gate with missing support must stop before any model change.
        try:
            state.refresh(q, model.residual_bank.prototypes,
                          torch.zeros_like(model.residual_bank.valid_mask()), pooled, 4)
        except GateError:
            pass
        else:
            raise AssertionError("Gate accepted a degenerate estimate")
        assert state.last_step.item() == 3
    else:
        torch.testing.assert_close(stable, pooled - .75 * (pooled @ q) @ q.T)
    assert averaged.module.reference is None
    torch.testing.assert_close(averaged.module.source_bank.prototypes, model.residual_bank.prototypes, atol=0, rtol=0)
    valley.update_and_evaluate(averaged, .8, 1.2, None)
    if model.use_direction_gate:
        saved_stats = averaged.module.finalize_direction_gate()
        assert saved_stats["step"] == 3
    with TemporaryDirectory(prefix="directdiff16-check-") as tmp:
        writer = cp.FoldCheckpoints(tmp, {"target": [0]}, SHAPE, model.hparams, model, 5001, 100)
        writer.checkpoint(model, valley, AveragedModel(model), sample, [dict(step=3, train_out=.8)])
        writer.save_inference(writer.final, "swad", averaged.to(device), sample, dict(start=0, end=3))
        assert cp.verify_inference(writer.best, device, tiny) == 0
        assert cp.verify_inference(writer.final, device, tiny) == 0

        clone = new(model.hparams, device)
        restore = LossValley(None, **SWAD)
        updates, _, segment = writer.resume(clone, restore, Path(tmp) / "results.jsonl", device)
        assert updates == 4 and clone.optimizer.state and segment is not None and len(restore.smooth_Q) == 1
        assert torch.equal(clone.residual_bank.counts, model.residual_bank.counts)
        for left, right in zip(clone.state_dict().values(), model.state_dict().values()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        model.update(x, y, x_2=aug, step=4)
        clone.update(x, y, x_2=aug, step=4)
        torch.testing.assert_close(cp.prediction(model, sample), cp.prediction(clone, sample), rtol=0, atol=0)
    print("PASS:", "Gate" if model.use_direction_gate else "Uniform",
          "IID/SWAD reload exact logits, complete rolling weights/optimizer/RNG/SWAD/source EMA recovery", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--group", choices=("Uniform", "Gate", "both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    x,y,aug = synthetic(args.device)
    if args.group in ("both", "Uniform"):
        gradient_and_invalid(new(config(False), args.device), x, y, aug)
        recovery(new(config(False), args.device), args.device, x, y, aug)
    if args.group in ("both", "Gate"):
        recovery(new(config(True), args.device), args.device, x, y, aug)
    print("PASS: DirectDiff16 training and checkpoint test suite", flush=True)


if __name__ == "__main__":
    main()
