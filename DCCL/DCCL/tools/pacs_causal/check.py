"""PACS training prerequisites. Run as a module with one visible GPU."""
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from domainbed.algorithms.dicm import DICM
from domainbed.lib.causal_checkpoints import FoldCheckpoints, prediction, shared_initialization, verify_inference
from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley


def settings(name):
    with Path(__file__).with_name(name + ".yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def real_initialization():
    hashes = []
    for name in ("P0", "P1"):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model = DICM((3, 224, 224), 7, 3, settings(name))
        hashes.append(shared_initialization(model))
        del model
    assert hashes[0] == hashes[1], hashes
    print("PASS: pretrained ResNet-50 and classifier states match bit for bit", hashes[0])


def small_model(shape, n_classes, n_domains, hparams):
    class Tiny(nn.Module):
        n_outputs = 16

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3 * 4 * 4, self.n_outputs)

        def forward(self, x):
            return self.layer(x.flatten(1))

    with patch("domainbed.algorithms.dicm.networks.Featurizer", side_effect=lambda *a, **kw: Tiny()):
        return DICM(shape, n_classes, n_domains, hparams)


def roundtrip(hp, device, directory):
    shape = ((3, 4, 4), 7, 3)
    torch.manual_seed(0)
    a = small_model(*shape, hp).to(device).train()
    cp = FoldCheckpoints(directory, {"target": [0]}, shape, hp, a, 5001, 100)
    first = AveragedModel(a)
    kwargs = dict(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    valley = LossValley(None, **kwargs)
    x = [torch.randn(7, *shape[0], device=device) for _ in range(3)]
    y = [torch.arange(7, device=device) for _ in range(3)]
    sample = x[0][:2]
    rows = []
    for step, validation_loss in enumerate((0.9, 1.0, 1.1, 0.8, 0.7, 0.8, 0.7)):
        a.update(x, y, x_2=[v + torch.randn_like(v) * 0.1 for v in x], step=step)
        first.update_parameters(a, step=step)
        valley.update_and_evaluate(first, 0.5, validation_loss, None)
        first = AveragedModel(a)
        rows.append(dict(step=step, test_in=0.3 + 0.01 * step, test_out=0.4,
                         train_out=0.5 - 0.01 * abs(step - 3), tr_outloss=validation_loss))
        cp.checkpoint(a, valley, first, sample, rows)
    observed_next_draw = torch.rand(3, device=device)
    assert valley.is_converged
    assert cp.best_row["step"] == 3
    final = valley.get_final_model().to(device)
    cp.save_inference(cp.final, "swad", final, sample,
                      {"start": final.start_step, "end": final.end_step})
    for path in [cp.best, cp.final]:
        print("PASS", hp["dicm_ablation"], path.name,
              "max |logit diff|", verify_inference(path, device, small_model))
    expected = prediction(a, sample)
    torch.manual_seed(0)
    b = small_model(*shape, hp).to(device).train()
    other = FoldCheckpoints(directory, {"target": [0]}, shape, hp, b, 5001, 100)
    restored_valley = LossValley(None, **kwargs)
    count, history, segment = other.resume(b, restored_valley, directory / "results.jsonl", device)
    assert b.optimizer.state and count == 7 and history == rows
    assert restored_valley.is_converged and len(restored_valley.smooth_Q) == 6
    assert len(restored_valley.converge_Q) == 3 and segment.n_averaged.item() == 0
    torch.testing.assert_close(torch.rand(3, device=device), observed_next_draw, atol=0, rtol=0)
    torch.testing.assert_close(prediction(b, sample), expected, atol=0, rtol=0)
    torch.testing.assert_close(prediction(restored_valley.get_final_model().to(device), sample),
                               prediction(final, sample))
    if a.use_projection:
        torch.testing.assert_close(a.residual_bank.prototypes, b.residual_bank.prototypes)
        torch.testing.assert_close(a.projection.subspace.basis, b.projection.subspace.basis)
    print("PASS:", hp["dicm_ablation"], "rolling resume restored optimizer, RNG, SWAD and model")


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA check:", torch.cuda.get_device_name(0), torch.cuda.mem_get_info(0))
    real_initialization()
    for name in ("P0", "P1"):
        with tempfile.TemporaryDirectory(prefix="dicm-causal-test-") as path:
            roundtrip(settings(name), "cuda", Path(path))


if __name__ == "__main__":
    main()
