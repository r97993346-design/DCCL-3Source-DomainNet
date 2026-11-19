"""Evaluation weights and rolling recovery for the independent PACS folds."""
import hashlib
import json
import os
from pathlib import Path

import torch

from domainbed.lib import stagewise
from domainbed.lib.swa_utils import AveragedModel


SCHEMA = 1


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        with temp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def shared_initialization(model):
    hashes = {}
    for component in ("featurizer", "classifier"):
        digest = hashlib.sha256()
        for name, value in sorted(getattr(model, component).state_dict().items()):
            data = value.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{data.dtype}:{data.shape}".encode())
            digest.update(data.tobytes())
        hashes[component] = digest.hexdigest()
    return hashes


def prediction(model, sample):
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            return model.predict(sample).detach().cpu()
    finally:
        model.train(training)


def qr_basis(model):
    projection = model.network[1]
    if not hasattr(projection, "subspace"):
        return None
    return projection.subspace.orthonormal_basis(detach=True).cpu()


def load_inference(path, device, model_class=None):
    """Read only checkpoints made by this experiment, from the trusted run directory."""
    if model_class is None:
        from domainbed.algorithms.dicm import DICM
        model_class = DICM
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != SCHEMA or value.get("kind") not in ("iid_best", "swad"):
        raise ValueError("Unsupported PACS inference checkpoint")
    # All trained and reference weights are loaded below. Avoid downloading weights.
    model = model_class(*value["shape"], {**value["hparams"], "pretrained": False})
    if value["kind"] == "swad":
        model = AveragedModel(model, device="cpu")
        model.start_step, model.end_step = value["metrics"]["start"], value["metrics"]["end"]
    model.load_state_dict(value["model"], strict=True)
    return model.to(device).eval(), value


def verify_inference(path, device, model_class=None):
    with torch.random.fork_rng(devices=[]):
        model, saved = load_inference(path, device, model_class)
    actual = prediction(model, saved["example_input"].to(device))
    expected = saved["example_logits"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if saved["q"] is not None:
        q = qr_basis(model.module if saved["kind"] == "swad" else model)
        # Compare the span using r x r matrices, allowing basis sign changes.
        torch.testing.assert_close(torch.linalg.svdvals(q.T @ saved["q"]),
                                   torch.ones(q.shape[1]), rtol=1e-4, atol=1e-5)
    return float((actual - expected).abs().max().item())


class FoldCheckpoints:
    def __init__(self, directory, identity, shape, hparams, algorithm, budget, interval):
        self.directory = Path(directory)
        self.latest = self.directory / "latest.pth"
        self.best = self.directory / "iid_best.pth"
        self.final = self.directory / "swad.pth"
        self.metadata = dict(schema=SCHEMA, identity=identity, shape=shape, hparams=dict(hparams),
                             initial_sha256=shared_initialization(algorithm),
                             source=stagewise.source_provenance(), budget=budget, interval=interval,
                             torch_version=str(torch.__version__),
                             reference="frozen ImageNet-pretrained ResNet-50; state included when used")
        self.best_row = None

    def save_inference(self, path, kind, model, sample, metrics, pending=None):
        if pending is None:
            example_input = sample.detach().cpu()
            example_logits = prediction(model, sample)
        else:
            example_input, example_logits = pending
        inspect = model.module if kind == "swad" else model
        atomic_save(dict(self.metadata, kind=kind, model=model.state_dict(), metrics=metrics,
                         q=qr_basis(inspect), example_input=example_input,
                         example_logits=example_logits), path)

    def checkpoint(self, algorithm, swad, segment, sample, records):
        row = records[-1]
        pending = None
        if self.best_row is None or row["train_out"] > self.best_row["train_out"]:
            self.best_row = row
            pending = sample.detach().cpu(), prediction(algorithm, sample)
        atomic_save(dict(self.metadata, kind="latest", model=algorithm.state_dict(),
                         optimizer=algorithm.optimizer.state_dict(), rng=stagewise.rng_state(),
                         swad=stagewise.pack_swad(swad, segment), records=records,
                         updates=row["step"] + 1, best_row=self.best_row, pending_iid=pending,
                         exact_data_loader_resume=False), self.latest)
        if pending is not None:
            self.save_inference(self.best, "iid_best", algorithm, sample, row, pending)

    def resume(self, algorithm, swad, results_path, device):
        saved = torch.load(self.latest, map_location="cpu", weights_only=False)
        for key, value in self.metadata.items():
            if saved.get(key) != value:
                raise ValueError(f"PACS resume mismatch: {key}")
        if saved.get("kind") != "latest":
            raise ValueError("Expected a rolling PACS checkpoint")
        algorithm.load_state_dict(saved["model"], strict=True)
        algorithm.optimizer.load_state_dict(saved["optimizer"])
        segment = stagewise.restore_swad(saved["swad"], swad, algorithm)
        if segment is not None:
            segment.to(device)
        self.best_row = saved["best_row"]
        if saved["pending_iid"] is not None:
            self.save_inference(self.best, "iid_best", algorithm, None, self.best_row,
                                saved["pending_iid"])
        if not self.best.is_file():
            raise ValueError("Missing source-selected IID checkpoint")
        # The log can be one evaluation ahead if interrupted between evaluation and save.
        rows = [dict(row, hparams=self.metadata["hparams"]) for row in saved["records"]]
        temp = results_path.with_suffix(".resume.tmp")
        temp.write_text("".join(json.dumps(row, default=float) + "\n" for row in rows),
                        encoding="utf-8")
        os.replace(temp, results_path)
        stagewise.restore_rng(saved["rng"])
        return saved["updates"], saved["records"], segment
