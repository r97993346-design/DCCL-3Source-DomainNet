"""Extract actual features from a source-selected PACS checkpoint."""
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import ImageFilter, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from domainbed.datasets import get_dataset
from domainbed.datasets.transforms import basic
from domainbed.lib.causal_checkpoints import load_inference, prediction, qr_basis
from domainbed.lib.stagewise import split_identity
from tools.pacs_causal.queue import DATA


REPRESENTATIONS = ("z", "E_beta", "S_raw", "S_removed", "E_full_res", "E_orth")
STYLES = ("gaussian_blur_1p6", "posterize_4bit")


def source_samples(saved):
    target, = saved["identity"]["test_envs"]
    args = SimpleNamespace(dataset="PACS", data_dir=DATA, seed=0, trial_seed=0,
                           holdout_fraction=0.2, label_ratio=1.0)
    dataset, train, validation = get_dataset([target], args, saved["hparams"])
    identity = split_identity(dataset, train, validation, args, [target])
    if saved["identity"] != identity:
        raise ValueError("Analysis dataset or source split differs from the checkpoint")
    rows = []
    classes = None
    for partitions, split in ((train, "train"), (validation, "validation")):
        for domain, (subset, _) in enumerate(partitions):
            if domain == target:
                continue
            base = subset.underlying_dataset
            if classes is None:
                classes = list(base.classes)
            assert classes == base.classes
            for key in subset.keys:
                path, label = base.samples[key]
                relative = Path(path).relative_to(DATA / "PACS").as_posix()
                sample_id = hashlib.sha256(relative.encode()).hexdigest()
                rows.append(dict(sample_id=sample_id, path=path, split=split,
                                 domain=domain, category=label))
    assert len({row["sample_id"] for row in rows}) == len(rows)
    return rows, classes


class ImageViews(Dataset):
    def __init__(self, rows, style=None):
        self.rows = rows
        self.style = style

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.rows[index]["path"]) as handle:
            image = handle.convert("RGB")
        if self.style == STYLES[0]:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.6))
        elif self.style == STYLES[1]:
            image = ImageOps.posterize(image, bits=4)
        elif self.style is not None:
            raise ValueError(self.style)
        return basic(image)


@torch.inference_mode()
def encode(model, rows, device, style=None):
    loader = DataLoader(ImageViews(rows, style), batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=device == "cuda")
    current, reference = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        current.append(model.featurizer(batch).float().cpu().numpy())
        reference.append(model.pre_featurizer(batch).float().cpu().numpy())
    return np.concatenate(current), np.concatenate(reference)


def project(x, q):
    return (x @ q) @ q.T


def decompose(z, z_ref, q, beta):
    sensitive = project(z - z_ref, q)
    stable = z - beta * sensitive
    return dict(z=z, E_beta=stable, S_raw=sensitive, S_removed=z - stable,
                E_full_res=z - sensitive, E_orth=z - project(z, q))


def balanced_indices(labels, domains, selection, rng, cap=None):
    """Use one common count per class/domain cell on each independent split."""
    cells = [np.flatnonzero(selection & (labels == y) & (domains == d))
             for y in np.unique(labels) for d in np.unique(domains)]
    count = min(map(len, cells))
    if cap is not None:
        count = min(count, cap)
    if count < 2:
        raise ValueError("At least two examples per source class/domain cell are required")
    return np.sort(np.concatenate([rng.choice(cell, count, replace=False) for cell in cells])), count


def save_samples(path, rows, domains, classes, balanced, tsne):
    b_indices, t_indices = set(balanced.tolist()), set(tsne.tolist())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "source_domain", "category", "domain_probe", "tsne"])
        for i, row in enumerate(rows):
            writer.writerow([row["sample_id"], row["split"], domains[row["domain"]],
                             classes[row["category"]], i in b_indices, i in t_indices])


def extract(checkpoint, device, directory):
    model, saved = load_inference(checkpoint, device)
    assert saved["kind"] == "iid_best" and model.use_projection and model.use_residual
    assert saved["hparams"]["dicm_ablation"] == "A2"
    if device == "cuda":
        # The recorded logits came from CUDA. Formal analysis uses the same GPU and settings.
        torch.testing.assert_close(prediction(model, saved["example_input"].to(device)),
                                   saved["example_logits"], rtol=1e-5, atol=1e-6)
    rows, classes = source_samples(saved)
    target, = saved["identity"]["test_envs"]
    domains = saved["identity"]["domains"]
    labels = np.asarray([r["category"] for r in rows])
    envs = np.asarray([r["domain"] for r in rows])
    train = np.asarray([r["split"] == "train" for r in rows])
    rng = np.random.default_rng(20261119 + target)
    domain_train, per_cell_train = balanced_indices(labels, envs, train, rng)
    domain_validation, per_cell_validation = balanced_indices(labels, envs, ~train, rng)
    tsne, per_cell_tsne = balanced_indices(labels, envs, ~train, rng, cap=24)
    intervention, _ = balanced_indices(labels, envs, ~train, rng, cap=48)

    q = qr_basis(model).numpy()
    assert q.shape[1] == saved["hparams"]["dicm_rank"]
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-5)
    z, z_ref = encode(model, rows, device)
    beta = float(model.projection.beta.item())
    reps = decompose(z, z_ref, q, beta)
    z_changes, residual_changes = {}, {}
    for style in STYLES:
        styled, styled_ref = encode(model, [rows[i] for i in intervention], device, style)
        dz = styled - z[intervention]
        z_changes[style] = dz
        residual_changes[style] = dz - (styled_ref - z_ref[intervention])

    alpha_results = {}
    with torch.inference_mode():
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            e = torch.from_numpy(z[~train] - alpha * reps["S_raw"][~train])
            output = torch.cat([model.classifier(each.to(device)).argmax(1).cpu()
                                for each in e.split(256)]).numpy()
            alpha_results[str(alpha)] = float((output == labels[~train]).mean())
    directory.mkdir(parents=True, exist_ok=True)
    save_samples(directory / "samples.csv", rows, domains, classes,
                 np.concatenate((domain_train, domain_validation)), tsne)
    np.savez(directory / "features.npz", **reps, z_ref=z_ref, q=q, y=labels, domain=envs,
             is_source_train=train, sample_id=np.asarray([r["sample_id"] for r in rows]),
             domain_train=domain_train, domain_validation=domain_validation, tsne_indices=tsne,
             intervention_indices=intervention,
             **{f"delta_r_{name}": value for name, value in residual_changes.items()},
             **{f"delta_z_{name}": value for name, value in z_changes.items()})
    counts = dict(source_train=int(train.sum()), source_validation=int((~train).sum()),
                  domain_train_per_cell=per_cell_train,
                  domain_validation_per_cell=per_cell_validation, tsne_per_cell=per_cell_tsne,
                  intervention_examples=len(intervention))
    return dict(representations=reps, z_ref=z_ref, q=q, beta=beta, y=labels, env=envs,
                train=train, domain_train=domain_train, domain_validation=domain_validation,
                tsne=tsne, intervention=intervention, dz=z_changes, dr=residual_changes,
                source=rows, classes=classes, domains=domains, counts=counts, target=target,
                alpha=alpha_results, checkpoint=str(checkpoint), provenance=saved["source"],
                split_sha256=saved["identity"]["split_sha256"])
