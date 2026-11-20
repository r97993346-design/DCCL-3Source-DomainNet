"""A fold-specific reference estimated only from the original source train split."""
from pathlib import Path
import hashlib
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from domainbed.datasets.transforms import basic
from domainbed.lib import stagewise


class SourceView(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split.keys)

    def __getitem__(self, index):
        image, _ = self.split.underlying_dataset[self.split.keys[index]]
        return basic(image)


@torch.no_grad()
def domain_balanced_reference(reference, splits, source_envs, device, workers=2):
    """Mean each source domain separately with deterministic evaluation preprocessing."""
    if reference.training or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("Fixed-reference estimation requires a frozen evaluation encoder")
    rng = stagewise.rng_state()
    centers, sizes = [], {}
    try:
        for env in source_envs:
            view = SourceView(splits[env][0])
            if len(view) == 0:
                raise ValueError(f"Empty reference estimation split for source {env}")
            loader = DataLoader(view, batch_size=64, shuffle=False, num_workers=workers,
                                generator=torch.Generator().manual_seed(0),
                                pin_memory=str(device).startswith("cuda"))
            total = None
            count = 0
            for images in loader:
                pooled = reference(images.to(device))
                summed = pooled.detach().double().sum(0).cpu()
                total = summed if total is None else total + summed
                count += len(images)
                if hasattr(reference, "clear_features"):
                    reference.clear_features()
            assert count == len(view)
            centers.append(total / count)
            sizes[str(env)] = count
        mean = torch.stack(centers).mean(0).float()
        if not torch.isfinite(mean).all():
            raise FloatingPointError("Non-finite frozen reference estimate")
        return mean, sizes
    finally:
        stagewise.restore_rng(rng)


def write_split_manifest(out_dir, args, dataset, train, val, test_envs, identity):
    basis = Path(args.data_dir) / args.dataset
    splits = []
    for role, tables in (("in", train), ("out", val)):
        for env, (subset, _) in enumerate(tables):
            label_source = env not in test_envs
            rows = []
            for item in subset.keys:
                path, label = subset.underlying_dataset.samples[item]
                relative = Path(path).relative_to(basis).as_posix()
                row = {"sample_id": hashlib.sha256(relative.encode()).hexdigest()}
                if label_source:
                    row.update(path=relative, category=int(label))
                rows.append(row)
            partition = ("source_train" if role == "in" else "source_validation") if label_source else f"test_{role}"
            splits.append(dict(partition=partition, domain=dataset.environments[env], rows=rows))
    bundle = dict(identity=identity, splits=splits)
    data = (json.dumps(bundle, ensure_ascii=False, separators=(",",":"), sort_keys=True) + "\n").encode()
    manifest = Path(out_dir) / "sample_manifest.json"
    if manifest.is_file():
        if manifest.read_bytes() != data:
            raise ValueError("Existing PACS sample manifest differs")
    else:
        tmp = manifest.with_suffix(f".tmp-{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(manifest)
    return dict(file=manifest.name, sha256=hashlib.sha256(data).hexdigest())
