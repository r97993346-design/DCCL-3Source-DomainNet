"""Optional finite-epoch, pooled-source training for CIPTDCCL.

Only source in-splits are pooled. Each epoch visits every training example
once, without replacement, including the final partial batch. Regrouping an
already sampled batch by its real domain preserves the existing algorithm's
list-of-source-batches interface without balancing or resampling the domains.
"""

from numbers import Integral

import torch
from torch.utils.data import ConcatDataset, DataLoader


def validate_cipt_training_options(hparams, requested_steps=None):
    sampling = str(hparams.get("cipt_train_sampling", "domain_balanced")).lower()
    epochs = hparams.get("cipt_train_epochs", 0)
    lr_unit = str(hparams.get("cipt_lr_step_unit", "step")).lower()
    if sampling not in ("domain_balanced", "mixed"):
        raise ValueError("cipt_train_sampling must be domain_balanced or mixed")
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 0:
        raise ValueError("cipt_train_epochs / --epochs must be a non-negative integer")
    if lr_unit not in ("step", "epoch"):
        raise ValueError("cipt_lr_step_unit must be step or epoch")
    if sampling == "mixed":
        if epochs < 1:
            raise ValueError("mixed sampling requires --epochs / cipt_train_epochs > 0")
        if requested_steps is not None:
            raise ValueError("--steps and mixed epoch training are mutually exclusive")
        if hparams.get("class_balanced", False):
            raise ValueError("mixed epoch training requires class_balanced=false")
    elif epochs or lr_unit == "epoch":
        raise ValueError("epoch training and epoch LR stepping require cipt_train_sampling=mixed")
    return sampling, int(epochs), lr_unit


def group_mixed_batch_by_domain(batch, source_envs):
    """Keep sampled membership and tensor alignment, including true domain IDs."""
    domains = batch["d"]
    groups = []
    for domain in source_envs:
        mask = domains == domain
        if mask.any():
            groups.append({key: value[mask] for key, value in batch.items()})
    if sum(len(group["d"]) for group in groups) != len(domains):
        raise ValueError("mixed training batch contains a non-source domain")
    return groups


class MixedSourceDataLoader:
    """One finite shuffled pass over the union of SOURCE training in-splits."""

    def __init__(self, in_splits, test_envs, batch_size, num_workers, seed):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("mixed batch_size must be a positive integer (TOTAL images)")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.source_envs = tuple(i for i in range(len(in_splits)) if i not in test_envs)
        if not self.source_envs:
            raise ValueError("mixed training requires at least one source domain")
        sources = []
        for domain in self.source_envs:
            dataset, weights = in_splits[domain]
            if len(dataset) == 0:
                raise ValueError("source domain {} has no training samples".format(domain))
            if weights is not None:
                raise ValueError("mixed training does not use class-balanced replacement sampling")
            sources.append(dataset)
        self.dataset = ConcatDataset(sources)
        # A private generator keeps evaluation-loader RNG consumption from
        # changing the source permutation for the following epoch.
        generator = torch.Generator().manual_seed(seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            generator=generator,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield group_mixed_batch_by_domain(batch, self.source_envs)


def repeat_epochs(loader):
    """Keep the trainer's iterator API while reshuffling at each finite epoch."""
    while True:
        yield from loader
