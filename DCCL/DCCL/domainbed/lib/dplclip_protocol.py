"""Metadata, dataset checks and checkpoint handling for the DPLCLIP baseline."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision

from domainbed.lib import misc

OFFICIAL_COMMIT = "53b1ff4456a8c27fbe8f0e03d8d2136ac85ccace"
DOMAINS = {
    "PACS": ("art_painting", "cartoon", "photo", "sketch"),
    "VLCS": ("CALTECH", "LABELME", "PASCAL", "SUN"),
    "OfficeHome": ("Art", "Clipart", "Product", "RealWorld"),
    "TerraIncognita": ("L100", "L38", "L43", "L46"),
}

DEFAULTS = {
    "clip_backbone": "ViT-B/16",
    "dpl_clip_path": "",
    "num_domain_tokens": 16,
    "mlp_width": 512,
    "mlp_depth": 3,
    "mlp_dropout": 0.1,
    "lr": 0.001,
    "momentum": 0.1,
    "weight_decay": 0.0,
    "optimizer": "sgd",
    "batch_size": 32,
    "test_batchsize": 64,
    "swad": None,
    "indomain_test": False,
    "data_augmentation": True,  # Registry default; upstream clip_transform takes precedence.
    "val_augment": False,
    "class_balanced": False,
    "label_ratio": 1.0,
    "sample_d": False,
    "mix": 0.0,
}


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy())
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def loader_generator(seed, split, env):
    return torch.Generator().manual_seed(misc.seed_hash(seed, "DPLCLIP", split, env))


def dataset_metadata(dataset, in_splits, out_splits, args):
    if tuple(dataset.environments) != DOMAINS[args.dataset]:
        raise ValueError(f"Unexpected {args.dataset} environment order: {dataset.environments}")
    class_names = list(dataset.datasets[0].classes)
    metadata = []
    for env, folder in enumerate(dataset.datasets):
        if folder.class_to_idx != {name: i for i, name in enumerate(class_names)}:
            raise ValueError(f"Class index mismatch in {dataset.environments[env]}")
        count = len(folder)
        seed = misc.seed_hash(args.trial_seed, env)
        expected = np.random.RandomState(seed).permutation(count)
        cutoff = int(args.holdout_fraction * count)
        in_keys = in_splits[env][0].keys
        out_keys = out_splits[env][0].keys
        if out_keys != expected[:cutoff].tolist() or sorted(in_keys) != sorted(expected[cutoff:]):
            raise ValueError(f"DomainBed split membership mismatch for environment {env}")
        digest = hashlib.sha256()
        for path, label in folder.samples:
            relative = Path(path).relative_to(folder.root).as_posix()
            digest.update(f"{relative}\t{label}\n".encode("utf-8"))
        metadata.append({
            "index": env, "name": dataset.environments[env], "count": count,
            "in_count": len(in_keys), "out_count": len(out_keys),
            "split_seed": seed, "sample_index_sha256": digest.hexdigest(),
            "in_order_sha256": hashlib.sha256(np.asarray(in_keys, dtype="<i8").tobytes()).hexdigest(),
        })
    return class_names, metadata


def begin_fold(args, hparams, dataset, in_splits, out_splits, n_steps, freq,
               train_workers, eval_workers, logger):
    import clip
    from clip.clip import _MODELS

    class_names, domains = dataset_metadata(dataset, in_splits, out_splits, args)
    hparams["class_names"] = class_names
    model_url = _MODELS[hparams["clip_backbone"]]
    expected_sha256 = model_url.split("/")[-2]
    if hparams["dpl_clip_path"]:
        path = Path(hparams["dpl_clip_path"])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Local CLIP checkpoint differs from the official OpenAI ViT-B/16")
    config = {
        "algorithm": "DPLCLIP", "dataset": args.dataset,
        "seed": args.seed, "trial_seed": args.trial_seed,
        "target_index": args.real_test_envs[0],
        "source_indices": [row["index"] for row in domains if row["index"] not in args.real_test_envs],
        "domains": domains, "steps": n_steps, "checkpoint_frequency": freq,
        "steps_are_zero_based_after_update": True,
        "eval_steps": sorted(set(range(0, n_steps, freq)) | {n_steps - 1}),
        "holdout_fraction": args.holdout_fraction, "train_workers": train_workers,
        "eval_workers": eval_workers, "hparams": dict(hparams),
        "data_dir": str(args.data_dir), "command": sys.argv,
        "selection": "first max source_val (unweighted mean of three source out splits)",
        "source_sampling": "independent replacement sampling per source in split",
        "eval_sampling": "sequential split-local batches, retain final short batch",
        "preprocessing": "OpenAI CLIP bicubic resize, center crop, RGB, tensor, normalize",
        "training_objective": "DPL paper Eq. 6, mean source-domain K-way CE",
        "clip_frozen": True, "clip_model_url": model_url, "clip_sha256": expected_sha256,
        "lr_scheduler": None, "hparams_seed": 0, "search": False,
        "official_dpl_commit": OFFICIAL_COMMIT,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "pillow": PIL.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "clip_module": clip.__file__,
            "gpu": torch.cuda.get_device_name(), "cudnn_deterministic": args.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(args.out_dir / "config.json", config)
    logger.info("DPLCLIP labels in index order: %s", class_names)
    logger.info("DPLCLIP split counts: %s", [(r["name"], r["in_count"], r["out_count"]) for r in domains])
    return config


def save_best(path, algorithm, config, record):
    temporary = path.with_suffix(".pth.tmp")
    state = {k: v.detach().cpu() for k, v in algorithm.prompt_generator.state_dict().items()}
    torch.save({"prompt_generator": state, "step": record["step"], "config": config}, temporary)
    temporary.replace(path)


def finalize_fold(args, algorithm, evaluator, records, logger):
    best = records.argmax("source_val")
    checkpoint = torch.load(args.out_dir / "best_prompt_generator.pth", map_location="cpu",
                            weights_only=True)
    if checkpoint["step"] != best["step"]:
        raise RuntimeError("Saved DPL checkpoint does not match the source-IID argmax")
    algorithm.prompt_generator.load_state_dict(checkpoint["prompt_generator"])
    accs, summaries = evaluator.evaluate(algorithm)
    for key, value in {**accs, **summaries}.items():
        if not np.isclose(value, best[key], rtol=0, atol=1e-7):
            raise RuntimeError(f"Restored DPL checkpoint differs for {key}: {value} vs {best[key]}")
    summary = {
        "dataset": args.dataset, "target_index": args.real_test_envs[0],
        "target_domain": DOMAINS[args.dataset][args.real_test_envs[0]], "seed": args.seed,
        "best_step": best["step"], "best_source_val": best["source_val"],
        "train_out": summaries["train_out"], "test_in": summaries["test_in"],
        "test_out": summaries["test_out"], "per_split_accuracy": accs,
        "evaluations": len(records), "restored_checkpoint_verified": True,
        "status": "complete",
    }
    write_json(args.out_dir / "best_summary.json", summary)
    logger.info("Best DPL source IID: step=%d, source_val=%.6f, test_in=%.6f, test_out=%.6f",
                best["step"], best["source_val"], summaries["test_in"], summaries["test_out"])
    return {"iid": summaries["test_out"], "iid (inD)": best["source_val"],
            "iid target in": summaries["test_in"]}, records
