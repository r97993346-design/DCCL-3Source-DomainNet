#!/usr/bin/env bash
# Read two exact existing checkpoints on the user's server, export non-CLIP
# state and readable evidence, and push only those files from a new worktree.
set -Eeuo pipefail

CIPT_PROJECT_DIR="${CIPT_PROJECT_DIR:-/home/hooasia/lgg/projects/repro_dccl_3domain/DCCL/DCCL}"
CIPT_PYTHON="${CIPT_PYTHON:-python}"
CIPT_EXPORT_REL="DCCL/DCCL/analysis_results/cipt_checkpoints_2_3"
CIPT_EXPORT_TMP=$(mktemp -d /tmp/cipt-checkpoints-23.XXXXXX)
CIPT_EXPORT_WORKTREE="$CIPT_EXPORT_TMP/repository"
CIPT_EXPORT_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
CIPT_EXPORT_BRANCH="analysis/cipt-checkpoints-2-3-${CIPT_EXPORT_STAMP}-${CIPT_EXPORT_TMP##*.}"
export CIPT_PROJECT_DIR CIPT_EXPORT_TMP

test -d "$CIPT_PROJECT_DIR/domainbed"
git -C "$CIPT_PROJECT_DIR" rev-parse --show-toplevel >/dev/null

"$CIPT_PYTHON" - <<'PY'
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import torch

torch.set_num_threads(2)
root = Path(os.environ['CIPT_PROJECT_DIR']).resolve()
out = Path(os.environ['CIPT_EXPORT_TMP']) / 'export'
out.mkdir()
jobs = [
    ('02_original_lr5e-5_art', 'b5b',
     'train_output/cipt_mechanism/PACS/261023_14-43-14_cipt_original_te0_seed0/checkpoints/TE0_5000.pth',
     'fa03ca305faf18feba0347058713910a3c8664bf750b5e6c9cc2e3f58b860a42'),
    ('03_subject80_lr5e-5_art', 'b5a',
     'train_output/cipt_mechanism/PACS/261025_23-42-59_cipt_subject80_te0_seed0/checkpoints/TE0_5000.pth',
     '6c02a511830eab9245868b83662efd091cb0190e821e04b5b70e7e0b57e17e25'),
]
parameters = [
    'causal_decomposition.causal_adapter.weight',
    'causal_decomposition.causal_adapter.bias',
    'causal_decomposition.spurious_adapter.weight',
    'causal_decomposition.spurious_adapter.bias',
    'text_features.prompt_learner.context',
    'tda.attention.in_proj_weight',
    'tda.attention.in_proj_bias',
    'tda.attention.out_proj.weight',
    'tda.attention.out_proj.bias',
    'tda.layer_norm.weight',
    'tda.layer_norm.bias',
]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               default=str, allow_nan=False), encoding='utf-8')

def tensor_stats(value):
    x = value.detach().float().cpu()
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Non-finite checkpoint tensor')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'norm': float(x.norm()), 'mean': float(x.mean()),
            'rms': float(x.square().mean().sqrt()), 'max_abs': float(x.abs().max())}

# Check both exact files before creating an upload branch.
for label, mode, rel, expected in jobs:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f'Missing {label}: {path}')
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'{label}: SHA256 differs from the uploaded experiment metadata. '
                         f'Expected {expected}; got {actual}. No upload performed.')

states, records = {}, []
kwargs = {'map_location': 'cpu'}
if 'weights_only' in inspect.signature(torch.load).parameters:
    kwargs['weights_only'] = False  # These are the user's own trusted training files.

for label, mode, rel, expected in jobs:
    path = root / rel
    checkpoint = torch.load(path, **kwargs)
    hp = dict(checkpoint['model_hparams'])
    args = dict(checkpoint['args'])
    if (args.get('dataset') != 'PACS' or checkpoint['test_envs'] != [0]
            or hp.get('cipt_objective') != 'legacy'
            or hp.get('cipt_template_mode') != mode
            or not math.isclose(float(hp.get('lr', 0)), 5e-5, rel_tol=1e-9)
            or not hp.get('cipt_use_tda', False)):
        raise ValueError(f'{label}: checkpoint settings do not match the requested experiment')
    state = checkpoint['model_dict']
    for key in parameters:
        if key not in state:
            raise KeyError(f'{label}: missing learned tensor {key}')
    # All three aliases of the frozen CLIP have a clip_model path component.
    # Preserve every other state key, including text banks and prompt buffers.
    kept = {key: value for key, value in state.items() if 'clip_model' not in key.split('.')}
    removed = [key for key in state if key not in kept]
    if not removed:
        raise ValueError(f'{label}: no frozen CLIP keys recognized; stop before exporting a large file')
    states[label] = {key: kept[key].detach().float().clone() for key in parameters}
    packed = {key: value for key, value in checkpoint.items() if key != 'model_dict'}
    packed.update(
        model_dict=kept,
        export_format='domainbed_cipt_without_frozen_clip_v1',
        source_checkpoint=str(path), source_checkpoint_sha256=expected,
        reload_note='model_dict omits frozen CLIP keys. Restore the original CLIP separately; '
                    'this is not a full native state_dict for strict loading.',
    )
    destination = out / (label + '.pth')
    torch.save(packed, destination)
    if destination.stat().st_size >= 50 * 1024 * 1024:
        raise ValueError(f'{label}: export unexpectedly exceeds 50 MiB; no upload performed')
    record = {
        'label': label, 'source_checkpoint': str(path),
        'source_checkpoint_sha256': expected, 'source_bytes': path.stat().st_size,
        'export_file': destination.name, 'export_bytes': destination.stat().st_size,
        'export_sha256': digest(destination), 'export_format': packed['export_format'],
        'model_hparams': hp, 'args': args, 'test_envs': checkpoint['test_envs'],
        'learned_parameter_statistics': {key: tensor_stats(kept[key]) for key in parameters},
        'retained_state_inventory': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                     if torch.is_tensor(value) else {'type': type(value).__name__}
                                     for key, value in kept.items()},
        'excluded_frozen_clip_key_count': len(removed),
    }
    write_json(out / (label + '.json'), record)
    records.append(record)
    print(f'Prepared {destination.name}: {destination.stat().st_size / 1024**2:.2f} MiB', flush=True)
    del checkpoint, state, kept, packed

comparison = {'original': jobs[0][0], 'subject80': jobs[1][0], 'parameters': {}}
for key in parameters:
    a, b = states[jobs[0][0]][key], states[jobs[1][0]][key]
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {key}')
    difference = b - a
    na, nb = float(a.norm()), float(b.norm())
    entry = {'shape': list(a.shape), 'exactly_equal': bool(torch.equal(a, b)),
             'max_abs_difference': float(difference.abs().max()),
             'difference_norm': float(difference.norm()),
             'relative_difference_to_original': float(difference.norm()) / na if na else None,
             'flattened_cosine': float((a.flatten() * b.flatten()).sum()) / (na * nb) if na and nb else None}
    if key.startswith('causal_decomposition.') and key.endswith('.weight'):
        eye = torch.eye(a.shape[0], a.shape[1])
        entry['original_distance_from_identity_reference'] = float((a - eye).norm())
        entry['subject80_distance_from_identity_reference'] = float((b - eye).norm())
    comparison['parameters'][key] = entry
write_json(out / 'pair_parameter_comparison.json', comparison)
manifest = {
    'purpose': 'The two exact lr=5e-5 PACS-to-Art checkpoints, numbered 2 and 3.',
    'files': [{'label': r['label'], 'source_checkpoint_sha256': r['source_checkpoint_sha256'],
               'export_file': r['export_file'], 'export_sha256': r['export_sha256']} for r in records],
    'torch_version': str(torch.__version__),
    'upload_checkout_commit_not_training_commit': subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    'note': 'CPU export and parameter comparison only. No inference or generalization evaluation was executed. '
            'Original checkpoints remain at their original paths. Frozen CLIP is excluded; all other state is retained.',
}
write_json(out / 'manifest.json', manifest)
print('Both checkpoints verified and exported.', flush=True)
PY

git -C "$CIPT_PROJECT_DIR" worktree add -b "$CIPT_EXPORT_BRANCH" "$CIPT_EXPORT_WORKTREE" HEAD
mkdir -p "$CIPT_EXPORT_WORKTREE/$CIPT_EXPORT_REL"
cp -a "$CIPT_EXPORT_TMP/export/." "$CIPT_EXPORT_WORKTREE/$CIPT_EXPORT_REL/"
git -C "$CIPT_EXPORT_WORKTREE" add -f -- "$CIPT_EXPORT_REL"
git -C "$CIPT_EXPORT_WORKTREE" commit --only -m "Upload original and subject80 CIPT lr5e-5 Art checkpoint exports" -- "$CIPT_EXPORT_REL"
git -C "$CIPT_EXPORT_WORKTREE" push origin "HEAD:refs/heads/$CIPT_EXPORT_BRANCH"
printf '\nUpload complete.\nGitHub branch: %s\nDirectory: %s\n' "$CIPT_EXPORT_BRANCH" "$CIPT_EXPORT_REL"
