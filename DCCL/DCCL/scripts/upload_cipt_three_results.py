#!/usr/bin/env python3
"""Collect the three selected PACS runs and push an isolated results branch.

Uses Python's standard library and Git only. Never runs training or changes
the checked-out branch, worktree, config, or normal Git index. Weights and
TensorBoard event files are inventoried rather than uploaded.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

REPOSITORY = 'r97993346-design/DCCL-3Source-DomainNet'
PROJECT = '/home/hooasia/lgg/projects/cipt_gpu0'
OLD_PROJECT = '/home/hooasia/lgg/projects/repro_dccl_3domain'
SPECS = [
    ('01_nr_3800', 'nr_result', 'gpu1_center_mse_3exp',
     '261101_17-54-27_pacs_es1_ind0_center_mse1_all80_nr1_lr5e5_constant_3800', 3800, True),
    ('02_nocon_3800', 'nocon_result', 'gpu1_pacs_nocon_after_queue_2exp',
     '261102_01-49-20_pacs_es1_ind0_center_mse1_all80_nocon_lr5e5_constant_3800', 3800, False),
    ('03_nocon_5000', 'old_result', 'rd_forms_all80_gpu1',
     '261027_22-02-36_pacs_all80_center_mse_e1_s1_rd1_swad', 5000, False),
]
SELECTED_RUNS = {
    '01_nr_3800': ('run_20261101_175424_UywWqT', '01_center_mse_lr5e5_constant_3800'),
    '02_nocon_3800': ('run_20261102_014917_2gr2Nj', '01_center_mse_nocon_lr5e5_constant_3800'),
    '03_nocon_5000': ('run_20261027_220233_ZOri6v', 'pacs_all80_center_mse_e1_s1_rd1_swad'),
}
SOURCE_FILES = [
    'train_all.py', 'domainbed/trainer.py', 'domainbed/evaluator.py',
    'domainbed/datasets/__init__.py', 'domainbed/lib/fast_data_loader.py',
    'domainbed/lib/cipt_lr_schedule.py', 'domainbed/hparams_registry.py',
    'domainbed/algorithms/cipt_env_kl.py', 'domainbed/algorithms/cipt_losses.py',
    'domainbed/algorithms/cipt_dccl_ablation.py',
    'domainbed/algorithms/cipt_neighbor_contrastive.py',
    'domainbed/algorithms/cipt_modules.py', 'domainbed/algorithms/cipt_prompt.py',
    'domainbed/algorithms/algorithms.py',
]
TEXT_EXTENSIONS = {'.txt', '.json', '.jsonl', '.yaml', '.yml', '.sh', '.csv', '.tsv', '.md'}
WEIGHT_EXTENSIONS = {'.pth', '.pt', '.ckpt', '.bin', '.safetensors', '.pkl', '.pickle', '.onnx'}
LIMIT = 90 * 1024 * 1024


def fail(message):
    raise SystemExit('ERROR: ' + message)


def git(root, *args, env=None, data=None, optional=False):
    process = subprocess.run(['git', '-C', str(root), *args], input=data,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if process.returncode and not optional:
        if args[0] in {'push', 'ls-remote', 'remote'}:
            fail('Git ' + args[0] + ' failed. Check the server GitHub connection and retry.')
        fail('Git ' + args[0] + ' failed: ' + process.stderr.decode(errors='replace').strip())
    return None if process.returncode else process.stdout


def trainer_dirs(project):
    root = Path(project).expanduser().resolve()
    return [p for p in [root / 'DCCL/DCCL', root / 'DCCL', root]
            if (p / 'train_all.py').is_file()]


def find_result(spec, args):
    key, option, family, leaf, _, _ = spec

    def is_result(directory):
        # TensorBoard writes PACS/runs/<same_unique_name>; the name alone
        # does not identify an experiment result directory.
        return (directory.is_dir() and not directory.is_symlink()
                and (directory / 'log.txt').is_file()
                and (directory / 'results.jsonl').is_file())

    explicit = getattr(args, option)
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not is_result(result):
            fail('--' + option.replace('_', '-') + ' must contain log.txt and results.jsonl: ' + str(result))
        return result
    projects = [args.old_project, args.project] if key.startswith('03') else [args.project]
    for project in projects:
        for train_dir in trainer_dirs(project):
            for parent in ['train_output/ablation', 'analysis_results']:
                family_dir = train_dir / parent / family
                run_name, experiment = SELECTED_RUNS[key]
                selected = family_dir / run_name / experiment / 'PACS' / leaf
                if is_result(selected):
                    return selected.resolve()
                candidates = sorted({p.resolve() for p in family_dir.glob('run_*/**/' + leaf)
                                     if is_result(p)})
                if len(candidates) == 1:
                    return candidates[0].resolve()
                if len(candidates) > 1:
                    fail('Multiple result directories for ' + key + '; provide --'
                         + option.replace('_', '-') + ' with one of:\n  '
                         + '\n  '.join(str(p) for p in candidates))
    fail('Cannot find ' + key + ' (' + leaf + '). Provide --' + option.replace('_', '-')
         + ' /absolute/path/to/the/directory/containing/log.txt')


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def csv_bytes(rows):
    if not rows:
        return b''
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def target_of(row):
    target = row.get('args', {}).get('real_test_envs')
    if isinstance(target, list) and len(target) == 1 and target[0] in range(4):
        return target[0]
    # evalmode=fast leaves a single target envN_in column.
    fields = [int(k[3:-3]) for k in row if re.fullmatch(r'env[0-3]_in', k)]
    if len(fields) == 1:
        return fields[0]
    fail('Cannot determine the target domain for a results.jsonl record.')


def validate_rows(content, spec):
    key, _, _, _, budget, contrastive = spec
    try:
        rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(key + ': invalid results.jsonl: ' + str(error))
    by_target = {te: [] for te in range(4)}
    for row in rows:
        for metric in ['step', 'train_out', 'test_in', 'test_out', 'tr_outloss']:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                fail(key + ': missing/nonfinite ' + metric)
        for metric in ['train_out', 'test_in', 'test_out']:
            if not 0 <= row[metric] <= 1:
                fail(key + ': accuracy outside [0,1]: ' + metric)
        hp = row.get('hparams', {})
        if not math.isclose(float(hp.get('lr', -1)), 5e-5, rel_tol=1e-10):
            fail(key + ': resolved LR is not 5e-5.')
        if hp.get('cipt_lr_schedule', 'constant') != 'constant':
            fail(key + ': resolved LR schedule is not constant.')
        if bool(hp.get('cipt_use_contrastive')) != contrastive:
            fail(key + ': resolved contrastive switch does not match the requested experiment.')
        if hp.get('cipt_env_constraint') != 'center_mse' or hp.get('cipt_env_selection') != 'all':
            fail(key + ': expected all80 center_mse; check the selected result directory.')
        by_target[target_of(row)].append(row)
    for te, values in by_target.items():
        if [r['step'] for r in values] != list(range(0, budget + 1, 100)):
            fail(key + ': target ' + str(te) + ' is incomplete or has duplicate/unexpected steps. '
                 'Expected 0,100,...,' + str(budget))
    return by_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--old-project', default=OLD_PROJECT)
    for name in ['nr-result', 'nocon-result', 'old-result']:
        parser.add_argument('--' + name)
    parser.add_argument('--dry-run', action='store_true', help='Validate and print; do not commit or push.')
    args = parser.parse_args()
    push_root = Path(git(args.project, 'rev-parse', '--show-toplevel').decode().strip())
    remote = git(push_root, 'remote', 'get-url', 'origin').decode().strip().rstrip('/')
    if not remote.endswith(('/' + REPOSITORY, '/' + REPOSITORY + '.git',
                            ':' + REPOSITORY, ':' + REPOSITORY + '.git')):
        fail('origin does not match ' + REPOSITORY)
    head = git(push_root, 'rev-parse', 'HEAD').decode().strip()
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    token = now.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    branch = 'analysis/pacs-center-three-results-' + token
    destination = 'DCCL/DCCL/analysis_results/pacs_center_mse_three_runs/run_' + token
    payload, tracked, groups, manifest_groups = {}, {}, {}, []

    def capture(path, target):
        if path.is_symlink():
            fail('Selected upload file is a symlink: ' + str(path))
        before = signature(path)
        if before[2] > LIMIT:
            fail('File exceeds 90 MiB: ' + str(path))
        data = path.read_bytes()
        if signature(path) != before:
            fail('File changed during snapshot: ' + str(path) + '. Retry after completion.')
        tracked[path] = before
        payload[target] = data
        return data

    for spec in SPECS:
        key = spec[0]
        result = find_result(spec, args)
        run_root, exp_root = result.parents[2], result.parents[1]
        source_repo = Path(git(result, 'rev-parse', '--show-toplevel').decode().strip())
        source_train = next((p for p in result.parents if (p / 'train_all.py').is_file()), None)
        if source_train is None:
            fail('Cannot locate train_all.py above ' + str(result))
        log_data = capture(result / 'log.txt', key + '/results/log.txt')
        if b'=== Summary ===' not in log_data:
            fail(key + ': log.txt has no final Summary. Wait for the selected run to finish.')
        records = capture(result / 'results.jsonl', key + '/results/results.jsonl')
        groups[key] = validate_rows(records, spec)
        excluded = []
        for p in sorted(result.rglob('*')):
            if not p.is_file() or p in tracked:
                continue
            if p.is_symlink() or p.suffix.lower() not in TEXT_EXTENSIONS:
                excluded.append({'path': str(p.relative_to(result)), 'bytes': p.stat().st_size,
                                 'reason': 'model_weight' if p.suffix.lower() in WEIGHT_EXTENSIONS else 'non_text_or_symlink'})
                continue
            capture(p, key + '/results/' + p.relative_to(result).as_posix())
        # Preserve the command and this queue's snapshots, without other experiment logs.
        for folder, subdir in [(exp_root, 'experiment_metadata'), (run_root, 'run_metadata')]:
            for p in sorted(folder.iterdir()):
                if p.is_file() and not p.is_symlink() and p.suffix.lower() in TEXT_EXTENSIONS:
                    capture(p, key + '/' + subdir + '/' + p.name)
        metadata = {
            'key': key, 'result_directory': str(result),
            'source_repository': str(source_repo),
            'source_head_at_upload': git(source_repo, 'rev-parse', 'HEAD').decode().strip(),
            'source_branch_at_upload': git(source_repo, 'branch', '--show-current').decode().strip(),
            'recorded_training_commit': None, 'excluded_files': excluded,
        }
        payload[key + '/source_at_upload/git_status.txt'] = git(source_repo, 'status', '--short')
        commit_file = run_root / 'git_commit.txt'
        commit = commit_file.read_text().strip() if commit_file.is_file() else ''
        if re.fullmatch(r'[0-9a-f]{40}', commit):
            metadata['recorded_training_commit'] = commit
        missing_source = []
        for rel in SOURCE_FILES:
            source_path = source_train / rel
            if source_path.is_file():
                capture(source_path, key + '/source_at_upload/' + rel)
            if metadata['recorded_training_commit']:
                git_path = (source_train.relative_to(source_repo) / rel).as_posix()
                historic = git(source_repo, 'show', commit + ':' + git_path, optional=True)
                if historic is not None:
                    payload[key + '/source_at_recorded_commit/' + rel] = historic
                else:
                    missing_source.append(rel)
        metadata['missing_files_at_recorded_commit'] = missing_source
        manifest_groups.append(metadata)
        print(key + ': four domains complete; last step=' + str(spec[4]) + '; ' + str(result), flush=True)

    iid_rows, cutoff_rows, metric_rows, hparams = [], [], [], {}
    metric_keys = ['train_out', 'test_in', 'test_out', 'tr_outloss', 'e_cls_loss',
                   'env_center_mse_loss', 'dccl_contrastive_loss', 'contrastive_weight_eff',
                   'nbr_pur_v', 'nbr_pur_e', 'nbr_gap', 'nbr_wmean', 'nbr_wmax']
    for key, by_te in groups.items():
        hparams[key] = by_te[0][0]['hparams']
        for te, values in by_te.items():
            chosen = max(values, key=lambda r: r['train_out'])  # first tie, same as Q.argmax
            iid_rows.append({'experiment': key, 'target': te, 'iid_step': chosen['step'],
                             'iid_train_out': chosen['train_out'], 'iid_test_in': chosen['test_in'],
                             'iid_test_out': chosen['test_out'], 'last_step': values[-1]['step'],
                             'last_test_in': values[-1]['test_in'], 'last_test_out': values[-1]['test_out']})
            for r in values:
                metric_rows.append({'experiment': key, 'target': te, 'step': r['step'],
                                    **{m: r.get(m) for m in metric_keys}})
            for budget in [500, 1000, 1500, 2000, 3000, 3800, 5000]:
                if budget > values[-1]['step']:
                    continue
                selected = max((r for r in values if r['step'] <= budget), key=lambda r: r['train_out'])
                cutoff_rows.append({'experiment': key, 'target': te, 'budget': budget,
                                    'selected_step': selected['step'], 'train_out': selected['train_out'],
                                    'test_in': selected['test_in'], 'test_out': selected['test_out']})
    prefixes = []
    for (left, gl), (right, gr) in itertools.combinations(groups.items(), 2):
        for te in range(4):
            shared = list(zip(gl[te], gr[te]))
            for metric in ['train_out', 'test_in', 'test_out', 'tr_outloss', 'e_cls_loss', 'env_center_mse_loss']:
                available = [(a, b) for a, b in shared if isinstance(a.get(metric), (int, float))
                             and isinstance(b.get(metric), (int, float))]
                changed = [(a, b) for a, b in available if not math.isclose(a[metric], b[metric], rel_tol=0, abs_tol=1e-10)]
                prefixes.append({'left': left, 'right': right, 'target': te, 'metric': metric,
                                 'compared_points': len(available), 'first_different_step': changed[0][0]['step'] if changed else '',
                                 'max_abs_difference': max((abs(a[metric]-b[metric]) for a, b in available), default=0)})
    payload['derived/iid_summary.csv'] = csv_bytes(iid_rows)
    payload['derived/checkpoint_metrics.csv'] = csv_bytes(metric_rows)
    payload['derived/cutoff_iid.csv'] = csv_bytes(cutoff_rows)
    payload['derived/prefix_comparison.csv'] = csv_bytes(prefixes)
    all_hp = sorted(set().union(*(hp.keys() for hp in hparams.values())))
    payload['derived/hparam_differences.json'] = json_bytes({
        k: {name: hp.get(k, '<absent>') for name, hp in hparams.items()}
        for k in all_hp if len({json.dumps(hp.get(k, '<absent>'), sort_keys=True) for hp in hparams.values()}) > 1
    })
    payload['README.md'] = (
        '# PACS Center MSE: three selected experiments\n\n'
        '01: NR=1, fixed LR=5e-5, 3800; 02: no contrastive, same LR, 3800; '
        '03: no contrastive, same LR, 5000. Raw records are validated before upload.\n\n'
        'Derived values are fractions (multiply by 100 for percent). IID selects each target\n'
        'independently by maximum source train_out, retaining the first tie. Both target\n'
        'test_in and test_out are reported at the same selected step. Cutoff tables reuse\n'
        'recorded trajectories; they are not separately rerun short-budget experiments.\n\n'
        'Prefix comparisons include loss values. Matching scalar records do not prove\n'
        'bit-identical parameters; model weights are not included. NR-vs-no-NR prefix\n'
        'differences are expected and are not a reproducibility test.\n\n'
        'source_at_upload records the current worktree and may differ from training.\n'
        'source_at_recorded_commit uses the saved commit, but cannot recover historical\n'
        'uncommitted edits. Missing historical files are listed in upload_manifest.json.\n\n'
        'Model weights and binary TensorBoard events are inventoried, not uploaded.\n'
    ).encode()
    manifest = {'repository': REPOSITORY, 'uploaded_at': now.isoformat(), 'branch': branch,
                'destination': destination, 'base_commit': head, 'experiments': manifest_groups,
                'files': [{'path': p, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                          for p, b in sorted(payload.items())]}
    payload['upload_manifest.json'] = json_bytes(manifest)
    # Ensure originals have not changed while the other experiments were collected.
    for path, original_stat in tracked.items():
        if signature(path) != original_stat:
            fail('Source file changed during collection: ' + str(path) + '. Retry after it stops writing.')
    print('Collected ' + str(len(payload)) + ' files; ' + str(sum(map(len, payload.values()))) + ' bytes.', flush=True)
    if args.dry_run:
        print('DRY RUN: all three runs validated. No commit or upload.')
        return
    git(push_root, 'var', 'GIT_AUTHOR_IDENT')
    git(push_root, 'var', 'GIT_COMMITTER_IDENT')
    with tempfile.TemporaryDirectory(prefix='cipt-three-upload-') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git(push_root, 'read-tree', head, env=env)
        entries = []
        for path, data in sorted(payload.items()):
            sha = git(push_root, 'hash-object', '-w', '--stdin', env=env, data=data).decode().strip()
            entries.append(('100644 ' + sha + '\t' + destination + '/' + path).encode() + b'\0')
        git(push_root, 'update-index', '-z', '--index-info', env=env, data=b''.join(entries))
        tree = git(push_root, 'write-tree', env=env).decode().strip()
        commit = git(push_root, 'commit-tree', tree, '-p', head, '-m',
                     'Upload PACS fixed-LR NR3800 no-con3800 no-con5000 results', env=env).decode().strip()
        print('Pushing ' + branch, flush=True)
        git(push_root, 'push', '--force-with-lease=refs/heads/' + branch + ':',
            'origin', commit + ':refs/heads/' + branch)
        remote_head = git(push_root, 'ls-remote', '--exit-code', 'origin', 'refs/heads/' + branch).decode().split()
        if not remote_head or remote_head[0] != commit:
            fail('Remote verification failed. Branch: ' + branch)
    print('Upload complete.')
    print('GitHub branch: ' + branch)
    print('Directory: ' + destination)
    print('Commit: ' + commit)
    print('https://github.com/' + REPOSITORY + '/tree/' + branch + '/' + destination)


if __name__ == '__main__':
    try:
        main()
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        fail(str(error))
