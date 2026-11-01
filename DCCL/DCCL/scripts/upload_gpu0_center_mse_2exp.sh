#!/usr/bin/env bash
# Upload the two selected experiment result directories using an isolated Git index.
# No checkout, pull, training, original-file edits, or use of the current Git index.
# Model weights are inventoried in upload_manifest.json rather than uploaded.
set -Eeuo pipefail
exec python3 - "$@" <<'PY'
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

REPOSITORY = 'r97993346-design/DCCL-3Source-DomainNet'
GROUPS = (
    '01_center_mse_nocon_lr2e3_constant_3000',
    '02_center_mse_nr1_lr2e3_constant_3000',
)
DEFAULT_RUN = ('/home/hooasia/lgg/projects/cipt_gpu0/DCCL/DCCL/train_output/'
               'ablation/gpu0_center_mse_lr2e3_2exp/run_20261101_182325_mtdf4W')
WEIGHTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.bin', '.pkl', '.pickle'}
LIMIT = 90 * 1024 * 1024

def die(message):
    raise SystemExit('ERROR: ' + message)

def git(*args, cwd=None, env=None, data=None):
    result = subprocess.run(['git', *args], cwd=cwd, env=env, input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        # Never print a remote URL that may contain embedded credentials.
        if args[0] in ('push', 'ls-remote', 'remote'):
            die('Git remote operation failed (' + args[0] +
                '). Check the server GitHub connection, then retry.')
        die('git ' + args[0] + ': ' + result.stderr.decode(errors='replace').strip())
    return result.stdout

def stamp(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=DEFAULT_RUN)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    if not run.is_dir():
        die('Experiment directory not found: ' + str(run))
    root = Path(git('rev-parse', '--show-toplevel', cwd=run).decode().strip())
    remote = git('remote', 'get-url', 'origin', cwd=root).decode().strip()
    if not remote.rstrip('/').endswith(('/' + REPOSITORY, '/' + REPOSITORY + '.git',
                                       ':' + REPOSITORY, ':' + REPOSITORY + '.git')):
        die('origin does not match ' + REPOSITORY + '; no upload attempted.')
    head = git('rev-parse', 'HEAD', cwd=root).decode().strip()
    source_branch = git('branch', '--show-current', cwd=root).decode().strip()
    if not (run / 'status.tsv').is_file():
        die('status.tsv is missing; cannot confirm both experiments completed.')
    status_bytes = (run / 'status.tsv').read_bytes()
    reader = csv.DictReader(io.StringIO(status_bytes.decode()), delimiter='\t')
    if not {'id', 'status'}.issubset(reader.fieldnames or []):
        die('Unexpected status.tsv columns; expected id and status.')
    latest = {}
    for row in reader:
        latest[row['id'].strip()] = row['status'].strip()
    for experiment_id in ('01', '02'):
        if latest.get(experiment_id) != 'SUCCESS':
            die('Experiment ' + experiment_id + ' is not SUCCESS in status.tsv. Retry after completion.')
    paths = [p for p in run.iterdir() if p.is_file() or p.is_symlink()]
    result_counts = {}
    for group in GROUPS:
        directory = run / group
        if not directory.is_dir() or directory.is_symlink():
            die('Experiment directory missing or symlinked: ' + str(directory))
        results = [p for p in directory.rglob('results.jsonl') if p.is_file() and p.stat().st_size]
        if not results:
            die('No nonempty results.jsonl found in ' + group)
        result_counts[group] = len(results)
        paths.extend(directory.rglob('*'))
    included, skipped = [], []
    for path in sorted(set(paths)):
        relative = path.relative_to(run)
        if any(part in {'.git', '__pycache__', '.cache'} for part in relative.parts):
            continue
        if path.is_symlink():
            skipped.append({'path': relative.as_posix(), 'reason': 'symlink'})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if path.suffix.lower() in WEIGHTS:
            skipped.append({'path': relative.as_posix(), 'bytes': size, 'reason': 'model_weight'})
            continue
        if path.suffix.lower() in {'.lock', '.pid', '.pyc'}:
            continue
        if size > LIMIT:
            die('Result file exceeds the 90 MiB upload limit: ' + str(relative))
        included.append((path, relative, stamp(path.stat())))
    train_relative = run.parents[3].relative_to(root)
    dest = train_relative / 'analysis_results' / 'gpu0_center_mse_lr2e3_2exp' / run.name
    print('Source:', run, flush=True)
    print('Results files:', json.dumps(result_counts), flush=True)
    print('Upload files:', len(included), '; excluded weights:',
          sum(p['reason'] == 'model_weight' for p in skipped), flush=True)
    print('Destination:', dest.as_posix(), flush=True)
    if args.dry_run:
        for _, relative, _ in included:
            print('  ' + relative.as_posix())
        print('DRY RUN: nothing committed or uploaded.')
        return
    git('var', 'GIT_AUTHOR_IDENT', cwd=root)
    git('var', 'GIT_COMMITTER_IDENT', cwd=root)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    branch = 'analysis/gpu0-center-mse-2exp-' + now.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    manifest = {
        'source': str(run), 'source_branch': source_branch, 'source_head': head,
        'uploaded_at': now.isoformat(), 'destination': dest.as_posix(),
        'experiments': list(GROUPS), 'results_jsonl_counts': result_counts,
        'files': [], 'excluded_files': skipped,
        'scope': 'Complete result files and root snapshots; model weights excluded.',
    }
    with tempfile.TemporaryDirectory(prefix='cipt-result-upload-') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git('read-tree', head, cwd=root, env=env)
        entries = []
        for path, relative, initial_stat in included:
            content = path.read_bytes()
            if stamp(path.stat()) != initial_stat:
                die('Source file changed while preparing upload: ' + str(relative) + '; retry after it stops writing.')
            sha = git('hash-object', '-w', '--stdin', cwd=root, env=env, data=content).decode().strip()
            target = dest / relative
            entries.append(('100644 ' + sha + '\t' + target.as_posix()).encode() + b'\0')
            manifest['files'].append({'path': relative.as_posix(), 'bytes': len(content),
                                      'sha256': hashlib.sha256(content).hexdigest(), 'git_blob': sha})
        if (run / 'status.tsv').read_bytes() != status_bytes:
            die('status.tsv changed during snapshot; retry after both experiments finish.')
        for path, relative, initial_stat in included:
            if stamp(path.stat()) != initial_stat:
                die('Source file changed before commit: ' + str(relative))
        manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
        manifest_sha = git('hash-object', '-w', '--stdin', cwd=root, env=env, data=manifest_data).decode().strip()
        entries.append(('100644 ' + manifest_sha + '\t' + (dest / 'upload_manifest.json').as_posix()).encode() + b'\0')
        git('update-index', '-z', '--index-info', cwd=root, env=env, data=b''.join(entries))
        tree = git('write-tree', cwd=root, env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', head, '-m',
                     'Upload GPU0 Center MSE two-experiment results ' + run.name,
                     cwd=root, env=env).decode().strip()
        print('Uploading branch:', branch, flush=True)
        git('push', 'origin', commit + ':refs/heads/' + branch, cwd=root)
        verified = git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + branch, cwd=root).decode().split()
        if not verified or verified[0] != commit:
            die('Push returned, but remote commit verification failed. Branch: ' + branch)
    print('Upload complete.')
    print('GitHub branch:', branch)
    print('Directory:', dest.as_posix())
    print('Commit:', commit)
    print('https://github.com/' + REPOSITORY + '/tree/' + branch + '/' + dest.as_posix())

try:
    main()
except (OSError, UnicodeError, csv.Error, ValueError) as error:
    die(str(error))
PY
