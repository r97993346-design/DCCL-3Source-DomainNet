#!/usr/bin/env python3
"""Linux RAM guard for explicitly selected, already-running CIPTDCCL jobs.

Example (run on the training SERVER):
  python -u memory_guard.py --pids 1148002 1143683 1145888

Uses only the Python standard library. Checks system MemAvailable, not MemFree.
Stops one selected training tree at a time, in the supplied PID order. No
checkpoint is created: SIGTERM/SIGKILL cannot add saving logic to a running job.
Ctrl+C stops only this guard. No process-group or username-wide signals are used.
Requires Linux pidfd support and Python >= 3.9 for PID-reuse-safe signaling.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

GIB = 1024 ** 3
PROC = Path('/proc')


def log(message):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    uid: int
    ppid: int


def process_identity(pid):
    """Read the kernel start time, so a reused PID is never accepted."""
    try:
        directory = PROC / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return ProcessIdentity(pid, int(fields[19]), directory.stat().st_uid,
                               int(fields[1]))
    except (OSError, ValueError, IndexError):
        return None


def same_process(expected, actual):
    return (actual is not None and expected.pid == actual.pid
            and expected.start_ticks == actual.start_ticks
            and expected.uid == actual.uid)


def is_alive(identity):
    return same_process(identity, process_identity(identity.pid))


def command_arguments(pid):
    return [part.decode(errors='replace') for part in
            (PROC / str(pid) / 'cmdline').read_bytes().split(b'\0') if part]


def process_ids():
    return [int(directory.name) for directory in PROC.iterdir()
            if directory.name.isdigit()]


def check_proc_namespace():
    proc_pid = int((PROC / 'self' / 'stat').read_text().split(' ', 1)[0])
    if proc_pid != os.getpid():
        raise RuntimeError('/proc is mounted for a different PID namespace; refusing to signal processes')


def validate_target(pid, run_prefix):
    identity = process_identity(pid)
    if identity is None:
        raise ValueError(f'PID {pid} does not exist or has already finished')
    if identity.uid != os.getuid():
        raise ValueError(f'PID {pid} belongs to another user; refusing to monitor it')
    if pid in (os.getpid(), os.getppid()):
        raise ValueError('The guard and its parent shell cannot be targets')
    try:
        arguments = command_arguments(pid)
    except OSError as error:
        raise ValueError(f'Cannot inspect PID {pid}: {error}') from error
    script_positions = [i for i, arg in enumerate(arguments)
                        if Path(arg).name == 'train_all.py']
    if not any(i + 1 < len(arguments)
               and arguments[i + 1].startswith(run_prefix) for i in script_positions):
        raise ValueError(f'PID {pid} is not train_all.py with run prefix {run_prefix!r}')
    if not is_alive(identity):
        raise ValueError(f'PID {pid} changed while being inspected; check the PID again')
    return identity, ' '.join(arguments)


def available_gib(meminfo=None):
    text = (PROC / 'meminfo').read_text() if meminfo is None else meminfo
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            fields = line.split()
            if len(fields) != 3 or fields[2] != 'kB':
                raise ValueError('Invalid MemAvailable units')
            value = int(fields[1])
            if value < 0:
                raise ValueError('Negative MemAvailable')
            return value * 1024 / GIB
    raise ValueError('MemAvailable is missing; no signals will be sent')


class PressureTimer:
    def __init__(self, threshold, critical, hold_seconds):
        self.threshold = threshold
        self.critical = critical
        self.hold_seconds = hold_seconds
        self.low_since = None

    def update(self, available, now):
        if available < self.critical:
            return 'critical'
        if available >= self.threshold:
            self.low_since = None
            return None
        if self.low_since is None:
            self.low_since = now
        if now - self.low_since >= self.hold_seconds:
            return 'sustained'
        return None


def descendants(root):
    snapshot = []
    for pid in process_ids():
        item = process_identity(pid)
        if item is not None and item.uid == root.uid:
            snapshot.append(item)
    selected = []
    parents = {root.pid}
    while True:
        children = [item for item in snapshot
                    if item.ppid in parents and item.pid not in parents]
        if not children:
            return selected
        selected.extend(children)
        parents.update(item.pid for item in children)


def send_signal(identity, number):
    """Pin the process with pidfd, then verify its captured identity."""
    descriptor = None
    try:
        descriptor = os.pidfd_open(identity.pid)
        if not is_alive(identity) or identity.uid != os.getuid():
            return False
        signal.pidfd_send_signal(descriptor, number)
        return True
    except ProcessLookupError:
        return False
    except OSError as error:
        log(f'Cannot signal PID {identity.pid}: {error}')
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stop_tree(root, grace_seconds):
    if not is_alive(root):
        return True
    children = descendants(root)
    log(f'Stopping training PID {root.pid} and its {len(children)} worker processes')
    # Stop the parent first to discourage spawning additional workers. Include
    # a second snapshot if it is still alive (for jobs with SIGTERM handlers).
    send_signal(root, signal.SIGTERM)
    if is_alive(root):
        known = {item.pid for item in children}
        children.extend(item for item in descendants(root) if item.pid not in known)
    identities = children + [root]
    for item in children:
        send_signal(item, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while any(is_alive(item) for item in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    for item in identities:
        if is_alive(item):
            log(f'PID {item.pid} did not exit; sending SIGKILL')
            send_signal(item, signal.SIGKILL)
    # A process in uninterruptible kernel sleep may not exit immediately.
    deadline = time.monotonic() + 1.0
    while any(is_alive(item) for item in identities) and time.monotonic() < deadline:
        time.sleep(0.1)
    survivors = [item.pid for item in identities if is_alive(item)]
    if survivors:
        log(f'Processes still present: {survivors}; manual investigation required')
        return False
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pids', nargs='+', type=int, required=True,
                        help='Training PIDs, in the order they may be stopped')
    parser.add_argument('--run-prefix', default='pacs_e1_s1_ind0_',
                        help='Required training run-name prefix (extra identity check)')
    parser.add_argument('--min-available-gb', type=float, default=12.)
    parser.add_argument('--critical-gb', type=float, default=6.)
    parser.add_argument('--hold-seconds', type=float, default=15.)
    parser.add_argument('--interval', type=float, default=3.)
    parser.add_argument('--grace-seconds', type=float, default=10.)
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be stopped and exit, without signaling')
    args = parser.parse_args(argv)
    numbers = (args.min_available_gb, args.critical_gb, args.hold_seconds,
               args.interval, args.grace_seconds)
    if not all(math.isfinite(value) for value in numbers):
        parser.error('Thresholds and intervals must be finite')
    if not (0 < args.critical_gb < args.min_available_gb):
        parser.error('Require 0 < critical-gb < min-available-gb')
    if args.hold_seconds < 0 or args.interval <= 0 or args.grace_seconds < 0:
        parser.error('Intervals must be positive; hold/grace may be zero')
    if not args.run_prefix or any(pid <= 1 for pid in args.pids):
        parser.error('Use a nonempty run prefix and training PIDs greater than 1')
    args.pids = list(dict.fromkeys(args.pids))
    return args


def main(argv=None):
    args = parse_args(argv)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Requires Linux and Python >= 3.9 with pidfd support')
    check_proc_namespace()
    # One guard per account avoids separate guards stopping several jobs at once.
    lock_path = Path(tempfile.gettempdir()) / f'cipt-memory-guard-{os.getuid()}.lock'
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another memory guard is already running for this account') from error
        targets = []
        for pid in args.pids:
            identity, command = validate_target(pid, args.run_prefix)
            targets.append(identity)
            log(f'Monitor target PID {pid}: {command}')
        available_gib()  # Fail at startup if the memory metric is unavailable.
        log(f'Guard PID={os.getpid()}; low<{args.min_available_gb:g} GiB for '
            f'{args.hold_seconds:g}s, critical<{args.critical_gb:g} GiB immediately')
        log('Stop order follows --pids. No checkpoint is saved. Ctrl+C stops ONLY the guard.')
        timer = PressureTimer(args.min_available_gb, args.critical_gb, args.hold_seconds)
        last_report = float('-inf')
        while True:
            targets = [item for item in targets if is_alive(item)]
            if not targets:
                log('All selected training jobs finished; guard exiting')
                return 0
            try:
                available = available_gib()
            except (OSError, ValueError) as error:
                log(f'Memory check failed: {error}; no signals sent')
                timer.low_since = None
                time.sleep(args.interval)
                continue
            now = time.monotonic()
            if now - last_report >= 30. or available < args.min_available_gb:
                log(f'MemAvailable={available:.2f} GiB; live PIDs={[p.pid for p in targets]}')
                last_report = now
            reason = timer.update(available, now)
            if reason:
                target = targets[0]
                log(f'{reason.upper()} memory pressure; selected PID {target.pid}')
                if args.dry_run:
                    log('DRY RUN: would terminate this training tree; no signals sent')
                    return 0
                if not stop_tree(target, args.grace_seconds):
                    return 2
                targets.pop(0)
                # Recheck memory before selecting any further job. For ordinary
                # low memory, require another full sustained-pressure interval.
                timer.low_since = None
            time.sleep(args.interval)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('Guard interrupted; remaining training jobs are left running')
        raise SystemExit(0)
    except (OSError, ValueError, RuntimeError) as error:
        log(f'ERROR: {error}')
        raise SystemExit(2)
