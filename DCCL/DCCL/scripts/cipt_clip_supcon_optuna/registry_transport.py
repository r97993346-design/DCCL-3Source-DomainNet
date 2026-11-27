"""Bounded Git transport for space-v2 admissions and archive verification."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from global_registry import REF, Registry, Unavailable

SYNC_GIT = Path("/media/hooasia/data/cipt_github_results_sync/git")


class BoundedRegistry(Registry):
    def git(self, cwd, *args, allow_failure=False):
        action = args[0] if args[0] != "-c" else next(word for word in args if word in ("commit", "push"))
        reads = action in ("ls-remote", "fetch")
        timeout = 50 if action == "fetch" else 20
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        env.setdefault("GIT_SSH_COMMAND", shlex.join([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=8", "-o", "ServerAliveCountMax=2",
        ]))
        for attempt in range(2 if reads else 1):
            try:
                with subprocess.Popen(["git", "-c", "core.hooksPath=/dev/null", *args],
                                      cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      start_new_session=True) as proc:
                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait(timeout=5)
                        raise
                result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
                if result.returncode == 0 or allow_failure:
                    return result
                problem = f"Git {action} returned {result.returncode}"
            except subprocess.TimeoutExpired:
                problem = f"Git {action} exceeded {timeout}s"
            except OSError as exc:
                problem = f"Git {action} could not start ({exc.errno})"
            if reads and attempt == 0:
                time.sleep(2)
        raise Unavailable(problem)

    def checkout(self, directory, ref=REF):
        if not ref.startswith("refs/heads/results/"):
            return super().checkout(directory, ref)
        refs = self.git(directory, "ls-remote", "--refs", self.remote, ref).stdout.splitlines()
        if not refs:
            return False
        tip = refs[0].split()[0].decode("ascii")
        suffix = ref.removeprefix("refs/heads/results/")
        cache = SYNC_GIT / suffix
        if cache.resolve().is_relative_to(SYNC_GIT) and (cache / ".git").is_dir():
            present = self.git(cache, "cat-file", "-e", tip + "^{commit}", allow_failure=True)
            if present.returncode == 0:
                # Read the pinned commit, never mutable files in the sync checkout.
                self.git(directory, "clone", "-q", "--shared", "--no-checkout", str(cache), ".")
                self.git(directory, "checkout", "-q", "--detach", tip)
                return True
        self.git(directory, "init", "-q")
        self.git(directory, "fetch", "-q", "--depth=1", self.remote, ref)
        self.git(directory, "checkout", "-q", "--detach", "FETCH_HEAD")
        return True

    def update(self, key, transform, *, tries=4):
        return super().update(key, transform, tries=tries)
