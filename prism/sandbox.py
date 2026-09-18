"""Cross-platform execution sandbox for verifying retrieved programs on sample tests (CodeT/LEVER-style signal).

Works on Windows and POSIX: one short-lived `python -I` subprocess per (program, test), wall-clock timeout,
psutil RSS watchdog, temp working dir, network disabled via a prelude that stubs `socket`.
Results are cached by (code-hash, test-hash) so re-runs and incremental rebuilds are free.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "cache" / "exec_cache.jsonl"

PRELUDE = r'''
import socket as _s, sys as _sys
def _blocked(*a, **k):
    raise OSError("network disabled in sandbox")
_s.socket = _blocked; _s.create_connection = _blocked
_sys.setrecursionlimit(10**6)
_src = open(_sys.argv[1], encoding="utf-8").read()
_sys.argv = _sys.argv[:1]
_g = {"__name__": "__main__", "__builtins__": __builtins__}
exec(compile(_src, "solution.py", "exec"), _g)
'''


def _h(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8", "replace")).hexdigest()


def outputs_match(got: str, want: str, tol: float = 1e-6) -> bool:
    """Token-wise comparison, whitespace-insensitive, float-tolerant, case-insensitive for yes/no style."""
    a, b = got.split(), want.split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y or x.lower() == y.lower():
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not math.isclose(fx, fy, rel_tol=tol, abs_tol=tol):
            return False
    return True


class Sandbox:
    def __init__(self, timeout: float = 4.0, mem_mb: int = 512, workers: int | None = None):
        self.timeout = timeout
        self.mem_mb = mem_mb
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self._tmp = Path(tempfile.mkdtemp(prefix="prism_sbx_"))
        (self._tmp / "prelude.py").write_text(PRELUDE, encoding="utf-8")
        self._lock = threading.Lock()
        self.cache: dict[str, bool] = {}
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            with open(CACHE_FILE, encoding="utf-8") as f:
                for line in f:
                    k, v = json.loads(line)
                    self.cache[k] = v

    def _run_once(self, code: str, stdin: str) -> str | None:
        """Return stdout, or None on crash/timeout/memory kill."""
        src = self._tmp / f"{_h(code)}.py"
        if not src.exists():
            src.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}  # SYSTEMROOT needed on Windows
        with tempfile.TemporaryDirectory(dir=self._tmp) as wd:
            try:
                p = subprocess.Popen([sys.executable, "-I", str(self._tmp / "prelude.py"), str(src)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=wd, env=env)
            except OSError:
                return None
            killed = threading.Event()

            def watchdog() -> None:
                try:
                    proc = psutil.Process(p.pid)
                    while p.poll() is None:
                        if proc.memory_info().rss > self.mem_mb * 2**20:
                            killed.set()
                            p.kill()
                            return
                        killed.wait(0.05)
                except psutil.Error:
                    pass

            wt = threading.Thread(target=watchdog, daemon=True)
            wt.start()
            try:
                out, _ = p.communicate(stdin.encode("utf-8"), timeout=self.timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                return None
            finally:
                killed.set()
            if p.returncode != 0:
                return None
            return out.decode("utf-8", "replace")

    def check(self, code: str, tests: list[tuple[str, str]], max_tests: int = 3) -> float:
        """Fraction of (up to max_tests) sample tests passed; stops at the first failure (fail-fast)."""
        tests = tests[:max_tests]
        if not tests:
            return float("nan")
        passed = 0
        for inp, want in tests:
            key = _h(code, inp, want)
            ok = self.cache.get(key)
            if ok is None:
                got = self._run_once(code, inp if inp.endswith("\n") else inp + "\n")
                ok = got is not None and outputs_match(got, want)
                with self._lock:
                    self.cache[key] = ok
                    with open(CACHE_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps([key, ok]) + "\n")
            if not ok:
                break
            passed += 1
        return passed / len(tests)

    def check_many(self, jobs: list[tuple[str, list[tuple[str, str]]]]) -> list[float]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda j: self.check(*j), jobs))
