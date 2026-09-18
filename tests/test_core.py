"""Cross-platform smoke tests (run on windows-latest + ubuntu-latest in CI)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism.metrics import evaluate  # noqa: E402
from prism.query import parse, reduced  # noqa: E402
from prism.sandbox import Sandbox, outputs_match  # noqa: E402

CF = """You are given n.

-----Input-----
One integer n.

-----Output-----
Print 2n.

-----Examples-----
Input
2

Output
4

Input
5

Output
10
"""

AT = """Given N, print N+1.

-----Constraints-----
1<=N<=10

-----Sample Input-----
3

-----Sample Output-----
4

Because 3+1=4.
"""


def test_parse_codeforces_and_atcoder():
    assert parse(CF).tests == [("2", "4"), ("5", "10")]
    assert parse(AT).tests == [("3", "4")]
    r = reduced(AT)
    assert r.index("Constraints") < r.index("Examples")


def test_outputs_match():
    assert outputs_match("1 2\n3\n", "1 2 3")
    assert outputs_match("0.3333333", "0.33333333")
    assert outputs_match("YES", "Yes")
    assert not outputs_match("1 2", "1 3")


def test_sandbox_pass_fail_and_safety():
    sb = Sandbox(timeout=2.0, workers=2)
    assert sb.check("print(int(input())*2)", parse(CF).tests) == 1.0
    assert sb.check("print(int(input())+1)", parse(CF).tests) == 0.0
    assert sb.check("import sys\nprint(int(open(0).read())+1)", parse(AT).tests) == 1.0
    assert sb.check("while True: pass", [("", "x")]) == 0.0
    assert sb.check("a=[0]*(10**10)", [("", "x")]) == 0.0
    assert sb.check("import socket\nsocket.socket()\nprint(1)", [("", "1")]) == 0.0


def test_metrics():
    m = evaluate([["d1", "d2"], ["d3", "d4"]], [["d1"], ["d4"]])
    assert m["mrr@10"] == 75.0 and m["recall@1"] == 50.0


def test_cast_chunking_and_versions(tmp_path):
    from prism.codebase import VersionedIndex
    repo = tmp_path / "r"
    repo.mkdir()
    g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (repo / "a.js").write_text("function playMusic(){ openApp(); startPlayer(); }\nfunction other(){ return 1 }\n")
    g("add", "."); g("commit", "-qm", "v1")
    (repo / "b.js").write_text("const x = () => { startPlayer(); openApp(); }\n")
    g("add", "."); g("commit", "-qm", "v2")
    vi = VersionedIndex(repo)
    c1 = vi.build("HEAD~1"); c2 = vi.build("HEAD")
    st = vi.stats[vi.resolve("HEAD")]
    assert st["reused_from_cache"] == 1 and st["parsed"] == 1
    calls = [c.calls for c in c2 if c.path == "a.js"][0]
    assert calls.index("openApp") < calls.index("startPlayer")
