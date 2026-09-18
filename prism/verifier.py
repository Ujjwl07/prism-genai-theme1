"""Learned execution verifier (LEVER, ICML'23 — adapted to *retrieved* programs).

Instead of a binary pass/fail bonus, a small logistic model scores each (query, candidate) from
(i) retrieval evidence and (ii) execution evidence on the query's sample tests. Execution features are designed for
problems that accept several valid answers (special judges): a correct solution then runs cleanly and prints
output of the right *shape* even when the exact tokens differ from the sample answer.
Trained on the APPS train split only; CPU-only (numpy at inference).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .sandbox import outputs_match

FEATURES = [
    "dense", "dense_gap_top1", "log_rank",
    "n_tests", "frac_tests_passed", "all_passed", "first_passed",
    "crash", "timeout", "memory", "empty_out",
    "n_lines_eq", "n_tokens_eq", "type_seq_eq", "multiset_eq", "first_token_eq",
    "numeric_close", "token_overlap",
]
_NUM = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _typ(t: str) -> str:
    if _NUM.match(t):
        return "f" if any(c in t for c in ".eE") else "i"
    return "w" + t.lower() if len(t) <= 3 else "w"  # keep short words (YES/NO/-1 style) distinguishable


def test_features(status: str, got: str, want: str) -> dict[str, float]:
    f = {"crash": float(status == "crash"), "timeout": float(status == "timeout"), "memory": float(status == "memory")}
    a, b = got.split(), want.split()
    f["empty_out"] = float(status == "ok" and not a)
    f["passed"] = float(status == "ok" and outputs_match(got, want))
    if status != "ok":
        for k in ("n_lines_eq", "n_tokens_eq", "type_seq_eq", "multiset_eq", "first_token_eq", "numeric_close",
                  "token_overlap"):
            f[k] = 0.0
        return f
    f["n_lines_eq"] = float(len(got.strip().splitlines()) == len(want.strip().splitlines()))
    f["n_tokens_eq"] = float(len(a) == len(b))
    f["type_seq_eq"] = float(len(a) == len(b) and all(_typ(x)[0] == _typ(y)[0] for x, y in zip(a, b)))
    f["multiset_eq"] = float(sorted(x.lower() for x in a) == sorted(y.lower() for y in b))
    f["first_token_eq"] = float(bool(a) and bool(b) and a[0].lower() == b[0].lower())
    close = 0.0
    if a and len(a) == len(b):
        errs = []
        for x, y in zip(a, b):
            try:
                fx, fy = float(x), float(y)
                errs.append(min(1.0, abs(fx - fy) / max(1.0, abs(fy))))
            except ValueError:
                errs.append(0.0 if x.lower() == y.lower() else 1.0)
        close = 1.0 - float(np.mean(errs))
    f["numeric_close"] = close
    sa, sb = set(x.lower() for x in a), set(y.lower() for y in b)
    f["token_overlap"] = len(sa & sb) / max(1, len(sa | sb))
    return f


def candidate_features(dense: float, top1: float, rank: int, details: list[tuple[str, str, str]],
                       n_tests_available: int) -> list[float]:
    per = [test_features(*d) for d in details]
    first = per[0] if per else None
    passed = sum(p["passed"] for p in per)
    row = {
        "dense": dense, "dense_gap_top1": top1 - dense, "log_rank": math.log1p(rank),
        "n_tests": float(n_tests_available), "frac_tests_passed": passed / max(1, min(n_tests_available, 5)),
        "all_passed": float(bool(per) and passed == len(per) and len(per) == min(n_tests_available, 5)),
        "first_passed": first["passed"] if first else 0.0,
    }
    for k in ("crash", "timeout", "memory", "empty_out", "n_lines_eq", "n_tokens_eq", "type_seq_eq", "multiset_eq",
              "first_token_eq", "numeric_close", "token_overlap"):
        row[k] = first[k] if first else 0.0  # the first sample is always run → comparable across candidates
    return [row[k] for k in FEATURES]


class Verifier:
    def __init__(self, mean: np.ndarray, std: np.ndarray, coef: np.ndarray, intercept: float):
        self.mean, self.std, self.coef, self.intercept = mean, std, coef, intercept

    def logits(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.coef + self.intercept

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist(),
                                          "coef": self.coef.tolist(), "intercept": self.intercept}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Verifier":
        d = json.loads(Path(path).read_text())
        assert d["features"] == FEATURES, "feature set changed; retrain the verifier"
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["coef"]), float(d["intercept"]))

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "Verifier":
        from sklearn.linear_model import LogisticRegression

        mean, std = X.mean(0), X.std(0) + 1e-6
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((X - mean) / std, y)
        return cls(mean, std, m.coef_[0], float(m.intercept_[0]))
