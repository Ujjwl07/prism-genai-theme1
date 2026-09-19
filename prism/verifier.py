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
# V8: problem-text (NL) features — LEVER's verifier conditions on the NL input, not only on execution results
NL_FEATURES = [
    "q_any_answer", "q_float_tol", "q_yes_no", "q_multi_test",
    "any_x_ran_ok", "any_x_n_tokens_eq", "any_x_type_seq_eq", "any_x_n_lines_eq",
]
ALL_FEATURES = FEATURES + NL_FEATURES

_ANY = re.compile(r"(print|output|find) any|any (of them|one of them|valid|correct|possible|suitable|such)|"
                  r"(multiple|several|many) (possible |valid |correct )?(answers|solutions)|"
                  r"if there are (several|multiple|many)|any order|вывести любой|любой из", re.I)
_FLOAT = re.compile(r"(absolute|relative) error|10\^\{?-\d|1e-\d|precision", re.I)
_YESNO = re.compile(r"\b(yes|no)\b.*\b(yes|no)\b|\"(yes|no)\"", re.I | re.S)
_MULTI = re.compile(r"(number of test ?cases|the first line contains (a single integer )?\$?t\$?|\bT\b test cases)", re.I)


def nl_features(query: str) -> dict[str, float]:
    return {"q_any_answer": float(bool(_ANY.search(query))), "q_float_tol": float(bool(_FLOAT.search(query))),
            "q_yes_no": float(bool(_YESNO.search(query))), "q_multi_test": float(bool(_MULTI.search(query)))}


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
                       n_tests_available: int, query: str = "") -> list[float]:
    """Feature vector aligned to ALL_FEATURES (a model picks the subset it was trained on)."""
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
    row.update(nl_features(query))
    any_ = row["q_any_answer"]
    ran_ok = float(bool(first) and not (first["crash"] or first["timeout"] or first["memory"]))
    row.update({"any_x_ran_ok": any_ * ran_ok, "any_x_n_tokens_eq": any_ * row["n_tokens_eq"],
                "any_x_type_seq_eq": any_ * row["type_seq_eq"], "any_x_n_lines_eq": any_ * row["n_lines_eq"]})
    return [row[k] for k in ALL_FEATURES]


def _cols(names: list[str]) -> list[int]:
    return [ALL_FEATURES.index(n) for n in names]


class Verifier:
    """Logistic verifier (V6). `features` = names it uses; inputs are rows aligned to ALL_FEATURES."""

    kind = "logreg"

    def __init__(self, mean, std, coef, intercept, features=None):
        self.mean, self.std, self.coef, self.intercept = mean, std, coef, intercept
        self.features = list(features or FEATURES)
        self._idx = _cols(self.features)

    def logits(self, X_all: np.ndarray) -> np.ndarray:
        X = X_all[:, self._idx]
        return ((X - self.mean) / self.std) @ self.coef + self.intercept

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"kind": "logreg", "features": self.features, "mean": self.mean.tolist(),
                                          "std": self.std.tolist(), "coef": self.coef.tolist(),
                                          "intercept": self.intercept}, indent=1))

    @classmethod
    def load(cls, path: str | Path):
        d = json.loads(Path(path).read_text())
        if d.get("kind") == "lgbm":
            return LTRVerifier.load(path)
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["coef"]), float(d["intercept"]), d["features"])

    @classmethod
    def fit(cls, X_all: np.ndarray, y: np.ndarray, C: float = 1.0, features=None, **_) -> "Verifier":
        from sklearn.linear_model import LogisticRegression

        feats = list(features or FEATURES)
        X = X_all[:, _cols(feats)]
        mean, std = X.mean(0), X.std(0) + 1e-6
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((X - mean) / std, y)
        return cls(mean, std, m.coef_[0], float(m.intercept_[0]), feats)


class LTRVerifier:
    """V8: gradient-boosted trees with a LambdaRank objective (LightGBM, NeurIPS'17), grouped by query.

    LightGBM is only needed for *training*. For inference the trees are exported to JSON and evaluated in numpy,
    so the runtime has no LightGBM/OpenMP dependency (LightGBM's libomp next to PyTorch's crashes on macOS).
    """

    kind = "lgbm"

    def __init__(self, trees: list[dict], features):
        self.trees, self.features = trees, list(features)
        self._idx = _cols(self.features)

    @staticmethod
    def _tree_predict(node: dict, X: np.ndarray) -> np.ndarray:
        if "leaf_value" in node:
            return np.full(len(X), node["leaf_value"])
        out = np.empty(len(X))
        f, thr = node["split_feature"], node["threshold"]
        assert node.get("decision_type", "<=") == "<=", "only numerical splits are exported"
        go_left = X[:, f] <= thr
        nan = np.isnan(X[:, f])
        if nan.any():
            go_left = np.where(nan, node.get("default_left", True), go_left)
        if go_left.any():
            out[go_left] = LTRVerifier._tree_predict(node["left_child"], X[go_left])
        if (~go_left).any():
            out[~go_left] = LTRVerifier._tree_predict(node["right_child"], X[~go_left])
        return out

    def logits(self, X_all: np.ndarray) -> np.ndarray:
        X = X_all[:, self._idx].astype(float)
        return np.sum([self._tree_predict(t, X) for t in self.trees], axis=0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"kind": "lgbm", "features": self.features, "trees": self.trees}))

    @classmethod
    def load(cls, path: str | Path) -> "LTRVerifier":
        d = json.loads(Path(path).read_text())
        return cls(d["trees"], d["features"])

    @classmethod
    def fit(cls, X_all: np.ndarray, y: np.ndarray, groups: np.ndarray, features=None, num_leaves: int = 15,
            n_estimators: int = 300, learning_rate: float = 0.05, min_child_samples: int = 50) -> "LTRVerifier":
        import lightgbm as lgb

        feats = list(features or ALL_FEATURES)
        order = np.argsort(groups, kind="stable")
        _, sizes = np.unique(groups[order], return_counts=True)
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
                           lambdarank_truncation_level=20)
        m.fit(X_all[order][:, _cols(feats)], y[order], group=sizes)
        trees = [t["tree_structure"] for t in m.booster_.dump_model()["tree_info"]]
        model = cls(trees, feats)
        # exactness check of the numpy export against LightGBM itself
        Xs = X_all[:2000][:, _cols(feats)]
        assert np.allclose(model.logits(X_all[:2000]), m.booster_.predict(Xs), atol=1e-6)
        return model
