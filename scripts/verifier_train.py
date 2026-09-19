"""V6: collect execution features on the APPS *train* split, cross-validate a LEVER-style verifier, fit & save it.

usage: python scripts/verifier_train.py [--limit 1200] [--exec-top-k 100] [--expand-k 1500]
Outputs: results/vf_train.npz (features), results/verifier.json (model), results/v6_cv.json (CV report)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.data import load_corpus, load_queries  # noqa: E402
from prism.metrics import evaluate  # noqa: E402
from prism.pipeline import Pipeline, PipelineConfig  # noqa: E402
from prism.query import parse  # noqa: E402
from prism.sandbox import Sandbox  # noqa: E402
from prism.verifier import ALL_FEATURES, FEATURES, LTRVerifier, Verifier  # noqa: E402


def rule_scores(X: np.ndarray, w: float = 1.0) -> np.ndarray:
    """V2 baseline on the same features: cosine + w * pass fraction."""
    return X[:, ALL_FEATURES.index("dense")] + w * X[:, ALL_FEATURES.index("frac_tests_passed")]


def ndcg_from_scores(qids, docs, scores, gold, extra) -> dict:
    ranked = []
    for q in gold:
        m = qids == q
        order = np.argsort(-scores[m])
        r = list(docs[m][order]) + [d for d in extra[q] if d not in set(docs[m])]
        ranked.append(r[:100])
    return evaluate(ranked, [[gold[q]] for q in gold])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--exec-top-k", type=int, default=100)
    ap.add_argument("--expand-k", type=int, default=1500)
    ap.add_argument("--out", default="results/verifier_v8.json")
    ap.add_argument("--report", default="results/v8_cv.json")
    args = ap.parse_args()

    corpus = load_corpus()
    qs = [q for q in load_queries("train")[: args.limit] if parse(q["text"]).tests]
    cfg = PipelineConfig(exec_top_k=args.exec_top_k, expand_k=args.expand_k)
    pipe = Pipeline(cfg, sandbox=Sandbox())
    pipe.index([d["id"] for d in corpus], [d["text"] for d in corpus])
    S = pipe.dense_scores([q["text"] for q in qs])
    t = time.time()
    traces = pipe.collect_features([q["text"] for q in qs], S)
    print(f"[v6] features for {len(qs)} queries in {time.time() - t:.0f}s", flush=True)

    rows, y, qid, did = [], [], [], []
    extra = {}
    for q, tr, top in zip(qs, traces, pipe._tops):
        extra[q["id"]] = [pipe.doc_ids[j] for j in top[:100]]
        for j, f in tr.features.items():
            rows.append(f)
            y.append(int(pipe.doc_ids[j] in q["rel"]))
            qid.append(q["id"])
            did.append(pipe.doc_ids[j])
    X, y, qid, did = np.array(rows), np.array(y), np.array(qid), np.array(did)
    np.savez("results/vf_train_all.npz", X=X, y=y, qid=qid, did=did)
    gold = {q["id"]: q["rel"][0] for q in qs}

    from sklearn.model_selection import GroupKFold

    allp = X[:, ALL_FEATURES.index("all_passed")]
    report = {"n_queries": len(qs), "n_candidates": int(len(y)), "positives": int(y.sum())}
    report["rule_w1.0"] = ndcg_from_scores(qid, did, rule_scores(X), gold, extra)
    print("rule", report["rule_w1.0"], flush=True)
    folds = list(GroupKFold(5).split(X, y, qid))
    candidates = {
        "logreg_v6feats_C0.01": lambda Xt, yt, gt: Verifier.fit(Xt, yt, C=0.01, features=FEATURES),
        "logreg_allfeats_C0.01": lambda Xt, yt, gt: Verifier.fit(Xt, yt, C=0.01, features=ALL_FEATURES),
        "logreg_allfeats_C0.1": lambda Xt, yt, gt: Verifier.fit(Xt, yt, C=0.1, features=ALL_FEATURES),
        "lgbm_v6feats": lambda Xt, yt, gt: LTRVerifier.fit(Xt, yt, gt, features=FEATURES),
        "lgbm_allfeats_l15": lambda Xt, yt, gt: LTRVerifier.fit(Xt, yt, gt, features=ALL_FEATURES),
        "lgbm_allfeats_l7": lambda Xt, yt, gt: LTRVerifier.fit(Xt, yt, gt, features=ALL_FEATURES, num_leaves=7),
        "lgbm_allfeats_l31_slow": lambda Xt, yt, gt: LTRVerifier.fit(Xt, yt, gt, features=ALL_FEATURES, num_leaves=31,
                                                                    learning_rate=0.03, n_estimators=500),
    }
    best = None
    for name, fit in candidates.items():
        oof = np.zeros(len(y))
        for tr_i, te_i in folds:
            oof[te_i] = fit(X[tr_i], y[tr_i], qid[tr_i]).logits(X[te_i])
        r = ndcg_from_scores(qid, did, oof + 100 * allp, gold, extra)  # V6b ordering: all-passed first
        report[name] = r
        print(name, r, flush=True)
        if best is None or r["ndcg@10"] > best[1]:
            best = (name, r["ndcg@10"])
    report["chosen"] = best[0]
    model = candidates[best[0]](X, y, qid)
    model.save(args.out)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print("chosen", best, "->", args.out, flush=True)


if __name__ == "__main__":
    main()
