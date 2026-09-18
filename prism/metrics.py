"""NDCG@k / MRR / Recall@k for rankings (matches MTEB/pytrec_eval definitions for binary relevance)."""
from __future__ import annotations

import math

import numpy as np


def evaluate(ranked: list[list[str]], rel: list[list[str]], k: int = 10) -> dict[str, float]:
    ndcg, mrr, r1, r10, r100 = [], [], [], [], []
    for docs, gold in zip(ranked, rel):
        gold = set(gold)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(docs[:k]) if d in gold)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
        ndcg.append(dcg / idcg)
        rr = 0.0
        for i, d in enumerate(docs[:k]):
            if d in gold:
                rr = 1 / (i + 1)
                break
        mrr.append(rr)
        r1.append(len(gold & set(docs[:1])) / len(gold))
        r10.append(len(gold & set(docs[:10])) / len(gold))
        r100.append(len(gold & set(docs[:100])) / len(gold))
    f = lambda xs: round(100 * float(np.mean(xs)), 2)
    return {"ndcg@10": f(ndcg), "mrr@10": f(mrr), "recall@1": f(r1), "recall@10": f(r10), "recall@100": f(r100)}


def rank_from_scores(scores: np.ndarray, doc_ids: list[str], top: int = 100) -> list[list[str]]:
    idx = np.argsort(-scores, axis=1)[:, :top]
    return [[doc_ids[j] for j in row] for row in idx]
