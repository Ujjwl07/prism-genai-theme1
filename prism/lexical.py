"""BM25 baseline (Robertson & Zaragoza, FnTIR 2009) with a code-aware tokenizer (identifier splitting)."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[A-Za-z]+|\d+")


def tokenize(text: str) -> list[str]:
    out = []
    for w in _TOK.findall(text):
        # split camelCase / snake_case pieces
        out.extend(p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w) or [w])
    return out


class BM25:
    def __init__(self, docs: list[str]):
        self.bm = BM25Okapi([tokenize(d) for d in docs])

    def scores(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.bm.get_scores(tokenize(q)) for q in queries]).astype(np.float32)
