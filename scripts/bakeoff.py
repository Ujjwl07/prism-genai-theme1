"""V1: encoder bake-off on the APPS test split (raw queries, no preprocessing).

usage: python scripts/bakeoff.py coderankembed codexembed-400m --device mps
Reports NDCG@10 / MRR@10 / Recall@k and token-overflow statistics.
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
from prism.encoders import Encoder  # noqa: E402
from prism.metrics import evaluate, rank_from_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate on first N queries (debug)")
    args = ap.parse_args()

    corpus = load_corpus()
    queries = load_queries("test")[: args.limit]
    doc_ids = [d["id"] for d in corpus]
    out = {}
    for m in args.models:
        from dataclasses import replace

        from prism.encoders import SPECS

        spec = SPECS[m] if args.max_len is None else replace(SPECS[m], max_len=args.max_len)
        enc = Encoder(spec, device=args.device)
        t0 = time.time()
        D = enc.encode([d["text"] for d in corpus], "doc")
        t1 = time.time()
        Q = enc.encode([q["text"] for q in queries], "query")
        t2 = time.time()
        res = evaluate(rank_from_scores(Q @ D.T, doc_ids), [q["rel"] for q in queries])
        qlen = enc.tokenizer_lengths([spec.query_prefix + q["text"] for q in queries])
        res["query_tokens_median"] = int(np.median(qlen))
        res["frac_queries_over_512"] = round(float((qlen > 512).mean()), 3)
        res["frac_queries_over_max_len"] = round(float((qlen > spec.max_len).mean()), 3)
        res["max_len"] = spec.max_len
        res["doc_encode_s"] = round(t1 - t0, 1)
        res["query_encode_s"] = round(t2 - t1, 1)
        out[f"{m}@{spec.max_len}"] = res
        print(m, json.dumps(res), flush=True)
        del enc
    path = Path("results/v1_bakeoff.json")
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
