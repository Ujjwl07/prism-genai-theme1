"""Run pipeline configurations on cached embeddings and report metrics (fast inner loop for the research log).

usage:
  python scripts/experiment.py --split test --views raw
  python scripts/experiment.py --split train --limit 1200 --views reduced --exec-grid
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
from prism.sandbox import Sandbox  # noqa: E402


def run(pipe: Pipeline, queries: list[dict], S: np.ndarray) -> dict:
    t = time.time()
    traces = pipe.rerank_with_execution([q["text"] for q in queries], S)
    res = evaluate([[d for d, _ in tr.ranking] for tr in traces], [q["rel"] for q in queries])
    res["verify_s_per_query"] = round((time.time() - t) / len(queries), 4)
    res["executed_frac"] = round(float(np.mean([tr.executed for tr in traces])), 3)
    res["refined_frac"] = round(float(np.mean([tr.refined for tr in traces])), 3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="codexembed-400m")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--views", default="raw")
    ap.add_argument("--fusion", default="sum")
    ap.add_argument("--exec-grid", action="store_true")
    ap.add_argument("--exec-top-k", type=int, default=0)
    ap.add_argument("--exec-weight", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--expand-k", type=int, default=0)
    ap.add_argument("--query-max-len", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only-with-tests", action="store_true", help="keep queries with parseable sample tests")
    args = ap.parse_args()

    corpus = load_corpus()
    queries = load_queries(args.split)[: args.limit]
    if args.only_with_tests:
        from prism.query import parse
        queries = [q for q in queries if parse(q["text"]).tests]
    print(f"{len(queries)} queries", flush=True)
    cfg = PipelineConfig(encoder=args.encoder, query_views=tuple(args.views.split(",")), fusion=args.fusion,
                         exec_top_k=args.exec_top_k, exec_weight=args.exec_weight, gate_margin=args.gate,
                         expand_k=args.expand_k, query_max_len=args.query_max_len)
    pipe = Pipeline(cfg, sandbox=Sandbox())
    pipe.index([d["id"] for d in corpus], [d["text"] for d in corpus])
    S = pipe.dense_scores([q["text"] for q in queries])
    out = {}
    base = PipelineConfig(**{**cfg.__dict__, "exec_top_k": 0})
    pipe.cfg = base
    out["dense_only"] = run(pipe, queries, S)
    print("dense_only", out["dense_only"], flush=True)
    grid = [(k, w) for k in (10, 20, 50, 100) for w in (0.02, 0.05, 0.1, 0.2, 1.0)] if args.exec_grid else \
        ([(args.exec_top_k, args.exec_weight)] if args.exec_top_k else [])
    for k, w in grid:
        pipe.cfg = PipelineConfig(**{**cfg.__dict__, "exec_top_k": k, "exec_weight": w})
        name = f"exec_k{k}_w{w}" + (f"_gate{args.gate}" if args.gate is not None else "") + \
            (f"_expand{args.expand_k}" if args.expand_k else "")
        out[name] = run(pipe, queries, S)
        print(name, out[name], flush=True)
    path = Path("results") / f"exp_{args.split}_{args.encoder}_{args.views.replace(',', '+')}{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print("->", path)


if __name__ == "__main__":
    main()
