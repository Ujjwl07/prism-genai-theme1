"""Reproduce the official MTEB evaluation JSON for CoIR AppsRetrieval with the full pipeline.

usage: python evaluate.py [--config results/best_config.json] [--no-exec] [--out results/appsretrieval_results.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from prism.mteb_model import run_mteb
from prism.pipeline import PipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/best_config.json")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--out", default="results/appsretrieval_results.json")
    args = ap.parse_args()
    cfg = PipelineConfig()
    if Path(args.config).exists():
        raw = json.loads(Path(args.config).read_text())
        names = {f.name for f in fields(PipelineConfig)}
        cfg = PipelineConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})
    print("config:", cfg)
    d = run_mteb(cfg, out_json=args.out, use_execution=not args.no_exec)
    s = d["scores"]["test"][0]
    print(f"NDCG@10 = {100 * s['ndcg_at_10']:.2f}   MRR@10 = {100 * s['mrr_at_10']:.2f}   -> {args.out}")


if __name__ == "__main__":
    main()
