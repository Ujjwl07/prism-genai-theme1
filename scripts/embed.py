"""Offline embedding job (sharded so it can fan out over CI runners).

usage: python scripts/embed.py --model codexembed-400m --what corpus|test|train --variant raw|reduced \
                               [--shard 0 --num-shards 1] [--max-len 1024] [--limit N]
Writes content-hash-keyed shards into cache/emb/<model>/ (merge = copy files together).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.data import ensure_data, load_corpus, load_queries  # noqa: E402
from prism.encoders import SPECS, Encoder  # noqa: E402
from prism.query import reduced  # noqa: E402


def texts_for(what: str, variant: str, limit: int | None) -> tuple[list[str], str]:
    if what == "corpus":
        return [d["text"] for d in load_corpus()], "doc"
    qs = load_queries(what)[:limit]
    if variant == "raw":
        return [q["text"] for q in qs], "query"
    if variant == "reduced":
        return [reduced(q["text"]) for q in qs], "query"
    if variant == "reduced_noex":
        return [reduced(q["text"], keep_examples=False) for q in qs], "query"
    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--what", required=True, choices=["corpus", "test", "train"])
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ensure_data()
    spec = SPECS[args.model] if args.max_len is None else replace(SPECS[args.model], max_len=args.max_len)
    texts, kind = texts_for(args.what, args.variant, args.limit)
    texts = texts[args.shard::args.num_shards]
    t = time.time()
    Encoder(spec, device="cpu").encode(texts, kind, batch_size=args.batch_size)
    dt = time.time() - t
    print(f"[embed] {args.model} {args.what}/{args.variant} shard {args.shard}/{args.num_shards}: "
          f"{len(texts)} texts in {dt:.1f}s ({len(texts) / max(dt, 1e-9):.2f}/s)", flush=True)


if __name__ == "__main__":
    main()
