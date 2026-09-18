"""Local copies of the CoIR APPS split (texts only; ids/metadata are never used as ranking signals)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_data() -> None:
    """Materialise corpus / test queries / train queries as JSONL (one-time, via MTEB + HF)."""
    DATA.mkdir(exist_ok=True)
    if not (DATA / "corpus.jsonl").exists() or not (DATA / "test_queries.jsonl").exists():
        import mteb

        task = mteb.get_task("AppsRetrieval")
        task.load_data()
        s = task.dataset["default"]["test"]
        with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as f:
            for x in s["corpus"]:
                f.write(json.dumps({"id": x["id"], "text": x["text"]}) + "\n")
        with open(DATA / "test_queries.jsonl", "w", encoding="utf-8") as f:
            for x in s["queries"]:
                f.write(json.dumps({"id": x["id"], "text": x["text"], "rel": list(s["relevant_docs"][x["id"]])}) + "\n")
    if not (DATA / "train_queries.jsonl").exists():
        from datasets import load_dataset

        qs = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
        qrels = load_dataset("CoIR-Retrieval/apps", "default", split="train")
        rel = {}
        for r in qrels:
            rel.setdefault(r["query-id"], []).append(r["corpus-id"])
        with open(DATA / "train_queries.jsonl", "w", encoding="utf-8") as f:
            for x in qs:
                if x["_id"] in rel:
                    f.write(json.dumps({"id": x["_id"], "text": x["text"], "rel": rel[x["_id"]]}) + "\n")


def load_corpus() -> list[dict]:
    return _read(DATA / "corpus.jsonl")


def load_queries(split: str = "test") -> list[dict]:
    return _read(DATA / f"{split}_queries.jsonl")
