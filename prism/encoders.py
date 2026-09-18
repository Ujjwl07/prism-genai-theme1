"""Dense encoders used in the bake-off / pipeline, with an on-disk embedding cache keyed by content hash.

Cache key = sha1(model, prefix, max_len, text); a new code version only re-embeds texts whose hash is new
(this is the P1 incremental-rebuild mechanism).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "emb"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_len: int = 1024
    dtype: str = "float32"


# Prefixes follow each model card / paper.
SPECS: dict[str, EncoderSpec] = {
    # CoRNStack (ICLR 2025)
    "coderankembed": EncoderSpec("coderankembed", "nomic-ai/CodeRankEmbed",
                                 query_prefix="Represent this query for searching relevant code: "),
    # CodeXEmbed (COLM 2025)
    "codexembed-400m": EncoderSpec("codexembed-400m", "Salesforce/SFR-Embedding-Code-400M_R",
                                   query_prefix="Instruct: Given Code or Text, retrieval relevant content\nQuery: "),
    "codexembed-2b": EncoderSpec("codexembed-2b", "Salesforce/SFR-Embedding-Code-2B_R",
                                 query_prefix="Instruct: Given Code or Text, retrieval relevant content\nQuery: ",
                                 dtype="bfloat16"),
    # CodeSage (ICLR 2024)
    "codesage-large-v2": EncoderSpec("codesage-large-v2", "codesage/codesage-large-v2"),
}


def default_device() -> str:
    dev = os.environ.get("PRISM_DEVICE")
    if dev:
        return dev
    return "cpu"


class Encoder:
    def __init__(self, spec: EncoderSpec | str, device: str | None = None):
        self.spec = SPECS[spec] if isinstance(spec, str) else spec
        self.device = device or default_device()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            kw = {"torch_dtype": getattr(torch, self.spec.dtype)}
            self._model = SentenceTransformer(self.spec.hf_id, device=self.device, trust_remote_code=True,
                                              model_kwargs=kw)
            self._model.max_seq_length = self.spec.max_len
        return self._model

    def tokenizer_lengths(self, texts: list[str]) -> np.ndarray:
        tok = self.model.tokenizer
        return np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])

    def _key(self, text: str, prefix: str) -> str:
        h = hashlib.sha1(f"{self.spec.hf_id}|{self.spec.max_len}|{prefix}|{text}".encode()).hexdigest()
        return h

    def encode(self, texts: list[str], kind: str = "doc", batch_size: int = 16, show: bool = True) -> np.ndarray:
        """Encode with a per-text cache (cache/emb/<model>/<hash>.npy packed into shards)."""
        prefix = self.spec.query_prefix if kind == "query" else self.spec.doc_prefix
        store = _Store(CACHE / self.spec.name)
        keys = [self._key(t, prefix) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in store]
        if missing:
            # sort by length for efficient batching
            order = sorted(missing, key=lambda i: len(texts[i]))
            vecs = self.model.encode([prefix + texts[i] for i in order], batch_size=batch_size,
                                     normalize_embeddings=True, show_progress_bar=show,
                                     convert_to_numpy=True).astype(np.float32)
            store.put([keys[i] for i in order], vecs)
        return store.get(keys)


class _Store:
    """Tiny append-only key->vector store: one .npz shard per write, index loaded lazily."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, tuple[int, int]] = {}
        self.shards: list[np.ndarray] = []
        for p in sorted(root.glob("shard_*.npz")):
            z = np.load(p, allow_pickle=False)
            sid = len(self.shards)
            self.shards.append(z["vecs"])
            for j, k in enumerate(z["keys"]):
                self.index[str(k)] = (sid, j)

    def __contains__(self, k: str) -> bool:
        return k in self.index

    def put(self, keys: list[str], vecs: np.ndarray) -> None:
        sid = len(self.shards)
        tag = hashlib.sha1("".join(keys).encode()).hexdigest()[:16]
        np.savez(self.root / f"shard_{tag}.npz", keys=np.array(keys), vecs=vecs)
        self.shards.append(vecs)
        for j, k in enumerate(keys):
            self.index[k] = (sid, j)

    def get(self, keys: list[str]) -> np.ndarray:
        return np.stack([self.shards[s][j] for s, j in (self.index[k] for k in keys)])
