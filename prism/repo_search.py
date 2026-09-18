"""Query-routed retrieval over a versioned JS/TS repository (demo + P1 + Bonus).

Router (query categorisation, cf. Adaptive-RAG NAACL'24): each query is classified as
  structural  "which files call X before Y"      -> ordered call-graph filter, then ranked
  usage       "where is X used / referenced"      -> exact identifier/string occurrences, ranked by dense sim
  semantic    anything else                      -> dense (code embedder) + BM25, fused with RRF (SIGIR'09)
plus an orthogonal version intent: latest (default) | old | all-versions (history).

Cross-version retrieval groups chunks by lineage (codebase.VersionedIndex.link_lineages) so near-identical
versions of the same code occupy ONE result slot; the best-matching version is shown with its siblings.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .codebase import Chunk, VersionedIndex
from .encoders import Encoder
from .lexical import BM25, tokenize

_STRUCT = re.compile(r"call(?:s|ing|ed)?\s+(?:the\s+)?(?:tool\s+|function\s+|method\s+)?[`'\"]?([\w.$]+)[`'\"]?"
                     r"\s+(?:tool\s+|function\s+)?(before|after)\s+(?:calling\s+)?(?:the\s+)?(?:tool\s+|function\s+|method\s+)?"
                     r"[`'\"]?([\w.$]+)[`'\"]?", re.I)
_USAGE = re.compile(r"(?:where|which\s+files?)\s+(?:is|are|does|do)?\s*(?:the\s+)?(.+?)\s+(?:used|called|referenced|invoked|defined)", re.I)
_OLD = re.compile(r"\b(old|older|previous|earlier|originally|before the|used to|legacy|initial)\b", re.I)
_ALL = re.compile(r"\b(all versions|history|evolution|every version|across versions|over time|changed)\b", re.I)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    why: str
    versions: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    route: str
    version_intent: str
    hits: list[Hit]
    ms: float


def classify(query: str) -> tuple[str, str]:
    route = "semantic"
    if _STRUCT.search(query):
        route = "structural"
    elif _USAGE.search(query):
        route = "usage"
    vi = "all" if _ALL.search(query) else "old" if _OLD.search(query) else "latest"
    return route, vi


def _short(name: str) -> str:
    return name.split(".")[-1].lower()


class RepoSearcher:
    def __init__(self, index: VersionedIndex, commits: list[str], encoder: str = "coderankembed"):
        """commits: oldest -> newest (already built in `index`)."""
        self.index = index
        self.commits = commits
        self.enc = Encoder(encoder)
        # unique chunks across versions (dedup by content hash; remember every commit that contains it)
        self.uniq: dict[str, Chunk] = {}
        self.seen_in: dict[str, list[str]] = {}
        for c in commits:
            for ch in index.by_commit[c]:
                if ch.hash not in self.uniq:
                    self.uniq[ch.hash] = ch
                self.seen_in.setdefault(ch.hash, []).append(c)
        self.chunks = list(self.uniq.values())
        self.pos = {c: i for i, c in enumerate(commits)}
        t = time.time()
        self.E = self.enc.encode([ch.header() + ch.text for ch in self.chunks], "doc", batch_size=16)
        self.embed_seconds = time.time() - t
        self.bm25 = BM25([f"{ch.path} {ch.symbol} {ch.text}" for ch in self.chunks])

    # ---------------------------------------------------------------------------------------------
    def _mask(self, version_intent: str, commit: str | None) -> np.ndarray:
        if version_intent == "all":
            return np.ones(len(self.chunks), bool)
        target = commit or (self.commits[0] if version_intent == "old" else self.commits[-1])
        return np.array([target in self.seen_in[ch.hash] for ch in self.chunks])

    def _semantic_scores(self, query: str) -> np.ndarray:
        q = self.enc.encode([query], "query", show=False)[0]
        dense = self.E @ q
        lex = self.bm25.scores([query])[0]
        rd = np.argsort(np.argsort(-dense))
        rl = np.argsort(np.argsort(-lex))
        return 1 / (60 + rd + 1) + 1 / (60 + rl + 1)

    def search(self, query: str, k: int = 10, commit: str | None = None) -> SearchResult:
        t = time.time()
        route, vint = classify(query)
        mask = self._mask(vint, commit)
        base = self._semantic_scores(query)
        why = ["semantic (dense+BM25, RRF)"] * len(self.chunks)
        if route == "structural":
            m = _STRUCT.search(query)
            a, rel, b = _short(m.group(1)), m.group(2).lower(), _short(m.group(3))
            if rel == "after":
                a, b = b, a
            ok = np.zeros(len(self.chunks), bool)
            for i, ch in enumerate(self.chunks):
                names = [_short(x) for x in ch.calls]
                if a in names and b in names and names.index(a) < max(j for j, n in enumerate(names) if n == b):
                    ok[i] = True
                    why[i] = f"calls {a}() before {b}() (call order in AST)"
            if ok.any():
                mask = mask & ok
        elif route == "usage":
            m = _USAGE.search(query)
            target = m.group(1).strip("`'\" ")
            toks = [w for w in re.split(r"[\s`'\"]+", target) if w]
            ident = max(toks, key=len) if toks else target
            pat = re.compile(re.escape(ident), re.I)
            ok = np.array([bool(pat.search(ch.text)) for ch in self.chunks])
            if not ok.any():  # fall back to sub-token match (e.g. "bluetooth settings deeplink")
                parts = [p for p in tokenize(target) if len(p) > 2]
                ok = np.array([sum(p in ch.text.lower() for p in parts) >= max(1, len(parts) - 1) for ch in self.chunks])
            for i in np.where(ok)[0]:
                why[i] = f"references '{ident}'"
            if ok.any():
                mask = mask & ok
        scores = np.where(mask, base, -np.inf)
        order = [int(i) for i in np.argsort(-scores) if np.isfinite(scores[i])]
        hits: list[Hit] = []
        taken: dict[str, Hit] = {}
        for i in order:
            ch = self.chunks[i]
            vers = [c[:8] for c in self.seen_in[ch.hash]]
            if vint == "all" and ch.lineage in taken:  # collapse near-duplicate versions into one slot
                taken[ch.lineage].versions.extend(v for v in vers if v not in taken[ch.lineage].versions)
                continue
            h = Hit(ch, float(scores[i]), why[i], vers)
            taken[ch.lineage] = h
            hits.append(h)
            if len(hits) >= k:
                break
        return SearchResult(route, vint, hits, 1000 * (time.time() - t))
