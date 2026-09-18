"""The retrieval pipeline: route -> reduce -> dense (multi-view RRF) -> [gated] execution verification.

Stage map (see RESEARCH_LOG.md):
  query-end   : prism.query.reduced / parse           (TOSEM'23 reformulation review, adapted as *reduction*)
  match-end   : dense bi-encoder(s) + RRF              (CodeXEmbed COLM'25 / CoRNStack ICLR'25; Cormack SIGIR'09)
  verification: sample-I/O execution of top-k          (Reiss ICSE'09, Stolee TOSEM'14, CodeT ICLR'23, LEVER ICML'23)
  routing     : confidence-gated extra compute         (Repoformer ICML'24, Adaptive-RAG NAACL'24)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .encoders import Encoder
from .query import parse, reduced
from .sandbox import Sandbox

VARIANTS = {
    "raw": lambda t: t,
    "reduced": lambda t: reduced(t),
    "reduced_noex": lambda t: reduced(t, keep_examples=False),
}


@dataclass
class PipelineConfig:
    encoder: str = "codexembed-400m"
    max_len: int = 1024
    query_views: tuple[str, ...] = ("raw",)
    fusion: str = "sum"            # "sum" of cosine scores or "rrf"
    rrf_k: int = 60
    exec_top_k: int = 20           # candidates verified by execution
    exec_weight: float = 1.0       # LEVER-style additive bonus for passing sample tests
    exec_max_tests: int = 3
    gate_margin: float | None = None  # skip execution when top1-top2 dense margin exceeds this (None = always run)
    expand_k: int = 0              # refine round: if nothing in top exec_top_k passes, verify ranks up to expand_k
    final_top_k: int = 100


@dataclass
class QueryTrace:
    """Per-query record for the demo UI / latency report."""
    ranking: list[tuple[str, float]]
    stage_ms: dict[str, float] = field(default_factory=dict)
    executed: bool = False
    refined: bool = False
    n_tests: int = 0
    passes: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: PipelineConfig, encoder: Encoder | None = None, sandbox: Sandbox | None = None):
        from dataclasses import replace

        from .encoders import SPECS

        self.cfg = cfg
        self.encoder = encoder or Encoder(replace(SPECS[cfg.encoder], max_len=cfg.max_len))
        self.sandbox = sandbox
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.D: np.ndarray | None = None

    # ---------------- indexing (content-hash cached => incremental across versions) ----------------
    def index(self, doc_ids: list[str], doc_texts: list[str]) -> float:
        t = time.time()
        self.doc_ids, self.doc_texts = list(doc_ids), list(doc_texts)
        self.D = self.encoder.encode(self.doc_texts, "doc", show=False)
        return time.time() - t

    # ---------------- first stage ----------------
    def dense_scores(self, queries: list[str]) -> np.ndarray:
        views = []
        for v in self.cfg.query_views:
            Q = self.encoder.encode([VARIANTS[v](q) for q in queries], "query", show=False)
            views.append(Q @ self.D.T)
        if len(views) == 1:
            return views[0]
        if self.cfg.fusion == "sum":
            return np.mean(views, axis=0)
        # reciprocal rank fusion
        fused = np.zeros_like(views[0])
        for S in views:
            ranks = np.argsort(np.argsort(-S, axis=1), axis=1)
            fused += 1.0 / (self.cfg.rrf_k + ranks + 1)
        return fused

    # ---------------- verification ----------------
    def rerank_with_execution(self, queries: list[str], S: np.ndarray) -> list[QueryTrace]:
        cfg = self.cfg
        traces: list[QueryTrace] = []
        jobs, where = [], []
        tops = np.argsort(-S, axis=1)[:, : max(cfg.final_top_k, cfg.exec_top_k, cfg.expand_k)]
        for qi, q in enumerate(queries):
            tests = parse(q).tests
            top = tops[qi]
            s = S[qi, top]
            tr = QueryTrace(ranking=[], n_tests=len(tests))
            run = bool(tests) and self.sandbox is not None and cfg.exec_top_k > 0 and cfg.exec_weight > 0
            if run and cfg.gate_margin is not None and len(s) > 1 and (s[0] - s[1]) > cfg.gate_margin:
                run = False
            tr.executed = run
            if run:
                for j in top[: cfg.exec_top_k]:
                    jobs.append((self.doc_texts[j], tests[: cfg.exec_max_tests]))
                    where.append((qi, int(j)))
            traces.append(tr)
        results = self.sandbox.check_many(jobs) if jobs else []
        bonus: dict[tuple[int, int], float] = {}
        for (qi, j), r in zip(where, results):
            if not math.isnan(r):
                bonus[(qi, j)] = r
                traces[qi].passes[self.doc_ids[j]] = r
        # refine round (bounded, one extra pass): widen verification only where no candidate passed
        if cfg.expand_k > cfg.exec_top_k:
            jobs, where = [], []
            for qi, tr in enumerate(traces):
                if tr.executed and not any(v == 1.0 for v in tr.passes.values()):
                    tests = parse(queries[qi]).tests
                    for j in tops[qi][cfg.exec_top_k: cfg.expand_k]:
                        jobs.append((self.doc_texts[j], tests[: cfg.exec_max_tests]))
                        where.append((qi, int(j)))
                    tr.refined = True
            for (qi, j), r in zip(where, self.sandbox.check_many(jobs) if jobs else []):
                if not math.isnan(r):
                    bonus[(qi, j)] = r
                    traces[qi].passes[self.doc_ids[j]] = r
        for qi, tr in enumerate(traces):
            top = tops[qi]
            scored = [(self.doc_ids[j], float(S[qi, j]) + cfg.exec_weight * bonus.get((qi, int(j)), 0.0))
                      for j in top]
            scored.sort(key=lambda x: -x[1])
            tr.ranking = scored[: cfg.final_top_k]
        return traces

    def search(self, queries: list[str]) -> list[QueryTrace]:
        t0 = time.time()
        S = self.dense_scores(queries)
        t1 = time.time()
        traces = self.rerank_with_execution(queries, S)
        t2 = time.time()
        n = max(len(queries), 1)
        for tr in traces:
            tr.stage_ms = {"dense": 1000 * (t1 - t0) / n, "verify": 1000 * (t2 - t1) / n}
        return traces
