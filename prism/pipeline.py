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
from .sandbox import Sandbox, outputs_match

VARIANTS = {
    "raw": lambda t: t,
    "reduced": lambda t: reduced(t),
    "reduced_noex": lambda t: reduced(t, keep_examples=False),
}


@dataclass
class PipelineConfig:
    encoder: str = "codexembed-400m"
    max_len: int = 1024            # document-side max tokens
    query_max_len: int = 1024      # query-side max tokens (512 halves query-encoding cost on CPU)
    query_views: tuple[str, ...] = ("raw",)
    fusion: str = "sum"            # "sum" of cosine scores or "rrf"
    rrf_k: int = 60
    exec_top_k: int = 20           # candidates verified by execution
    exec_weight: float = 1.0       # LEVER-style additive bonus for passing sample tests
    exec_max_tests: int = 3
    gate_margin: float | None = None  # skip execution when top1-top2 dense margin exceeds this (None = always run)
    expand_k: int = 0              # refine round: if nothing in top exec_top_k passes, verify ranks up to expand_k
    final_top_k: int = 100
    verifier: str | None = None    # path to a trained LEVER-style verifier (V6); None = pass/fail bonus rule
    verifier_max_tests: int = 5
    query_encoder: str | None = None  # V7b: e.g. "codexembed-400m-int8" (same weights, int8 query side)
    early_exit_batch: int = 0         # V7d: verify in batches of this size, stop after the first full pass (0 = off)


@dataclass
class QueryTrace:
    """Per-query record for the demo UI / latency report."""
    ranking: list[tuple[str, float]]
    stage_ms: dict[str, float] = field(default_factory=dict)
    executed: bool = False
    refined: bool = False
    n_tests: int = 0
    passes: dict[str, float] = field(default_factory=dict)
    features: dict[int, list[float]] = field(default_factory=dict)  # doc index -> verifier features (V6)


class Pipeline:
    def __init__(self, cfg: PipelineConfig, encoder: Encoder | None = None, sandbox: Sandbox | None = None):
        from dataclasses import replace

        from .encoders import SPECS

        self.cfg = cfg
        self.encoder = encoder or Encoder(replace(SPECS[cfg.encoder], max_len=cfg.max_len))
        # query encoder shares the loaded model; only the truncation length (and cache key) differs
        if cfg.query_encoder:  # separate (e.g. int8) model for queries; documents keep fp32 cached vectors
            self.q_encoder = Encoder(replace(SPECS[cfg.query_encoder], max_len=cfg.query_max_len))
        else:
            self.q_encoder = self.encoder if cfg.query_max_len == cfg.max_len else \
                Encoder(replace(SPECS[cfg.encoder], max_len=cfg.query_max_len))
            if self.q_encoder is not self.encoder:
                self.q_encoder._model_from = self.encoder
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
            Q = self.q_encoder.encode([VARIANTS[v](q) for q in queries], "query", show=False)
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
    # ---------------- verification with a learned verifier (V6) ----------------
    def collect_features(self, queries: list[str], S: np.ndarray) -> list[QueryTrace]:
        """Stage-1 verify top exec_top_k; refine to expand_k where no candidate passes all samples.
        Every executed candidate gets a feature vector (prism.verifier.candidate_features)."""
        from .verifier import candidate_features

        cfg = self.cfg
        width = max(cfg.final_top_k, cfg.exec_top_k, cfg.expand_k)
        tops = np.argsort(-S, axis=1)[:, :width]
        traces = [QueryTrace(ranking=[], n_tests=len(parse(q).tests)) for q in queries]
        tests_of = [parse(q).tests for q in queries]

        def run(pairs: list[tuple[int, int, int]]) -> None:  # (query, doc, rank)
            res = self.sandbox.details_many([(self.doc_texts[j], tests_of[qi]) for qi, j, _ in pairs],
                                            cfg.verifier_max_tests)
            for (qi, j, r), det in zip(pairs, res):
                top1 = float(S[qi, tops[qi][0]])
                traces[qi].features[j] = candidate_features(float(S[qi, j]), top1, r, det, len(tests_of[qi]))
                n = min(len(tests_of[qi]), cfg.verifier_max_tests)
                ok = sum(1 for st, got, want in det if st == "ok" and outputs_match(got, want))
                traces[qi].passes[self.doc_ids[j]] = ok / n if n else 0.0

        if cfg.early_exit_batch > 0:  # V7d anytime verification (interactive latency)
            b, limit = cfg.early_exit_batch, max(cfg.exec_top_k, cfg.expand_k)
            for qi in range(len(queries)):
                if not (tests_of[qi] and self.sandbox is not None and cfg.exec_top_k > 0):
                    continue
                traces[qi].executed = True
                for start in range(0, limit, b):
                    run([(qi, int(j), start + r) for r, j in enumerate(tops[qi][start: start + b])])
                    if any(v == 1.0 for v in traces[qi].passes.values()):
                        break
                    traces[qi].refined = start + b >= cfg.exec_top_k
            self._tops = tops
            return traces
        first = []
        for qi in range(len(queries)):
            if tests_of[qi] and self.sandbox is not None and cfg.exec_top_k > 0:
                traces[qi].executed = True
                first += [(qi, int(j), r) for r, j in enumerate(tops[qi][: cfg.exec_top_k])]
        run(first)
        if cfg.expand_k > cfg.exec_top_k:
            more = []
            for qi, tr in enumerate(traces):
                if tr.executed and not any(v == 1.0 for v in tr.passes.values()):
                    tr.refined = True
                    more += [(qi, int(j), cfg.exec_top_k + r) for r, j in enumerate(tops[qi][cfg.exec_top_k: cfg.expand_k])]
            run(more)
        self._tops = tops
        return traces

    def rank_with_verifier(self, traces: list[QueryTrace], S: np.ndarray, verifier) -> list[QueryTrace]:
        """Executed candidates: all-samples-passed first, then by verifier logit; non-executed after (dense order)."""
        from .verifier import FEATURES

        for qi, tr in enumerate(traces):
            top = self._tops[qi]
            if tr.features:
                idx = list(tr.features)
                F = np.array([tr.features[j] for j in idx])
                lg = verifier.logits(F)
                # V6b: exact verification is a hard filter (all samples passed first); the verifier orders within groups
                allp = F[:, FEATURES.index("all_passed")]
                scored = [(self.doc_ids[j], 100.0 + 50.0 * float(a) + float(v)) for j, v, a in zip(idx, lg, allp)]
                seen = set(idx)
                scored += [(self.doc_ids[j], float(S[qi, j])) for j in top if int(j) not in seen]
            else:
                scored = [(self.doc_ids[j], float(S[qi, j])) for j in top]
            scored.sort(key=lambda x: -x[1])
            tr.ranking = scored[: self.cfg.final_top_k]
        return traces

    def rerank_with_execution(self, queries: list[str], S: np.ndarray) -> list[QueryTrace]:
        cfg = self.cfg
        if cfg.verifier:
            from .verifier import Verifier

            if not hasattr(self, "_verifier"):
                self._verifier = Verifier.load(cfg.verifier)
            return self.rank_with_verifier(self.collect_features(queries, S), S, self._verifier)
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
