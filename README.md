# PRISM GenAI Hackathon 2026 · Theme 01 — Agentic Code Intelligence

**Execution-verified, agentic code retrieval that runs on a laptop CPU.**
Given a natural-language query, rank code snippets by relevance — for competitive-programming problem statements
(CoIR `AppsRetrieval`, the screening benchmark) and for a real JS/TS voice-assistant codebase with versions.

> Every design decision is traced to a peer-reviewed paper and every experiment is logged in
> [`RESEARCH_LOG.md`](RESEARCH_LOG.md) (V0 = literature review, V1… = experiments, results and genuine failures).

## Results (CoIR AppsRetrieval, test split, 3,765 queries / 8,765 snippets)

| Pipeline stage | NDCG@10 | MRR@10 | Recall@1 |
|---|---|---|---|
| BM25 (code-aware tokens) | 1.50 | 1.26 | 0.90 |
| CodeRankEmbed 137M (dense) | 23.53 | 20.59 | 15.67 |
| CodeXEmbed-400M (dense) | 50.78 | 46.16 | 37.53 |
| + execution verification of top-100 | 80.77 | 79.70 | 77.03 |
| + gated refinement to rank 1,500 | 88.91 | 87.69 | 84.65 |
| **+ learned verifier, exact-pass first (submitted, V6b)** | **90.68** | **89.51** | **86.43** |

Configuration selected on the APPS **train** split (`results/best_config.json`, RESEARCH_LOG V2/V2b/V6).
Official JSON: `results/appsretrieval_results.json` (MTEB 2.21.0), also attached to the GitHub Release
`PRISM_GENAI_HACKATHON_Y2026`.

**Interactive latency (CPU):** repository search 0.13–0.21 s/query (warm). Problem→code search uses the demo profile
(int8 512-token queries — 1.8× faster than fp32/1024 on an Intel Xeon at no measured accuracy loss — plus the learned
verifier with early-exit verification): 0.7–1.4 s end-to-end on an M2 laptop (RESEARCH_LOG V7).
The 3.9 h benchmark run is a one-time offline cost (≈1.8 M sandboxed executions, all cached).

## Submission materials

- Presentation: `CollegeName_TeamName_Submission_ppt.pptx` (built by `node deck/build_deck.js` from `results/`)
- Demo video (≤ 5 min): _link to be added_
- Research log: [`RESEARCH_LOG.md`](RESEARCH_LOG.md)

## How it works

```
OFFLINE (per code version, content-hash cache → incremental rebuilds)
  snippets ──► dense code embeddings (CodeXEmbed-400M, CPU)         [APPS]
  repo@commit ─► cAST chunks (tree-sitter) ─► call graph + embeddings + BM25   [JS/TS repo]
QUERY TIME
  query ─► route (semantic | structural | usage) + version intent (latest | old | all)
        ─► reduce & restructure long problem statements, extract sample I/O
        ─► dense retrieval (top-100)
        ─► verify: run top-k candidates on the query's own sample tests in a sandbox
        ─► LEVER-style score fusion ─► ranking (+ per-stage latency)
```

| Stage | Built on (peer-reviewed) |
|---|---|
| Problem framing (query/code/match end) | Survey of Source Code Search, TOSEM'24 |
| Query reduction / restructuring | Query-reformulation review, TOSEM'23 |
| Dense code retrieval | CodeXEmbed (COLM'25), CoRNStack (ICLR'25), CodeSage (ICLR'24) |
| Execution verification of retrieved code + learned verifier | Reiss ICSE'09, Stolee et al. TOSEM'14, CodeT ICLR'23, LEVER ICML'23 |
| Confidence-gated extra compute | Repoformer ICML'24, Adaptive-RAG NAACL'24 |
| Fixed, bounded agent loop | Agentless FSE'25 |
| AST chunking for repositories | cAST, EMNLP'25 Findings |
| Near-duplicate versions → lineages | Toma ICSE'24, TreeCen ASE'22 |
| Rank fusion | RRF, SIGIR'09 |

## Quick start (Windows / Linux / macOS, CPU only)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements-demo.txt
python scripts/fetch_cache.py            # optional: download precomputed embeddings (Release asset)
python evaluate.py                       # reproduce the MTEB JSON  (results/appsretrieval_results.json)
git clone --depth 60 https://github.com/leon-ai/leon demo_repos/leon
python app.py                            # demo UI at http://127.0.0.1:7860
```
Windows shortcut: `run.bat`. Docker: `docker build -t prism . && docker run -p 7860:7860 prism`.

Without the precomputed cache, embedding the 8,765-snippet corpus on a 4-core laptop CPU takes a while
(see latency section in the log); everything is cached by content hash, so it is paid once.

## Repository layout

```
prism/query.py        section parsing, query reduction, sample-I/O extraction
prism/encoders.py     dense encoders + content-hash embedding cache (incremental)
prism/sandbox.py      cross-platform execution sandbox (timeout, memory watchdog, no network)
prism/pipeline.py     retrieval pipeline + execution verification + gating
prism/mteb_model.py   MTEB SearchProtocol adapter (official JSON)
prism/codebase.py     cAST chunking, call graph, versioned incremental index, lineages
prism/repo_search.py  query router (semantic/structural/usage) + version-aware search
app.py                Gradio demo
scripts/              bake-off, embedding jobs, experiments
tests/                cross-platform tests (CI on windows-latest + ubuntu-latest)
```

## Integrity notes
- Only `text` fields are used for ranking. Dataset ids and `meta_information` (which contains problem URLs) are
  never read by the pipeline.
- Tuning uses the APPS **train** split only; the test split is used for reporting.
