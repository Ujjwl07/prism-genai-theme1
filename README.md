# PRISM GenAI Hackathon 2026 · Theme 01 — Agentic Code Intelligence

**Execution-verified, agentic code retrieval that runs on a laptop CPU.**
Given a natural-language query, rank code snippets by relevance — for competitive-programming problem statements
(CoIR `AppsRetrieval`, the screening benchmark) and for a real JS/TS voice-assistant codebase with versions.

> Every design decision is traced to a peer-reviewed paper and every experiment is logged in
> [`RESEARCH_LOG.md`](RESEARCH_LOG.md) (V0 = literature review, V1… = experiments, results and genuine failures).

## Results (CoIR AppsRetrieval, test split, 3,765 queries / 8,765 snippets)

See the table at the end of `RESEARCH_LOG.md` (kept in sync) and `results/`.
The submitted MTEB JSON is attached to the GitHub Release `PRISM_GENAI_HACKATHON_Y2026`
and also stored at `results/appsretrieval_results.json`.

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
| Execution verification of retrieved code | Reiss ICSE'09, Stolee et al. TOSEM'14, CodeT ICLR'23, LEVER ICML'23 |
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
