# Research Log — Theme 01: Agentic Code Intelligence (Samsung PRISM GenAI Hackathon 2026)

**Rules for this log**
1. This file is **append-only**. Each change to the approach is a new version (V0, V1, V2, …). Earlier entries are never rewritten.
2. Every design decision cites a **peer-reviewed paper from a top venue**. We cite the peer-reviewed version, not the arXiv preprint, whenever one exists.
3. Only **genuine, method-level bottlenecks and failures** are logged. Implementation bugs, typos and install problems are fixed and not recorded.

**Task.** Rank code snippets for a natural-language query. Screening metric: NDCG@10 and MRR on the CoIR `AppsRetrieval` test split (MTEB).
**Runtime constraint.** Must run on CPU only, on Windows laptops with Intel x86 CPUs and integrated GPU. Development happens on a MacBook Air M2.

---

## V0 — Literature review and baseline design (2026-09-18)

### V0.1 The benchmark, from its own paper
- **CoIR: A Comprehensive Benchmark for Code Information Retrieval Models.** Li et al., *ACL 2025* (Long). [arXiv 2407.02883](https://arxiv.org/abs/2407.02883) · [ACL Anthology](https://aclanthology.org/2025.acl-long.1072.pdf)
  - APPS is a "code contest retrieval" task. Queries are problem statements (Codeforces, AtCoder, Kattis, Codewars). The corpus is Python solutions.
  - Split sizes (CoIR Table 2): about **5k train** queries, about **3.8k test** queries, a corpus of about **9k**, no dev split. Average query length is about 1.4k and average document length about 575. So queries are about 2.5× longer than documents.
  - Each query has **exactly one** relevant document, so NDCG@10 and MRR depend entirely on where that one solution lands.
  - **Consequence:** there is no official dev split, so all tuning happens on a held-out slice of the **train** split. The test split is only used for reporting.

### V0.2 Published APPS scores (NDCG@10), and what they rule out
Sources: **CodeXEmbed** (Liu et al., *COLM 2025*, Table 1, [arXiv 2411.12644](https://arxiv.org/abs/2411.12644)) and **CoRNStack** (Suresh et al., *ICLR 2025*, Table 8, [arXiv 2412.01007](https://arxiv.org/abs/2412.01007)). Baseline rows in both come from the CoIR leaderboard.

| Model | Params | APPS NDCG@10 |
|---|---|---|
| UniXcoder | 123M | 1.36 |
| CodeT5+ (embedding) | 110M | 3.3 |
| BGE-Base / BGE-M3 | 110M / 567M | 4.05 / 7.37 |
| OpenAI-Ada-002 | – | 8.70 |
| E5-base | 110M | 11.52 |
| Jina-Code-v2 | 161M | 16.4 |
| CodeSage-Small / Base / Large | 130M / 356M / 1.3B | 17.3 / 27.6 / 32.7 |
| CodeRankEmbed (CoRNStack) | 137M | 21.1 |
| E5-Mistral | 7B | 21.33 |
| Voyage-Code-002 | – | 26.52 |
| CodeSage-large-v2 | 1.3B | 50.45 |
| CodeXEmbed-400M | 400M | 48.57 |
| CodeXEmbed-2B | 2B | 74.99 |
| CodeXEmbed-7B | 7B | 85.22 |

**What we will NOT do, and why:**
- **Use UniXcoder, GraphCodeBERT or CodeT5-era encoders as candidates.** UniXcoder (*ACL 2022*) scores 1.36 and CodeT5+ scores 3.3 on APPS. Those papers are background for the "related work" slide only.
- **Assume that a model tuned for code is enough.** Jina-Code-v2 (16.4) and CodeSage-Small (17.3) lose badly to CodeXEmbed-2B (74.99), a larger model trained with multi-task text+code retrieval data. We choose the model by **measuring it on APPS**, not by its label.

**Taken forward:** the candidate set for the V1 bake-off is CodeXEmbed-400M, CodeXEmbed-2B, CodeSage-large-v2 and CodeRankEmbed (a small reference). The deciding factor is the accuracy vs. CPU latency trade-off on an Intel laptop, so we measure both.

### V0.3 Component-by-component review
Each entry lists what the paper already did, what we reuse, what we change, and what we deliberately don't do.

#### (a) How the problem space is organised
- **A Survey of Source Code Search: A 3-Dimensional Perspective.** *ACM TOSEM 2024*. [doi:10.1145/3656341](https://doi.org/10.1145/3656341)
  - Sorts 68 papers into **query-end** (reformulation), **code-end** (representation) and **match-end** (matching/ranking) work.
  - **Use:** our pipeline is organised on these three axes, and they map onto the five improvement axes in the problem statement: categorise query, preprocess query, categorise snippets, pre/post-process snippets, multi-pass retrieval.

#### (b) Query end: reformulation
- **A Systematic Review of Automated Query Reformulations in Source Code Search.** Rahman & Roy, *ACM TOSEM 2023*. [doi:10.1145/3607179](https://doi.org/10.1145/3607179)
  - Surveys 70 studies. Most reformulation methods **expand** short, vague queries by adding terms.
  - **Why this doesn't fit APPS directly:** APPS queries are the opposite, long and noisy, and they get cut off at encoder limits. We apply reformulation as **reduction and restructuring**: keep the task, input/output format and constraints, move narrative to the end, and extract the sample I/O as structured data.
- **HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels.** Gao et al., *ACL 2023*. [aclanthology 2023.acl-long.99](https://aclanthology.org/2023.acl-long.99/)
  - Generates a hypothetical document with an LLM and retrieves with its embedding. Big zero-shot gains over Contriever.
  - **Change:** the "hypothetical document" becomes a hypothetical *Python solution*. Generating on CPU at query time is expensive, so HyDE only runs **when first-stage confidence is low** (see (d)).
  - **Known risk (from ReCo, below):** generated code differs in style from the corpus code, which hurts matching. That's why we pair it with ReCo.

#### (c) Code end: representation and rewriting
- **ReCo: Rewriting the Code — A Simple Method for LLM-Augmented Code Search.** Li, Zhou & Shen, *ACL 2024* (Oral). [aclanthology 2024.acl-long.75](https://aclanthology.org/2024.acl-long.75/)
  - Generation-augmented retrieval (GAR) generates example code for the query, but that code is written in a different style from the code in the codebase. ReCo **rewrites the corpus code** into a normalised style. Reported gains: up to +35.7% for sparse, +27.6% for zero-shot dense and +23.6% for fine-tuned dense retrieval.
  - **Change:** the rewrite runs **once, offline**, with a small local code LLM (llama.cpp, so it works on Windows and Mac), cached by content hash. A new code version only rewrites the snippets that changed, which feeds directly into P1.
- **cAST: Structural Chunking via Abstract Syntax Tree.** *EMNLP 2025 Findings*. [doi:10.18653/v1/2025.findings-emnlp.430](https://doi.org/10.18653/v1/2025.findings-emnlp.430)
  - Recursive AST-node splitting with sibling merging, instead of line-based chunking. Reports +4.3 Recall@5 on RepoEval.
  - **Not used for APPS:** each APPS document is already one complete, short solution, and MTEB scores whole documents. **Used for** the JavaScript demo repository and for versioned indexing (P1).

#### (d) Match end: how much compute each query gets
- **Repoformer: Selective Retrieval for Repository-Level Code Completion.** Wu et al., *ICML 2024* (Oral). [PMLR / ACM DL 10.5555/3692070.3694253](https://dl.acm.org/doi/10.5555/3692070.3694253)
  - Always retrieving is inefficient and sometimes harmful. Deciding selectively gives up to a **70% inference speed-up with no loss** in quality.
- **Adaptive-RAG.** Jeong et al., *NAACL 2024*. [aclanthology 2024.naacl-long.389](https://aclanthology.org/2024.naacl-long.389/)
  - A small classifier sends each query to no retrieval, single-step or iterative retrieval based on how complex it is.
  - **Change (applies to both):** every APPS query needs retrieval, so we don't gate *whether* to retrieve. We gate **how much extra work** to do: stop at the first stage, or go on to rerank, execution check and refine. The routing signal is first-stage confidence (margin between top scores, agreement between views), calibrated on the train split. This is how we stay within an Intel-CPU latency budget.

#### (e) Match end: fusion and reranking
- **Reciprocal Rank Fusion.** Cormack, Clarke & Büttcher, *SIGIR 2009*. Combines rankings without training. Used to fuse the raw-code, ReCo-code and HyDE result lists.
- **ColBERTv2.** Santhanam et al., *NAACL 2022*. [doi:10.18653/v1/2022.naacl-main.272](https://doi.org/10.18653/v1/2022.naacl-main.272) Late interaction sits between a bi-encoder and a cross-encoder in cost, which is a good fit for CPU.
- **RankGPT: Is ChatGPT Good at Search?** Sun et al., *EMNLP 2023*. LLM listwise reranking. Too slow for large LLMs on CPU. We only consider it with a small local model and a small k.
- **CoRNStack.** *ICLR 2025*. Releases a code reranker trained on high-quality contrastive code data. It's a candidate for the rerank stage.
- **RocketQAv2.** Ren et al., *EMNLP 2021*. [doi:10.18653/v1/2021.emnlp-main.224](https://doi.org/10.18653/v1/2021.emnlp-main.224) Trains retriever and reranker jointly via listwise distillation.
  - **Plan:** if a cross-encoder reranker is too slow on an Intel CPU, we distil it into the bi-encoder along the lines of RocketQAv2 rather than drop it.

#### (f) Match end: verifying by execution (our main new idea, and its prior art)
**Prior art that already exists, so we don't claim it as new:**
- **Semantics-Based Code Search.** Reiss, *ICSE 2009*. The user supplies keywords plus test cases. Candidates are retrieved by keywords, transformed, and **kept only if they compile and pass the tests**.
- **Solving the Search for Source Code.** Stolee, Elbaum & Dobos, *ACM TOSEM 2014*. [doi:10.1145/2581377](https://doi.org/10.1145/2581377) Code search driven by **concrete input/output examples**, matched via constraint solving.
- **CodeT: Code Generation with Generated Tests.** Chen et al., *ICLR 2023*. Reranks generated programs by running them against tests (dual execution agreement).
- **LEVER: Learning to Verify Language-to-Code Generation with Execution.** Ni et al., *ICML 2023*. A learned verifier combines the NL input, the program and its **execution result**, and adds +4.6 to +10.9 points.

**What we change, and where our contribution actually lies:**
- Reiss and Stolee used specifications the *user wrote*, over a keyword or SMT-based search engine. CodeT and LEVER rerank *generated* programs. We **rerank *retrieved* programs from a dense retriever**, using sample I/O that is **already in the natural-language query** and that we extract automatically. We combine the pass/fail signal with the retrieval score LEVER-style (weights learned on the train split, not the test split).
- **Scope, stated honestly:** this only applies to queries that contain executable examples (APPS-style). For other queries the router skips it.

#### (g) Agent control loop
- **Agentless: Demystifying LLM-Based Software Engineering Agents.** Xia et al., *FSE 2025* (ACM SIGSOFT Distinguished Paper). [doi:10.1145/3715754](https://doi.org/10.1145/3715754) A fixed localise → repair → validate pipeline beat open-source agents on SWE-bench Lite (32.0%) at about $0.70 per issue.
- **AutoCodeRover.** Zhang et al., *ISSTA 2024*. [doi:10.1145/3650212.3680384](https://doi.org/10.1145/3650212.3680384) An agent with tools for searching code structure (class/method search).
- **SWE-agent.** Yang et al., *NeurIPS 2024*. Shows how much the design of the agent's tool interface matters.
  - **Decision:** we will **not** build a free-form LLM agent loop. Following Agentless, "agentic" here means a **fixed, bounded** plan → search → verify → refine loop with at most one extra round. That makes it reproducible, cheap on CPU and easy to evaluate. AutoCodeRover's structural search tools inspire the JS demo's structural queries (call graph).

#### (h) Long queries
- **mGTE: Generalized Long-Context Text Representation and Reranking Models.** Zhang et al., *EMNLP 2024 Industry*. [doi:10.18653/v1/2024.emnlp-industry.103](https://doi.org/10.18653/v1/2024.emnlp-industry.103)
  - **Use:** reference for handling long inputs. We compare (i) plain truncation, (ii) the cleaned query from (b), and (iii) the chosen encoder's own longer context limit. We measure how many APPS queries exceed each limit in V1 instead of assuming.

#### (i) Versions and near-duplicates (P1 and Bonus, hands-on round)
- **Machine Learning is All You Need: A Simple Token-based Approach for Effective Code Clone Detection (Toma).** *ICSE 2024*. [doi:10.1145/3597503.3639114](https://doi.org/10.1145/3597503.3639114) Cheap token-type-sequence features for clone detection.
- **TreeCen: Building Tree Graph for Scalable Semantic Code Clone Detection.** *ASE 2022*. [doi:10.1145/3551349.3556927](https://doi.org/10.1145/3551349.3556927)
  - **Use:** group near-identical chunks across commits into **lineages**. Rank lineages first, then pick the best version inside each one. This keeps near-duplicate versions from filling the top 10, and uses cheap features so it runs on CPU.

#### (j) Fine-tuning (optional, only if there's cloud GPU time; inference stays on CPU)
- **CoRNStack** (*ICLR 2025*): fine-tuning on high-quality contrastive code data raised average CoIR by +5.6 to +16.7 across three ~130M encoders (Table 9).
- **CodeXEmbed** (*COLM 2025*): in-domain training changes APPS only slightly for the 400M model (48.57 → 45.91) and slightly more for 2B (74.99 → 76.86).
  - **Takeaway:** for the models we'd actually use, in-domain fine-tuning gives **small gains on APPS**. It's lower priority than query reduction, ReCo and execution verification. Only the APPS **train** split would ever be used.

### V0.4 Baseline design coming out of the review
```
OFFLINE (per code version, content-hash cache)
  snippets → normalise → ReCo rewrite (local small LLM) → embed raw + rewritten → index
QUERY TIME
  query → route + reduce/restructure + extract sample I/O
        → dense top-100 (RRF over views)
        → confident? yes → return
                     no  → rerank top-k → execution verification → gated HyDE refine (≤1 round) → return
```
Cross-platform rules: PyTorch-CPU / ONNX Runtime / OpenVINO for encoders, llama.cpp for local LLMs, a `subprocess` + `psutil` sandbox (no Unix-only `resource` module), and a Windows + Linux CI matrix.

### V0.5 Open questions to settle by measurement in V1
1. Real CPU latency of the candidate encoders on a 4-core Intel CPU (Windows CI) vs. the M2, in fp32 vs. int8.
2. What fraction of APPS test queries exceed each encoder's token limit?
3. What fraction of APPS queries contain sample I/O that can be parsed and executed? This sets how much execution verification can help at most.
4. Can the installed MTEB version take a custom search/rerank stage and still write the official result JSON?

### V0 genuine bottlenecks identified (from the literature, before any experiments)
- **Vocabulary and modality gap:** APPS queries describe a story, and the code has meaningless identifiers. That's why general text encoders score in single digits (V0.2).
- **Query/document length asymmetry:** queries are about 2.5× longer than documents, so we risk cutting off exactly the constraint and format lines that distinguish one problem from another.
- **CPU budget:** the strongest published model that could plausibly fit (CodeXEmbed-2B) is about 5× bigger than the 400M option, so model choice is limited by Intel-CPU latency, not only accuracy.

**Next → V1:** set up MTEB, run the encoder bake-off on the APPS test split, measure CPU latency on the M2 and a Windows Intel runner, and answer the four open questions above.

### V0 team decisions (2026-09-18)
- **Execution-based verification: GO.** It is presented transparently as APPS-specific verification, with the prior art in V0.3(f) credited.
- **Cloud GPU (Colab/Kaggle): allowed for one-time offline jobs only** (optional fine-tuning on the APPS train split, precomputing ReCo rewrites). Anything the judges run stays CPU-only (Windows, Intel).

---

## V1 — Measured facts and first-stage bake-off (2026-09-18)

### Research basis
- Candidate encoders and their published APPS numbers: see V0.2 (CodeXEmbed COLM'25; CoRNStack ICLR'25).
- Lexical baseline: BM25 (Robertson & Zaragoza, *Foundations and Trends in IR* 2009) with identifier splitting.

### Setup
- Data: CoIR `AppsRetrieval` via MTEB 2.21 (revision `f22508f9`). Measured: **8,765 corpus docs** (5,000 are solutions to *train* problems and act as distractors; 3,765 are solutions to test problems), **3,765 test queries**, exactly one relevant doc per query.
- Lengths (characters): query median **1,601** (p90 2,592), doc median **332** (p90 1,128).
- Encoders run with sentence-transformers, `max_len=1024`, fp32, **CPU only**, on GitHub-hosted x86 runners (2 vCPU). The same kind of hardware as the judges' Intel laptops.
- Metrics computed by `prism/metrics.py` (same definitions as MTEB/pytrec_eval for binary relevance).

### Results (APPS test, raw queries, dense only)
| Method | NDCG@10 | MRR@10 | R@1 | R@10 | R@100 | Paper (NDCG@10) |
|---|---|---|---|---|---|---|
| BM25 (code-aware tokens) | 1.50 | 1.26 | 0.90 | 2.26 | 9.38 | – |
| BM25 on reduced query | 1.49 | 1.26 | 0.88 | 2.26 | 9.93 | – |
| CodeRankEmbed (137M) | 23.53 | 20.59 | 15.67 | 33.04 | 56.92 | 21.1 |
| **CodeXEmbed-400M** | **50.78** | **46.16** | **37.53** | **65.52** | **86.37** | 48.57 |

Dense recall of the gold doc for CodeXEmbed-400M: R@20 = 73.04, R@50 = 81.46, R@100 = 86.37.

### Genuine bottlenecks & failures
1. **Lexical matching fails completely on APPS (NDCG@10 1.5).** Problem statements ("Vasya has n candies…") and solutions (`n=int(input())`) share almost no vocabulary. This matches the V0 modality-gap analysis, and it's why BM25 is excluded from the APPS pipeline. It is kept only for the JS repository, where identifiers do carry meaning.
2. **CPU throughput on laptop-class hardware.** On a 2-vCPU x86 runner, CodeXEmbed-400M embeds **0.38 corpus docs/s** and **0.16 queries/s** (queries are ~4× longer). CodeRankEmbed manages 1.36 docs/s. On the 8 GB M2, local CPU runs were in the same range. This is dominated by compute (≈2·params·tokens FLOPs per text), not by implementation overhead, and it has two consequences:
   - CodeXEmbed-2B (75.0 NDCG@10 in its paper) would cost about 6× more per text, which is impractical for query-time use on the judges' CPU laptops. **We keep the 400M model** and try to close the gap with verification (V2), not with model size.
   - Corpus embeddings are computed **once**, cached by content hash, and shipped as a release asset. Only query-time encoding happens on the judge's machine (~2–6 s/query for the 400M model on a laptop CPU).
3. **Development-machine memory.** Running the 400M encoder and a full-corpus BM25 side by side on the 8 GB dev laptop exhausted memory. All heavy jobs now run as sharded GitHub Actions jobs (`.github/workflows/embed.yml`, `task.yml`).

### Answers to the V0.5 open questions
- Q2 (truncation): answered in V3 (query reduction), together with its effect on accuracy.
- Q3: **97.6%** of test queries (3,674/3,765) contain sample I/O that `prism/query.py` can parse (Codeforces, AtCoder and Russian-language formats).
- Q4: MTEB 2.21 exposes `SearchProtocol` (`index`/`search`), so the **whole pipeline**, execution verification included, can write the official JSON (`prism/mteb_model.py`).

**Next → V2:** execution-based verification of the dense top-k.

---

## V2 — Execution-based verification of retrieved code (2026-09-18)

### Research basis
- **Reiss, "Semantics-Based Code Search", ICSE 2009**: filters keyword-search results by running user-supplied tests.
- **Stolee, Elbaum & Dobos, "Solving the Search for Source Code", ACM TOSEM 2014**: code search driven by input/output examples.
- **CodeT, ICLR 2023** and **LEVER, ICML 2023**: rerank *generated* programs using execution results. LEVER combines the model's score with an execution-based verifier.
- **APPS, Hendrycks et al., NeurIPS 2021 Datasets & Benchmarks**: the benchmark's official harness runs stdin programs wrapped in a function.
- What these papers already established, so we don't redo it: running code is a much stronger correctness signal than lexical or semantic similarity. We don't train a LEVER-style verifier because the pass/fail signal on the sample tests is already very sharp (see failures below). A learned verifier would add compute without a clear gain.

### What we changed / adapted
- **Test source.** No user-written specs (unlike Reiss and Stolee) and no LLM-generated tests (unlike CodeT). We **extract the sample I/O already written in the natural-language query** (`prism/query.py`), covering Codeforces, AtCoder, CodeChef and Russian formats. Coverage: **97.6%** of test queries.
- **Candidates.** We verify **retrieved** programs (the dense top-k), not generated ones.
- **Scoring.** LEVER-style additive fusion, `score = cosine(q, d) + w · pass_fraction(d)`, with fail-fast over up to 3 sample tests.
- **Sandbox.** One `python -I` subprocess per run, with a wall-clock timeout (4 s), a psutil RSS watchdog (512 MB) and sockets stubbed out. Results are cached by (execution-semantics hash, code, test). The same code runs on Windows and Linux (CI).

### Setup
- Encoder: CodeXEmbed-400M (V1), raw queries.
- **Tuning on the train split only**: the first 1,200 train queries, restricted to the **568** with parseable stdin tests. Many early train problems are LeetCode *function-style* tasks with no stdin I/O, while the test split is 97.6% stdin-style, so this restriction makes the tuning set match the test distribution.

### Results — train tuning grid (568 queries, NDCG@10)
| top-k verified | w=0.02 | w=0.05 | w=0.1 | w=0.2 | w=1.0 |
|---|---|---|---|---|---|
| dense only | 42.26 | | | | |
| 10 | 49.26 | 52.13 | 52.54 | 52.54 | 52.54 |
| 20 | 52.07 | 56.88 | 57.89 | 57.89 | 57.89 |
| 50 | 53.92 | 63.40 | 65.87 | 65.89 | 65.89 |
| 100 | 54.03 | 66.28 | 71.32 | 71.53 | **71.53** |

- The weight plateaus at w ≥ 0.2, i.e. a program that passes should outrank any program that doesn't. **Chosen: w = 1.0**, the robust end of the plateau.
- Accuracy increases with k all the way to 100, so the limit is how many candidates get verified, not how the scores are fused → motivates V2b.

### Genuine bottlenecks & failures
1. **Correct solutions failing their own sample tests.** In a 150-query sample, 13 of 150 gold solutions (8.7%) "crashed". The cause is data format, not a sandbox bug: APPS stores stdin solutions with **top-level `return` / `nonlocal`**, which only compile inside the function wrapper the official APPS harness uses. **Fix:** when plain compilation fails, the sandbox wraps the program the way the APPS harness does. The gold pass rate on the same sample went from **86.7% to 90.0%**. The remaining ~10% are **problems that accept several correct answers** (e.g. "print any valid permutation"), which can't be verified without the problem's own checker. This is an **inherent limit** of sample-based verification.
2. **False positives.** In **8.6%** of test queries at least one *non-gold* program in the top-20 also passes all sample tests (near-duplicate problems among the 5,000 train-problem distractors, or weak samples). Among programs that pass, the dense score decides the order.
3. **Verification depth vs. recall.** With k = 20, 27% of gold solutions are never verified (dense R@20 = 73.0%); R@100 = 86.4%.
4. **Function-style (LeetCode/Codewars) problems can't be executed.** They have no stdin/stdout I/O, and the function name isn't in the CoIR text. These queries fall back to dense ranking. This affects **2.4%** of test queries but a large share of the early train split.

---

## V2b — Confidence-gated refinement round (2026-09-18)

### Research basis
- **Repoformer, ICML 2024** and **Adaptive-RAG, NAACL 2024**: spend extra retrieval effort only when it's needed.
- **Agentless, FSE 2025**: a fixed, bounded pipeline beats open-ended agent loops.

### What we changed / adapted
- One bounded extra round: **if none of the top-k verified candidates passes all sample tests**, keep verifying down the dense ranking to rank `expand_k`. Queries that already have a passing candidate stop, and that's most of them.

### Results — train (568 queries, w = 1.0)
| Configuration | NDCG@10 | MRR@10 | R@1 | R@10 | queries refined | verify s/query (M2, 3–4 workers, cold cache) |
|---|---|---|---|---|---|---|
| k=100 (no refine) | 71.53 | 69.83 | 65.85 | 76.76 | 0% | 1.58 |
| k=20 → 100 | 70.45 | 69.09 | 65.85 | 74.65 | 43% | n/a (warm cache) |
| k=100 → 300 | 77.66 | 75.83 | 71.48 | 83.27 | 26.6% | 1.7 |
| k=100 → 500 | 80.41 | 78.57 | 74.30 | 86.09 | 26.6% | 1.6 |
| **k=100 → 1500** | **82.71** | **80.82** | **76.58** | **88.56** | 26.6% | 5.8 |
| k=100 → 3000 | 83.41 | 81.52 | 77.29 | 89.26 | 26.6% | 7.3 |

- **k=20→100 matches plain k=100 (−1.1) with about half the executions.** That's the latency setting for the interactive demo.
- The returns shrink beyond 1,500 (+0.7 for 2× the refine cost). **Chosen for the screening run: k=100, expand_k=1500, w=1.0**, selected on train only.

### Genuine bottlenecks & failures
- **Refinement cost is concentrated on the hardest ~27% of queries.** These are the queries whose gold solution can't be verified (special-judge problems) or isn't retrievable at all. For them the refine round checks up to 1,400 extra programs before giving up, which costs about 20 s on a laptop CPU. That's fine offline (the screening JSON), but for interactive use the `expand_k` budget has to be capped. This is the latency/accuracy trade-off we report.

---

## V3 — Query reduction / restructuring (2026-09-18)

### Research basis
- **Systematic review of automated query reformulation in code search, ACM TOSEM 2023**: most techniques *expand* short queries. APPS queries have the opposite problem, so we test *reduction*: keep the statement, constraints and I/O spec first, examples next, and drop notes (`prism/query.reduced`).
- **mGTE, EMNLP 2024 Industry**: when inputs exceed the context window, what gets truncated matters.

### Setup
Dense-only CodeXEmbed-400M on the train split (1,200 queries), `max_len = 1024`.

### Results (train, NDCG@10)
| Query view | NDCG@10 | MRR@10 | R@10 |
|---|---|---|---|
| raw | 48.25 | 43.85 | 62.17 |
| reduced | 48.61 | 44.10 | 62.92 |
| raw + reduced (score mean) | 48.42 | 43.96 | 62.50 |
| raw + reduced (RRF) | 48.31 | 43.87 | 62.33 |

### Genuine finding
- **At 1,024 tokens, reduction barely matters (+0.36).** Measured with the model's tokenizer, test queries have a median of **469 tokens** (p90 760): only **2.1%** exceed 1,024 tokens, but **41.2%** exceed 512. The V0 worry about truncation mostly disappears at 1,024 tokens. It would only matter at 512, which halves query-encoding cost on CPU (being tested: 512-token train embeddings on CI).
- **Decision:** keep raw queries for the screening configuration. Reduction stays available as a switch (`query_views`).

---

## V4 — Versioned repository search: P1 and Bonus (2026-09-18)

### Research basis
- **cAST (EMNLP 2025 Findings)** for AST split-then-merge chunking. **Toma (ICSE 2024)** for token-type abstraction in clone detection. **Adaptive-RAG (NAACL 2024)** for query routing. **RRF (SIGIR 2009)** for dense + lexical fusion. **AutoCodeRover (ISSTA 2024)** for structure-aware code-search tools (our call-order filter).

### Setup
Leon open-source voice assistant (JS/TS), 3 versions spanning 60 commits (705 → 723 files). Encoder CodeRankEmbed (137M, chosen for repo-scale CPU cost). tree-sitter JS/TS parsers.

### Results
| Measurement | Value |
|---|---|
| Cold index of version 1 (parse + cAST chunks + call lists) | 705 files → 3,096 chunks in 4.7 s (CI runner) |
| Next version (incremental, blob-sha cache) | **53** of 710 files re-parsed, 0.83 s |
| Latest version (incremental) | **73** of 723 files re-parsed, 1.27 s |
| Unique chunks across the 3 versions (dedup by content hash) | 3,677 (vs 9,408 total) |
| Arbitrary later commit (HEAD~5) via demo "Rebuild" | 1 file re-parsed, 0.44 s |
| Query latency (route + dense + BM25 + RRF), warm | 130–210 ms |
| Structural query "call runWithProfileContext before ensureActiveProfileRuntime" | top-1 `server/src/core/socket-server.ts:978-1035` (verified order in AST) |

### Genuine bottlenecks & failures
1. **Lineage linking was quadratic.** Comparing every new chunk with every previous chunk took **86 s** for 3 versions. Restricting candidates to *orphaned chunks of the same file*, plus a `quick_ratio` pre-filter, brought it to **4.5 s**. The limitation: a chunk that moves to a *different file* and is renamed at the same time is not linked. It becomes a new lineage.
2. **Repo embedding cost on CPU.** Embedding the 3,677 unique chunks took 5,133 s on a 2-vCPU runner (0.7 chunks/s, 137M model). That's why vectors are cached by chunk hash and shipped. After that, a new commit only embeds its changed chunks.
3. **Test files often rank high for "how does X work" queries.** They mention the same identifiers as the implementation. We haven't down-weighted them yet (possible next step: a path prior).
