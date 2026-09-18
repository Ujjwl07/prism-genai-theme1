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
