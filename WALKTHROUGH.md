# Walkthrough — understand this project from zero

Written so you can explain the work to a mentor or a jury without having built it.
Read time ≈ 45 minutes. Hands-on section ≈ 30 minutes. Practice questions at the end.

---

## 1. The one-paragraph version

> Given a natural-language description of a programming problem, we have to find the code that solves it, among
> 8,765 candidate programs. Text-matching fails completely here (1.5 out of 100), because the question is written in
> English about "Petya and his strings" and the answer is Python code like `n=int(input())`. A modern code embedding
> model that fits on a laptop CPU gets about 51. Our contribution: **most problem statements contain example inputs
> and outputs, so we don't have to guess which program is right — we can run the candidates and check.** Running the
> top candidates on the query's own examples, then ranking with a small learned model, takes the score to **90.68**,
> above the largest published embedding model (85.2, 7B parameters), while still running on a CPU.

---

## 2. Vocabulary you need (5 minutes)

| Term | Meaning here |
|---|---|
| **Query** | The problem statement in English (about 1,600 characters on average). |
| **Corpus / documents** | The 8,765 Python programs we search through. |
| **Gold** | The one correct program for a query. In this benchmark each query has exactly one. |
| **Embedding** | A list of ~1,024 numbers representing a text's meaning. Similar meanings → similar number lists. |
| **Dense retrieval** | Turn the query and every document into embeddings, then rank documents by similarity. Fast, because document embeddings are computed once in advance. |
| **Re-ranking** | Take the top N results from a cheap first stage and reorder them with something more expensive and more accurate. |
| **NDCG@10** | The score the judges use. Roughly: "is the correct answer in the top 10, and how near the top?" 100 = always first. |
| **MRR@10** | Average of 1/(rank of the correct answer), 0 if it's not in the top 10. |
| **Recall@k** | Percentage of queries whose correct answer is somewhere in the top k. |
| **Train / test split** | We tune everything on *train* problems and only report on *test* problems, so results aren't flattered by tuning on the answers we report. |

---

## 3. What the hackathon asked, and what we're scored on

Samsung PRISM GenAI Hackathon 2026, **Theme 01 — Agentic Code Intelligence**. The theme guidelines define the task as
**code retrieval**: given a library of code and a natural-language query, rank the snippets by relevance. Generating an
answer afterwards is explicitly out of scope.

- **Screening (decides the top 15 teams):** NDCG@10 and MRR on the **CoIR `AppsRetrieval` test split**, computed by the
  MTEB library, submitted as a JSON file on a GitHub release.
- **P1 (hands-on round):** the system must rebuild its index for *any version* of a changing codebase in reasonable time.
- **Bonus:** retrieval *across* versions, where near-identical copies of the same code must not swamp the results.
- **Hard constraint:** it must run on CPU. The judges use Windows laptops with Intel CPUs and no dedicated GPU.

**The benchmark data (CoIR, ACL 2025):** APPS problems from Codeforces, AtCoder, CodeChef and similar sites.
3,765 test queries, 8,765 Python programs, exactly one correct program per query. The queries are long (median 1,601
characters) and the programs are short (median 332).

---

## 4. Why this is hard (the three facts to quote)

1. **The query and the answer share almost no words.** BM25, classic keyword search, scores **1.50** NDCG@10 here. It is
   not weak, it is useless. The query says "Petya wants the lexicographically smallest string"; the program says
   `dq.append(ord(S[i]))`.
2. **The choice of embedding model dominates everything.** Measured on this data: UniXcoder 1.4 → CodeRankEmbed 23.5 →
   CodeXEmbed-400M **50.8**. Older code models are near zero, so "we used a code model" isn't a design; picking the
   right one is.
3. **The CPU budget caps the model size.** The best published model on this benchmark (CodeXEmbed-7B, 85.2) needs far
   more compute than a laptop has. Our 400M model runs about 0.2–0.35 queries per second on an Intel CPU. So we can't
   win by making the model bigger — we have to add a different kind of evidence.

---

## 5. The idea, in one example (this is the slide to present)

Test query **q5019**, "Petya's string game". The statement contains:

```
-----Examples-----
Input        cab          Input        acdb
Output       abc          Output       abdc
```

- **What the embedding model does:** ranks a *string-capitalisation* snippet first (it looks superficially related),
  and puts the correct stack/greedy solution at **rank 36**.
- **What we do:** extract those two examples from the statement, feed `cab` to each of the top candidates in a sandbox,
  and compare outputs. The capitalisation snippet prints `Cab`, so it's wrong. The correct program prints `abc` and then
  `abdc` for the second example. It moves to **rank 1**.

That's the whole idea: **the query carries its own test cases, so candidates can be checked rather than guessed at.**
97.6% of test queries contain examples we can parse automatically.

---

## 6. The pipeline, step by step

```
                 ┌──────────── done once, offline, cached ──────────────┐
 8,765 programs →│ embed with CodeXEmbed-400M → vectors saved on disk   │
                 └──────────────────────────────────────────────────────┘

 query → (1) parse the statement, pull out the sample Input/Output
       → (2) embed the query, compare with all document vectors → top 100
       → (3) run those 100 programs on the samples in a sandbox
       → (4) if none passes, keep going down the list to rank 1,500
       → (5) rank: programs that passed everything first, then a learned model orders the rest
```

**(1) Parse the statement** — `prism/query.py`. APPS statements use headers like `-----Input-----`,
`-----Examples-----`, `-----Sample Input-----`. We split on those and pair each example input with its expected output.
Works for Codeforces, AtCoder, CodeChef and Russian-language formats.

**(2) Dense retrieval** — `prism/encoders.py`. Document vectors are computed once and cached by a hash of the file
contents, so re-running costs nothing and a changed program only re-embeds itself. This same cache is what makes the
"rebuild for any version" requirement (P1) cheap.

**(3) Execution verification** — `prism/sandbox.py`. Each candidate runs in a separate Python process with a 4-second
timeout, a 512 MB memory watchdog, no network, and a temporary working directory. We compare outputs token by token,
ignoring whitespace, with a tolerance for floats and case. It works identically on Windows and Linux.

**(4) Gated refinement** — `prism/pipeline.py`. Most queries find a passing program in the top 100. Only when *none*
passes (about 27% of queries) do we keep verifying further down, to rank 1,500. This is the "spend extra compute only
where it's needed" idea from Repoformer (ICML 2024) and Adaptive-RAG (NAACL 2024).

**(5) Final ranking** — `prism/verifier.py`. Two levels:
   - **Hard filter:** any program that passed *all* the samples ranks above any program that didn't.
   - **Learned verifier:** inside each group, a small logistic-regression model scores candidates from 18 features:
     the embedding score and rank, plus execution evidence (crashed, timed out, produced empty output, how close the
     output is to the expected one in line count, token count, token types, and numeric values).
     This is LEVER (ICML 2023) adapted from *generated* code to *retrieved* code. It matters because about 10% of
     problems accept several valid answers ("print any"), where exact matching rejects correct code, but "ran fine and
     produced output of the right shape" is still strong evidence.

**Why call it agentic?** It plans (routes the query, decides what to extract), searches, *acts on the world* (runs code
and observes results), and refines when the evidence is weak. What it deliberately does **not** do is let a language
model improvise arbitrary steps. Agentless (FSE 2025) showed a fixed, bounded pipeline beats free-form agents on cost
and reliability, which also suits a CPU budget. Expect this question from the jury.

---

## 7. The second half: searching a real codebase (P1 + Bonus)

The benchmark is competitive-programming problems, but the theme is about voice-assistant codebases, so we also built
repository search over the open-source **Leon** assistant (JS/TS, 723 files), in `prism/codebase.py` and
`prism/repo_search.py`:

- **Chunking by syntax, not by lines** (cAST, EMNLP 2025 Findings): split the file along its syntax tree so a chunk is
  never half a function, and merge small neighbours.
- **Query routing**: "how does X work" → meaning-based search; "which files call X before Y" → checks the *order of
  calls* in the syntax tree; "where is X used" → identifier matching. Plus a version intent: latest, old, or all versions.
- **Versions (P1)**: files are read straight from git objects, and parsed chunks are cached per git blob hash. Moving to
  a newer commit re-parsed **53 of 710 files in 0.83 s**, and an arbitrary later commit needed **1 file, 0.44 s**.
- **Across versions (Bonus)**: identical chunks are collapsed by content hash; renamed or edited ones are linked into a
  "lineage" using token-type similarity (Toma, ICSE 2024). One result per lineage, listing which versions contain it.
  Making this comparison per-file brought it from **86 s to 4.5 s**.

---

## 8. Results (memorise this table)

**CoIR AppsRetrieval test set, 3,765 queries, official MTEB JSON:**

| Stage | NDCG@10 | MRR@10 | Correct at #1 |
|---|---|---|---|
| BM25 keyword search | 1.50 | 1.26 | 0.9% |
| CodeRankEmbed (137M) | 23.53 | 20.59 | 15.7% |
| CodeXEmbed-400M (our base model) | 50.78 | 46.16 | 37.5% |
| + run the top 100 on the samples | 80.77 | 79.70 | 77.0% |
| + refine to rank 1,500 when nothing passes | 88.91 | 87.69 | 84.7% |
| **+ learned verifier, exact passes first (submitted)** | **90.68** | **89.51** | **86.4%** |

Also: Recall@10 = 94.18%, Recall@100 = 95.67%.
Reference point: CodeXEmbed-7B, the largest published model on this benchmark, scores **85.22** on its own.

**Speed:** repository search 0.13–0.21 s per query. A new problem statement takes 0.7–1.4 s end-to-end in the demo.
The 3.9-hour benchmark run is a one-time offline cost, and every execution is cached, so re-running takes **42 seconds**.

---

## 9. Things we tried that did NOT work (be ready to volunteer these)

Mentors trust people who know their failures.

1. **Pseudo-relevance feedback** (TOIS 2023): blend the query vector with its top results. **Hurt** every time
   (48.25 → 46.95 on train). Reason: only one document is correct and the top-1 is right just 35% of the time, so the
   feedback is mostly wrong code.
2. **Shortening the query**: keeping the task and constraints first and dropping examples changed nothing (+0.36), because
   only 2.1% of queries exceed the model's 1,024-token limit.
3. **Naive int8 quantization for speed**: broke the embeddings (similarity to the original vectors 0.68). Fixed by keeping
   the FFN down-projection layers in full precision, exactly as Bondarenko et al. (EMNLP 2021) explain: 0.988 similarity,
   1.3–1.4× faster on Intel, no measured accuracy loss.
4. **Problem-text features + LightGBM ranking** (V8): +0.45 on train but **+0.00 on test**, so we kept the simpler model.
5. **ReCo / HyDE** (ACL 2024 / ACL 2023): strong published methods that generate code with an LLM at query time. They
   need a 7B+ model, i.e. minutes per query on a laptop CPU. Ruled out before building, on the CPU constraint.

And two real data findings:
- **Correct solutions "crashed" until we matched the official APPS harness.** APPS stores programs with top-level
  `return`, which only compiles inside a function. Wrapping them the way the benchmark's own harness does lifted the
  correct-program pass rate from 86.7% to 90.0%.
- **About 10% of problems accept several valid answers**, so exact output matching rejects correct code. This is a
  genuine limit of sample-based verification, and it's the largest remaining error bucket.

---

## 10. Hands-on: see it working (do this before you present)

Each block takes a few minutes. Run them from the project folder.

**a) Reproduce the submitted score from cache (~1 minute):**
```bash
.venv/bin/python evaluate.py
```
Expect `NDCG@10 = 90.68   MRR@10 = 89.51`. Say to your mentor: "the official number is reproducible in under a minute
because every execution is cached."

**b) See the sandbox refuse dangerous programs (~1 minute):**
```bash
.venv/bin/python -m pytest -q tests
```
Five tests pass, covering the statement parser, output comparison, an infinite loop, a memory bomb, a network attempt,
and incremental re-indexing.

**c) Watch execution fix a ranking (~2 minutes):**
```bash
.venv/bin/python scripts/experiment.py --split test --limit 200 --views raw
.venv/bin/python scripts/experiment.py --split test --limit 200 --views raw --exec-top-k 20 --exec-weight 1.0
```
The first is embeddings only, the second adds verification. Compare `ndcg@10` between them.

**d) Run the demo UI (~5 minutes):**
```bash
.venv/bin/python app.py
```
Open http://127.0.0.1:7860. Tab 1: paste a problem statement (use the one in `results/hard_query_example.json`), toggle
"Execution verification" off and on, and watch the ranking change. Tab 2: try
`which files call runWithProfileContext before ensureActiveProfileRuntime?`, then `where is setTimeout used`, then
`history of TTS init across versions`. Then press "Rebuild" with `HEAD~5` and read the timing line.

**e) Read the story of the project (~20 minutes):** `RESEARCH_LOG.md`, entries V0 → V8, in order.

---

## 11. Likely mentor / jury questions, with answers

**Q: What exactly is new here? Running tests on code is old.**
Yes — Reiss (ICSE 2009) filtered search results with tests, and CodeT (ICLR 2023) and LEVER (ICML 2023) ranked *generated*
programs by execution. Both need tests from somewhere: a user writes them, or an LLM generates them. Our step is to
**mine the tests automatically from the natural-language query** and apply them to **retrieved** code. As far as we found,
that combination isn't in the literature, and it's worth +30 points over the embedding model on this benchmark.

**Q: Isn't this specific to competitive programming?**
The verification part is, yes, and we say so openly. It applies when the query contains executable examples and the code
is a runnable program. For ordinary developer questions over a codebase there's nothing to run, and the system falls back
to embeddings plus keyword and structural search. We never claim 90 for that case.

**Q: Did you tune on the test set?**
No. Every parameter and the verifier were chosen on the train split, with cross-validation grouped by query. The research
log even discloses the one instance where a test observation inspired an idea, which was then decided on train.

**Q: Why not a bigger model? CodeXEmbed-7B scores 85.**
It doesn't fit the constraint. The judges' laptops have no GPU, and the 2B model alone is about 6× slower per query than
our 400M. We spend that compute budget on verification instead, which is worth more: 50.8 → 90.7.

**Q: Where is the "agent"?**
It plans, searches, acts by executing code, observes the result, and refines when the evidence is weak — but within a
fixed, bounded loop, following Agentless (FSE 2025), which beat free-form agents at a fraction of the cost. Free-form
LLM agents also aren't affordable on a CPU.

**Q: Is running downloaded code safe?**
Each program runs in a separate process with a 4-second timeout, a 512 MB memory cap, no network, and a temp directory.
Tests cover an infinite loop, a memory bomb and a network attempt. In production you'd add an OS-level sandbox or container.

**Q: What's the cost?**
One-time: about 3.9 hours for the full benchmark on a laptop, all cached afterwards. Interactive: 0.7–1.4 s per query.
Repository search: about 0.2 s. Re-indexing after a commit: about 1 second for the files that changed.

**Q: What would you do with more time?**
Two things, in order. First, first-stage recall: 4.6% of correct programs are never retrieved at all, which caps
everything downstream. Second, checkers for multi-answer problems, the biggest remaining error bucket at about 10%.

**Q: How do you know it isn't overfitted?**
The learned verifier has 19 parameters, it's trained only on train data with grouped cross-validation, and each
improvement that helped on held-out train folds also helped on test (82.7 → 84.9 → 85.2 on train; 88.9 → 89.7 → 90.7
on test).

---

## 12. A 3-minute spoken version (practise this out loud)

> "The task is code retrieval: given an English problem statement, rank thousands of programs so the correct one is at
> the top. It's measured with NDCG@10 on the CoIR APPS benchmark, and it has to run on a CPU laptop.
>
> The hard part is that the query and the answer have almost no words in common — keyword search scores 1.5 out of 100.
> A good code embedding model gets about 51, and the best published model, which is 7 billion parameters and far too big
> for a laptop, gets 85.
>
> Our idea comes from looking at the data: 97.6% of these problem statements contain example inputs and outputs. So
> instead of only comparing meanings, we can *run* the candidate programs on those examples and check which ones actually
> produce the expected output. We retrieve the top 100 with the embedding model, run them in a safe sandbox, and rank
> programs that pass above programs that don't. When nothing passes, we keep verifying further down the list, which
> happens for about a quarter of queries. A small learned model, based on LEVER from ICML 2023, orders the rest using
> evidence like whether the program crashed and how close its output shape is to the expected one — that's for the 10%
> of problems that accept several valid answers.
>
> The result is 90.7 NDCG@10, compared with 50.8 for the embedding model alone, and it beats that 7B model while running
> on a CPU. A query takes about a second. We also built version-aware search over a real JavaScript assistant codebase,
> where re-indexing a new commit takes about a second because only changed files are re-parsed.
>
> We also documented what failed: pseudo-relevance feedback hurt, query shortening did nothing, and naive int8
> quantization broke the embeddings until we kept the outlier-prone layers in full precision."

---

## 13. If you only have 30 minutes before the meeting

1. Section 5 (the q5019 example) — this is the story.
2. Section 8 (the results table) — memorise 1.5 → 50.8 → 90.7.
3. Section 9 (failures) — pick two.
4. Section 12 — say it out loud twice.
