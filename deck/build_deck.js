// Builds the submission deck from the numbers in ../results (run: node deck/build_deck.js)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const R = (f) => { try { return JSON.parse(fs.readFileSync(path.join(__dirname, "..", "results", f), "utf8")); } catch { return null; } };
const final = R("appsretrieval_results.json");
const fin = final ? final.scores.test[0] : null;
const FINAL_NDCG = fin ? (100 * fin.ndcg_at_10).toFixed(2) : "—";
const FINAL_MRR = fin ? (100 * fin.mrr_at_10).toFixed(2) : "—";
const FINAL_R1 = fin ? (100 * fin.recall_at_1).toFixed(1) : "—";
const hard = R("hard_query_example.json");
const TEAM = process.env.TEAM || "CollegeName_TeamName";
const MEMBERS = process.env.MEMBERS || "Team members: <name · year · branch>";

// palette: charcoal terminal + "pass" mint + "fail" coral
const C = { ink: "161A1F", ink2: "232A31", paper: "FFFFFF", mist: "EEF2F4", slate: "5B6770", text: "1D2329",
  mint: "2FBF84", mintSoft: "DDF5EA", coral: "E4574C", coralSoft: "FBE3E1", amber: "E9A23B", line: "D5DCE1" };
const H = "Arial", B = "Calibri", M = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Execution-verified agentic code retrieval";

const W = 13.33;
function title(s, t, sub, dark = false) {
  s.addText(t, { x: 0.6, y: 0.4, w: W - 1.2, h: 0.8, fontFace: H, fontSize: 34, bold: true, color: dark ? C.paper : C.text, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.6, y: 1.15, w: W - 1.2, h: 0.45, fontFace: B, fontSize: 16, color: dark ? "AEB8C0" : C.slate, margin: 0, isTextBox: true });
}
function codeCard(s, x, y, w, h, code, label, tone) {
  // motif: a small code window with three dots and a pass/fail pill
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.ink2 }, line: { color: C.ink2 }, rectRadius: 0.08,
    shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.18 } });
  ["E4574C", "E9A23B", "2FBF84"].forEach((c, i) => s.addShape(pres.shapes.OVAL, { x: x + 0.18 + i * 0.2, y: y + 0.14, w: 0.12, h: 0.12, fill: { color: c }, line: { color: c } }));
  if (label) {
    const col = tone === "pass" ? C.mint : tone === "fail" ? C.coral : C.slate;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + w - 2.05, y: y + 0.08, w: 1.9, h: 0.26, fill: { color: col }, line: { color: col }, rectRadius: 0.13 });
    s.addText(label, { x: x + w - 2.05, y: y + 0.08, w: 1.9, h: 0.26, fontFace: B, fontSize: 10, bold: true, color: C.paper, align: "center", valign: "middle", margin: 0, isTextBox: true });
  }
  s.addText(code, { x: x + 0.2, y: y + 0.42, w: w - 0.4, h: h - 0.55, fontFace: M, fontSize: 10, color: "D8DEE3", valign: "top", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, num, lab, col) {
  s.addText(num, { x, y, w, h: 0.9, fontFace: H, fontSize: 48, bold: true, color: col || C.text, margin: 0, isTextBox: true });
  s.addText(lab, { x, y: y + 0.9, w, h: 0.6, fontFace: B, fontSize: 13, color: C.slate, margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, head, body, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || C.mist }, line: { color: fill || C.mist }, rectRadius: 0.1 });
  s.addText(head, { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: 0.4, fontFace: H, fontSize: 16, bold: true, color: C.text, margin: 0, isTextBox: true });
  s.addText(body, { x: x + 0.25, y: y + 0.65, w: w - 0.5, h: h - 0.8, fontFace: B, fontSize: 13, color: C.text, valign: "top", margin: 0, isTextBox: true });
}

// 1 ─ Title ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.ink };
  s.addText("THEME 01 · AGENTIC CODE INTELLIGENCE", { x: 0.6, y: 0.6, w: 8, h: 0.4, fontFace: B, fontSize: 14, bold: true, color: C.mint, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("Retrieve, then verify:", { x: 0.6, y: 1.5, w: 7.2, h: 0.9, fontFace: H, fontSize: 44, bold: true, color: C.paper, margin: 0, isTextBox: true });
  s.addText("execution-verified code retrieval that runs on a laptop CPU", { x: 0.6, y: 2.4, w: 7.2, h: 1.3, fontFace: H, fontSize: 28, color: "C9D1D7", margin: 0, isTextBox: true });
  s.addText([{ text: TEAM, options: { bold: true, breakLine: true } }, { text: MEMBERS }],
    { x: 0.6, y: 5.6, w: 7.2, h: 0.9, fontFace: B, fontSize: 14, color: "AEB8C0", margin: 0, isTextBox: true });
  codeCard(s, 8.3, 1.3, 4.4, 2.3, "query ─► dense top-100\n       ─► run on the query's\n          own sample tests\n       ─► passing code first", "✓ passes samples", "pass");
  stat(s, 8.3, 4.2, 2.2, FINAL_NDCG, "NDCG@10 on CoIR APPS test", C.mint);
  stat(s, 10.6, 4.2, 2.2, "50.78", "dense-only baseline (same model)", "C9D1D7");
  s.addNotes("Title. Samsung PRISM GenAI Hackathon 2026, Theme 01. One-line pitch: retrieve with a CPU-sized code embedder, then verify candidates by executing them on the sample tests contained in the query.");
}

// 2 ─ Problem in our words ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "The problem, in our words", "Rank thousands of code snippets for one natural-language query — fast, on CPU, without an LLM reading everything.");
  card(s, 0.6, 1.9, 3.9, 2.35, "Different languages", "Queries tell a story (“Petya received a string…”). Code says n=int(input()). Keyword overlap is near zero: BM25 scores 1.5 NDCG@10.");
  card(s, 4.72, 1.9, 3.9, 2.35, "Long vs. short", "Query median 1,601 chars vs. code median 332 chars; 8,765 candidates, exactly one correct per query.");
  card(s, 8.84, 1.9, 3.9, 2.35, "Laptop CPU budget", "Judges run on Intel laptops without a GPU. A 400M embedder does ~0.4 docs/s on 2 vCPUs; a 2B model is ~6× slower.");
  s.addChart(pres.charts.BAR, [{ name: "NDCG@10", labels: ["BM25", "CodeRankEmbed 137M", "CodeXEmbed 400M"], values: [1.5, 23.53, 50.78] }], {
    x: 0.6, y: 4.5, w: 7.6, h: 2.6, barDir: "bar", chartColors: [C.slate], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFormatCode: "0.0",
    catAxisLabelColor: C.text, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Measured first-stage retrieval, APPS test (NDCG@10)", titleFontSize: 12, titleColor: C.slate, catAxisLabelFontSize: 11 });
  s.addText("Screening metric: NDCG@10 and MRR on CoIR AppsRetrieval (3,765 test queries), via MTEB.", { x: 8.6, y: 4.8, w: 4.1, h: 1.2, fontFace: B, fontSize: 14, color: C.text, margin: 0, isTextBox: true });
  s.addText("Generation after retrieval is out of scope — the job is ranking.", { x: 8.6, y: 6.0, w: 4.1, h: 0.8, fontFace: B, fontSize: 13, italic: true, color: C.slate, margin: 0, isTextBox: true });
}

// 3 ─ Key insight ──────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Insight: the query carries its own tests", "97.6% of APPS test queries contain sample Input/Output we can parse — so candidates can be checked, not just compared.");
  const q = hard ? hard.query.split("-----Examples-----")[1] || "" : "";
  codeCard(s, 0.6, 1.9, 3.9, 3.0, "-----Examples-----" + q.slice(0, 160), "extracted tests", "");
  codeCard(s, 4.72, 1.9, 3.9, 3.0, hard ? hard.dense_top1.slice(0, 300) : "", "dense #1 · ✗ fails", "fail");
  codeCard(s, 8.84, 1.9, 3.9, 3.0, hard ? hard.gold.slice(0, 360) : "", "dense #" + (hard ? hard.dense_rank : "?") + " → #1 · ✓", "pass");
  s.addText([
    { text: "Hard query " + (hard ? hard.qid : "") + " (“Petya's string game”): ", options: { bold: true } },
    { text: "the embedder ranks a string-capitalisation snippet first and the correct stack/greedy solution #" + (hard ? hard.dense_rank : "?") + ". Running the top candidates on “cab → abc” and “acdb → abdc” puts the correct program at #1." }],
    { x: 0.6, y: 5.2, w: 12.1, h: 1.0, fontFace: B, fontSize: 14, color: C.text, margin: 0, isTextBox: true });
  s.addText("Prior art we build on: test-filtered code search (Reiss, ICSE'09), I/O-example search (Stolee et al., TOSEM'14), execution reranking of generated code (CodeT, ICLR'23; LEVER, ICML'23). New here: tests mined automatically from the NL query, applied to dense-retrieved code.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.8, fontFace: B, fontSize: 12, color: C.slate, margin: 0, isTextBox: true });
}

// 4 ─ Architecture ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Architecture: plan → search → verify → refine", "Fixed pipeline, not a free-form agent (Agentless, FSE'25): reproducible and cheap on CPU.");
  const steps = [
    ["1  Route & parse", "classify query; split sections; extract sample I/O"],
    ["2  Dense search", "CodeXEmbed-400M, cached corpus vectors; top-100"],
    ["3  Verify", "run top-100 on sample tests in a sandbox (fail-fast)"],
    ["4  Refine (gated)", "only if nothing passed: verify down to rank 1,500"],
    ["5  Fuse & rank", "score = cosine + 1.0 × pass-fraction (LEVER-style)"]];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: 2.25, h: 1.9, fill: { color: i === 2 || i === 3 ? C.mintSoft : C.mist }, line: { color: i === 2 || i === 3 ? C.mint : C.line, width: 1 }, rectRadius: 0.1 });
    s.addText(st[0], { x: x + 0.15, y: 2.1, w: 1.95, h: 0.5, fontFace: H, fontSize: 15, bold: true, color: C.text, margin: 0, isTextBox: true });
    s.addText(st[1], { x: x + 0.15, y: 2.65, w: 1.95, h: 1.15, fontFace: B, fontSize: 12, color: C.text, valign: "top", margin: 0, isTextBox: true });
    if (i < 4) s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + 2.29, y: 2.85, w: 0.17, h: 0.2, rotate: 90, fill: { color: C.slate }, line: { color: C.slate } });
  });
  card(s, 0.6, 4.35, 3.9, 2.6, "Offline, once per version", "Corpus embeddings and execution results cached by content hash → a new code version only re-embeds / re-runs what changed.");
  card(s, 4.72, 4.35, 3.9, 2.6, "Sandbox (Windows + Linux)", "python -I subprocess, 4 s timeout, 512 MB watchdog, no network, APPS-harness wrapping. Tested on windows-latest + ubuntu-latest CI.");
  card(s, 8.84, 4.35, 3.9, 2.6, "Official output", "The whole pipeline is an MTEB SearchProtocol model, so the submitted JSON is produced by mteb.evaluate itself.");
}

// 5 ─ Research grounding ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Every stage is grounded in peer-reviewed work", "Full literature review, decisions and genuine failures: RESEARCH_LOG.md (V0 → V5).");
  const rows = [
    ["Stage", "Built on", "What we changed"],
    ["Query parsing / reduction", "Query-reformulation review, TOSEM'23", "Reduce long statements instead of expanding short ones"],
    ["Dense retrieval", "CodeXEmbed COLM'25 · CoRNStack ICLR'25", "Chose 400M: best accuracy that fits a laptop CPU"],
    ["Execution verification", "Reiss ICSE'09 · Stolee TOSEM'14 · CodeT ICLR'23 · LEVER ICML'23", "Tests mined from the query; verify retrieved (not generated) code"],
    ["Gated refinement", "Repoformer ICML'24 · Adaptive-RAG NAACL'24", "Extra compute only when no candidate passes"],
    ["Bounded agent loop", "Agentless FSE'25", "Fixed pipeline, ≤ 1 refine round"],
    ["Repo chunking", "cAST EMNLP'25 Findings", "AST split-merge for JS/TS, call graph per chunk"],
    ["Versions / near-duplicates", "Toma ICSE'24 · TreeCen ASE'22", "Token-type lineages collapse versions into one slot"],
    ["APPS execution semantics", "APPS, NeurIPS'21 D&B", "Harness-style wrapping of top-level return code"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.paper : C.text, fill: { color: i === 0 ? C.ink : i % 2 ? C.paper : C.mist } } }))),
    { x: 0.6, y: 1.85, w: 12.1, colW: [2.9, 4.3, 4.9], fontFace: B, fontSize: 12, border: { type: "solid", color: C.line, pt: 0.5 }, rowH: 0.52, valign: "middle" });
}

// 6 ─ Results / ablation ───────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Results on CoIR AppsRetrieval (test, 3,765 queries)", "Configuration selected on the APPS train split only; test used once for reporting.");
  s.addChart(pres.charts.BAR, [{ name: "NDCG@10", labels: ["BM25", "CodeRankEmbed", "CodeXEmbed-400M", "+ verify top-100", "+ gated refine (final)"],
    values: [1.5, 23.53, 50.78, 80.77, fin ? +FINAL_NDCG : 0] }], {
    x: 0.6, y: 1.8, w: 7.8, h: 5.2, barDir: "col", chartColors: [C.slate, C.slate, C.slate, C.mint, C.mint], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    dataLabelFontSize: 12, dataLabelFontBold: true, catAxisLabelColor: C.text, catAxisLabelFontSize: 11, valAxisHidden: true, valAxisMaxVal: 100,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "NDCG@10 (%)", titleFontSize: 12, titleColor: C.slate });
  stat(s, 8.9, 1.9, 3.8, FINAL_NDCG, "NDCG@10 — final pipeline (vs 50.78 dense-only)", C.mint);
  stat(s, 8.9, 3.55, 3.8, FINAL_MRR, "MRR@10", C.text);
  stat(s, 8.9, 5.2, 3.8, FINAL_R1 + "%", "queries with the correct solution at rank 1 (37.5% dense-only)", C.text);
}

// 7 ─ Tuning on train ──────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "How we chose the configuration (train split)", "568 train queries with stdin tests; weight w ≥ 0.2 plateaus → w = 1.0 (a passing program outranks any non-passing one).");
  s.addChart(pres.charts.LINE, [{ name: "verify top-k", labels: ["dense", "k=10", "k=20", "k=50", "k=100", "→300", "→500", "→1500", "→3000"],
    values: [42.26, 52.54, 57.89, 65.89, 71.53, 77.66, 80.41, 82.71, 83.41] }], {
    x: 0.6, y: 1.9, w: 7.6, h: 4.9, chartColors: [C.mint], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 8, showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.0",
    dataLabelFontSize: 11, catAxisLabelColor: C.text, valAxisLabelColor: C.slate, valAxisMinVal: 30, valAxisMaxVal: 90, valGridLine: { color: "E6EAED", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Train NDCG@10 vs. verification depth (→N = gated refine to rank N)", titleFontSize: 12, titleColor: C.slate });
  card(s, 8.6, 1.9, 4.1, 2.3, "Gating saves compute", "Refinement triggers on only 26.6% of queries. k=20→100 matches plain k=100 (70.5 vs 71.5) with ~half the executions — used for the live demo.");
  card(s, 8.6, 4.45, 4.1, 2.35, "Diminishing returns", "→1500 adds +2.3 over →500; →3000 adds only +0.7 more for ~1.3× the cost. Chosen: k=100, refine to 1,500.");
}

// 8 ─ Codebase search: P1 + Bonus ──────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Beyond APPS: version-aware repo search", "Demo on the open-source Leon voice assistant (723 files, 3 versions). Router: semantic · structural · usage × latest / old / all versions.");
  stat(s, 0.6, 1.9, 2.9, "53 / 710", "files re-parsed when moving to the next commit (rest from blob cache)", C.mint);
  stat(s, 3.7, 1.9, 2.9, "0.8 s", "incremental re-index vs 4.7 s cold (parse + cAST chunking)", C.text);
  stat(s, 6.8, 1.9, 2.9, "4.5 s", "cross-version lineage linking (was 86 s before per-file matching)", C.text);
  codeCard(s, 9.9, 1.9, 2.85, 1.9, "which files call\nrunWithProfileContext\nbefore\nensureActiveProfileRuntime?", "structural", "");
  card(s, 0.6, 4.3, 3.9, 2.65, "P1 · any version, fast", "Files are read from git objects (no checkout); parses cached per blob sha, vectors per chunk hash — only changed code is re-processed.");
  card(s, 4.72, 4.3, 3.9, 2.65, "Bonus · across versions", "Identical chunks dedup by hash; renamed/edited ones join a lineage by token-type similarity (Toma-style). One result slot per lineage, listing its versions.");
  card(s, 8.84, 4.3, 3.9, 2.65, "Structural & usage queries", "Call order from the AST (“X before Y”), identifier/string usage filters, dense + BM25 (RRF) ranking; 130–210 ms per query.");
}

// 9 ─ CPU & Windows ────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Built for the judges' hardware", "Everything the evaluator runs uses cross-platform components; heavy work is precomputed once and shipped as a cache.");
  const rows = [["Component", "Choice", "Why"],
    ["Embeddings", "PyTorch CPU (sentence-transformers)", "No CUDA/MPS/MLX; 400M params fit laptop RAM"],
    ["Sandbox", "subprocess + psutil watchdog", "No Unix-only `resource`; works on Windows"],
    ["Code parsing", "tree-sitter wheels, Python ast", "Prebuilt for Windows x86"],
    ["Caches", "content-hash keyed, release asset", "Pay corpus cost once; incremental after"],
    ["CI", "GitHub Actions windows-latest + ubuntu-latest", "Every push tested on real Windows x86"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.paper : C.text, fill: { color: i === 0 ? C.ink : i % 2 ? C.paper : C.mist } } }))),
    { x: 0.6, y: 1.9, w: 7.9, colW: [1.7, 3.2, 3.0], fontFace: B, fontSize: 12, border: { type: "solid", color: C.line, pt: 0.5 }, rowH: 0.55, valign: "middle" });
  stat(s, 9.0, 1.9, 3.7, "0.16 q/s", "400M query encoding on a 2-vCPU x86 runner (1024 tokens)", C.text);
  stat(s, 9.0, 3.6, 3.7, "~1.6 s", "mean verification time of top-100 per query on the M2 (fail-fast, 3 workers)", C.text);
  s.addText("Run: run.bat (Windows) · python app.py · docker run -p 7860:7860 prism", { x: 9.0, y: 5.6, w: 3.7, h: 1.2, fontFace: M, fontSize: 11, color: C.slate, margin: 0, isTextBox: true });
}

// 10 ─ Limitations ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.paper };
  title(s, "Limitations and genuine failures", "From RESEARCH_LOG.md — measured, not hypothetical.");
  card(s, 0.6, 1.9, 5.95, 1.55, "Multiple valid answers (~10% of gold solutions)", "“Print any valid permutation” problems need a checker; exact-match verification rejects correct code.", C.coralSoft);
  card(s, 6.75, 1.9, 5.95, 1.55, "Unretrievable gold (6% of train queries)", "Gold still missing from the final top-100 after refinement: if the embedder never surfaces it, verification cannot help.", C.coralSoft);
  card(s, 0.6, 3.65, 5.95, 1.55, "Weak samples → false positives (8.6%)", "Another program also passes all samples; dense score breaks the tie.", C.coralSoft);
  card(s, 6.75, 3.65, 5.95, 1.55, "Function-style tasks (2.4% of test)", "LeetCode/Codewars problems have no stdin I/O and no function name in the text → dense-only ranking.", C.coralSoft);
  card(s, 0.6, 5.4, 12.1, 1.5, "Cost concentrates on the hardest queries", "The gated refine round checks up to 1,400 extra programs for ~27% of queries (~20 s on a laptop CPU). Fine offline for the benchmark; interactive use caps the budget (k=20→100).", C.mist);
}

// 11 ─ Close ───────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide(); s.background = { color: C.ink };
  s.addText("Retrieve with a CPU-sized embedder.\nVerify with the query's own tests.", { x: 0.6, y: 1.4, w: 8.2, h: 2.0, fontFace: H, fontSize: 34, bold: true, color: C.paper, margin: 0, isTextBox: true });
  s.addText([{ text: "Next steps (worklet): ", options: { bold: true, color: C.mint } },
    { text: "learned checkers for multi-answer problems · LLM-generated tests for queries without samples (CodeT-style) · int8 OpenVINO encoders for Intel laptops · repo-level test execution for the JS assistant." }],
    { x: 0.6, y: 3.8, w: 8.2, h: 1.6, fontFace: B, fontSize: 16, color: "C9D1D7", margin: 0, isTextBox: true });
  stat(s, 9.3, 1.5, 3.4, FINAL_NDCG, "NDCG@10 (APPS test)", C.mint);
  stat(s, 9.3, 3.3, 3.4, "+" + (fin ? (FINAL_NDCG - 50.78).toFixed(1) : "—"), "points over the same embedder alone", C.paper);
  s.addText("github.com/Ujjwl07/prism-genai-theme1 · tag PRISM_GENAI_HACKATHON_Y2026", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontFace: B, fontSize: 12, color: "AEB8C0", margin: 0, isTextBox: true });
}

const out = path.join(__dirname, "..", `${TEAM}_Submission_ppt.pptx`);
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
