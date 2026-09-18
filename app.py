"""Gradio demo: (1) problem-statement -> code retrieval with execution verification (APPS corpus),
(2) routed, version-aware search over a JS/TS repository.

Run:  python app.py [--repo demo_repos/leon] [--no-apps]
CPU-only; works on Windows/Linux/macOS.
"""
from __future__ import annotations

import argparse
import html
import subprocess
import time

import gradio as gr

from prism.codebase import VersionedIndex
from prism.data import ensure_data, load_corpus
from prism.pipeline import Pipeline, PipelineConfig
from prism.repo_search import RepoSearcher
from prism.sandbox import Sandbox

STATE: dict = {}


def build_apps() -> None:
    ensure_data()
    corpus = load_corpus()
    # interactive latency budget (RESEARCH_LOG V2b): verify top-20, refine to 100 only if nothing passes
    # interactive profile (RESEARCH_LOG V3b/V6b/V7): int8 512-token queries, learned verifier, early-exit verification
    pipe = Pipeline(PipelineConfig(exec_top_k=100, expand_k=300, query_max_len=512, query_encoder="codexembed-400m-int8",
                                   verifier="results/verifier.json", early_exit_batch=10), sandbox=Sandbox())
    t = time.time()
    pipe.index([d["id"] for d in corpus], [d["text"] for d in corpus])
    STATE["apps"] = pipe
    STATE["apps_index_s"] = time.time() - t
    pipe.q_encoder.encode(["warm-up"], "query", show=False)  # load weights before the first user query


def apps_search(problem: str, use_exec: bool, k: int):
    pipe: Pipeline = STATE["apps"]
    k_exec = pipe.cfg.exec_top_k
    pipe.cfg.exec_top_k = k_exec if use_exec else 0
    t = time.time()
    try:
        tr = pipe.search([problem])[0]
    finally:
        pipe.cfg.exec_top_k = k_exec
    total = 1000 * (time.time() - t)
    rows = []
    for rank, (did, score) in enumerate(tr.ranking[:k], 1):
        code = pipe.doc_texts[pipe.doc_ids.index(did)]
        p = tr.passes.get(did)
        badge = "—" if p is None else ("✅ passes samples" if p == 1.0 else f"❌ {p:.0%} of samples")
        rows.append(f"<details {'open' if rank <= 3 else ''}><summary><b>#{rank}</b> {did} · score {score:.3f} · {badge}"
                    f"</summary><pre>{html.escape(code[:3000])}</pre></details>")
    info = (f"**Latency:** {total:.0f} ms total · dense {tr.stage_ms['dense']:.0f} ms · verify {tr.stage_ms['verify']:.0f} ms"
            f" · sample tests parsed: {tr.n_tests} · execution {'ran' if tr.executed else 'skipped'}")
    return info, "\n".join(rows)


def build_repo(repo: str, n_versions: int = 3) -> None:
    revs = subprocess.run(["git", "-C", repo, "log", "--format=%H", "-n", "60"], capture_output=True,
                          text=True, check=True).stdout.split()
    # oldest, middle, newest of the fetched history (same selection as scripts/repo_demo_check.py → cache hits)
    commits = [revs[-1], revs[len(revs) // 2], revs[0]][:n_versions] if n_versions == 3 else list(reversed(revs[:n_versions]))
    vi = VersionedIndex(repo)
    for c in commits:
        vi.build(c)
    vi.link_lineages()
    STATE["repo"] = RepoSearcher(vi, commits)
    STATE["repo_vi"] = vi
    STATE["repo_commits"] = commits
    STATE["repo"].enc.encode(["warm-up"], "query", show=False)


def repo_search(query: str, version: str, k: int):
    rs: RepoSearcher = STATE["repo"]
    commit = None if version in ("auto", "") else next(c for c in STATE["repo_commits"] if c.startswith(version))
    r = rs.search(query, k=int(k), commit=commit)
    out = [f"**Route:** {r.route} · **version intent:** {r.version_intent} · {r.ms:.0f} ms"]
    for i, h in enumerate(r.hits, 1):
        c = h.chunk
        out.append(f"<details {'open' if i <= 3 else ''}><summary><b>#{i}</b> <code>{c.path}:{c.start_line}-{c.end_line}</code>"
                   f" · {html.escape(c.symbol)} · {html.escape(h.why)} · in {len(h.versions)} version(s): "
                   f"{', '.join(h.versions[:4])}</summary><pre>{html.escape(c.text[:3000])}</pre></details>")
    return "\n".join(out)


def rebuild(rev: str):
    """Index any commit incrementally and make it searchable (only new chunks are embedded)."""
    vi: VersionedIndex = STATE["repo_vi"]
    c = vi.resolve(rev)
    vi.build(c)
    t = time.time()
    commits = [x for x in STATE["repo_commits"] if x != c] + [c]
    vi.link_lineages()
    old = len(STATE["repo"].chunks)
    STATE["repo"] = RepoSearcher(vi, commits, encoder=STATE["repo"].enc)
    STATE["repo_commits"] = commits
    return (f"Indexed `{c[:8]}`: {vi.stats[c]} · searchable now as the latest version "
            f"({len(STATE['repo'].chunks) - old} new unique chunks embedded, {time.time() - t:.1f} s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="demo_repos/leon")
    ap.add_argument("--no-apps", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if not args.no_apps:
        build_apps()
    build_repo(args.repo)

    with gr.Blocks(title="PRISM Agentic Code Retrieval") as demo:
        gr.Markdown("# Agentic, execution-verified code retrieval (CPU-only)")
        if not args.no_apps:
            with gr.Tab("Problem → code (APPS)"):
                q = gr.Textbox(lines=12, label="Problem statement (with sample Input/Output if available)")
                with gr.Row():
                    ex = gr.Checkbox(True, label="Execution verification")
                    k = gr.Slider(1, 20, 10, step=1, label="top-k")
                b = gr.Button("Search", variant="primary")
                info, res = gr.Markdown(), gr.HTML()
                b.click(apps_search, [q, ex, k], [info, res])
        with gr.Tab("Codebase search (versioned)"):
            q2 = gr.Textbox(label="Query", placeholder="which files call loadConfig before start? / where is setTimeout used?")
            with gr.Row():
                ver = gr.Dropdown(["auto"] + [c[:8] for c in STATE["repo_commits"]], value="auto", label="Version")
                k2 = gr.Slider(1, 20, 10, step=1, label="top-k")
            b2 = gr.Button("Search", variant="primary")
            res2 = gr.HTML()
            b2.click(repo_search, [q2, ver, k2], [res2])
            gr.Markdown("### Rebuild index for any commit (incremental)")
            rev = gr.Textbox(label="git revision", value="HEAD")
            b3 = gr.Button("Rebuild")
            st = gr.Markdown()
            b3.click(rebuild, [rev], [st])
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
