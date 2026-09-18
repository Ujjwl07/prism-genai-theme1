"""Build a 3-version index of the demo repo and run representative queries of every route."""
import subprocess, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.codebase import VersionedIndex
from prism.repo_search import RepoSearcher

repo = sys.argv[1] if len(sys.argv) > 1 else "demo_repos/leon"
revs = subprocess.run(["git", "-C", repo, "log", "--format=%H", "-n", "60"], capture_output=True, text=True).stdout.split()
commits = [revs[-1], revs[len(revs) // 2], revs[0]]
vi = VersionedIndex(repo)
for c in commits:
    vi.build(c)
t = time.time(); vi.link_lineages(); print("lineage linking s", round(time.time() - t, 1))
print(json.dumps(vi.stats, indent=1))
rs = RepoSearcher(vi, commits)
print("embed seconds", round(rs.embed_seconds, 1), "unique chunks", len(rs.chunks))
for q in ["how does the assistant pick which LLM provider to use?",
          "where is the ASR speech recognition result sent to the brain",
          "which files call loadConfig before start?",
          "where is setTimeout used",
          "how did the old version handle wake word detection",
          "history of the TTS synthesizer initialisation across versions"]:
    r = rs.search(q, k=5)
    print(f"\n### {q}  [route={r.route}, version={r.version_intent}, {r.ms:.0f} ms]")
    for h in r.hits:
        print(f"  {h.score:.4f} {h.chunk.path}:{h.chunk.start_line}-{h.chunk.end_line} {h.chunk.symbol[:40]} | {h.why} | versions={len(h.versions)}")
