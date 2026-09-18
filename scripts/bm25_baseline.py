"""V1 lexical baseline: BM25 (code-aware tokenizer) on raw and reduced APPS test queries."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.data import ensure_data, load_corpus, load_queries
from prism.lexical import BM25
from prism.metrics import evaluate
from prism.query import reduced
import numpy as np

ensure_data()
c = load_corpus(); q = load_queries()
bm = BM25([d["text"] for d in c]); ids = [d["id"] for d in c]
out = {}
for name, f in [("raw", lambda t: t), ("reduced", reduced)]:
    ranked = []
    for x in q:
        s = bm.bm.get_scores(__import__("prism.lexical", fromlist=["tokenize"]).tokenize(f(x["text"])))
        ranked.append([ids[j] for j in np.argsort(-s)[:100]])
    out["bm25_" + name] = evaluate(ranked, [x["rel"] for x in q]); print(name, out["bm25_" + name], flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v1_bm25.json").write_text(json.dumps(out, indent=2))
