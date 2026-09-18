"""V7b: CPU speed + fidelity of fp32 vs int8 (down_proj kept fp32) CodeXEmbed-400M query encoding."""
import json, platform, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.data import ensure_data, load_queries
from prism.encoders import Encoder

ensure_data()
qs = [q["text"] for q in load_queries("test")[:32]]
out, vec = {"machine": platform.machine(), "system": platform.system(), "threads": torch.get_num_threads()}, {}
for name in ("codexembed-400m", "codexembed-400m-int8"):
    for L in (512, 1024):
        e = Encoder(name); m = e.model; m.max_seq_length = L
        m.encode(qs[:2])
        t = time.time(); V = m.encode([e.spec.query_prefix + x for x in qs], batch_size=4, normalize_embeddings=True)
        out[f"{name}@{L}_q_per_s"] = round(len(qs) / (time.time() - t), 3); vec[(name, L)] = V
        print(out, flush=True)
for L in (512, 1024):
    c = (vec[("codexembed-400m", L)] * vec[("codexembed-400m-int8", L)]).sum(1)
    out[f"cos@{L}"] = [round(float(c.mean()), 4), round(float(c.min()), 4)]
Path("results").mkdir(exist_ok=True)
Path(f"results/bench_quant_{platform.system()}.json").write_text(json.dumps(out, indent=1))
print(out)
