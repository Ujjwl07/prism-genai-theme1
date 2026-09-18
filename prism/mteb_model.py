"""MTEB SearchProtocol adapter so the full pipeline (not just an encoder) produces the official result JSON."""
from __future__ import annotations

from typing import Any

from mteb.models.model_meta import ModelMeta

from .pipeline import Pipeline, PipelineConfig
from .sandbox import Sandbox


class PrismSearch:
    """Implements mteb.models.models_protocols.SearchProtocol (index + search)."""

    def __init__(self, cfg: PipelineConfig | None = None, use_execution: bool = True):
        self.cfg = cfg or PipelineConfig()
        self.pipe = Pipeline(self.cfg, sandbox=Sandbox() if use_execution else None)
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": f"prism/agentic-exec-retriever-{self.cfg.encoder}",
            "revision": "PRISM_GENAI_HACKATHON_Y2026",
            "release_date": "2026-09-25",
            "languages": ["eng-Latn", "python-Code"],
            "open_weights": True,
            "framework": ["PyTorch", "Sentence Transformers"],
            "similarity_fn_name": "cosine",
            "use_instructions": True,
        })

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None) -> None:
        # Only the code text is used — ids and metadata (e.g. problem URLs) are never ranking signals.
        self.pipe.index(list(corpus["id"]), list(corpus["text"]))

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs, top_ranked=None,
               num_proc=None) -> dict[str, dict[str, float]]:
        qids, qtexts = list(queries["id"]), list(queries["text"])
        traces = self.pipe.search(qtexts)
        return {qid: dict(tr.ranking[:top_k]) for qid, tr in zip(qids, traces)}


def run_mteb(cfg: PipelineConfig, out_json: str = "appsretrieval_results.json", use_execution: bool = True) -> dict[str, Any]:
    import json

    import mteb

    model = PrismSearch(cfg, use_execution=use_execution)
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 8}, overwrite_strategy="always")
    task_result = list(result.task_results)[0]
    d = task_result.to_dict()
    with open(out_json, "w") as f:
        json.dump(d, f, indent=2)
    return d
