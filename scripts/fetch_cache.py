"""Download precomputed caches (embeddings, execution results) from the GitHub Release, if not present.

The cache is content-hash keyed, so it is only an accelerator: anything missing is recomputed locally on CPU.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/Ujjwl07/prism-genai-theme1/releases/download/PRISM_GENAI_HACKATHON_Y2026/cache.zip"


def main() -> None:
    marker = ROOT / "cache" / ".fetched"
    if marker.exists():
        print("[fetch_cache] cache already present")
        return
    try:
        print(f"[fetch_cache] downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(ROOT)
        marker.parent.mkdir(exist_ok=True)
        marker.write_text("ok")
        print("[fetch_cache] done")
    except Exception as e:  # offline / private repo: fall back to local computation
        print(f"[fetch_cache] skipped ({e}); embeddings will be computed locally on CPU", file=sys.stderr)


if __name__ == "__main__":
    main()
