#!/usr/bin/env bash
# Merge embedding shards produced by the offline-embed workflow into cache/emb/ (content-hash keyed => safe merge).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
for run in "$@"; do gh run download "$run" -D "$tmp/$run"; done
mkdir -p cache/emb
find "$tmp" -name 'shard_*.npz' | while read -r f; do
  model=$(basename "$(dirname "$f")"); mkdir -p "cache/emb/$model"; cp -n "$f" "cache/emb/$model/" || true
done
rm -rf "$tmp"; du -sh cache/emb/*
