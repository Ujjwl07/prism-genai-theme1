"""Repository-side indexing for JS/TS codebases: cAST chunking, call graph, versioned & incremental indexing,
cross-version lineages.

- Chunking follows cAST (EMNLP'25 Findings): split recursively along AST nodes until a chunk fits a size budget
  (non-whitespace chars), then greedily merge adjacent small siblings. Chunks never cut a syntactic unit.
- Versions are read straight from git objects (`git show <commit>:<path>`) — no checkout. Parsing is cached per
  git blob sha and embeddings per chunk content hash, so moving to a new commit only re-processes changed files
  (P1: rebuild for any version in reasonable time).
- Lineages (Bonus): chunks of the same (file, symbol) across commits form a lineage; unmatched chunks are linked
  by token-type-sequence similarity in the spirit of Toma (ICSE'24) clone detection, which is robust to renames.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tree_sitter as ts
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "codebase"
EXTS = {".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "ts", ".tsx": "tsx"}
SKIP_DIRS = ("node_modules/", "dist/", "build/", ".min.", "vendor/")

_LANGS = {
    "js": ts.Language(tsjs.language()),
    "ts": ts.Language(tsts.language_typescript()),
    "tsx": ts.Language(tsts.language_tsx()),
}
_SYMBOL_NODES = {"function_declaration", "method_definition", "class_declaration", "generator_function_declaration",
                 "lexical_declaration", "variable_declaration", "interface_declaration", "type_alias_declaration",
                 "enum_declaration", "export_statement", "abstract_class_declaration"}


@dataclass
class Chunk:
    commit: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    text: str
    calls: list[str] = field(default_factory=list)   # callee names in source order
    hash: str = ""
    lineage: str = ""

    def header(self) -> str:
        return f"// {self.path} :: {self.symbol}\n"


def _nws(b: bytes) -> int:
    return len(re.sub(rb"\s", b"", b))


def _symbol_name(node: ts.Node) -> str:
    n = node.child_by_field_name("name")
    if n is not None:
        return n.text.decode("utf-8", "replace")
    if node.type in ("lexical_declaration", "variable_declaration"):
        for c in node.named_children:
            if c.type == "variable_declarator" and c.child_by_field_name("name") is not None:
                return c.child_by_field_name("name").text.decode("utf-8", "replace")
    if node.type == "export_statement":
        for c in node.named_children:
            s = _symbol_name(c)
            if s:
                return s
    return ""


def _enclosing_symbol(node: ts.Node) -> str:
    names = []
    cur = node
    while cur is not None:
        if cur.type in _SYMBOL_NODES or cur.type in ("class_declaration",):
            s = _symbol_name(cur)
            if s:
                names.append(s)
        cur = cur.parent
    return ".".join(reversed(names))


def _calls(node: ts.Node) -> list[str]:
    out = []
    stack = [node]
    order = []
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(reversed(n.children))
    for n in order:
        if n.type in ("call_expression", "new_expression"):
            f = n.child_by_field_name("function") or n.child_by_field_name("constructor")
            if f is not None:
                name = f.text.decode("utf-8", "replace")
                # keep the last member name (this.a.b.run -> run) plus full dotted path
                out.append(name.split("(")[0].strip())
    return out


def cast_chunks(src: bytes, lang: str, max_nws: int = 1200) -> list[tuple[int, int, ts.Node | None]]:
    """cAST split-then-merge. Returns (start_byte, end_byte, representative node) spans."""
    parser = ts.Parser(_LANGS[lang])
    root = parser.parse(src).root_node

    def split(node: ts.Node) -> list[tuple[int, int, ts.Node]]:
        if _nws(src[node.start_byte:node.end_byte]) <= max_nws or not node.children:
            return [(node.start_byte, node.end_byte, node)]
        pieces: list[tuple[int, int, ts.Node]] = []
        for c in node.children:
            pieces.extend(split(c))
        return pieces

    spans = []
    for c in root.children:
        spans.extend(split(c))
    # greedy merge of adjacent siblings while under budget
    merged: list[list] = []
    for s, e, n in spans:
        if merged and _nws(src[merged[-1][0]:e]) <= max_nws:
            merged[-1][1] = e
            if merged[-1][2] is None or merged[-1][2].type in ("comment", "import_statement", ";", "}", "{"):
                merged[-1][2] = n
        else:
            merged.append([s, e, n])
    return [(s, e, n) for s, e, n in merged if _nws(src[s:e]) > 0]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def list_files(repo: Path, commit: str) -> list[tuple[str, str]]:
    """[(blob_sha, path)] of source files at commit."""
    out = []
    for line in _git(repo, "ls-tree", "-r", commit).splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and Path(path).suffix in EXTS and not any(s in path for s in SKIP_DIRS):
            out.append((sha, path))
    return out


def chunk_file(src: bytes, path: str, commit: str) -> list[Chunk]:
    lang = EXTS[Path(path).suffix]
    chunks = []
    for s, e, node in cast_chunks(src, lang):
        text = src[s:e].decode("utf-8", "replace")
        start_line = src[:s].count(b"\n") + 1
        end_line = start_line + text.count("\n")
        sym = _enclosing_symbol(node) if node is not None else ""
        if not sym and node is not None:
            sym = _symbol_name(node)
        parser = ts.Parser(_LANGS[lang])
        calls = _calls(parser.parse(src[s:e]).root_node)
        chunks.append(Chunk(commit, path, start_line, end_line, sym or f"<top-level@{start_line}>", text, calls,
                            hashlib.sha1(text.encode()).hexdigest()))
    return chunks


class VersionedIndex:
    """Chunks for any number of commits, with per-blob parse cache (incremental)."""

    def __init__(self, repo: str | Path):
        self.repo = Path(repo)
        self.cache_dir = CACHE / hashlib.sha1(str(self.repo.resolve()).encode()).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.by_commit: dict[str, list[Chunk]] = {}
        self.stats: dict[str, dict] = {}

    def resolve(self, rev: str) -> str:
        return _git(self.repo, "rev-parse", rev).strip()

    def build(self, rev: str) -> list[Chunk]:
        import time

        commit = self.resolve(rev)
        t = time.time()
        files = list_files(self.repo, commit)
        reused = parsed = 0
        chunks: list[Chunk] = []
        for sha, path in files:
            cf = self.cache_dir / f"{sha}_{hashlib.sha1(path.encode()).hexdigest()[:8]}.json"
            if cf.exists():
                raw = json.loads(cf.read_text(encoding="utf-8"))
                reused += 1
            else:
                src = _git_bytes(self.repo, "cat-file", "blob", sha)
                try:
                    raw = [asdict(c) for c in chunk_file(src, path, "")]
                except Exception:  # unparsable file: index as one chunk
                    txt = src.decode("utf-8", "replace")
                    raw = [asdict(Chunk("", path, 1, txt.count("\n") + 1, "<file>", txt, [],
                                        hashlib.sha1(txt.encode()).hexdigest()))]
                cf.write_text(json.dumps(raw), encoding="utf-8")
                parsed += 1
            for r in raw:
                r["commit"] = commit
                chunks.append(Chunk(**r))
        for c in chunks:
            c.lineage = f"{c.path}::{c.symbol}"
        self.by_commit[commit] = chunks
        self.stats[commit] = {"files": len(files), "parsed": parsed, "reused_from_cache": reused,
                              "chunks": len(chunks), "seconds": round(time.time() - t, 3)}
        return chunks

    def link_lineages(self, sim: float = 0.85) -> None:
        """Merge lineages across commits: exact (path, symbol) first; otherwise token-type similarity (Toma-style)."""
        commits = list(self.by_commit)
        known: dict[str, list[str]] = {}
        for c in commits:
            for ch in self.by_commit[c]:
                known.setdefault(ch.lineage, []).append(ch.hash)
        for i in range(1, len(commits)):
            prev = {ch.lineage: ch for ch in self.by_commit[commits[i - 1]]}
            prev_types = {k: token_types(v.text) for k, v in prev.items()}
            cur_lineages = {ch.lineage for ch in self.by_commit[commits[i]]}
            for ch in self.by_commit[commits[i]]:
                if ch.lineage in prev:
                    continue
                tt = token_types(ch.text)
                best, best_s = None, 0.0
                for k, pt in prev_types.items():
                    if k in cur_lineages or abs(len(pt) - len(tt)) > 0.3 * max(len(tt), 1):
                        continue
                    s = difflib.SequenceMatcher(None, pt, tt, autojunk=False).ratio()
                    if s > best_s:
                        best, best_s = k, s
                if best is not None and best_s >= sim:
                    ch.lineage = best


_TT = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|`(?:\\.|[^`])*`|\S")
_KW = set("""break case catch class const continue debugger default delete do else export extends finally for
function if import in instanceof let new return super switch this throw try typeof var void while with yield async
await static get set of null true false undefined interface type enum implements private public protected readonly""".split())


def token_types(text: str) -> list[str]:
    """Toma-style abstraction: identifiers -> ID, literals -> LIT, keep keywords/operators."""
    out = []
    for t in _TT.findall(text):
        if t in _KW:
            out.append(t)
        elif t[0].isalpha() or t[0] in "_$":
            out.append("ID")
        elif t[0].isdigit() or t[0] in "\"'`":
            out.append("LIT")
        else:
            out.append(t)
    return out
