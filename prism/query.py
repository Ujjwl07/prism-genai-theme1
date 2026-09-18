"""Query-end processing: section parsing, reduction/restructuring, sample-I/O extraction.

APPS problem statements are structured with `-----Header-----` lines (Codeforces / AtCoder / CodeChef
style). We parse them into sections so we can (a) build a reduced query that keeps the task, I/O format
and constraints, and (b) extract executable sample tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADER = re.compile(r"^-{3,}\s*(.+?)\s*:?\s*-{3,}\s*$", re.M)

_EX_HEAD = re.compile(r"^(examples?|sample|примеры?|пример)", re.I)
_IN_HEAD = re.compile(r"^(input|входные данные)$", re.I)
_OUT_HEAD = re.compile(r"^(output|выходные данные)$", re.I)
_NOTE_HEAD = re.compile(r"^(note|notes|explanation|примечание)", re.I)


@dataclass
class ParsedQuery:
    statement: str
    input_spec: str = ""
    output_spec: str = ""
    constraints: str = ""
    notes: str = ""
    examples_text: str = ""
    tests: list[tuple[str, str]] = field(default_factory=list)


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(header, body)], first header is '' for the statement."""
    out, pos, head = [], 0, ""
    for m in HEADER.finditer(text):
        out.append((head, text[pos:m.start()].strip("\n")))
        head, pos = m.group(1).strip(), m.end()
    out.append((head, text[pos:].strip("\n")))
    return out


def _examples_from_block(block: str) -> list[tuple[str, str]]:
    """Codeforces style: 'Input\\n...\\n\\nOutput\\n...' repeated."""
    lines = block.split("\n")
    tests, i = [], 0
    while i < len(lines):
        if _IN_HEAD.match(lines[i].strip()):
            j = i + 1
            while j < len(lines) and not _OUT_HEAD.match(lines[j].strip()):
                j += 1
            if j >= len(lines):
                break
            k = j + 1
            while k < len(lines) and not _IN_HEAD.match(lines[k].strip()):
                k += 1
            inp = "\n".join(lines[i + 1:j]).strip("\n")
            out = "\n".join(lines[j + 1:k]).strip("\n")
            tests.append((inp, out))
            i = k
        else:
            i += 1
    return tests


def parse(text: str) -> ParsedQuery:
    secs = split_sections(text)
    pq = ParsedQuery(statement=secs[0][1])
    ex_blocks = []
    pending_sample_in: str | None = None
    for head, body in secs[1:]:
        h = head.lower()
        if h.startswith("sample input"):
            pending_sample_in = body.strip("\n")
            ex_blocks.append(f"Sample Input\n{body}")
        elif h.startswith("sample output"):
            # AtCoder: output then optional free-text explanation after a blank line
            out = body.split("\n\n")[0].strip("\n")
            if pending_sample_in is not None:
                pq.tests.append((pending_sample_in, out))
                pending_sample_in = None
            ex_blocks.append(f"Sample Output\n{body}")
        elif _EX_HEAD.match(h):
            pq.tests.extend(_examples_from_block(body))
            ex_blocks.append(body)
        elif _IN_HEAD.match(h) or h.startswith("input"):
            pq.input_spec += body + "\n"
        elif _OUT_HEAD.match(h) or h.startswith("output"):
            pq.output_spec += body + "\n"
        elif h.startswith("constraint"):
            pq.constraints += body + "\n"
        elif _NOTE_HEAD.match(h):
            pq.notes += body + "\n"
        else:
            pq.statement += f"\n\n{head}\n{body}"
    pq.examples_text = "\n\n".join(ex_blocks)
    return pq


def reduced(text: str, keep_examples: bool = True, keep_notes: bool = False) -> str:
    """Restructured query: task + constraints + I/O spec first, then (optionally) examples; notes dropped.

    Rationale (V2): encoders truncate at max_len; putting the discriminative parts first protects them.
    """
    pq = parse(text)
    parts = [pq.statement.strip()]
    if pq.constraints.strip():
        parts.append("Constraints:\n" + pq.constraints.strip())
    if pq.input_spec.strip():
        parts.append("Input:\n" + pq.input_spec.strip())
    if pq.output_spec.strip():
        parts.append("Output:\n" + pq.output_spec.strip())
    if keep_examples and pq.examples_text.strip():
        parts.append("Examples:\n" + pq.examples_text.strip())
    if keep_notes and pq.notes.strip():
        parts.append("Note:\n" + pq.notes.strip())
    return "\n\n".join(parts)
