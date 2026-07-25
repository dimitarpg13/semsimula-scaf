#!/usr/bin/env python3
"""Lint Markdown for GitHub KaTeX + Mermaid rendering hazards.

Implements the rules from
``semsimula-paper/companion_notes/GitHub_Markdown_LaTeX_Rendering_Cheatsheet.md``.
Every rule there was confirmed by an observed rendering failure on github.com,
so this is a port of empirical findings, not a style checker.

Usage:
    python tools/lint_markdown.py docs/*.md
    python tools/lint_markdown.py --list-rules
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FATAL = "FATAL"
WARN = "WARN"


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    message: str
    excerpt: str = ""

    def render(self) -> str:
        head = f"{self.path}:{self.line}: [{self.severity}] {self.rule} — {self.message}"
        return f"{head}\n    {self.excerpt}" if self.excerpt else head


# ---------------------------------------------------------------------------
# Block segmentation
# ---------------------------------------------------------------------------

def _segment(text: str):
    """Split into (kind, start_line, lines) where kind is prose/mermaid/code/math."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            start, body = i, []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            kind = "mermaid" if lang == "mermaid" else "code"
            out.append((kind, start + 2, body))
        elif stripped == "$$":
            start, body = i, []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                body.append(lines[i])
                i += 1
            i += 1
            out.append(("math", start + 2, body))
        else:
            start, body = i, []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("```") or s == "$$":
                    break
                body.append(lines[i])
                i += 1
            out.append(("prose", start + 1, body))
    return out


def _strip_code(line: str) -> str:
    """Blank out backtick code spans, preserving column positions.

    Essential for avoiding false positives on documents that *quote* bad
    examples inside code spans — the cheatsheet itself being the worst case.
    """
    return re.sub(r"`[^`]*`", lambda m: " " * len(m.group()), line)


def _math_spans(line: str) -> tuple[list[str], list[str]]:
    """Return ``(inline, display)`` math spans on a prose line.

    The distinction matters: GitHub's Markdown emphasis pass is aggressive
    inside inline ``$...$`` but comparatively relaxed inside ``$$...$$``, so
    rules §5 and §12 apply only to the inline list. Single-line ``$$...$$`` is
    counted as display here because the block segmenter only recognises ``$$``
    on a line of its own.
    """
    without_code = _strip_code(line)
    display = re.findall(r"\$\$(.+?)\$\$", without_code)
    remainder = re.sub(r"\$\$.+?\$\$", " ", without_code)
    inline = re.findall(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)", remainder)
    return inline, display


# ---------------------------------------------------------------------------
# Part I — KaTeX rules
# ---------------------------------------------------------------------------

def check_katex(path: str, seg) -> list[Finding]:
    f: list[Finding] = []
    for kind, start, body in seg:
        if kind == "code":
            continue
        for off, line in enumerate(body):
            ln = start + off
            if kind == "prose":
                inline, display = _math_spans(line)
            elif kind == "math":
                inline, display = [], [line]
            else:
                inline, display = [], []
            maths = inline + display
            joined = " ".join(maths)

            # 1 — spacing commands render as punctuation
            for cmd, shows in ((r"\;", ";"), (r"\,", ",")):
                if cmd in joined:
                    f.append(Finding(path, ln, "§1", FATAL,
                                     f"{cmd!r} in math renders as {shows!r}; remove it",
                                     line.strip()))
            # 11 — negative thin space
            if re.search(r"\\!", joined):
                f.append(Finding(path, ln, "§11", FATAL,
                                 r"\! in math can break \left(...\right); remove it",
                                 line.strip()))
            # 2 — \operatorname is blocked
            if r"\operatorname" in joined:
                f.append(Finding(path, ln, "§2", FATAL,
                                 r"\operatorname is blocked; use \mathrm{..}(..) or \text{..}",
                                 line.strip()))
            # 13 — literal < > inside math
            for m in maths:
                if re.search(r"(?<!\\)<", m) or re.search(r"(?<![\\=\-])>", m):
                    f.append(Finding(path, ln, "§13", FATAL,
                                     r"literal < or > in math; use \lt and \gt",
                                     line.strip()))
                    break
            # 5a — lone * inside math
            for m in maths:
                if re.search(r"(?<!\\)\*", m):
                    f.append(Finding(path, ln, "§5a", FATAL,
                                     r"bare * in math pairs as italic; use \ast",
                                     line.strip()))
                    break
            # 6 — \|...\| double-bar norm
            if re.search(r"\\\|", joined):
                f.append(Finding(path, ln, "§6", FATAL,
                                 r"\|...\| in math; use \lVert ... \rVert",
                                 line.strip()))
            # 10 — \tag in display math
            if kind == "math" and r"\tag" in line:
                f.append(Finding(path, ln, "§10", FATAL,
                                 r"\tag{} forces vertical layout; number in prose",
                                 line.strip()))
            # 19 — \left/\middle/\right constructs
            if r"\middle" in joined:
                f.append(Finding(path, ln, "§19", FATAL,
                                 r"\middle fails on GitHub; use \lbrace .. \mid .. \rbrace",
                                 line.strip()))
            if re.search(r"\\left\\lVert", joined):
                f.append(Finding(path, ln, "§19", FATAL,
                                 r"\left\lVert fails; use \Big\lVert or plain \lVert",
                                 line.strip()))
            if re.search(r"\\underbrace\{\s*\\left\\\{", joined):
                f.append(Finding(path, ln, "§19", FATAL,
                                 r"\underbrace{\left\{..\right\}}; use \lbrace .. \rbrace",
                                 line.strip()))
            # 9 — \ddot x without braces
            if re.search(r"\\d?dot\s+[A-Za-z]", joined):
                f.append(Finding(path, ln, "§9", WARN,
                                 r"\dot/\ddot without braces; use \ddot{x}",
                                 line.strip()))
            # 5 — many underscores in one inline span (display blocks are exempt)
            for m in inline:
                if m.count("_") > 2:
                    f.append(Finding(path, ln, "§5", WARN,
                                     f"{m.count('_')} underscores in one inline span; "
                                     "split it or move to a display block",
                                     line.strip()))
                    break
            # 12 — }_x subscript patterns, 2+ on a line (inline only)
            n = sum(len(re.findall(r"\}_[A-Za-z0-9]", m)) for m in inline)
            if n >= 2:
                f.append(Finding(path, ln, "§12", FATAL,
                                 f"{n} '}}_x' patterns on one line pair as italic; "
                                 r"escape each as }\_x",
                                 line.strip()))
            # 7 — display line starting with "- "
            if kind == "math" and line.strip().startswith("- "):
                f.append(Finding(path, ln, "§7", FATAL,
                                 "display-math line starts with '- ' and becomes a bullet",
                                 line.strip()))
            if kind != "prose":
                continue
            # 3 — inline math inside a table cell
            if line.strip().startswith("|") and line.count("|") >= 2 and inline:
                f.append(Finding(path, ln, "§3", WARN,
                                 "inline math in a table cell is often not parsed; "
                                 "use Unicode (~, ≈, ≤)",
                                 line.strip()))
            # 4 — math inside an italic span (code spans do not count)
            bare = _strip_code(line)
            for pat, kindname in ((r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", "*...*"),
                                  (r"(?<![_\w])_([^_\n]+?)_(?![_\w])", "_..._")):
                for span in re.findall(pat, bare):
                    if "$" in span:
                        f.append(Finding(path, ln, "§4", FATAL,
                                         f"math inside an italic {kindname} span "
                                         "renders as raw text",
                                         line.strip()))
    return f


# ---------------------------------------------------------------------------
# Part II — Mermaid rules
# ---------------------------------------------------------------------------

_NODE_DEF = re.compile(r'(\w+)\s*(\[\[|\[|\(\(|\(|\{\{|\{|>)\s*("?)(.*?)\3\s*(\]\]|\]|\)\)|\)|\}\}|\})')
_PIPE_LABEL = re.compile(r"\|([^|]*)\|")
_ARROW = re.compile(r"(-->|-\.->|---|-\.-|==>)")


def check_mermaid(path: str, seg) -> list[Finding]:
    f: list[Finding] = []
    for kind, start, body in seg:
        if kind != "mermaid":
            continue
        declared_inline: set[str] = set()
        in_subgraph = False
        for off, raw in enumerate(body):
            ln = start + off
            line = raw.strip()
            if not line or line.startswith("%%"):
                continue
            if line.startswith("subgraph"):
                in_subgraph = True
                # 16 — subgraph ID ["Title"]
                if re.search(r'subgraph\s+\w+\s*\[\s*"', line):
                    f.append(Finding(path, ln, "§16", FATAL,
                                     'subgraph ID ["Title"] is invalid; drop the quotes',
                                     line))
                continue
            if line == "end":
                in_subgraph = False
                continue

            # 24 — brackets inside pipe-form edge labels
            for lbl in _PIPE_LABEL.findall(line):
                if re.search(r"[()\[\]{}]", lbl):
                    f.append(Finding(path, ln, "§24", FATAL,
                                     f"bracket in pipe-form edge label {lbl!r}; "
                                     "edge labels are unquoted, use plain words",
                                     line))
            # 17 — dotted edge with inline label
            if re.search(r"-\.\s*[^|>]*\s*\.-", line):
                f.append(Finding(path, ln, "§17", WARN,
                                 "inline dotted-edge label; prefer pipe form -.->|text|",
                                 line))
            # 20 — advanced node shapes
            if re.search(r"\(\(\s*\"", line):
                f.append(Finding(path, ln, "§20", FATAL,
                                 'double-circle (("text")) crashes GitHub; use ("text")',
                                 line))
            if re.search(r"\[\s*/|/\s*\]|\[\s*\\\\|\\\\\s*\]", line):
                f.append(Finding(path, ln, "§20", FATAL,
                                 "parallelogram shape crashes GitHub; use [\"text\"]",
                                 line))
            if re.search(r"\{\{", line):
                f.append(Finding(path, ln, "§20", WARN,
                                 "hexagon {{..}} is unreliable on GitHub", line))
            # 23a — chained arrows
            if len(_ARROW.findall(line)) >= 3:
                f.append(Finding(path, ln, "§23", FATAL,
                                 f"{len(_ARROW.findall(line))} arrows on one line; "
                                 "use one edge per line",
                                 line))
            # 23b — inline node definition as target of a dotted edge
            if "-.->" in line or re.search(r"-\..*\.->", line):
                tail = re.split(r"-\.->\|[^|]*\||-\.->|\.->", line)[-1]
                if _NODE_DEF.search(tail):
                    f.append(Finding(path, ln, "§23", FATAL,
                                     "inline node definition as dotted-edge target; "
                                     "pre-declare the node on its own line",
                                     line))
            # 23c — node defined inside a subgraph block
            if in_subgraph and _NODE_DEF.search(line):
                f.append(Finding(path, ln, "§23", WARN,
                                 "node defined inside subgraph; declare it outside and "
                                 "reference the bare id",
                                 line))
            # 21 — unquoted node label containing --
            for _id, _o, q, lbl, _c in _NODE_DEF.findall(line):
                if not q and re.search(r"--|-\.", lbl):
                    f.append(Finding(path, ln, "§21", FATAL,
                                     f"unquoted node label {lbl!r} contains '--'; quote it",
                                     line))

            # Label-content rules (§14, §15, §18, §22)
            for _id, _o, _q, lbl, _c in _NODE_DEF.findall(line):
                if not lbl:
                    continue
                declared_inline.add(_id)
                if "{" in lbl or "}" in lbl:
                    f.append(Finding(path, ln, "§14", FATAL,
                                     f"brace in node label {lbl!r}; rephrase", line))
                if "[" in lbl or "]" in lbl:
                    f.append(Finding(path, ln, "§15", FATAL,
                                     f"nested bracket in node label {lbl!r}; use parens",
                                     line))
                if "$" in lbl:
                    f.append(Finding(path, ln, "§22", FATAL,
                                     f"math delimiter in node label {lbl!r}; use ASCII",
                                     line))
                if "<br/>" in lbl:
                    f.append(Finding(path, ln, "§18", WARN,
                                     "<br/> in label; use <br>", line))
                if "?" in lbl:
                    f.append(Finding(path, ln, "§18", WARN,
                                     f"'?' in label {lbl!r}; drop or reword", line))
                if ".." in lbl:
                    f.append(Finding(path, ln, "§18", FATAL,
                                     f"'..' in label {lbl!r} is a range token; spell out",
                                     line))
                if re.search(r"-[A-Za-z]", lbl):
                    f.append(Finding(path, ln, "§18", FATAL,
                                     f"hyphen before a letter in label {lbl!r}; "
                                     "use a space",
                                     line))
                if "=" in lbl:
                    f.append(Finding(path, ln, "§18", WARN,
                                     f"'=' in label {lbl!r}; use a word", line))
                if "_" in lbl:
                    f.append(Finding(path, ln, "§18", WARN,
                                     f"underscore in label {lbl!r} can act as italic; "
                                     "use a space",
                                     line))
                if re.search(r"[^\x00-\x7F]", lbl):
                    bad = "".join(sorted({c for c in lbl if ord(c) > 127}))
                    f.append(Finding(path, ln, "§18", WARN,
                                     f"non-ASCII {bad!r} in label; spell out in ASCII",
                                     line))
    return f


RULES = {
    "§1": "spacing commands render as punctuation",
    "§2": "\\operatorname is blocked",
    "§3": "inline math in table cells",
    "§4": "inline math inside italic spans",
    "§5": "many underscores in one inline span",
    "§5a": "lone * inside math",
    "§6": "\\|...\\| double-bar norm",
    "§7": "display math line starting with '- '",
    "§9": "\\ddot x without braces",
    "§10": "\\tag{} in display math",
    "§11": "\\! negative thin space",
    "§12": "}_x subscripts, 2+ per line",
    "§13": "literal < > inside math",
    "§14": "braces in Mermaid node labels",
    "§15": "nested brackets in Mermaid node labels",
    "§16": 'subgraph ID ["Title"]',
    "§17": "inline dotted-edge labels",
    "§18": "Mermaid label cautionary cleanups",
    "§19": "\\left/\\middle/\\right constructs",
    "§20": "advanced Mermaid node shapes",
    "§21": "'--' in unquoted node labels",
    "§22": "math delimiters in node labels",
    "§23": "chained arrows / inline dotted targets / subgraph defs",
    "§24": "brackets in pipe-form edge labels",
}


def lint(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    seg = _segment(text)
    return check_katex(str(path), seg) + check_mermaid(str(path), seg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--list-rules", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    args = ap.parse_args(argv)

    if args.list_rules:
        for k, v in RULES.items():
            print(f"  {k:5s} {v}")
        return 0
    if not args.paths:
        ap.error("no paths given")

    findings: list[Finding] = []
    for p in args.paths:
        findings.extend(lint(p))

    fatal = [x for x in findings if x.severity == FATAL]
    warn = [x for x in findings if x.severity == WARN]
    for x in sorted(findings, key=lambda x: (x.path, x.line)):
        print(x.render())

    n = len(args.paths)
    print(f"\n{n} file(s): {len(fatal)} fatal, {len(warn)} warning(s)")
    if fatal:
        return 1
    return 1 if (args.strict and warn) else 0


if __name__ == "__main__":
    sys.exit(main())
