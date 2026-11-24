#!/usr/bin/env python3
"""Every result cell is followed by a story cell; story cells compute their numbers; no stock phrases."""
import re
import sys
from pathlib import Path

import nbformat

FORBID = re.compile(
    r"\b(delve|leverage[sd]?|tapestry|in conclusion|cutting[- ]edge|seamless(ly)?|unlock(s|ed)?|"
    r"it is important to note|it'?s worth noting|game[- ]chang\w*|holistic|robust solution)\b", re.I)
RESULT_HINT = re.compile(r"nb\.save_fig\(|plt\.show\(|\bdisplay\(")
LITERAL = re.compile(r"nb\.story\(\s*f?(\"\"\"|''')(.*?)\1", re.S)
HOLE = re.compile(r"\{[^{}]*\}")
problems: list[str] = []
for p in sorted(Path("notebooks").glob("*.ipynb")):
    cells = nbformat.read(p, as_version=4).cells
    for i, c in enumerate(cells):
        tags = set(c.get("metadata", {}).get("tags", []))
        if c.cell_type == "code" and "story" not in tags and "result" not in tags and RESULT_HINT.search(c.source):
            problems.append(f"{p.name} cell {i}: shows a result but is not tagged 'result'")
        if "result" in tags:
            nxt = cells[i + 1] if i + 1 < len(cells) else None
            if nxt is None or "story" not in set(nxt.get("metadata", {}).get("tags", [])):
                problems.append(f"{p.name} cell {i}: result cell not followed by a story cell")
        if "story" in tags:
            m = LITERAL.search(c.source)
            if c.cell_type != "code" or not m:
                problems.append(f"{p.name} cell {i}: story cell must call nb.story(f\"\"\"...\"\"\")")
                continue
            text = m.group(2)
            holes = HOLE.findall(text)
            static = HOLE.sub("", text)
            if len(holes) < 2:
                problems.append(f"{p.name} cell {i}: story has fewer than 2 computed values")
            if re.search(r"\d", static):
                m = re.search(r".{0,25}\d.{0,25}", static)
                problems.append(f"{p.name} cell {i}: typed digit in story text (compute it): {m.group(0)!r}" if m else f"{p.name} cell {i}: typed digit in story text (compute it)")
            if FORBID.search(text):
                problems.append(f"{p.name} cell {i}: stock phrase: {FORBID.search(text).group(0)!r}")
            words = len(static.split())
            if words > 170:
                problems.append(f"{p.name} cell {i}: story too long ({words} words of prose)")
print("\n".join(problems) or "story check: OK")
sys.exit(1 if problems else 0)