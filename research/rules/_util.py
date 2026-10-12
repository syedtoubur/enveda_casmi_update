"""Helpers shared by the preflight rules."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKTREE = Path("C:/casmi/lines/exploit")
TEMPLATE_HINTS = ("MEASURED (source) or ASSUMED", "<", "each factor MEASURED")


def text(idea: Path, name: str) -> str:
    p = idea / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def lines(idea: Path, *names: str):
    """(file, lineno, line) over the named files, skipping unfilled template lines."""
    for n in names:
        for i, ln in enumerate(text(idea, n).splitlines(), 1):
            if not any(h in ln for h in TEMPLATE_HINTS[:1]) and not re.fullmatch(r"\s*-?\s*\*\*[^*]+\*\*:\s*<.*>\s*", ln):
                yield n, i, ln
