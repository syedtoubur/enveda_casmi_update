"""Inventory of dead ideas and necropsy.md skeletons (deterministic, no model calls).

    python research/bin/necropsy.py --list                      # table of dead ideas
    python research/bin/necropsy.py --json                      # same inventory as JSON
    python research/bin/necropsy.py --skeleton research/<line>/<idea>   # write necropsy.md if absent
    python research/bin/necropsy.py --all-skeletons             # skeleton for every dead idea lacking one

A necropsy is a side file: nothing here touches STATUS. The M-tier `/necropsy` skill fills a skeleton in; the H-tier
`/necropsy-synth` skill reads all of them and proposes revivals, which rule R12 then checks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]

DEAD_STATE = re.compile(r"^(status:\s*)?(dead|revert|killed-[\w-]+)\b", re.I)
REASON_KEYS = ("reason", "dead_reason", "dead", "why", "result", "decision")
CAUSES = ("bug", "leak-only", "non-discriminative", "pool-reach", "noise-level", "c1c2-harm", "infra", "unknown")
CAUSE_RULES = [
    ("leak-only", re.compile(r"leak", re.I)),
    ("non-discriminative", re.compile(r"fire|FPR|discrimin|non-discriminating", re.I)),
    ("pool-reach", re.compile(r"reach", re.I)),
    ("noise-level", re.compile(r"noise|within noise|\bCI\b")),
    ("c1c2-harm", re.compile(r"C[12]\b.*(harm|drop|negative)", re.I)),
    ("infra", re.compile(r"Traceback|FAIL.*G4|not valid")),
]

SKELETON = """# Necropsy: {name}
cause: {cause}  # guess, confirm
evidence:            # file:line citations only (gates.md, harness_result.json, result.md, stress_result.json)
-
what_was_learned:    # what this rules in/out, in one or two sentences
revival_paths:       # each: change -> why it addresses the cause -> cheapest measurement
-
combine_with:        # other dead/near-miss ideas acting on a DIFFERENT cohort/stratum, with the numbers
-
"""


def guess_cause(reason: str) -> str:
    for cause, rx in CAUSE_RULES:
        if rx.search(reason or ""):
            return cause
    return "unknown"


def _status(idea: Path) -> tuple[str, dict]:
    lines = (idea / "STATUS").read_text(encoding="utf-8", errors="replace").splitlines()
    meta = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip()] = v.strip()
    return (lines[0].strip() if lines else ""), meta


def _gates_fail(idea: Path) -> str:
    p = idea / "gates.md"
    if not p.exists():
        return ""
    last = ""
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "FAIL" in ln:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            last = cells[3] if len(cells) > 3 and cells[3] else ln.strip()
    return last


def _history(research: Path) -> dict[str, str]:
    """idea folder name -> last campaign history reason."""
    out: dict[str, str] = {}
    for p in sorted(research.glob("campaigns/*/state.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for h in d.get("history", []) or []:
            if isinstance(h, dict) and h.get("idea") and h.get("reason"):
                out[Path(str(h["idea"])).name] = str(h["reason"])
    return out


def inventory(research: Path = RESEARCH) -> list[dict]:
    hist = _history(research)
    rows = []
    for st in sorted(research.glob("*/*/STATUS")):
        idea = st.parent
        if idea.parent.name == "_template":
            continue
        state, meta = _status(idea)
        if not DEAD_STATE.match(state):
            continue
        reason = next((meta[k] for k in REASON_KEYS if meta.get(k)), "") or _gates_fail(idea) or hist.get(idea.name, "")
        rows.append({"path": idea.relative_to(research.parent).as_posix(), "line": idea.parent.name, "status": state,
                     "reason": reason, "cause_guess": guess_cause(reason),
                     "has_necropsy": (idea / "necropsy.md").exists()})
    return rows


def is_filled(idea: Path) -> bool:
    """A necropsy counts as done once its cause is confirmed (the skeleton's `# guess` mark is gone)."""
    out = idea / "necropsy.md"
    return out.exists() and "# guess" not in out.read_text(encoding="utf-8", errors="replace")


def skeleton(idea: Path, research: Path = RESEARCH) -> bool:
    """Write necropsy.md if absent. True when written."""
    out = idea / "necropsy.md"
    if out.exists():
        return False
    state, meta = _status(idea) if (idea / "STATUS").exists() else ("", {})
    reason = next((meta[k] for k in REASON_KEYS if meta.get(k)), "") or _gates_fail(idea) \
        or _history(research).get(idea.name, "")
    out.write_text(SKELETON.format(name=idea.name, cause=guess_cause(reason)), encoding="utf-8")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skeleton")
    ap.add_argument("--all-skeletons", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.skeleton:
        p = Path(a.skeleton)
        if not p.is_absolute():
            p = (RESEARCH.parent / p) if (RESEARCH.parent / p).exists() else Path.cwd() / p
        print(("wrote " if skeleton(p.resolve()) else "exists ") + str(p / "necropsy.md"))
    elif a.all_skeletons:
        n = 0
        for r in inventory():
            if not r["has_necropsy"]:
                n += skeleton(RESEARCH.parent / r["path"])
        print(f"{n} skeleton(s) written")
    elif a.json:
        print(json.dumps(inventory(), indent=2))
    else:
        rows = inventory()
        for r in rows:
            print(f"{r['path']} | {r['line']} | {r['status']} | has_necropsy={r['has_necropsy']} | "
                  f"reason: {r['reason'][:100] or '-'}")
        print(f"{len(rows)} dead idea(s), {sum(r['has_necropsy'] for r in rows)} with necropsy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
