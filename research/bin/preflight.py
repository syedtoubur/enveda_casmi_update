"""Static checks every idea passes before an LLM spends tokens on it (replaces the Gemini critic's checklist).

    python research/bin/preflight.py research/exploit/<idea> [--tags spike,spec]   # 0 = clean, 1 = findings
    python research/bin/preflight.py --all          # every non-instrument idea folder
    python research/bin/preflight.py --selftest     # each rule flags exactly its FLAGS (regression proof)

Rules live in research/rules/R*.py: ID, TAGS, LESSON, SRC, FLAGS, check(idea) -> [finding]. A new mistake becomes a
new rule plus a fixture (research/rules/fixtures/<ID>_*/ = the files that broke). FLAGS lists every folder the rule
flags, relative to research/ (real ideas and fixtures); --selftest fails when that set changes, so a rule can never
silently start or stop flagging past work. New rules are listed at the next campaign grill, where the human can veto.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
RULES = RESEARCH / "rules"


def load_rules(tags: set[str] | None = None) -> list:
    sys.path.insert(0, str(RULES))
    out = []
    for p in sorted(RULES.glob("R*.py")):
        spec = importlib.util.spec_from_file_location(p.stem, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if not tags or tags & set(m.TAGS):
            out.append(m)
    return out


def is_instrument(idea: Path) -> bool:
    st = idea / "STATUS"
    return st.exists() and "kind: instrument" in st.read_text(encoding="utf-8")


def ideas() -> list[Path]:
    return sorted(p.parent for p in RESEARCH.glob("*/*/STATUS")
                  if p.parent.parent.name != "_template" and not is_instrument(p.parent))


def fixtures() -> list[Path]:
    return sorted(p for p in (RULES / "fixtures").glob("*") if p.is_dir())


def run(idea: Path, rules) -> list[str]:
    return [f"[{r.ID}] {f}" for r in rules for f in r.check(idea)]


def selftest() -> int:
    bad = 0
    targets = ideas() + fixtures()
    for r in load_rules():
        got = {t.relative_to(RESEARCH).as_posix() for t in targets if r.check(t)}
        want = set(r.FLAGS)
        if got != want:
            bad += 1
            print(f"FAIL {r.ID}: flags {sorted(got)} but FLAGS = {sorted(want)}")
        else:
            print(f"ok   {r.ID}: {r.LESSON[:70]}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="?")
    ap.add_argument("--tags", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    rules = load_rules(set(filter(None, a.tags.split(","))) or None)
    targets = ideas() if a.all else [Path(a.idea).resolve()]
    n = 0
    for t in targets:
        found = run(t, rules)
        n += len(found)
        print(f"{t.relative_to(RESEARCH).as_posix() if t.is_relative_to(RESEARCH) else t}: "
              f"{'clean' if not found else str(len(found)) + ' finding(s)'}")
        for f in found:
            print("  ", f)
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
