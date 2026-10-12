"""Match Kaggle submissions to idea folders and record scores.

    python research/bin/poll_lb.py [--write]

Reads `kaggle competitions submissions -c <competition> --csv`. A submission belongs to an idea
when its description contains the idea's kernel slug (STATUS `kernel:`): the human pastes the
slug from the submit card into Kaggle's submission description. Unmatched submissions among the
15 most recent are listed so the Reporter can ask.
Prints a table; with --write it records the scores in STATUS (`lb:` first run, `lb_rep:` the
replicate) and sets the decision with decide(), never an LLM:
- the line's control (config `lines.<line>.control`) -> `scored`;
- a probe whose control has no LB yet -> `scored-unpaired` (re-decided once the control is scored);
- G4 counters not fired (values > 0 in the G4 log, kaggle_push.check_log) -> `revert`;
- LB >= control + KEEP_MARGIN (L05) -> `keep-pending`, else `revert`;
- with a replicate: mean(lb, lb_rep) >= control + KEEP_MARGIN -> `keep`, else `revert`.
STATUS `lb_skip: n` ignores the n oldest matched submissions (a broken earlier kernel version).
An open idea with no matched submission is decided from the `lb`/`lb_rep` already in its STATUS.
The Reporter only records the result (EXPERIMENTS row, lessons).
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess

from _common import RESEARCH, load_config, read_status, write_status

KEEP_MARGIN = 0.004  # LB noise floor for identical code (L05)
OPEN = ("ready-to-submit", "submitted", "scored", "scored-unpaired", "keep-pending")


def fetch(cfg) -> list[dict]:
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", cfg["competition"], "--csv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)
    return list(csv.DictReader(io.StringIO(r.stdout)))


def match(subs: list[dict], ideas: dict[str, str]) -> list[tuple[str, dict]]:
    out = []
    for s in subs:
        text = " ".join(str(v) for v in s.values()).lower()
        for idea, slug in ideas.items():
            if slug and slug.split("/")[-1].lower() in text:
                out.append((idea, s))
    return out


def decide(lbs: list[float], control_lb: float | None, fired: bool,
           margin: float = KEEP_MARGIN) -> tuple[str, str]:
    """(new state, reason) for a probe with LB scores `lbs` (oldest first)."""
    if control_lb is None:
        return "scored-unpaired", "paired control not scored yet (L05, L06)"
    if not fired:
        return "revert", "G4 counters did not fire (values not > 0)"
    bar = control_lb + margin
    if len(lbs) >= 2:
        m = sum(lbs[:2]) / 2
        return ("keep" if m >= bar else "revert"), f"mean {m:.4f} vs bar {bar:.3f} (replicated)"
    return ("keep-pending" if lbs[0] >= bar else "revert"),         f"{lbs[0]:.3f} vs bar {bar:.3f}" + ("; submit one replicate" if lbs[0] >= bar else "")


def counters_fired(idea) -> bool:
    """Re-check the G4 log with the value rule (kaggle_push.check_log); no counters = not proven."""
    import kaggle_push
    cf = idea / "kaggle_push" / "counters.txt"
    counters = [c for c in cf.read_text(encoding="utf-8").splitlines() if c] if cf.exists() else []
    logs = list((idea / "kaggle_output").glob("*.log"))
    if not counters or not logs:
        return False
    text = "".join(kaggle_push.log_text_of(p) for p in logs)
    return not kaggle_push.check_log(text, counters, [])


def _score(s: dict) -> str:
    return s.get("publicScore") or s.get("public_score") or ""


def main(argv=None, fetcher=fetch) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    ideas = {}
    for st in RESEARCH.glob("*/*/STATUS"):
        _, meta = read_status(st.parent)
        if meta.get("kernel"):
            ideas[st.parent.relative_to(RESEARCH).as_posix()] = meta["kernel"]
    subs = fetcher(cfg)
    matched = match(subs, ideas)
    seen = {id(s) for _, s in matched}
    for s in subs[:15]:
        if id(s) not in seen:
            print(f"UNMATCHED	{s.get('date', '')}	{s.get('description', '')}	LB={_score(s)}")
    scores: dict[str, list[float]] = {}
    meta_only: set[str] = set()
    for idea, s in sorted(matched, key=lambda t: str(t[1].get("date", ""))):
        print(f"{idea}	{s.get('date', '')}	{s.get('status', '')}	LB={_score(s)}")
        if _score(s):
            scores.setdefault(idea, []).append(float(_score(s)))
    controls = {(RESEARCH.parent / c).resolve(): ln for ln, lc in cfg["lines"].items()
                if (c := lc.get("control"))}
    control_lb: dict[str, float] = {}
    for d, ln in controls.items():
        _, meta = read_status(d) if (d / "STATUS").exists() else ("", {})
        rel = d.relative_to(RESEARCH).as_posix() if d.is_relative_to(RESEARCH) else ""
        got = scores.get(rel, [])[int(meta.get("lb_skip", 0) or 0):]
        if got or meta.get("lb"):
            control_lb[ln] = got[0] if got else float(meta["lb"])
    # ideas whose submissions carry no slug: decide from the lb/lb_rep already recorded in STATUS
    for st in RESEARCH.glob("*/*/STATUS"):
        rel = st.parent.relative_to(RESEARCH).as_posix()
        state, meta = read_status(st.parent)
        if rel not in scores and state in OPEN and meta.get("lb"):
            scores[rel] = [float(meta[k]) for k in ("lb", "lb_rep") if meta.get(k)]
            meta_only.add(rel)
    for idea, got in scores.items():
        d = RESEARCH / idea
        state, meta = read_status(d)
        if idea not in meta_only:
            got = got[int(meta.get("lb_skip", 0) or 0):]
        if not got or state not in OPEN:
            continue
        upd = {"lb": f"{got[0]:.3f}"} | ({"lb_rep": f"{got[1]:.3f}"} if len(got) > 1 else {})
        if d.resolve() in controls:
            new, why = "scored", "line control"
        else:
            line = meta.get("line", d.parent.name)
            new, why = decide(got, control_lb.get(line), counters_fired(d))
        print(f"DECIDE	{idea}	{state} -> {new}	{why}")
        if a.write:
            write_status(d, new, decision=why, **upd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
