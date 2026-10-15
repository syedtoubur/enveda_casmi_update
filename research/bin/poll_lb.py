"""Match Kaggle submissions to idea folders and record scores.

    python research/bin/poll_lb.py [--write]
    python research/bin/poll_lb.py --claim research/exploit/<idea> [--ref <submission ref>]   # right after submitting

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
import re
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


def match(subs: list[dict], ideas: dict[str, str], refs: dict[str, set[str]] | None = None) -> list[tuple[str, dict]]:
    """Each submission belongs to at most one idea (loop audit W13):
    1. a submission `ref` claimed by an idea (STATUS `lb_refs:`, written by `--claim`) is exact;
    2. otherwise the kernel slug as a whole token of the description (`casmi-foo` never matches `casmi-foo-v2`);
       a description naming two ideas is ambiguous and matches none."""
    refs = refs or {}
    by_ref = {r: idea for idea, rs in refs.items() for r in rs}
    out = []
    for s in subs:
        ref = str(s.get("ref", ""))
        if ref and ref in by_ref:
            out.append((by_ref[ref], s))
            continue
        text = str(s.get("description", "")).lower()
        hits = [idea for idea, slug in ideas.items() if slug and re.search(
            rf"(?<![\w-]){re.escape(slug.split('/')[-1].lower())}(?![\w-])", text)]
        if len(hits) == 1:
            out.append((hits[0], s))
        elif len(hits) > 1:
            print(f"AMBIGUOUS\t{s.get('date', '')}\t{s.get('description', '')}\tmatches {', '.join(hits)}")
    return out


def claim(idea_rel: str, subs: list[dict], taken: set[str], ref: str | None = None) -> str | None:
    """Bind one submission ref to an idea: the given ref, or the newest submission no idea has claimed or matched."""
    if ref:
        return ref
    for s in sorted(subs, key=lambda x: str(x.get("date", "")), reverse=True):
        if str(s.get("ref", "")) and str(s["ref"]) not in taken:
            return str(s["ref"])
    return None


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
    ap.add_argument("--claim", metavar="IDEA", help="bind the newest unclaimed submission (or --ref) to this idea")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    ideas, refs = {}, {}
    for st in RESEARCH.glob("*/*/STATUS"):
        _, meta = read_status(st.parent)
        rel = st.parent.relative_to(RESEARCH).as_posix()
        if meta.get("kernel"):
            ideas[rel] = meta["kernel"]
        if meta.get("lb_refs"):
            refs[rel] = set(meta["lb_refs"].split(","))
    subs = fetcher(cfg)
    if a.claim:
        d = (RESEARCH.parent / a.claim).resolve()
        rel = d.relative_to(RESEARCH).as_posix()
        taken = {r for rs in refs.values() for r in rs} | {str(s.get("ref", "")) for _, s in match(subs, ideas, refs)}
        ref = claim(rel, subs, taken, a.ref)
        if not ref:
            print("[poll_lb] no unclaimed submission to bind")
            return 1
        _, meta = read_status(d)
        have = [r for r in meta.get("lb_refs", "").split(",") if r]
        write_status(d, "submitted" if read_status(d)[0] == "ready-to-submit" else None,
                     lb_refs=",".join(have + [ref] if ref not in have else have))
        print(f"[poll_lb] {rel} <- submission {ref}")
        return 0
    matched = match(subs, ideas, refs)
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
        if not meta.get("lb_pipeline"):        # W2: an LB number is evidence only for the pipeline it was scored on
            base = cfg["lines"].get(meta.get("line", d.parent.name), {}).get("base_notebook", "?")
            upd["lb_pipeline"] = f"{str(base).replace(chr(92), '/').rsplit('/', 1)[-1]}+{d.name}"
        if d.resolve() in controls or meta.get("kind") == "control":
            new, why = "scored", "control (never decided)"
        else:
            line = meta.get("line", d.parent.name)
            new, why = decide(got, control_lb.get(line), counters_fired(d))
        print(f"DECIDE	{idea}	{state} -> {new}	{why}")
        if a.write:
            write_status(d, new, decision=why, **upd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
