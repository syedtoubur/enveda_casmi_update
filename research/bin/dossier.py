"""What the human reads to decide a submit (grill 2026-10-13): the ranked submit queue, a card per idea, a digest per cycle.

    python research/bin/dossier.py                       # refresh research/submit_queue.md and every queued card.md
    python research/bin/dossier.py --digest <campaign>   # also (re)write that campaign's digest for its last cycle

- `research/submit_queue.md`: probes in `ready-to-submit`, ranked by the scorecard composite (heavy path: ceiling),
  with kernel + version, the leak-free ("other") Δ on the target cohort, flags and a recommendation. Then the probes
  still in G4, and the ones already submitted with their LB.
- `<idea>/card.md`: claim, change, scorecard table, attacks (with deviations), gate decision, G4 result, kernel.
- `<campaign>/digest_cycle<N>.md`: what the cycle tried, what died and why, model calls, and the four loop metrics
  (LB hit rate per slot, harness-vs-LB pairs, idea novelty, unattended cycles).
Writes only these files; never changes STATUS.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import _common
from _common import read_status
WAITING = ("g4-pushed", "g4-fix", "g4-timeout")
SUBMITTED = ("submitted", "scored", "scored-unpaired", "keep-pending", "keep", "revert")


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(x, nd=4) -> str:
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "-"


def ideas() -> list[Path]:
    return sorted(p.parent for p in _common.RESEARCH.glob("*/*/STATUS") if p.parent.parent.name not in ("_template", "campaigns"))


def score_of(idea: Path) -> dict:
    """Ranking numbers of one idea from harness_result.json (scorecard-1, heavy path, or a legacy compare)."""
    r = _json(idea / "harness_result.json") or {}
    if r.get("heavy_path"):
        return dict(rank=r.get("ceiling"), kind="ceiling", target="-", delta=r.get("ceiling"), ci=None, other=None,
                    flags=["heavy path: not replayed"])
    if r.get("schema") == "scorecard-1":
        t = r["cohorts"].get(r["target"], {})
        return dict(rank=r.get("composite"), kind="composite", target=r["target"], delta=t.get("delta"),
                    ci=t.get("ci90"), other=t.get("delta_other"), flags=r.get("flags", []))
    if "delta" in r:
        return dict(rank=r.get("delta"), kind="legacy delta", target=r.get("shard", "?"), delta=r.get("delta"),
                    ci=r.get("ci90"), other=None, flags=["legacy single-shard result"])
    return dict(rank=None, kind="none", target="-", delta=None, ci=None, other=None, flags=["no harness result"])


def recommend(s: dict, meta: dict) -> str:
    if meta.get("override"):
        return "human override: submit only as a calibration test"
    if s["kind"] == "ceiling":
        return "consider: heavy path, the LB is the only measurement"
    lo = (s["ci"] or [None])[0]
    if isinstance(s["rank"], (int, float)) and s["rank"] > 0 and isinstance(lo, (int, float)) and lo > 0:
        if isinstance(s["other"], (int, float)) and s["other"] <= 0:
            return "consider: the gain sits on leak-hit molecules only (calibration.md)"
        return "submit"
    return "consider: weak or uncertain harness signal"


def kernel_ref(meta: dict) -> str:
    k = meta.get("kernel", "")
    return f"`{k}` v{meta['kernel_version']}" if k and meta.get("kernel_version") else (f"`{k}` (latest)" if k else "-")


def write_card(idea: Path) -> Path:
    st, meta = read_status(idea)
    s = score_of(idea)
    r = _json(idea / "harness_result.json") or {}
    hyp = idea / "hypothesis.md"
    claim = next((ln.strip("-* ") for ln in hyp.read_text(encoding="utf-8").splitlines()
                  if re.search(r"claim", ln, re.I)), "") if hyp.exists() else ""
    L = [f"# Card: {idea.name}", "", f"- STATUS: `{st}`; line {meta.get('line', idea.parent.name)}; "
         f"branch {meta.get('branch', '-')}", f"- kernel: {kernel_ref(meta)}",
         f"- claim: {claim or '(see hypothesis.md)'}", f"- knobs: {r.get('knobs', r.get('b', '-'))}",
         f"- gate: {meta.get('gate') or meta.get('reason') or '-'}",
         f"- override: {meta.get('override', 'none')}", f"- recommendation: **{recommend(s, meta)}**", ""]
    if r.get("schema") == "scorecard-1":
        L += ["| cohort | n | Δ | 90% CI | up/down | leak-hit Δ | other Δ | displaced |", "|---|---|---|---|---|---|---|---|"]
        for c, x in r["cohorts"].items():
            tag = " (target)" if c == r["target"] else " (advisory)" if x.get("advisory") else ""
            L.append(f"| {c}{tag} | {x.get('n')} | {_f(x.get('delta'))} | [{_f(x['ci90'][0])}, {_f(x['ci90'][1])}] | "
                     f"{x.get('up')}/{x.get('down')} | {_f(x.get('delta_leak_hit'))} | {_f(x.get('delta_other'))} | "
                     f"{x.get('displaced')} |")
        L += ["", f"Composite {_f(r.get('composite'))}."]
    elif s["kind"] == "ceiling":
        L.append(f"Heavy path, measured ceiling {_f(s['delta'])}.")
    L += ["", "## Flags", *([f"- {f}" for f in s["flags"]] or ["- none"])]
    stress = _json(idea / "stress_result.json")
    if isinstance(stress, list):
        L += ["", "## Attacks", "| id | status | blocking | deviation |", "|---|---|---|---|"]
        for a in stress:
            if isinstance(a, dict) and "id" in a:
                dev = a.get("deviation")
                L.append(f"| {a['id']} | {a.get('status') or ('died' if a.get('died') else 'survived')} | "
                         f"{'yes' if a.get('blocking') else 'no'} | {dev.get('proxy', '') if isinstance(dev, dict) else ''} |")
    gates = idea / "gates.md"
    g4 = [ln for ln in gates.read_text(encoding="utf-8").splitlines() if ln.startswith("| G4")] if gates.exists() else []
    L += ["", "## G4", *(g4[-3:] or ["- not run yet"]), "",
          "KEEP is decided by `poll_lb.decide()` against the paired control; only the human submits."]
    out = idea / "card.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def write_queue() -> Path:
    rows, waiting, done = [], [], []
    for d in ideas():
        st, meta = read_status(d)
        if meta.get("kind") in ("control", "instrument"):
            continue
        if st == "ready-to-submit":
            write_card(d)
            rows.append((d, meta, score_of(d)))
        elif st in WAITING:
            waiting.append((d, st, meta))
        elif st in SUBMITTED:
            done.append((d, st, meta))
    rows.sort(key=lambda x: -(x[2]["rank"] if isinstance(x[2]["rank"], (int, float)) else -9))
    rel = lambda d: d.relative_to(_common.REPO).as_posix()   # noqa: E731
    L = [f"# Submit queue (generated {time.strftime('%Y-%m-%d %H:%M')} by research/bin/dossier.py; do not edit)", "",
         "Ranked by the scorecard composite (equal C1/C2/C3 weights until refit; heavy path: measured ceiling). "
         "\"other Δ\" is the target-cohort Δ on molecules BASE has not memorised, the honest part of the signal "
         "(calibration.md). Submit the kernel version shown, then bind it: `python research/bin/poll_lb.py --claim <idea folder>` (Kaggle's submission list carries no kernel name).", "",
         "| # | idea | kernel | rank by | score | target Δ [90% CI] | other Δ | flags | recommendation |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i, (d, meta, s) in enumerate(rows, 1):
        ci = f"[{_f(s['ci'][0])}, {_f(s['ci'][1])}]" if s["ci"] else ""
        L.append(f"| {i} | [{d.name}]({rel(d)}/card.md) | {kernel_ref(meta)} | {s['kind']} | {_f(s['rank'])} | "
                 f"{s['target']} {_f(s['delta'])} {ci} | {_f(s['other'])} | {len(s['flags'])} | {recommend(s, meta)} |")
    if not rows:
        L.append("| - | nothing ready to submit | | | | | | | |")
    L += ["", "## In G4 (not submittable yet)", *([f"- {d.name}: `{st}`" for d, st, _ in waiting] or ["- none"]),
          "", "## Submitted", *([f"- {d.name}: `{st}` LB {meta.get('lb', '?')} ({meta.get('decision', '')})"
                                 for d, st, meta in done] or ["- none"])]
    queue = _common.RESEARCH / "submit_queue.md"
    queue.write_text("\n".join(L) + "\n", encoding="utf-8")
    return queue


def loop_metrics(history: list[dict]) -> dict:
    """The four loop metrics the grill chose (2026-10-13)."""
    sub = [h for h in history if h.get("lb")]
    keep = [h for h in sub if h.get("now") in ("keep", "keep-pending")]
    novel = 0
    for h in history:
        hyp = _common.REPO / h["idea"] / "hypothesis.md"
        if hyp.exists() and re.search(r"novelty\s*:\s*(new|novel)", hyp.read_text(encoding="utf-8"), re.I):
            novel += 1
    calib = _common.RESEARCH / "harness" / "calibration.md"
    pairs = sum(1 for ln in calib.read_text(encoding="utf-8").splitlines() if re.match(r"\| 20\d\d-", ln)) \
        if calib.exists() else 0
    unattended = sum(1 for h in history if not h.get("override") and h.get("status") != "needs-human")
    return dict(hit_rate=f"{len(keep)}/{len(sub)} submitted kept" if sub else "no submissions yet",
                calibration_pairs=pairs, novelty=f"{novel}/{len(history)} ideas tagged novel",
                unattended=f"{unattended}/{len(history)} cycles closed without a human")


def write_digest(campaign: Path) -> Path | None:
    st = _json(campaign / "state.json")
    if not st or not st.get("history"):
        return None
    for x in st["history"]:                      # live STATUS / LB, not the snapshot taken when the cycle closed
        d = _common.REPO / x["idea"]
        if (d / "STATUS").exists():
            now, meta = read_status(d)
            x.update(now=now, lb=meta.get("lb", ""), override=x.get("override") or meta.get("override", ""))
    n = len(st["history"])
    h = st["history"][-1]
    idea = _common.REPO / h["idea"]
    s = score_of(idea) if idea.exists() else {"kind": "none", "rank": None, "flags": []}
    m = loop_metrics(st["history"])
    L = [f"# Digest: {campaign.name}, cycle {n} ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"- idea: `{h['idea']}` (branch {h.get('branch')}) closed as `{h['status']}`"
         + (f", now `{h['now']}`" if h.get("now") and h.get("now") != h["status"] else ""),
         f"- reason: {h.get('reason') or '-'}", f"- override: {h.get('override') or 'none'}",
         f"- harness: {s['kind']} {_f(s['rank'])}; flags: {'; '.join(s['flags']) or 'none'}",
         f"- model calls so far: {st.get('calls', {})}", "", "## Loop metrics",
         *[f"- {k}: {v}" for k, v in m.items()], "", "## All cycles",
         "| # | idea | closed as | now | LB | reason |", "|---|---|---|---|---|---|",
         *[f"| {i} | {x['idea'].rsplit('/', 1)[-1]} | {x['status']} | {x.get('now', '')} | {x.get('lb', '')} | "
           f"{(x.get('reason') or '')[:80]} |" for i, x in enumerate(st["history"], 1)], "",
         "Submit decisions: `research/submit_queue.md`. Lesson and rule drafts: `research/proposals/`."]
    out = campaign / f"digest_cycle{n}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", metavar="CAMPAIGN")
    a = ap.parse_args(argv)
    print(write_queue())
    if a.digest:
        c = Path(a.digest) if Path(a.digest).is_absolute() else _common.REPO / a.digest
        print(write_digest(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
