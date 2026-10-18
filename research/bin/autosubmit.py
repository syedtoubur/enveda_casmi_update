"""Auto-submit within budget (conductor Phase 3, plan decision 12).

    python research/bin/autosubmit.py --once [--dry-run]

Spends every daily Kaggle submission slot except `autosubmit.reserve_for_human`, and only on ideas that
passed every code gate. It never selects final submissions on Kaggle (`final_pick_note()` is text only).

Off (prints `autosubmit: off (<why>)`, exit 0) when `autosubmit.enabled` is false, `daily_submissions` is
null, or `research/conductor/conductor.yaml` has `paused: true`.

Budget: `min(daily - reserve, max_auto_per_day or inf) - auto_submitted_today`, and also
`daily - count_today > reserve` so the human's slot survives even when the human already submitted.
`count_today` = rows of `kaggle competitions submissions` dated today in UTC (Kaggle's day rolls over at
UTC midnight); `auto_submitted_today` comes from `research/conductor/autosubmit_log.jsonl`.

Candidates, in priority order:
  a. replicate: STATUS `keep-pending`, no `lb_rep`, replicate not yet submitted -> same kernel version again;
  b. control: a line control with no `lb` that a `scored-unpaired` probe is waiting on (paired control, L05);
  c. queue: `ready-to-submit` ideas ranked like dossier.write_queue(), eligible only when every code gate holds
     (see `eligibility`). Human overrides are never auto-submitted.
Each submit is `kaggle competitions submit <competition> -k <kernel> -v <version> -f submission.csv -m <msg>`,
then `poll_lb.main(["--claim", <idea>])` binds the submission to the idea (STATUS -> submitted).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import _common
from _common import load_config, read_status, write_status

LOG_NAME = "autosubmit_log.jsonl"


def conductor_dir() -> Path:
    return _common.RESEARCH / "conductor"


def log_path() -> Path:
    return conductor_dir() / LOG_NAME


def is_paused() -> bool:
    p = conductor_dir() / "conductor.yaml"
    if not p.exists():
        return False
    try:
        import yaml
        return bool((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("paused"))
    except Exception:  # noqa: BLE001 - an unreadable file must not unpause the loop
        return True


def off_reason(cfg: dict) -> str | None:
    a = cfg.get("autosubmit") or {}
    if not a.get("enabled"):
        return "autosubmit.enabled is false"
    if a.get("daily_submissions") is None:
        return "autosubmit.daily_submissions is null"
    if is_paused():
        return "conductor paused"
    return None


def utc_today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def count_today(subs: list[dict], today: str | None = None) -> int:
    """Submissions dated today in UTC (Kaggle's `date` column is UTC)."""
    today = today or utc_today()
    return sum(1 for s in subs if str(s.get("date", ""))[:10] == today)


def read_log() -> list[dict]:
    p = log_path()
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def append_log(row: dict) -> None:
    log_path().parent.mkdir(parents=True, exist_ok=True)
    with open(log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def auto_today(today: str | None = None) -> int:
    """Auto submissions logged today (UTC); failed attempts are logged with ok=false and not counted."""
    today = today or utc_today()
    return sum(1 for r in read_log() if r.get("ok", True) and str(r.get("time", ""))[:10] == today)


def budget(cfg: dict, n_today: int, n_auto: int) -> int:
    a = cfg["autosubmit"]
    daily, reserve = int(a["daily_submissions"]), int(a.get("reserve_for_human", 1))
    cap = a.get("max_auto_per_day")
    b = min(daily - reserve, int(cap) if cap is not None else float("inf")) - n_auto
    if daily - n_today <= reserve:        # the human's slot is never consumed
        return 0
    return max(0, min(int(b), daily - n_today - reserve))


def _g4_pass(idea: Path) -> str:
    """'' when the newest G4 row of gates.md is PASS, else the reason."""
    g = idea / "gates.md"
    rows = [ln for ln in g.read_text(encoding="utf-8").splitlines() if ln.startswith("| G4 |")] if g.exists() else []
    if not rows:
        return "no G4 row in gates.md"
    cells = [c.strip() for c in rows[-1].split("|")]
    return "" if len(cells) > 3 and cells[3] == "PASS" else f"latest G4 row is not PASS ({cells[3] if len(cells) > 3 else '?'})"


def _draft_ok(cfg: dict, idea: Path, meta: dict) -> str:
    import kaggle_push
    why = kaggle_push.firing_check(cfg, idea, meta)
    return "; ".join(why)


def eligibility(cfg: dict, idea: Path, meta: dict, firing=_draft_ok) -> list[str]:
    """Reasons a ready-to-submit idea may NOT be auto-submitted ([] = eligible)."""
    import dossier
    import gate
    why = []
    if meta.get("kind") in ("control", "instrument"):
        return [f"kind {meta['kind']}"]
    if meta.get("override"):
        why.append("human override (human-submitted only)")
    rec = dossier.recommend(dossier.score_of(idea), meta)
    if rec != "submit":
        why.append(f"recommendation is not 'submit': {rec}")
    ev = gate.evaluate(idea, cfg.get("gates", {}), meta)
    if not ev["ok"] and not meta.get("gate"):
        why.append(f"gate failed: {ev['why']}")
    g4 = _g4_pass(idea)
    if g4:
        why.append(f"G4: {g4}")
    else:
        fd = firing(cfg, idea, meta)
        if fd:
            why.append(f"G4 draft vs control: {fd}")
    line = meta.get("line", idea.parent.name)
    ctrl = (cfg.get("lines", {}).get(line) or {}).get("control")
    if not ctrl:
        why.append(f"line {line} has no control")
    else:
        cd = _common.REPO / ctrl
        if not (cd / "STATUS").exists():
            why.append(f"control {ctrl} has no STATUS")
        elif not read_status(cd)[1].get("lb"):
            why.append(f"control {ctrl} has no lb (paired control not scored)")
    if not meta.get("kernel"):
        why.append("no kernel in STATUS")
    if not meta.get("kernel_version"):
        why.append("no kernel_version in STATUS")
    return why


def _rel(d: Path) -> str:
    return d.resolve().relative_to(_common.REPO.resolve()).as_posix()


def candidates(cfg: dict, idea_dirs: list[Path] | None = None, firing=_draft_ok) -> tuple[list[dict], list[dict]]:
    """(candidates in priority order, skipped [{idea, why}])."""
    import dossier
    dirs = idea_dirs if idea_dirs is not None else dossier.ideas()
    info = {d: read_status(d) for d in dirs}
    reps, ctrls, queue, skipped = [], [], [], []
    seen_ctrl: set[Path] = set()

    def cand(kind, d, meta):
        return dict(kind=kind, idea=_rel(d), dir=d, kernel=meta.get("kernel", ""),
                    version=meta.get("kernel_version", ""), name=d.name)

    for d, (st, meta) in info.items():
        if st == "keep-pending" and not meta.get("lb_rep"):
            nrefs = len([r for r in meta.get("lb_refs", "").split(",") if r])
            if nrefs >= 2:
                skipped.append(dict(idea=_rel(d), why="replicate already submitted, awaiting poll_lb"))
            elif not (meta.get("kernel") and meta.get("kernel_version")):
                skipped.append(dict(idea=_rel(d), why="replicate: no kernel/kernel_version"))
            else:
                reps.append(cand("a", d, meta))
    for d, (st, meta) in info.items():
        if st != "scored-unpaired":
            continue
        line = meta.get("line", d.parent.name)
        ctrl = (cfg.get("lines", {}).get(line) or {}).get("control")
        cd = (_common.REPO / ctrl).resolve() if ctrl else None
        if cd is None or cd in seen_ctrl or not (cd / "STATUS").exists():
            continue
        cst, cmeta = read_status(cd)
        if cmeta.get("lb"):
            continue
        if cst != "ready-to-submit" or not (cmeta.get("kernel") and cmeta.get("kernel_version")):
            skipped.append(dict(idea=_rel(cd), why=f"control not submittable (state {cst}, kernel/version needed)"))
            continue
        if cmeta.get("override") or cmeta.get("autosubmit_error"):
            skipped.append(dict(idea=_rel(cd), why="control has override or earlier autosubmit_error"))
            continue
        seen_ctrl.add(cd)
        ctrls.append(cand("b", cd, cmeta))
    ranked = []
    for d, (st, meta) in info.items():
        if st != "ready-to-submit" or meta.get("kind") in ("control", "instrument"):
            continue
        if d.resolve() in seen_ctrl:
            continue
        why = eligibility(cfg, d, meta, firing)
        if meta.get("autosubmit_error"):
            why.append(f"earlier autosubmit_error: {meta['autosubmit_error'][:80]}")
        if why:
            skipped.append(dict(idea=_rel(d), why="; ".join(why)))
            continue
        r = dossier.score_of(d)["rank"]
        ranked.append((-(r if isinstance(r, (int, float)) else -9), cand("c", d, meta)))
    queue = [c for _, c in sorted(ranked, key=lambda t: t[0])]
    return reps + ctrls + queue, skipped


def kaggle_submit(cfg: dict, c: dict) -> tuple[bool, str]:
    msg = f"{c['name']} | {c['kernel']}@v{c['version']} | auto"
    cmd = ["kaggle", "competitions", "submit", cfg["competition"], "-k", c["kernel"], "-v", str(c["version"]),
           "-f", "submission.csv", "-m", msg]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    bad = r.returncode != 0 or "error" in out.lower() or "401" in out or "403" in out
    return (not bad), out[-300:]


def claim(rel: str, tries: int = 4, wait_s: float = 20.0) -> int:
    """Bind the newest unclaimed submission; Kaggle may list a fresh submission only after a few seconds."""
    import time
    import poll_lb
    rc = 1
    for i in range(tries):
        rc = poll_lb.main(["--claim", rel])
        if rc == 0:
            return rc
        if i < tries - 1:
            time.sleep(wait_s)
    return rc


def run_once(cfg: dict | None = None, dry_run: bool = False, fetcher=None, submit=kaggle_submit,
             claimer=claim, idea_dirs: list[Path] | None = None, firing=_draft_ok, out=print) -> dict:
    cfg = cfg or load_config()
    why = off_reason(cfg)
    if why:
        out(f"autosubmit: off ({why})")
        return dict(off=why, submitted=[])
    import poll_lb
    subs = (fetcher or poll_lb.fetch)(cfg)
    n_today, n_auto = count_today(subs), auto_today()
    b = budget(cfg, n_today, n_auto)
    cands, skipped = candidates(cfg, idea_dirs, firing)
    a = cfg["autosubmit"]
    out(f"autosubmit: daily {a['daily_submissions']}, reserve {a.get('reserve_for_human', 1)}, "
        f"today (UTC) {n_today}, auto today {n_auto}, budget {b}")
    for c in cands:
        out(f"  candidate [{c['kind']}] {c['idea']} {c['kernel']}@v{c['version']}")
    for s in skipped:
        out(f"  skipped {s['idea']}: {s['why']}")
    if not cands:
        out("  no candidates")
    done = []
    if dry_run:
        out("autosubmit: dry run, nothing submitted")
        return dict(budget=b, candidates=cands, skipped=skipped, submitted=[])
    for c in cands:
        if b <= 0:
            out("  budget exhausted")
            break
        ok, msg = submit(cfg, c)
        row = dict(time=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), idea=c["idea"],
                   kernel=c["kernel"], version=c["version"], kind=c["kind"], ok=ok)
        if not ok:
            row["error"] = msg
            append_log(row)
            write_status(c["dir"], None, autosubmit_error=" ".join(msg.split())[:200])
            out(f"  FAILED {c['idea']}: {msg}")
            continue
        append_log(row)
        b -= 1
        done.append(c["idea"])
        rc = claimer(c["idea"])
        out(f"  submitted {c['idea']} (claim rc={rc})")
    return dict(budget=b, candidates=cands, skipped=skipped, submitted=done)


def _lb_mean(meta: dict) -> float | None:
    v = [float(meta[k]) for k in ("lb", "lb_rep") if meta.get(k)]
    return sum(v) / len(v) if v else None


def final_pick_note(idea_dirs: list[Path] | None = None) -> str:
    """Markdown paragraph recommending the two final submissions. Text for the digest; selects nothing on Kaggle."""
    import dossier
    dirs = idea_dirs if idea_dirs is not None else dossier.ideas()
    keeps = []
    for d in dirs:
        st, meta = read_status(d)
        m = _lb_mean(meta)
        if st == "keep" and m is not None:
            keeps.append((m, d, meta))
    if not keeps:
        return "**Final picks:** no idea is in `keep` yet, so nothing is recommended. Nothing is selected on Kaggle."
    keeps.sort(key=lambda t: -t[0])
    best = keeps[0]
    bl = best[2].get("line", best[1].parent.name)

    def fam(meta):
        return meta.get("family") or meta.get("branch") or ""
    diff = next((k for k in keeps[1:] if k[2].get("line", k[1].parent.name) != bl
                 or (fam(k[2]) and fam(k[2]) != fam(best[2]))), None)
    fmt = lambda t: (f"`{t[1].name}` (line {t[2].get('line', t[1].parent.name)}, "   # noqa: E731
                     f"LB {t[2].get('lb', '?')}" + (f", replicate {t[2]['lb_rep']}" if t[2].get("lb_rep") else "")
                     + f", mean {t[0]:.4f}, kernel {t[2].get('kernel', '?')} v{t[2].get('kernel_version', '?')})")
    s = f"**Final picks (suggestion only; the human selects on Kaggle):** best keep by mean LB is {fmt(best)}."
    if diff:
        s += f" The most different high-LB keep is {fmt(diff)}, a different line or family, to hedge the private split."
    else:
        s += " No second keep differs in line or family yet, so there is no hedge pick."
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass (the only mode)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run_once(dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
