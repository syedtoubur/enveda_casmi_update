"""The LB-slot gate between harness-done and build (grill 2026-10-13; gate_report.md supersede block).

Rank-only until calibrated: the gate never asks "is the gain big enough?". It only kills ideas that
show no gain or real harm; everything else is queued and ranked by the scorecard composite, and
the human decides at submit. A written prior.md never passes the gate by itself.

    harness_result.json (scorecard-1): target, cohorts{name: {delta, ci90, advisory}}, displaced,
        n_target, composite, heavy_path, ceiling
    stress_result.json: [{id, died, status, blocking, deviation{original, proxy, accepted_by}}]

Dead when: a death condition fired; a blocking attack is UNTESTED; target delta <= 0 or its 90% CI
upper bound < 0; a non-advisory cohort's CI upper bound < -harm_margin; rank-1 displacements above
displace_max_frac of the target cohort; a heavy-path idea without a measured ceiling > 0.
A STATUS `override:` line (campaign.py --override) passes a dead idea and is recorded as such.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"harm_margin": 0.004, "displace_max_frac": 0.01}


def _load(path: Path):
    """(data, error). Never raises: an agent-written file may be missing or malformed."""
    if not path.exists():
        return None, f"no {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (ValueError, OSError) as e:
        return None, f"{path.name} unreadable: {e}"


def attack_findings(stress) -> tuple[list[str], list[str]]:
    """(dead reasons, flags) from stress_result.json rows."""
    dead, flags = [], []
    rows = stress if isinstance(stress, list) else []
    for r in rows:
        if not isinstance(r, dict) or "id" not in r:
            continue
        if r.get("died") is True:
            dead.append(f"attack {r['id']} died")
        elif str(r.get("status", "")).upper() == "UNTESTED" and r.get("blocking"):
            dead.append(f"blocking attack {r['id']} UNTESTED ({r.get('why', 'no reason given')})")
        if isinstance(r.get("deviation"), dict):
            d = r["deviation"]
            flags.append(f"attack {r['id']} ran a proxy ({d.get('proxy', '?')}) instead of "
                         f"{d.get('original', '?')}; accepted by {d.get('accepted_by', 'nobody')}")
    return dead, flags


def _ci(c: dict) -> tuple[float, float] | None:
    ci = c.get("ci90")
    try:
        lo, hi = float(ci[0]), float(ci[1])
        return lo, hi
    except (TypeError, ValueError, IndexError):
        return None


def scorecard_findings(r: dict, hcfg: dict) -> tuple[list[str], list[str]]:
    """(dead reasons, flags) from a harness_result.json; recomputed, never read from its `verdict`."""
    h = {**DEFAULTS, **(hcfg or {})}
    dead, flags = [], []
    if r.get("heavy_path"):
        ceil = r.get("ceiling")
        if not isinstance(ceil, (int, float)) or ceil <= 0:
            dead.append(f"heavy path without a measured ceiling > 0 (ceiling {ceil})")
        else:
            flags.append(f"heavy path: ranked by measured ceiling {ceil:+.4f}, not replayed")
        return dead, flags
    cohorts = r.get("cohorts")
    if not isinstance(cohorts, dict):  # legacy single-shard result: the target is its own shard
        name = r.get("shard") or "legacy-shard"
        cohorts = {name: {"delta": r.get("delta"), "ci90": r.get("ci90")}}
        r = {**r, "target": name}
    target = r.get("target")
    t = cohorts.get(target) if target else None
    if not isinstance(t, dict) or not isinstance(t.get("delta"), (int, float)):
        return [f"no delta for target cohort {target!r}"], flags
    ci = _ci(t)
    if t["delta"] <= 0:
        dead.append(f"target {target} delta {t['delta']:+.4f} <= 0")
    if ci and ci[1] < 0:
        dead.append(f"target {target} CI {ci[0]:+.4f},{ci[1]:+.4f} entirely below 0")
    for name, c in cohorts.items():
        if not isinstance(c, dict):
            continue
        ci = _ci(c)
        if ci is None:
            flags.append(f"{name}: no CI")
            continue
        if name != target and not c.get("advisory") and ci[1] < -h["harm_margin"]:
            dead.append(f"harm on {name}: CI {ci[0]:+.4f},{ci[1]:+.4f} below -{h['harm_margin']}")
        elif ci[0] < 0:
            flags.append(f"{name}: CI lower bound {ci[0]:+.4f} < 0{' (advisory cohort)' if c.get('advisory') else ''}")
    n, disp = r.get("n_target"), r.get("displaced")
    if isinstance(n, int) and isinstance(disp, int) and n > 0 and disp > h["displace_max_frac"] * n:
        dead.append(f"{disp} rank-1 truths displaced > {h['displace_max_frac']:.0%} of {n}")
    return dead, flags


def evaluate(idea: Path, hcfg: dict, meta: dict) -> dict:
    """{ok, dead, flags, override, why} for one idea folder."""
    res, err = _load(idea / "harness_result.json")
    dead, flags = ([err], []) if res is None or not isinstance(res, dict) else scorecard_findings(res, hcfg)
    stress, serr = _load(idea / "stress_result.json")
    if stress is not None:
        d2, f2 = attack_findings(stress)
        dead, flags = dead + d2, flags + f2
    elif serr and not serr.startswith("no "):
        flags.append(serr)
    override = meta.get("override", "")
    ok = not dead or bool(override)
    if dead and override:
        why = f"OVERRIDE ({override}) of: {'; '.join(dead)}"
    elif dead:
        why = "; ".join(dead)
    else:
        why = "queued (rank-only)" + (f"; flags: {'; '.join(flags)}" if flags else "")
    return {"ok": ok, "dead": dead, "flags": flags, "override": override, "why": why}
