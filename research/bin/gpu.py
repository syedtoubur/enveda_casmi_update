"""Weekly Kaggle GPU pool the loop may spend without asking (grill 2026-10-13).

One shared pool: config `kaggle.weekly_gpu_h` (30 h) per ISO week for G4 probe runs, harness dumps (instruments) and
training. Anything but a G4 run must leave `kaggle.g4_reserve_h` free, and a training run may not plan more than
`kaggle.per_run_h_max` (4 h). Colab runs are outside this pool: they need a human-approved training_request.md.

    python research/bin/gpu.py          # this week's spend, by kind
"""
from __future__ import annotations

import datetime as dt
import json

import _common

KINDS = ("probe", "control", "instrument", "training")


def ledger_path():
    return _common.RESEARCH / "gpu_ledger.json"


def week(t: dt.date | None = None) -> str:
    y, w, _ = (t or dt.date.today()).isocalendar()
    return f"{y}-W{w:02d}"


def load() -> list[dict]:
    try:
        return json.loads(ledger_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def used(wk: str | None = None) -> float:
    wk = wk or week()
    return sum(float(r.get("hours", 0)) for r in load() if r.get("week") == wk)


def record(idea: str, kernel: str, hours: float, kind: str) -> None:
    rows = load()
    rows.append(dict(week=week(), date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), idea=idea, kernel=kernel,
                     kind=kind, hours=round(hours, 3)))
    ledger_path().write_text(json.dumps(rows, indent=1), encoding="utf-8")


def estimate(cfg, kind: str, meta: dict) -> float:
    k = cfg.get("kaggle", {})
    return float(meta.get("gpu_h_est") or (k.get("est_h") or {}).get(kind, 1.0))


def refuse_reason(cfg, kind: str, est_h: float) -> str:
    """'' when the run fits the pool, else why not."""
    k = cfg.get("kaggle", {})
    cap, reserve, per_run = float(k.get("weekly_gpu_h", 30)), float(k.get("g4_reserve_h", 8)), float(k.get("per_run_h_max", 4))
    if kind == "training" and est_h > per_run:
        return f"training run planned at {est_h:.1f} GPU-h > per-run cap {per_run:.0f} h: write a Colab training_request.md"
    room = cap - (0 if kind in ("probe", "control") else reserve)
    u = used()
    if u + est_h > room:
        return (f"weekly GPU pool: {u:.1f} h used + {est_h:.1f} h planned > {room:.0f} h allowed for {kind} runs "
                f"(pool {cap:.0f} h, G4 reserve {reserve:.0f} h); wait for next week or ask the human")
    return ""


def log_hours(log_text_json: str) -> float | None:
    """Run time of a Kaggle log (a JSON list of {time: seconds since start, ...}), in hours."""
    try:
        return max(float(e.get("time", 0)) for e in json.loads(log_text_json)) / 3600
    except (ValueError, TypeError, AttributeError):
        return None


if __name__ == "__main__":
    rows = [r for r in load() if r.get("week") == week()]
    by = {}
    for r in rows:
        by[r["kind"]] = by.get(r["kind"], 0) + r["hours"]
    print(f"{week()}: {used():.2f} GPU-h used; " + ", ".join(f"{k} {v:.2f}" for k, v in by.items()))
