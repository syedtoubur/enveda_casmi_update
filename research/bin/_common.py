"""Shared helpers for research/bin scripts: config, STATUS file, gate log, paths."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or RESEARCH / "config.yaml").read_text(encoding="utf-8"))


def idea_dir(arg: str | Path) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = (REPO / p) if (REPO / p).exists() else Path.cwd() / p
    p = p.resolve()
    if not (p / "STATUS").exists():
        raise SystemExit(f"not an idea folder (no STATUS): {p}")
    return p


def read_status(idea: Path) -> tuple[str, dict]:
    lines = (idea / "STATUS").read_text(encoding="utf-8").splitlines()
    state = lines[0].strip() if lines else ""
    meta = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            meta[k.strip()] = v.strip()
    return state, meta


def write_status(idea: Path, state: str | None = None, **updates) -> None:
    old_state, meta = read_status(idea)
    meta.update({k: str(v) for k, v in updates.items()})
    body = [state or old_state] + [f"{k}: {v}" for k, v in meta.items()]
    (idea / "STATUS").write_text("\n".join(body) + "\n", encoding="utf-8")


def log_gate(idea: Path, gate: str, result: str, evidence: str, cu: float | str = "") -> None:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(idea / "gates.md", "a", encoding="utf-8") as f:
        f.write(f"| {gate} | {now} | {result} | {evidence} | {cu} |\n")


def win_to_wsl(p: Path) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        s = f"/mnt/{s[0].lower()}{s[2:]}"
    return s
