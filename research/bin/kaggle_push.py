"""Push a notebook as a private Kaggle kernel version (G4). Never submits.

    python research/bin/kaggle_push.py research/exploit/20261010-foo path/to/probe.ipynb \
        [--slug casmi-foo] [--preset sota_0433_inference_preset] [--wait 7200] \
        [--counter "TRACK4 .*" ...]

kernel-metadata.json is built from the preset in specs/Dataset_intro/dataset_catalog.json.
The commit run takes the fast-commit path (10-20 molecules, L24). With --wait it polls
`kaggle kernels status`, downloads the output to <idea>/kaggle_output/, and greps the log for
the --counter regexes into gates.md.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from _common import REPO, idea_dir, load_config, log_gate, read_status, write_status


def kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle"] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def build_metadata(cfg, preset: str, slug: str, code_file: str) -> dict:
    cat = json.loads((REPO / cfg["catalog"]).read_text(encoding="utf-8"))
    p = cat["presets"][preset]
    gpu = str(p.get("enable_gpu", "true")).lower() == "true"
    md = {
        "id": f"{cfg['kaggle_user']}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,  # L24: offline
        "dataset_sources": list(p.get("dataset_sources", [])),
        "competition_sources": [cfg["competition"]],  # catalog preset slug is wrong
        "kernel_sources": list(p.get("kernel_sources", [])),
        "model_sources": list(p.get("model_sources", [])),
    }
    if gpu:
        md["machine_shape"] = "NvidiaTeslaT4"  # L16: T4 on
    return md


def parse_status(text: str) -> str:
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?(\w+)', text, re.I)
    return m.group(1).lower() if m else "unknown"


def grep_counters(log_text: str, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        rx = re.compile(pat)
        hits += [ln.strip() for ln in log_text.splitlines() if rx.search(ln)][-3:]
    return hits


def main(argv=None, run=kaggle, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("notebook")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--wait", type=float, default=0, help="seconds to poll for completion")
    ap.add_argument("--counter", action="append", default=[])
    a = ap.parse_args(argv)

    cfg = load_config()
    idea = idea_dir(a.idea)
    _, meta = read_status(idea)
    line = meta.get("line", idea.parent.name)
    preset = a.preset or cfg["lines"][line]["kaggle_preset"]
    slug = a.slug or ("casmi-" + re.sub(r"^\d{8}-", "", idea.name))[:50]
    nb = Path(a.notebook).resolve()

    stage = idea / "kaggle_push"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    shutil.copy2(nb, stage / nb.name)
    md = build_metadata(cfg, preset, slug, nb.name)
    (stage / "kernel-metadata.json").write_text(json.dumps(md, indent=2), encoding="utf-8")

    r = run(["kernels", "push", "-p", str(stage)])
    if r.returncode != 0:
        log_gate(idea, "G4", "FAIL", f"push: {(r.stdout + r.stderr).strip()[-200:]}")
        print(r.stdout, r.stderr)
        return 1
    write_status(idea, "g4-pushed", kernel=md["id"])
    print(r.stdout.strip())
    if not a.wait:
        log_gate(idea, "G4", "PUSHED", md["id"])
        return 0

    deadline = time.time() + a.wait
    status = "unknown"
    while time.time() < deadline:
        status = parse_status(run(["kernels", "status", md["id"]]).stdout)
        if status in ("complete", "error", "cancelacknowledged"):
            break
        sleep(60)
    out = idea / "kaggle_output"
    shutil.rmtree(out, ignore_errors=True)
    run(["kernels", "output", md["id"], "-p", str(out)], timeout=1800)
    log_text = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in out.glob("*.log")) if out.exists() else ""
    hits = grep_counters(log_text, a.counter)
    ok = status == "complete"
    log_gate(idea, "G4", "PASS" if ok else f"FAIL ({status})",
             f"{md['id']}; counters: {' / '.join(hits) or 'none matched'}")
    if ok:
        write_status(idea, "ready-to-submit")
    print(f"[kaggle_push] {md['id']} status={status}")
    for h in hits:
        print("  ", h)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
