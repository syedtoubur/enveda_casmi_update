"""Push a notebook as a private Kaggle kernel version (G4). Never submits.

    python research/bin/kaggle_push.py research/exploit/20261010-foo path/to/probe.ipynb \
        [--slug casmi-foo] [--preset sota_0433_inference_preset] [--wait 7200] \
        [--counter "TRACK4 .*" ...]

kernel-metadata.json is built from the preset in specs/Dataset_intro/dataset_catalog.json.
The commit run takes the fast-commit path (10-20 molecules, L24). With --wait it polls
`kaggle kernels status`, downloads only the log/csv/meta files to <idea>/kaggle_output/, and greps
the log for the --counter regexes into gates.md.

Refuses to push (L30, L31) when:
- a file or dataset the notebook loads is not in the pushed sources (audit_sources), or
- the line has a `control:` idea whose gates.md has no G4 PASS yet (control pushes are exempt).
G4 FAILs on a COMPLETE run if the log matches a `gates.g4_fail_markers` regex or a --counter
regex never matches.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from _common import REPO, RESEARCH, idea_dir, load_config, log_gate, read_status, write_status


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


FIND_RX = [r"""(?:\bfind|_find)\(\s*f?['"]([^'"]+)['"]""",
           r"""glob(?:\.glob)?\(\s*f?['"]([^'"]+)['"]""",
           r"""['"](/kaggle/input/[^'"]+)['"]"""]
COMP_FILES = {"test.parquet", "train.parquet", "sample_submission.csv"}


def notebook_code(nb: Path) -> str:
    cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
    code = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                     for c in cells if c.get("cell_type") == "code")
    # helper modules are often embedded as escaped strings (%%writefile, exec): decode them too
    return code + "\n" + code.encode("utf-8", "replace").decode("unicode_escape", "ignore")


def audit_sources(code: str, catalog: dict, sources: list[str]) -> tuple[list[str], list[str]]:
    """(missing, unresolved). missing: '<ref> -> needs <slug|slug>' for refs no pushed source covers."""
    files: dict[str, list[str]] = {}
    for group in catalog.get("datasets", {}).values():
        for d in group.values():
            files[d["slug"]] = [Path(f).name for f in d.get("key_files", [])]
    have = set(sources)
    names = {slug.split("/")[1]: slug for slug in files}
    names.update({s.split("/")[1]: s for s in sources})
    refs = {m for rx in FIND_RX for m in re.findall(rx, code)}
    # dataset names used as path filters, e.g. `if 'casmi26-fp-models-v2' in p`
    refs |= {n for n in names if re.search(rf"""['"/]{re.escape(n)}['"/]""", code)}
    missing, unresolved = [], []
    for ref in sorted(refs):
        base = ref.rstrip("\\/").split("/")[-1]
        if "{" in ref or len(base) < 3:
            continue
        need = sorted({slug for n, slug in names.items() if n in ref.split("/")})
        if not need:
            if base in COMP_FILES:
                continue
            need = sorted(s for s, fs in files.items() if any(fnmatch.fnmatch(f, base) for f in fs))
        if not need:
            unresolved.append(ref)
        elif not have & set(need):
            missing.append(f"{ref} -> needs {' | '.join(need)}")
    return missing, unresolved


def control_passed(cfg, line: str) -> tuple[bool, str]:
    ctrl = cfg["lines"].get(line, {}).get("control")
    if not ctrl:
        return True, "no control configured"
    gates = REPO / ctrl / "gates.md"
    ok = gates.exists() and any(ln.startswith("| G4 |") and "| PASS |" in ln
                                for ln in gates.read_text(encoding="utf-8").splitlines())
    return ok, ctrl


def log_text_of(path: Path) -> str:
    """Kaggle logs are a JSON list of {stream_name, time, data}; fall back to raw text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(e.get("data", "") for e in json.loads(raw))
    except (ValueError, AttributeError):
        return raw


def parse_status(text: str) -> str:
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?(\w+)', text, re.I)
    return m.group(1).lower() if m else "unknown"


def grep_counters(log_text: str, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        rx = re.compile(pat)
        hits += [ln.strip() for ln in log_text.splitlines() if rx.search(ln)][-3:]
    return hits


def check_log(log_text: str, counters: list[str], markers: list[str]) -> list[str]:
    """Reasons a COMPLETE run still fails G4: a fail marker fired, or a counter never printed."""
    bad = [f"marker {m!r}: {grep_counters(log_text, [m])[0][:120]}"
           for m in markers if grep_counters(log_text, [m])]
    bad += [f"counter {c!r} not printed" for c in counters if not grep_counters(log_text, [c])]
    return bad


DONE = ("complete", "error", "cancelacknowledged")
# top-level log/csv plus the ICE/GL meta files; not the vendored packages under ice_site/
OUTPUT_FILES = r"^[^/\\]+\.(log|csv)$|_work[/\\][^/\\]+meta\.json$"


def finalize(cfg, idea: Path, kernel: str, status: str, run=kaggle) -> int:
    """Download the log of a finished kernel, scan it, and log G4. 0 = PASS, 2 = FAIL."""
    stage = idea / "kaggle_push"
    counters = [c for c in (stage / "counters.txt").read_text(encoding="utf-8").splitlines() if c] \
        if (stage / "counters.txt").exists() else []
    out = idea / "kaggle_output"
    shutil.rmtree(out, ignore_errors=True)
    run(["kernels", "output", kernel, "-p", str(out),
         "--file-pattern", OUTPUT_FILES], timeout=1800)
    log_text = "".join(log_text_of(p) for p in out.glob("*.log")) if out.exists() else ""
    hits = grep_counters(log_text, counters)
    _, meta = read_status(idea)
    required = cfg["lines"].get(meta.get("line", idea.parent.name), {}).get("g4_required") or []
    bad = check_log(log_text, counters + required, cfg.get("gates", {}).get("g4_fail_markers", []))
    ok = status == "complete" and not bad
    result = "PASS" if ok else f"FAIL ({status}{'; ' + ' / '.join(bad) if bad else ''})"
    log_gate(idea, "G4", result, f"{kernel}; counters: {' / '.join(hits) or 'none matched'}")
    write_status(idea, "ready-to-submit" if ok else "g4-failed")
    card = idea / "submit_card.md"
    if card.exists():
        with card.open("a", encoding="utf-8") as f:
            f.write(f"- **G4 result ({time.strftime('%Y-%m-%d %H:%M')})**: {result}; "
                    f"counters: {' / '.join(hits) or 'none'}\n")
    print(f"[kaggle_push] {kernel} status={status} -> {result}")
    for h in hits:
        print("  ", h)
    return 0 if ok else 2


def check(cfg, ideas: list[Path], run=kaggle) -> int:
    """One status poll per g4-pushed idea; finalize the finished ones. No waiting, no Claude."""
    worst = 0
    for idea in ideas:
        state, meta = read_status(idea)
        if state != "g4-pushed" or not meta.get("kernel"):
            continue
        status = parse_status(run(["kernels", "status", meta["kernel"]]).stdout)
        if status not in DONE:
            print(f"[kaggle_push] {meta['kernel']} still {status}")
            continue
        worst = max(worst, finalize(cfg, idea, meta["kernel"], status, run))
    return worst


def main(argv=None, run=kaggle, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="?")
    ap.add_argument("notebook", nargs="?")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--wait", type=float, default=0, help="seconds to poll for completion")
    ap.add_argument("--counter", action="append", default=[])
    ap.add_argument("--check", action="store_true",
                    help="finalize G4 for <idea>, or for every g4-pushed idea when none is given")
    a = ap.parse_args(argv)

    cfg = load_config()
    if a.check:
        ideas = [idea_dir(a.idea)] if a.idea else sorted(RESEARCH.glob("*/*/STATUS"))
        return check(cfg, [i if i.is_dir() else i.parent for i in ideas], run)
    if not (a.idea and a.notebook):
        ap.error("idea and notebook are required unless --check")
    idea = idea_dir(a.idea)
    _, meta = read_status(idea)
    line = meta.get("line", idea.parent.name)
    preset = a.preset or cfg["lines"][line]["kaggle_preset"]
    slug = a.slug or ("casmi-" + re.sub(r"^\d{8}-", "", idea.name))[:50]
    nb = Path(a.notebook).resolve()

    if meta.get("kind") != "control":
        ok, ctrl = control_passed(cfg, line)
        if not ok:
            log_gate(idea, "G4", "BLOCKED", f"control {ctrl} has no G4 PASS yet (L31)")
            print(f"[kaggle_push] refused: control {ctrl} has no G4 PASS yet")
            return 6
    cat = json.loads((REPO / cfg["catalog"]).read_text(encoding="utf-8"))
    missing, unresolved = audit_sources(notebook_code(nb), cat,
                                        cat["presets"][preset].get("dataset_sources", []))
    if missing:
        log_gate(idea, "G4", "BLOCKED", "sources (L30): " + "; ".join(missing))
        print("[kaggle_push] refused: notebook needs files the preset does not mount:")
        print("\n".join("  " + m for m in missing))
        return 5
    if unresolved:
        print("[kaggle_push] warning, not in the catalog (check by hand):", ", ".join(unresolved))

    stage = idea / "kaggle_push"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    shutil.copy2(nb, stage / nb.name)
    md = build_metadata(cfg, preset, slug, nb.name)
    (stage / "kernel-metadata.json").write_text(json.dumps(md, indent=2), encoding="utf-8")
    (stage / "counters.txt").write_text("\n".join(a.counter) + "\n", encoding="utf-8")

    r = run(["kernels", "push", "-p", str(stage)])
    if r.returncode != 0:
        log_gate(idea, "G4", "FAIL", f"push: {(r.stdout + r.stderr).strip()[-200:]}")
        print(r.stdout, r.stderr)
        return 1
    print(r.stdout.strip())
    if "not valid" in r.stdout + r.stderr:  # Kaggle pushes anyway, silently dropping the bad sources
        log_gate(idea, "G4", "FAIL", f"push dropped sources: {(r.stdout + r.stderr).strip()[:300]}")
        write_status(idea, "g4-failed", kernel=md["id"])
        return 1
    write_status(idea, "g4-pushed", kernel=md["id"])
    log_gate(idea, "G4", "PUSHED", md["id"] + (f"; unresolved refs: {', '.join(unresolved)}"
                                               if unresolved else ""))
    if not a.wait:
        return 0

    deadline = time.time() + a.wait
    status = "unknown"
    while time.time() < deadline:
        status = parse_status(run(["kernels", "status", md["id"]]).stdout)
        if status in DONE:
            break
        sleep(60)
    if status not in DONE:
        print(f"[kaggle_push] {md['id']} still {status}; finish later with --check")
        return 0
    return finalize(cfg, idea, md["id"], status, run)


if __name__ == "__main__":
    raise SystemExit(main())
