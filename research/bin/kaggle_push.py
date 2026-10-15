"""Push a notebook as a private Kaggle kernel version (G4). Never submits.

    python research/bin/kaggle_push.py research/exploit/20261010-foo path/to/probe.ipynb \
        [--slug casmi-foo] [--preset sota_0433_inference_preset] [--wait 7200] \
        [--counter "TRACK4 fired" ...] [--counter "?ELIGIBLE"] [--source owner/catalog-dataset ...]
    python research/bin/kaggle_push.py --check [<idea>]     # backup: one poll, finalize finished runs
    python research/bin/kaggle_push.py --watch <idea>       # what the push starts, detached

kernel-metadata.json is built from the preset in specs/Dataset_intro/dataset_catalog.json, plus
any --source (must be a catalog dataset; kept in STATUS `extra_sources` for later pushes).
The commit run takes the fast-commit path (10-20 molecules, L24).

Without --wait the push starts a detached watcher (L34) that outlives Claude: it polls
`kaggle kernels status` every `kaggle.watch_interval_s` with a heartbeat in
<idea>/kaggle_watch/watch.log, and gives up after `kaggle.watch_timeout_s` (STATUS g4-timeout,
which --check still finalizes). A finished run is finalized: only the log/csv/meta files go to
<idea>/kaggle_output/, the --counter regexes go to gates.md, and after_g4 hands off:
- PASS: toast; a line control also queues that line's waiting probes (one call_claude drain).
- FAIL: toast, g4_fix_brief.md, STATUS g4-fix and a fresh /run-idea fix agent, at most
  `gates.g4_fix_rounds` times; then needs-human with a questions.md entry.

Refuses to push (L30, L31) when:
- a file or dataset the notebook loads is not in the pushed sources (audit_sources), or
- the line has a `control:` idea whose gates.md has no G4 PASS yet (control pushes are exempt).
G4 FAILs on a COMPLETE run if the log matches a `gates.g4_fail_markers` regex, a --counter
regex never matches, or the number after its last match is not > 0 (`?<regex>`: may be 0).
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import call_claude
import gpu
from _common import REPO, RESEARCH, idea_dir, load_config, log_gate, read_status, write_status

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # children of a detached process get no console
FIX_SCOPE = """\
- The fix agent may change the patch code in the line worktree (`E:/casmi/lines/<line>`), then rerun G1 and G2.
- It may add push sources **that already exist in `dataset_catalog.json`** (`kaggle_push.py --source <owner/slug>`).
- A dataset missing from the catalog, or a fix that would change the idea itself: write to `questions.md`, set STATUS `needs-human`, stop.
- For a **control** idea only sources and metadata may change, never the notebook code.
- The catalog itself is never edited.
"""


def kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle"] + args, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          encoding="utf-8", errors="replace", timeout=timeout, creationflags=NO_WINDOW)


def build_metadata(cfg, preset: str, slug: str, code_file: str, extra: list[str] = ()) -> dict:
    cat = json.loads((REPO / cfg["catalog"]).read_text(encoding="utf-8"))
    p = cat["presets"][preset]
    gpu = str(p.get("enable_gpu", "true")).lower() == "true"
    sources = list(p.get("dataset_sources", []))
    md = {
        "id": f"{cfg['kaggle_user']}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,  # L24: offline
        "dataset_sources": sources + [s for s in extra if s not in sources],
        "competition_sources": [cfg["competition"]],  # catalog preset slug is wrong
        "kernel_sources": list(p.get("kernel_sources", [])),
        "model_sources": list(p.get("model_sources", [])),
    }
    if gpu:
        md["machine_shape"] = "NvidiaTeslaT4"  # L16: T4 on
    return md


def control_image(cfg, line: str) -> str | None:
    """The line control's pinned Kaggle image (control STATUS `docker_image:`). A probe runs on its control's image;
    otherwise Kaggle uses its latest image and the comparison is no longer paired (E006 asserts Python 3.12)."""
    ctrl = cfg["lines"].get(line, {}).get("control")
    if not ctrl or not (REPO / ctrl / "STATUS").exists():
        return None
    return read_status(REPO / ctrl)[1].get("docker_image") or None


def push_preconditions(cfg, idea: Path, line: str) -> str:
    """'' when a probe may be pushed (loop audit W7, N5, N6), else why not:
    - the line's control pins a Kaggle image (control STATUS `docker_image:`), so the pair runs on the same image;
    - the newest G2 row is not a FAIL;
    - a campaign idea (it has spec.md) is preflight-clean now, not only when it was specced."""
    if cfg["lines"].get(line, {}).get("control") and not control_image(cfg, line):
        return (f"control {cfg['lines'][line]['control']} has no `docker_image:` in STATUS; a probe on Kaggle's "
                "latest image is not paired (E006 asserts Python 3.12)")
    gates = idea / "gates.md"
    g2 = [ln for ln in gates.read_text(encoding="utf-8").splitlines() if ln.startswith("| G2")] if gates.exists() else []
    if g2 and "| FAIL" in g2[-1]:
        return f"newest G2 row is a FAIL: {g2[-1][:200]}"
    if (idea / "spec.md").exists():
        import preflight
        found = preflight.run(idea, preflight.load_rules())
        if found:
            return f"preflight not clean: {found[0][:200]}" + (f" (+{len(found) - 1} more)" if len(found) > 1 else "")
    return ""


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


NUM = r"-?\d+(?:\.\d+)?"


def counter_value(log_text: str, pat: str) -> float | None:
    """The counter's value at the LAST match of `pat`, or None.
    - a capture group: its number (`lists (\\d+)`);
    - a pattern that itself matches digits (`\\d`, `[0-9`) and ends on one: the number it ended on, so
      `engine lists \\d+` reads 12 from `engine lists 12 619s`, not the runtime 619;
    - otherwise the first number within 30 characters after the match:
      `[X] eligible molecules 1` -> 1; `... | molecules reranked 0 | ...` -> 0."""
    rx, val = re.compile(pat), None
    digits = bool(re.search(r"\\d|\[0-9", pat))
    for ln in log_text.splitlines():
        m = rx.search(ln)
        if not m:
            continue
        if m.groups() and m.group(1) is not None and re.fullmatch(NUM, m.group(1)):
            val = float(m.group(1))
        elif digits and (n := re.search(rf"({NUM})$", m.group(0))):
            val = float(n.group(1))
        else:
            n = re.match(rf"\D{{0,30}}?({NUM})", ln[m.end():])
            val = float(n.group(1)) if n else None
    return val


def read_submission(path: Path) -> dict[str, str]:
    """molecule_id -> the ranked candidate string of a submission.csv ({} when absent)."""
    import csv
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {r["molecule_id"]: r.get("smiles", "") for r in csv.DictReader(f)}


def draft_diff(probe_csv: Path, control_csv: Path) -> tuple[int, int]:
    """(molecules whose ranked list differs, molecules in both drafts). A probe whose draft equals the
    control's draft changed nothing on the draft slice: its G4 counters prove it ran, not that it fired."""
    p, c = read_submission(probe_csv), read_submission(control_csv)
    common = set(p) & set(c)
    return sum(p[k] != c[k] for k in common), len(common)


def control_draft(cfg, line: str) -> Path | None:
    """The line control's own draft submission.csv (its G4 commit run), or None."""
    lc = cfg["lines"].get(line, {})
    path = lc.get("control_draft") or (f"{lc['control']}/kaggle_output/submission.csv" if lc.get("control") else None)
    return REPO / path if path else None


def firing_check(cfg, idea: Path, meta: dict) -> list[str]:
    """Reasons the probe's own change did not demonstrably fire (L36): its draft must differ from the
    control's draft on at least `g4_min_changed` molecules (STATUS overrides `gates.g4_min_changed`),
    beyond the control's own run-to-run noise `gates.g4_draft_noise`."""
    g = cfg.get("gates", {})
    if meta.get("kind") in ("control", "instrument") or not g.get("g4_draft_diff", False):
        return []
    line = meta.get("line", idea.parent.name)
    ctrl = control_draft(cfg, line)
    if ctrl is None or not ctrl.is_file():
        return [f"no control draft for line {line} ({ctrl}); push the control once to capture it"]
    changed, common = draft_diff(idea / "kaggle_output" / "submission.csv", ctrl)
    if not common:
        return ["draft submission.csv missing or shares no molecule with the control draft"]
    need = max(int(meta.get("g4_min_changed", g.get("g4_min_changed", 1))), int(g.get("g4_draft_noise", 0)) + 1)
    return [] if changed >= need else [f"draft identical to control on {common - changed}/{common} molecules "
                                       f"(changed {changed} < {need}): the change never fired"]


def check_log(log_text: str, counters: list[str], markers: list[str],
              required: list[str] = ()) -> list[str]:
    """Reasons a COMPLETE run still fails G4: a fail marker fired, a required line or counter never
    printed, or a counter's value is not > 0 (a printed `molecules reranked 0` proves nothing fired, L05).
    A counter written as `?<regex>` may legitimately be 0 (or have no number) and only has to print."""
    bad = [f"marker {m!r}: {grep_counters(log_text, [m])[0][:120]}"
           for m in markers if grep_counters(log_text, [m])]
    bad += [f"required {r!r} not printed" for r in required if not grep_counters(log_text, [r])]
    for c in counters:
        pat = c[1:] if c.startswith("?") else c
        if not grep_counters(log_text, [pat]):
            bad.append(f"counter {pat!r} not printed")
        elif not c.startswith("?"):
            v = counter_value(log_text, pat)
            if v is None or v <= 0:
                bad.append(f"counter {pat!r} value {v if v is not None else 'missing'} (must be > 0)")
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
    logs = sorted(out.glob("*.log")) if out.exists() else []
    log_text = "".join(log_text_of(p) for p in logs)
    hours = max([h for p in logs if (h := gpu.log_hours(p.read_text(encoding="utf-8", errors="replace")))], default=None)
    hits = grep_counters(log_text, [c.lstrip("?") for c in counters])
    _, meta = read_status(idea)
    if hours is not None:
        gpu.record(idea.relative_to(REPO).as_posix(), kernel, hours, meta.get("kind", "probe"))
    required = cfg["lines"].get(meta.get("line", idea.parent.name), {}).get("g4_required") or []
    bad = check_log(log_text, counters, cfg.get("gates", {}).get("g4_fail_markers", []), required)
    unfired = firing_check(cfg, idea, meta) if status == "complete" and not bad else []
    bad += unfired
    ok = status == "complete" and not bad
    result = "PASS" if ok else f"FAIL ({status}{'; ' + ' / '.join(bad) if bad else ''})"
    log_gate(idea, "G4", result, f"{kernel}; counters: {' / '.join(hits) or 'none matched'}")
    instrument = meta.get("kind") == "instrument"        # harness dumps: never submitted, never auto-fixed
    write_status(idea, ("instrument-ready" if instrument else "ready-to-submit") if ok else "g4-failed")
    card = idea / "submit_card.md"
    if card.exists():
        with card.open("a", encoding="utf-8") as f:
            f.write(f"- **G4 result ({time.strftime('%Y-%m-%d %H:%M')})**: {result}; "
                    f"counters: {' / '.join(hits) or 'none'}\n")
    print(f"[kaggle_push] {kernel} status={status} -> {result}", flush=True)
    for h in hits:
        print("  ", h)
    reasons = bad if status == "complete" else [f"kernel status {status}"] + bad
    no_ctrl = any(r.startswith("no control draft") for r in unfired)  # only a control push fixes it
    after_g4(cfg, idea, ok, kernel, log_text, reasons, fatal=instrument or no_ctrl)
    return 0 if ok else 2


# ---------------- hand-off after a verdict (tests replace these four hooks)

def toast(title: str, body: str) -> None:
    """Best-effort Windows toast through PowerShell's WinRT API. Never raises."""
    print(f"[toast] {title}: {body}", flush=True)
    if os.name != "nt":
        return
    ps = (r"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
          r"$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
          r"$n = $t.GetElementsByTagName('text');"
          r"$null = $n.Item(0).AppendChild($t.CreateTextNode($env:CASMI_TOAST_TITLE));"
          r"$null = $n.Item(1).AppendChild($t.CreateTextNode($env:CASMI_TOAST_BODY));"
          r"$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe';"
          r"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show("
          r"[Windows.UI.Notifications.ToastNotification]::new($t))")
    env = dict(os.environ, CASMI_TOAST_TITLE=title, CASMI_TOAST_BODY=body)
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                            base64.b64encode(ps.encode("utf-16-le")).decode()],
                           env=env, capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
        if r.returncode:
            print(f"[toast] failed rc={r.returncode}: {r.stderr.strip()[:300]}", flush=True)
    except Exception as e:  # noqa: BLE001 - a toast must never break G4
        print(f"[toast] failed: {e}", flush=True)


def _call_claude(args: list[str]) -> None:
    subprocess.run([sys.executable, str(Path(call_claude.__file__).resolve())] + args, cwd=REPO,
                   stdin=subprocess.DEVNULL, timeout=120, creationflags=NO_WINDOW)


def queue_ideas(rels: list[str]) -> None:
    """Append ideas to the Claude queue in one write, then start one detached drain (L34)."""
    cfg = load_config()
    q = call_claude.queue_read(cfg)
    call_claude.queue_write(cfg, q + [r for r in rels if r not in q])
    _call_claude(["--drain", "--detach"])


def queue_fix(rel: str) -> None:
    """A fresh headless /run-idea for one idea; the call_claude lock and queue serialize it."""
    _call_claude([rel, "--detach"])


def spawn_watcher(idea: Path) -> int:
    """Start `--watch <idea>` as a detached process that outlives this one and any Claude session."""
    wd = idea / "kaggle_watch"
    wd.mkdir(exist_ok=True)
    args = [sys.executable, "-u", str(Path(__file__).resolve()), "--watch",
            idea.relative_to(REPO).as_posix()]
    # NO_WINDOW, not DETACHED: the watcher owns a hidden console that its children (kaggle,
    # powershell, call_claude) inherit; a console-less parent makes Windows pop a window per child
    flags = NO_WINDOW | 0x00000200 if os.name == "nt" else 0  # NO_WINDOW | NEW_GROUP
    breakaway = 0x01000000 if os.name == "nt" else 0  # leave the caller's job object if allowed
    with open(wd / "watch.log", "a", encoding="utf-8") as log:
        kw = dict(cwd=REPO, close_fds=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            p = subprocess.Popen(args, creationflags=flags | breakaway, **kw)
        except OSError:  # the job forbids breakaway
            p = subprocess.Popen(args, creationflags=flags, **kw)
    print(f"[kaggle_push] watcher pid {p.pid}; heartbeat in {wd / 'watch.log'}")
    return p.pid


def waiting_probes(control: Path) -> list[str]:
    """Other ideas on the control's line whose STATUS is one call_claude would run."""
    out = []
    for st in sorted(control.parent.glob("*/STATUS")):
        state, meta = read_status(st.parent)
        if st.parent != control and state in call_claude.WAITING_STATES and meta.get("kind") != "control":
            out.append(st.parent.relative_to(REPO).as_posix())
    return out


def _context_blocks(lines: list[str], patterns: list[str], ctx: int = 20, cap: int = 300) -> list[str]:
    rxs = [re.compile(p) for p in patterns]
    keep = sorted({j for i, ln in enumerate(lines) if any(rx.search(ln) for rx in rxs)
                   for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1))})
    out, prev = [], None
    for j in keep:
        if prev is not None and j != prev + 1:
            out.append("...")
        out.append(lines[j])
        prev = j
    return out[:cap]


def build_fix_brief(cfg, idea: Path, kernel: str, log_text: str, reasons: list[str], round_n: int) -> Path:
    """A compact, self-contained brief for a fresh fix agent. Never reads a raw .ipynb (CLAUDE.md §3)."""
    _, meta = read_status(idea)
    line = meta.get("line", idea.parent.name)
    lcfg = cfg["lines"].get(line, {})
    control = meta.get("kind") == "control"
    rounds = cfg.get("gates", {}).get("g4_fix_rounds", 2)
    args_p = idea / "kaggle_push" / "push_args.json"
    pa = json.loads(args_p.read_text(encoding="utf-8")) if args_p.exists() else {}
    md = [f"# G4 fix brief: {idea.name}", "",
          f"- kernel: `{kernel}` (round {round_n} of {rounds}; log in `kaggle_output/`)",
          f"- kind: {meta.get('kind', 'probe')}; line: {line}; worktree: `{lcfg.get('worktree')}`"]
    if pa:
        cmd = ["python research/bin/kaggle_push.py", idea.relative_to(REPO).as_posix(), f'"{pa["notebook"]}"',
               "--preset", pa["preset"], "--slug", pa["slug"]]
        cmd += [f'--counter "{c}"' for c in pa.get("counters", [])]
        cmd += [f"--source {s}" for s in pa.get("extra_sources", [])]
        md += ["", "## Re-push command (no --wait; add `--source <owner/slug>` for a catalog dataset)",
               "```", " ".join(cmd), "```"]
    spec = idea / "spec.md"
    if not control and spec.exists():
        sl = spec.read_text(encoding="utf-8", errors="replace").splitlines()
        heads = [i for i, ln in enumerate(sl) if ln.startswith("## ")]
        end = heads[2] if len(heads) > 2 else len(sl)  # title + Problem Statement + Solution
        md += ["", "## The one change (spec.md, up to the 3rd section)",
               *[ln.replace("## ", "### ", 1) if ln.startswith("## ") else ln for ln in sl[:min(end, 40)]]]
    md += ["", "## Fail reasons", *[f"- {r}" for r in reasons or ["(none recorded)"]]]
    lines = log_text.splitlines()
    markers = cfg.get("gates", {}).get("g4_fail_markers", []) + ["Traceback"]
    ctx = _context_blocks(lines, markers)
    if ctx:
        md += ["", "## Marker context (±20 lines)", "```", *ctx, "```"]
    md += ["", "## Log tail (last 80 lines)", "```", *(lines[-80:] or ["(no log downloaded)"]), "```"]
    gates = idea / "gates.md"
    if gates.exists():
        md += ["", "## gates.md", gates.read_text(encoding="utf-8", errors="replace").strip()]
    base = REPO / str(lcfg.get("base_notebook") or "")
    pushed = Path(pa.get("notebook", ""))
    if not control and base.is_file() and pushed.is_file():
        try:
            r = subprocess.run([sys.executable, str(REPO / ".agents/skills/analyze-notebook/scripts/compare.py"),
                                str(base), str(pushed)], cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300, creationflags=NO_WINDOW)
            md += ["", "## compare.py base vs pushed", "```", *(r.stdout + r.stderr).splitlines()[:200], "```"]
        except Exception as e:  # noqa: BLE001
            md += ["", f"compare.py failed: {e}"]
    md += ["", "## Fix scope", FIX_SCOPE]
    path = idea / "g4_fix_brief.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def after_g4(cfg, idea: Path, ok: bool, kernel: str, log_text: str = "", reasons: list[str] = (),
             fatal: bool = False) -> None:
    """Hand a G4 verdict on: toast, queue probes after a control PASS, or start a fix round."""
    _, meta = read_status(idea)
    rel = idea.relative_to(REPO).as_posix()
    if ok:
        write_status(idea, None, g4_fixes=0)
        body = "ready to submit"
        if meta.get("kind") == "control":
            probes = waiting_probes(idea)
            if probes:
                queue_ideas(probes)
            log_gate(idea, "G4 queue", f"{len(probes)} probes", ", ".join(probes) or "none")
            body += f"; queued: {', '.join(p.rsplit('/', 1)[-1] for p in probes) or 'none'}"
        if meta.get("kind") not in ("control", "instrument"):
            try:
                import dossier
                dossier.write_queue()
                body += "; added to research/submit_queue.md"
            except Exception as e:  # noqa: BLE001 - the queue must never break G4
                print(f"[kaggle_push] dossier failed: {e}", flush=True)
        toast(f"{kernel} G4 PASS", body)
        return
    n = int(meta.get("g4_fixes", 0) or 0)
    rounds = cfg.get("gates", {}).get("g4_fix_rounds", 2)
    brief = build_fix_brief(cfg, idea, kernel, log_text, list(reasons), n + 1)
    brief_rel = brief.relative_to(REPO).as_posix()
    if fatal or n >= rounds:
        write_status(idea, "needs-human")
        why = "not fixable within the fix scope" if fatal else f"{n} automatic fix rounds used"
        with open(idea / "questions.md", "a", encoding="utf-8") as f:
            f.write(f"\n## G4 needs a human ({time.strftime('%Y-%m-%d %H:%M')})\n"
                    f"- kernel `{kernel}`: {why}.\n- reasons: {'; '.join(reasons) or 'see gates.md'}\n"
                    f"- brief: `{brief_rel}`\n")
        log_gate(idea, "G4", "NEEDS-HUMAN", f"{why}; {brief_rel}")
        toast(f"{kernel} G4 FAIL", f"needs a human: {why}")
        return
    write_status(idea, "g4-fix", g4_fixes=n + 1)
    log_gate(idea, "G4 fix", f"round {n + 1}", brief_rel)
    queue_fix(rel)
    toast(f"{kernel} G4 FAIL", f"fix agent round {n + 1} started")


# ---------------- the watcher

def _lock_path(idea: Path) -> Path:
    return idea / "kaggle_watch" / "watch.lock"


def watch_live(cfg, idea: Path, push_id: str, clock=time.time) -> bool:
    """True while another live watcher owns this push."""
    try:
        info = json.loads(_lock_path(idea).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    limit = cfg.get("kaggle", {}).get("watch_timeout_s", 3600) + 300
    pid = int(info.get("pid", -1))
    return (info.get("push_id", "") == push_id and pid != os.getpid()
            and clock() - info.get("t", 0) < limit and call_claude._pid_alive(pid))


def watch(cfg, idea: Path, run=kaggle, sleep=time.sleep, clock=time.time) -> int:
    """Poll one pushed kernel until it finishes, then finalize. Exits quietly when superseded."""
    state, meta = read_status(idea)
    kernel, push_id = meta.get("kernel"), meta.get("push_id", "")
    if state != "g4-pushed" or not kernel:
        print(f"[watch] nothing to watch: STATUS={state}", flush=True)
        return 0
    if watch_live(cfg, idea, push_id, clock):
        print(f"[watch] another watcher owns {kernel}; exiting", flush=True)
        return 0
    lock = _lock_path(idea)
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "t": clock(), "push_id": push_id}), encoding="utf-8")
    kcfg = cfg.get("kaggle", {})
    interval, timeout = kcfg.get("watch_interval_s", 30), kcfg.get("watch_timeout_s", 3600)
    t0 = clock()
    try:
        while True:
            state, meta = read_status(idea)
            if state != "g4-pushed" or meta.get("push_id", "") != push_id:
                print(f"[watch] superseded (STATUS={state}); exiting", flush=True)
                return 0
            try:
                r = run(["kernels", "status", kernel])
                status = parse_status(r.stdout) if r.returncode == 0 else "unknown"
            except Exception as e:  # noqa: BLE001 - a transient CLI failure is not a verdict
                print(f"[watch] status call failed: {e}", flush=True)
                status = "unknown"
            print(f"[watch] {kernel} {status} t+{int(clock() - t0)}s", flush=True)
            if status in DONE:
                return finalize(cfg, idea, kernel, status, run)
            if clock() - t0 >= timeout:
                write_status(idea, "g4-timeout")
                log_gate(idea, "G4", "TIMEOUT", f"{kernel} still {status} after {int(timeout)}s; "
                                                 "kaggle_push.py --check finalizes it later")
                toast(f"{kernel} G4 TIMEOUT", f"still {status} after {int(timeout // 60)} min")
                return 4
            sleep(interval)
    finally:
        try:
            if json.loads(lock.read_text(encoding="utf-8")).get("pid") == os.getpid():
                lock.unlink()
        except (OSError, ValueError):
            pass


def check(cfg, ideas: list[Path], run=kaggle) -> int:
    """One status poll per g4-pushed/g4-timeout idea; finalize the finished ones. No waiting, no Claude."""
    worst = 0
    for idea in ideas:
        state, meta = read_status(idea)
        if state not in ("g4-pushed", "g4-timeout") or not meta.get("kernel"):
            continue
        if watch_live(cfg, idea, meta.get("push_id", "")):
            print(f"[kaggle_push] {meta['kernel']}: a watcher owns it; skipping")
            continue
        status = parse_status(run(["kernels", "status", meta["kernel"]]).stdout)
        if status not in DONE:
            print(f"[kaggle_push] {meta['kernel']} still {status}")
            continue
        worst = max(worst, finalize(cfg, idea, meta["kernel"], status, run))
    return worst


def catalog_slugs(cat: dict) -> set[str]:
    return {d["slug"] for group in cat.get("datasets", {}).values() for d in group.values() if "slug" in d}


def main(argv=None, run=kaggle, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="?")
    ap.add_argument("notebook", nargs="?")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--wait", type=float, default=0, help="seconds to poll for completion (no watcher)")
    ap.add_argument("--counter", action="append", default=[])
    ap.add_argument("--source", action="append", default=[],
                    help="extra dataset source; must be a dataset slug in the catalog")
    ap.add_argument("--check", action="store_true",
                    help="finalize G4 for <idea>, or for every g4-pushed/g4-timeout idea when none is given")
    ap.add_argument("--watch", metavar="IDEA", help="poll one pushed idea until done (started by the push)")
    a = ap.parse_args(argv)

    cfg = load_config()
    if a.watch:
        return watch(cfg, idea_dir(a.watch), run, sleep)
    if a.check:
        ideas = [idea_dir(a.idea)] if a.idea else sorted(RESEARCH.glob("*/*/STATUS"))
        return check(cfg, [i if i.is_dir() else i.parent for i in ideas], run)
    if not (a.idea and a.notebook):
        ap.error("idea and notebook are required unless --check or --watch")
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
    if meta.get("kind") not in ("control", "instrument"):
        why = push_preconditions(cfg, idea, line)
        if why:
            log_gate(idea, "G4", "BLOCKED", why)
            print(f"[kaggle_push] refused: {why}")
            return 10
    kind = meta.get("kind", "probe")
    why = gpu.refuse_reason(cfg, kind, gpu.estimate(cfg, kind, meta))
    if why:
        log_gate(idea, "G4", "BLOCKED", why)
        with open(idea / "questions.md", "a", encoding="utf-8") as f:
            f.write(f"\n- **GPU pool** ({time.strftime('%Y-%m-%d %H:%M')}): {why}\n")
        print(f"[kaggle_push] refused: {why}")
        return 8
    cat = json.loads((REPO / cfg["catalog"]).read_text(encoding="utf-8"))
    unknown = [s for s in a.source if s not in catalog_slugs(cat)]
    if unknown:
        log_gate(idea, "G4", "BLOCKED", f"--source not in the catalog: {', '.join(unknown)}")
        print(f"[kaggle_push] refused: --source not in the catalog: {', '.join(unknown)}")
        return 5
    extra = [s for s in meta.get("extra_sources", "").split(",") if s]
    extra += [s for s in a.source if s not in extra]
    sources = cat["presets"][preset].get("dataset_sources", [])
    missing, unresolved = audit_sources(notebook_code(nb), cat, sources + extra)
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
    md = build_metadata(cfg, preset, slug, nb.name, extra)
    if img := control_image(cfg, line):
        md["docker_image"] = img
    (stage / "kernel-metadata.json").write_text(json.dumps(md, indent=2), encoding="utf-8")
    (stage / "counters.txt").write_text("\n".join(a.counter) + "\n", encoding="utf-8")
    (stage / "push_args.json").write_text(json.dumps(
        {"notebook": str(nb), "preset": preset, "slug": slug, "counters": a.counter,
         "extra_sources": extra}, indent=2), encoding="utf-8")
    print(f"[kaggle_push] {len(md['dataset_sources'])} dataset sources")

    r = run(["kernels", "push", "-p", str(stage)])
    if r.returncode != 0:
        log_gate(idea, "G4", "FAIL", f"push: {(r.stdout + r.stderr).strip()[-200:]}")
        print(r.stdout, r.stderr)
        return 1
    print(r.stdout.strip())
    if "successfully pushed" not in (r.stdout + r.stderr).lower():
        # the CLI exits 0 on e.g. "Kernel push error: Maximum batch GPU session count of 2 reached."
        msg = f"push: {(r.stdout + r.stderr).strip()[-200:]}"
        log_gate(idea, "G4", "FAIL", msg)
        write_status(idea, "g4-failed", kernel=md["id"], reason=msg[:150])
        print("[kaggle_push] push did not go through; no watcher started")
        return 1
    if "not valid" in r.stdout + r.stderr:  # Kaggle pushes anyway, silently dropping the bad sources
        msg = f"push dropped sources: {(r.stdout + r.stderr).strip()[:300]}"
        log_gate(idea, "G4", "FAIL", msg)
        write_status(idea, "g4-failed", kernel=md["id"])
        after_g4(cfg, idea, False, md["id"], reasons=[msg], fatal=True)  # only a catalog edit fixes it
        return 1
    ver = re.search(r"Kernel version (\d+) successfully pushed", r.stdout + r.stderr, re.I)
    write_status(idea, "g4-pushed", kernel=md["id"], push_id=f"{time.time():.6f}",
                 **({"kernel_version": ver.group(1)} if ver else {}),
                 **({"docker_image": md["docker_image"]} if md.get("docker_image") else {}),
                 **({"extra_sources": ",".join(extra)} if extra else {}))
    log_gate(idea, "G4", "PUSHED", md["id"] + (f"; unresolved refs: {', '.join(unresolved)}"
                                               if unresolved else ""))
    if not a.wait:
        spawn_watcher(idea)
        return 0

    deadline = time.time() + a.wait
    status = "unknown"
    while time.time() < deadline:
        status = parse_status(run(["kernels", "status", md["id"]]).stdout)
        print(f"[kaggle_push] {md['id']} {status}", flush=True)
        if status in DONE:
            break
        sleep(cfg.get("kaggle", {}).get("watch_interval_s", 30))
    if status not in DONE:
        print(f"[kaggle_push] {md['id']} still {status}; finish later with --check")
        return 0
    return finalize(cfg, idea, md["id"], status, run)


if __name__ == "__main__":
    raise SystemExit(main())
