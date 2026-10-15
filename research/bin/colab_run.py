"""Run one script on a Colab T4 through the Colab CLI in WSL, and always stop the session.

    python research/bin/colab_run.py research/explore/20261010-foo spike/check.py --kind smoke \
        [--env KEY=VAL ...] [--fetch /content/out.json:out.json ...]

- The script path is relative to the idea folder (or absolute).
- Drive is mounted at /content/drive; scripts must not call drive.mount() (check
  os.path.exists('/content/drive/MyDrive') instead). Gotchas: specs/specs_scratch/archive/colab_spike_result.md.
- The exec timeout is capped so the run cannot exceed the CU cap left for this kind.
- stdout goes to <idea>/colab_<kind>.log; CU spent is added to STATUS cu_used and logged in gates.md.
- The script must exist locally before a session starts. After drivemount a probe waits up to
  MOUNT_WAIT_S for /content/drive/MyDrive, remounting once. A run whose output holds a Python
  error counts as FAIL even when the CLI exits 0. After `gates.colab_infra_retries` + 1 FAIL
  rows for a gate, colab_run refuses (rc 4): fall back to the Kaggle draft run.
- Every script gets CASMI_RESUME_DIR=<drive_root>/runs/<idea>/<kind>: checkpoint there (with RNG state) and resume
  from it, because a session can die mid-run (L27). `--fetch` files are pulled even when the run fails.
- `--kind train` (grill 2026-10-13) runs only after the human approved the idea's training_request.md (STATUS
  `colab_train: approved`) and confirmed Drive (`python research/bin/colab_run.py --check-mount`).

    python research/bin/colab_run.py --check-login     # free: token holds the full Drive scope?
    python research/bin/colab_run.py --check-mount     # live: start a session, mount Drive, probe, stop (~0.05 CU)
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from _common import idea_dir, load_config, log_gate, read_status, win_to_wsl, write_status

MOUNT_WAIT_S = 120
MOUNT_PROBE = f"""import os, time
for _ in range({MOUNT_WAIT_S // 5}):
    if os.path.isdir('/content/drive/MyDrive'):
        print('MOUNT_OK'); break
    time.sleep(5)
else:
    print('MOUNT_MISSING', os.listdir('/content') if os.path.isdir('/content') else None)
"""
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ERROR_RX = re.compile(r"Traceback \(most recent call last\)|^\s*\w*(?:Error|Exception)\b:", re.M)


def has_error(text: str) -> bool:
    return bool(ERROR_RX.search(ANSI.sub("", text)))


def gate_fails(idea: Path, gate: str) -> int:
    g = idea / "gates.md"
    return sum(1 for ln in g.read_text(encoding="utf-8").splitlines()
               if ln.startswith(f"| {gate} |") and "| FAIL" in ln) if g.exists() else 0


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
LOGIN_CMD = ('wsl -- bash -lc "mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.json.bak; '
             'OAUTHLIB_RELAX_TOKEN_SCOPE=1 ~/.local/bin/colab sessions"  (tick every Drive permission)')


def login_problem(shell=subprocess.run) -> str:
    """'' if the CLI token holds the full Drive scope, else why not. Free: reads a local file, no session."""
    r = shell(["wsl.exe", "--", "bash", "-lc", "cat ~/.config/colab-cli/token.json"],
              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    try:
        scopes = json.loads(r.stdout).get("scopes") or []
    except ValueError:
        return "not logged in (no readable ~/.config/colab-cli/token.json)"
    return "" if DRIVE_SCOPE in scopes else "token lacks the full Drive scope (has: " + ", ".join(
        x.rsplit("/", 1)[-1] for x in scopes) + ")"


def ask_login(idea: Path, gate: str, why: str) -> None:
    log_gate(idea, gate, "BLOCKED", f"colab login: {why}; no session started")
    with (idea / "questions.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- **Colab login needed** ({gate}, {time.strftime('%Y-%m-%d %H:%M')}): {why}. "
                f"Run `! {LOGIN_CMD}`, then re-queue this idea. Meanwhile it falls back to the Kaggle draft run.\n")


PRELUDE = 'export OAUTHLIB_RELAX_TOKEN_SCOPE=1; export PATH="$HOME/.local/bin:$PATH"; '


def wsl(colab: str, args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    sh = PRELUDE + " ".join([colab] + [shlex.quote(a) for a in args])
    return subprocess.run(["wsl.exe", "--", "bash", "-lc", sh], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def budget_seconds(cfg, kind: str, cu_used_kind: float) -> float:
    c = cfg["colab"]
    cap = {"smoke": c["cu_cap_smoke"], "val": c["cu_cap_val"], "train": c.get("cu_cap_train", 0.0)}[kind]
    left = max(0.0, cap - cu_used_kind)
    return min(float(c["exec_timeout_s"]), left / c["cu_per_hour"] * 3600)


def mount(c: dict, session: str, probe: Path, runner) -> None:
    """drivemount + probe, remounting once; raises RuntimeError with the likely cause."""
    for attempt in (1, 2):
        r = runner(c["cli"], ["drivemount", "-s", session], 600)
        if r.returncode != 0 or has_error(r.stdout + "\n" + r.stderr):
            raise RuntimeError(
                f"drivemount failed: {ANSI.sub('', r.stderr)[-300:]} -- a Google 400 page plus "
                "'mount failed' means the CLI token lacks the full Drive scope (only drive.file); "
                "re-login granting full Drive access (see research/README.md, Colab login)")
        p = runner(c["cli"], ["exec", "-s", session, "-f", win_to_wsl(probe),
                              "--timeout", str(MOUNT_WAIT_S + 30)], MOUNT_WAIT_S + 120)
        if "MOUNT_OK" in p.stdout:
            return
    raise RuntimeError(f"Drive not mounted after 2 drivemounts. drivemount said: "
                       f"{r.stdout[-300:]}{r.stderr[-300:]} probe said: {p.stdout[-300:]}")


def check_mount(cfg, runner=wsl, shell=subprocess.run) -> int:
    """Live Drive check after a re-login: 0 = mounted, 7 = login needed, 8 = mount failed."""
    import _common
    why = login_problem(shell)
    if why:
        print(f"[colab_run] LOGIN NEEDED: {why}\n  run: {LOGIN_CMD}")
        return 7
    c, session = cfg["colab"], "casmi-mount-check"
    probe = _common.RESEARCH / "queue" / "_mount_probe.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(MOUNT_PROBE, encoding="utf-8")
    try:
        r = runner(c["cli"], ["new", "--gpu", c["gpu"], "-s", session], 600)
        if r.returncode != 0:
            raise RuntimeError(f"colab new failed: {r.stdout[-300:]}{r.stderr[-300:]}")
        mount(c, session, probe, runner)
        print("[colab_run] MOUNT OK: Colab sessions see /content/drive/MyDrive")
        return 0
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[colab_run] MOUNT FAILED: {e}")
        return 8
    finally:
        runner(c["cli"], ["stop", "-s", session], 300)


def main(argv=None, runner=wsl, shell=subprocess.run) -> int:
    argv0 = argv if argv is not None else sys.argv[1:]
    if argv0 == ["--check-login"]:
        why = login_problem(shell)
        print("[colab_run] login OK" if not why else f"[colab_run] LOGIN NEEDED: {why}\n  run: {LOGIN_CMD}")
        return 0 if not why else 7
    if argv0 == ["--check-mount"]:
        return check_mount(load_config(), runner, shell)
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("script")
    ap.add_argument("--kind", choices=["smoke", "val", "spike", "train"], required=True)
    ap.add_argument("--gate", default=None, help="gate label for gates.md (default by kind)")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--fetch", action="append", default=[], help="remote:local (local relative to idea)")
    a = ap.parse_args(argv)

    cfg = load_config()
    c = cfg["colab"]
    idea = idea_dir(a.idea)
    _, meta = read_status(idea)
    bucket = {"smoke": "smoke", "spike": "smoke", "val": "val", "train": "train"}[a.kind]
    key = f"cu_{bucket}"
    used_kind = float(meta.get(key, 0) or 0)
    secs = budget_seconds(cfg, bucket, used_kind)
    gate = a.gate or {"smoke": "G3a", "val": "G3b", "spike": "spike", "train": "train"}[a.kind]
    if a.kind == "train" and meta.get("colab_train") != "approved":
        log_gate(idea, gate, "BLOCKED", "Colab training needs the human's approval of training_request.md "
                                        "(STATUS colab_train: approved); no session started")
        print("[colab_run] training not approved (write training_request.md; the human sets colab_train: approved)")
        return 9
    if secs < 60:
        log_gate(idea, gate, "BLOCKED", f"CU cap reached ({key}={used_kind})")
        print(f"[colab_run] CU cap reached for {key}; not starting")
        return 4

    script = Path(a.script)
    script = script if script.is_absolute() else idea / script
    if not script.exists():
        log_gate(idea, gate, "FAIL", f"script not found: {script} (no session started)", "0")
        print(f"[colab_run] script not found: {script}")
        return 1
    if gate_fails(idea, gate) > int(cfg.get("gates", {}).get("colab_infra_retries", 1)):
        log_gate(idea, gate, "BLOCKED", "Colab retries used up; fall back to the Kaggle draft run")
        print("[colab_run] retries used up for", gate)
        return 4
    if meta.get("colab") == "off":  # the human declined Colab for this cycle in the grill
        log_gate(idea, gate, "BLOCKED", "colab: off in STATUS; no session started")
        print("[colab_run] colab: off for this idea")
        return 7
    why = login_problem(shell)
    if why:
        ask_login(idea, gate, why)
        print(f"[colab_run] LOGIN NEEDED: {why}")
        return 7
    session = ("casmi-" + idea.name)[:40]
    logf = idea / f"colab_{a.kind}.log"
    t0 = time.time()
    rc, result, started = 1, "FAIL", False
    try:
        r = runner(c["cli"], ["new", "--gpu", c["gpu"], "-s", session], 600)
        if r.returncode != 0:
            raise RuntimeError(f"colab new failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        t0 = time.time()  # bill from runtime start
        probe = idea / "colab" / "_mount_probe.py"
        probe.parent.mkdir(exist_ok=True)
        probe.write_text(MOUNT_PROBE, encoding="utf-8")
        mount(c, session, probe, runner)
        started = True
        resume = f"{c['drive_root']}/runs/{idea.name}/{a.kind}"
        args = ["exec", "-s", session, "-f", win_to_wsl(script), "--timeout", f"{secs:.0f}",
                "--env", f"CASMI_RESUME_DIR={resume}"]
        for e in a.env:
            args += ["--env", e]
        r = runner(c["cli"], args, secs + 300)
        logf.write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""),
                        encoding="utf-8")
        rc = r.returncode
        if rc == 0 and has_error(r.stdout + "\n" + r.stderr):
            rc = 3  # the CLI exits 0 even when the cell raised
        result = "RAN" if rc == 0 else "FAIL"
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        logf.write_text(f"{logf.read_text() if logf.exists() else ''}\n[colab_run] {e}\n",
                        encoding="utf-8")
    finally:
        if started:                    # pull outputs even after a failed or timed-out run (they die with the session)
            for spec in a.fetch:
                remote, local = spec.split(":", 1)
                try:
                    runner(c["cli"], ["download", "-s", session, remote, win_to_wsl(idea / local)], 600)
                except Exception as e:  # noqa: BLE001
                    print(f"[colab_run] fetch {remote} failed: {e}")
        runner(c["cli"], ["stop", "-s", session], 300)
        cu = (time.time() - t0) / 3600 * c["cu_per_hour"]
        _, meta = read_status(idea)
        write_status(idea, None,
                     cu_used=f"{float(meta.get('cu_used', 0) or 0) + cu:.2f}",
                     **{key: f"{used_kind + cu:.2f}"})
        log_gate(idea, gate, result, f"{logf.name} (session {session} stopped)", f"{cu:.2f}")
    print(f"[colab_run] {result} rc={rc} log={logf}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
