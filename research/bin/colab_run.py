"""Run one script on a Colab T4 through the Colab CLI in WSL, and always stop the session.

    python research/bin/colab_run.py research/explore/20261010-foo spike/check.py --kind smoke \
        [--env KEY=VAL ...] [--fetch /content/out.json:out.json ...]

- The script path is relative to the idea folder (or absolute).
- Drive is mounted at /content/drive; scripts must not call drive.mount() (check
  os.path.exists('/content/drive/MyDrive') instead). Gotchas: specs/specs_scratch/archive/colab_spike_result.md.
- The exec timeout is capped so the run cannot exceed the CU cap left for this kind.
- stdout goes to <idea>/colab_<kind>.log; CU spent is added to STATUS cu_used and logged in gates.md.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

from _common import idea_dir, load_config, log_gate, read_status, win_to_wsl, write_status

PRELUDE = 'export OAUTHLIB_RELAX_TOKEN_SCOPE=1; export PATH="$HOME/.local/bin:$PATH"; '


def wsl(colab: str, args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    sh = PRELUDE + " ".join([colab] + [shlex.quote(a) for a in args])
    return subprocess.run(["wsl.exe", "--", "bash", "-lc", sh], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def budget_seconds(cfg, kind: str, cu_used_kind: float) -> float:
    c = cfg["colab"]
    cap = c["cu_cap_smoke"] if kind == "smoke" else c["cu_cap_val"]
    left = max(0.0, cap - cu_used_kind)
    return min(float(c["exec_timeout_s"]), left / c["cu_per_hour"] * 3600)


def main(argv=None, runner=wsl) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("script")
    ap.add_argument("--kind", choices=["smoke", "val", "spike"], required=True)
    ap.add_argument("--gate", default=None, help="gate label for gates.md (default by kind)")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--fetch", action="append", default=[], help="remote:local (local relative to idea)")
    a = ap.parse_args(argv)

    cfg = load_config()
    c = cfg["colab"]
    idea = idea_dir(a.idea)
    _, meta = read_status(idea)
    key = f"cu_{'smoke' if a.kind in ('smoke', 'spike') else 'val'}"
    used_kind = float(meta.get(key, 0) or 0)
    secs = budget_seconds(cfg, "smoke" if a.kind in ("smoke", "spike") else "val", used_kind)
    gate = a.gate or {"smoke": "G3a", "val": "G3b", "spike": "spike"}[a.kind]
    if secs < 60:
        log_gate(idea, gate, "BLOCKED", f"CU cap reached ({key}={used_kind})")
        print(f"[colab_run] CU cap reached for {key}; not starting")
        return 4

    script = Path(a.script)
    script = script if script.is_absolute() else idea / script
    session = ("casmi-" + idea.name)[:40]
    logf = idea / f"colab_{a.kind}.log"
    t0 = time.time()
    rc, result = 1, "FAIL"
    try:
        r = runner(c["cli"], ["new", "--gpu", c["gpu"], "-s", session], 600)
        if r.returncode != 0:
            raise RuntimeError(f"colab new failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        t0 = time.time()  # bill from runtime start
        r = runner(c["cli"], ["drivemount", "-s", session], 600)
        if r.returncode != 0:
            raise RuntimeError(f"drivemount failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        args = ["exec", "-s", session, "-f", win_to_wsl(script), "--timeout", f"{secs:.0f}"]
        for e in a.env:
            args += ["--env", e]
        r = runner(c["cli"], args, secs + 300)
        logf.write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""),
                        encoding="utf-8")
        for spec in a.fetch:
            remote, local = spec.split(":", 1)
            runner(c["cli"], ["download", "-s", session, remote, win_to_wsl(idea / local)], 600)
        rc = r.returncode
        result = "RAN" if rc == 0 else "FAIL"
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        logf.write_text(f"{logf.read_text() if logf.exists() else ''}\n[colab_run] {e}\n",
                        encoding="utf-8")
    finally:
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
