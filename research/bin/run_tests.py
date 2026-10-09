"""Run the repo test suite inside a line's worktree (G2), never in the main checkout.

    python research/bin/run_tests.py exploit [--idea research/exploit/20261010-foo]

The headless allowlist only matches `sh build/run_all_tests.sh` relative to cwd, which is the
main checkout; that tested the wrong code and regenerated notebooks there (L29). This wrapper
sets cwd to the worktree. With --idea, the tail of the output is logged to gates.md as G2.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from _common import idea_dir, load_config, log_gate


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line", choices=["exploit", "explore"])
    ap.add_argument("--idea", default=None)
    a = ap.parse_args(argv)

    wt = Path(load_config()["lines"][a.line]["worktree"])
    if not (wt / "build" / "run_all_tests.sh").exists():
        print(f"[run_tests] no build/run_all_tests.sh in {wt}")
        return 2
    sh = shutil.which("sh") or shutil.which("bash") or "sh"
    r = run([sh, "build/run_all_tests.sh"], cwd=wt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-6000:])
    if a.idea:
        tail = " / ".join(ln.strip() for ln in out.strip().splitlines()[-3:])
        log_gate(idea_dir(a.idea), "G2 suite", "PASS" if r.returncode == 0 else "FAIL",
                 f"run_tests.py {a.line} (cwd {wt}): {tail[:300]}")
    print(f"[run_tests] {a.line} rc={r.returncode} cwd={wt}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
