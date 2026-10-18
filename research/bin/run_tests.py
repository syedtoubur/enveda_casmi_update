"""Run a line's test suite inside its worktree (G2), never in the main checkout, and keep going past failures.

    python research/bin/run_tests.py exploit [--idea research/exploit/20261010-foo] [--legacy]

The test list comes from the worktree's `build/run_all_tests.sh` (every `run` / `soft` / `python <test>.py` line);
each test runs on its own with cwd = the worktree (L29), plus the repo's harness tests. Each test ends as:
- PASS;
- SKIP-INPUT: it died only because an input file or module is absent from this checkout (FileNotFoundError,
  ModuleNotFoundError, "No such file"); listed by name, never hidden (loop audit W8);
- FAIL: anything else.
G2 is PASS iff no test FAILs. With --idea the verdict and the failing/skipped names go to gates.md, and kaggle_push
refuses a probe whose newest G2 row is a FAIL (N6). `--legacy` runs the old stop-at-first-failure shell script.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from _common import REPO, idea_dir, load_config, log_gate

TEST_LINE = re.compile(r"^\s*(?:run|soft|python)\s+((?:build|scratch_pkg|research)/\S+\.py)([^|#\n]*)", re.M)
MISSING = re.compile(r"FileNotFoundError|ModuleNotFoundError|No such file or directory|cannot find the (?:file|path)")
REPO_TESTS = ["research/harness/test_harness.py", "research/harness/test_suite.py", "research/harness/test_metrics.py",
              "research/bin/test_campaign.py", "research/bin/test_providers.py",
              "research/bin/test_conductor.py", "research/bin/test_autosubmit.py", "research/bin/test_desk.py",
              "research/bin/test_necropsy.py", "research/bin/test_campaign_status.py"]


def tests_of(script: str) -> list[list[str]]:
    out = []
    for m in TEST_LINE.finditer(script):
        args = [x for x in m.group(2).split() if not x.startswith("$")]
        if [m.group(1)] + args not in out:
            out.append([m.group(1)] + args)
    return out


def classify(rc: int, out: str) -> str:
    if rc == 0:
        return "PASS"
    tail = "\n".join(out.strip().splitlines()[-15:])
    return "SKIP-INPUT" if MISSING.search(tail) else "FAIL"


def keep_going(wt: Path, run=subprocess.run, timeout: float = 1200) -> tuple[bool, list[tuple[str, str, str]]]:
    script = (wt / "build" / "run_all_tests.sh").read_text(encoding="utf-8", errors="replace")
    plan = [(wt, t) for t in tests_of(script) if (wt / t[0]).exists()]
    plan += [(REPO, [t]) for t in REPO_TESTS if (REPO / t).exists()]
    res = []
    for cwd, t in plan:
        try:
            r = run([sys.executable] + t, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=timeout)
            out = (r.stdout or "") + (r.stderr or "")
            res.append((" ".join(t), classify(r.returncode, out), out.strip().splitlines()[-1][:160] if out.strip() else ""))
        except subprocess.TimeoutExpired:
            res.append((" ".join(t), "FAIL", f"timeout {timeout:.0f}s"))
    return not any(v == "FAIL" for _, v, _ in res), res


def legacy(wt: Path, run=subprocess.run):
    sh = shutil.which("sh") or shutil.which("bash") or next(
        (c for c in (r"C:\Program Files\Git\bin\sh.exe", r"C:\Program Files\Git\bin\bash.exe") if Path(c).exists()), "sh")
    return run([sh, "build/run_all_tests.sh"], cwd=wt, capture_output=True, text=True,
               encoding="utf-8", errors="replace", timeout=3600)


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line", choices=sorted(load_config()["lines"]))
    ap.add_argument("--idea", default=None)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args(argv)

    wt = Path(load_config()["lines"][a.line]["worktree"])
    if not (wt / "build" / "run_all_tests.sh").exists():
        print(f"[run_tests] no build/run_all_tests.sh in {wt}")
        return 2
    if a.legacy:
        r = legacy(wt, run)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-6000:])
        ok, detail = r.returncode == 0, " / ".join(ln.strip() for ln in out.strip().splitlines()[-3:])[:300]
    else:
        ok, res = keep_going(wt, run)
        for name, v, last in res:
            print(f"{v:10} {name}  {last}")
        fails = [n for n, v, _ in res if v == "FAIL"]
        skips = [n for n, v, _ in res if v == "SKIP-INPUT"]
        detail = (f"{sum(v == 'PASS' for _, v, _ in res)}/{len(res)} pass; FAIL: {', '.join(fails) or 'none'}; "
                  f"SKIP-INPUT: {', '.join(skips) or 'none'}")[:400]
        print(detail)
    if a.idea:
        log_gate(idea_dir(a.idea), "G2 suite", "PASS" if ok else "FAIL", f"run_tests.py {a.line} (cwd {wt}): {detail}")
    print(f"[run_tests] {a.line} {'PASS' if ok else 'FAIL'} cwd={wt}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
