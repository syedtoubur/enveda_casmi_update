"""Hand an idea to headless Claude Code, one Claude job at a time.

    python research/bin/call_claude.py research/exploit/20261010-foo [--detach] [--dry-run]

- If another Claude job holds the lock, the idea is appended to the queue and this exits 0.
- When the job ends, the next queued idea runs in the same process (drain loop).
- If Claude stops on the Pro usage limit, STATUS becomes `paused-quota` and the idea stays
  at the head of the queue; re-running this script later (or `--drain`) resumes it.
- Claude runs on the subscription (no API key): never set ANTHROPIC_API_KEY for this.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from _common import REPO, idea_dir, load_config, read_status, write_status

STALE_LOCK_S = 12 * 3600
LIMIT_RE = re.compile(r"usage limit|limit reached|rate[_ ]limit|hit your limit", re.I)
WAITING_STATES = {"ready-for-judge", "approved", "implementing", "g1", "g2", "g3a", "g3b", "g4",
                  "paused-quota", "judging", "g4-fix"}


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would TerminateProcess on Windows
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Lock:
    def __init__(self, path: Path):
        self.path = path

    def acquire(self, owner: str) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    self.path.unlink(missing_ok=True)
                    continue
                return False
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "idea": owner, "t": time.time()}, f)
            return True
        return False

    def _stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return True
        return (not _pid_alive(int(info.get("pid", -1)))) or time.time() - info.get("t", 0) > STALE_LOCK_S

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def _queue(cfg) -> Path:
    return REPO / cfg["claude"]["queue_file"]


def queue_read(cfg) -> list[str]:
    q = _queue(cfg)
    return [l.strip() for l in q.read_text().splitlines() if l.strip()] if q.exists() else []


def queue_write(cfg, items: list[str]) -> None:
    _queue(cfg).parent.mkdir(parents=True, exist_ok=True)
    _queue(cfg).write_text("".join(i + "\n" for i in items))


def build_cmd(cfg, rel: str, worktree: Path | None, prompt: str | None = None,
              model: str | None = None, settings: str | None = None) -> list[str]:
    """Headless, allowlisted Claude command. Default prompt: /run-idea <rel>; campaign.py passes its own
    prompt (/replan, /spec, /attack, /build, /scout, ...), the step's model and, for web steps, the web settings."""
    cmd = ["claude", "-p", prompt or f"/run-idea {rel}",
           "--output-format", "stream-json", "--verbose",
           "--permission-mode", "dontAsk", "--permission-prompts", "none",
           "--settings", str(REPO / (settings or cfg["claude"]["settings"])),
           "--strict-mcp-config"]
    if worktree is not None:
        cmd += ["--add-dir", str(worktree)]
    if model:
        cmd += ["--model", model]
    return cmd


def run_prompt(cfg, prompt: str, log: Path, model: str | None = None, worktree: Path | None = None,
               settings: str | None = None) -> str:
    """Run one headless Claude prompt, streaming to `log`.
    Returns 'done' | 'paused' (usage limit) | 'error' (non-zero exit; campaign.py counts it as no progress)."""
    cmd = build_cmd(cfg, "", worktree, prompt=prompt, model=model, settings=settings)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    hit = False
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "call_claude", "start": dt.datetime.now().isoformat(), "cmd": cmd}) + "\n")
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for ln in p.stdout:
            f.write(ln)
            hit |= is_limit_line(ln)
        rc = p.wait()
        f.write(json.dumps({"type": "call_claude", "end": dt.datetime.now().isoformat(), "rc": rc}) + "\n")
    return "paused" if hit else ("error" if rc else "done")


def is_limit_line(ln: str) -> bool:
    """True for a failed `result` event (or non-JSON stderr) that mentions the usage limit.
    Tool output that merely contains the words never counts."""
    try:
        obj = json.loads(ln)
    except ValueError:
        return bool(LIMIT_RE.search(ln))
    if not isinstance(obj, dict) or obj.get("type") != "result":
        return False
    failed = obj.get("is_error") or obj.get("subtype") not in (None, "success")
    return bool(failed and LIMIT_RE.search(json.dumps(obj)))


def run_one(cfg, rel: str, dry_run: bool = False) -> str:
    """Run Claude on one idea. Returns 'done' | 'paused' | 'skipped'."""
    idea = idea_dir(rel)
    state, meta = read_status(idea)
    if state not in WAITING_STATES:
        print(f"[call_claude] skip {rel}: STATUS={state}")
        return "skipped"
    line = meta.get("line", idea.parent.name)
    wt = Path(cfg["lines"][line]["worktree"])
    cmd = build_cmd(cfg, rel, wt if wt.exists() else None)
    if dry_run:
        print(" ".join(cmd))
        return "done"
    if state == "paused-quota":
        state = meta.get("resume_from", "ready-for-judge")
        write_status(idea, state)
    if state == "ready-for-judge":
        write_status(idea, "judging")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    log = idea / "claude_run.jsonl"
    hit_limit = False
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "call_claude", "start": dt.datetime.now().isoformat(),
                            "cmd": cmd}) + "\n")
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for ln in p.stdout:
            f.write(ln)
            hit_limit |= is_limit_line(ln)
        rc = p.wait()
        f.write(json.dumps({"type": "call_claude", "end": dt.datetime.now().isoformat(),
                            "rc": rc}) + "\n")
    if hit_limit:
        write_status(idea, "paused-quota", resume_from=read_status(idea)[0])
        print(f"[call_claude] usage limit hit; {rel} paused")
        return "paused"
    print(f"[call_claude] {rel} finished rc={rc} STATUS={read_status(idea)[0]}")
    return "done"


def drain(cfg, first: str | None, dry_run: bool = False) -> int:
    lock = Lock(REPO / cfg["claude"]["lock_file"])
    q = queue_read(cfg)
    if first and first not in q:
        q.append(first)
        queue_write(cfg, q)
    if not lock.acquire(first or "drain"):
        print(f"[call_claude] Claude busy; queued ({len(q)} waiting)")
        return 0
    try:
        while True:
            q = queue_read(cfg)
            if not q:
                return 0
            outcome = run_one(cfg, q[0], dry_run)
            if outcome == "paused":
                return 3  # leave it at the head; resume later
            queue_write(cfg, [i for i in queue_read(cfg) if i != q[0]])
    finally:
        lock.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="?", help="research/<line>/<idea>; omit with --drain")
    ap.add_argument("--drain", action="store_true", help="only process the queue")
    ap.add_argument("--detach", action="store_true", help="run in a detached background process")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    rel = None
    if a.idea:
        rel = idea_dir(a.idea).relative_to(REPO).as_posix()
    elif not a.drain:
        ap.error("idea or --drain required")
    if a.detach:
        args = [sys.executable, __file__] + ([rel] if rel else ["--drain"])
        # NO_WINDOW | NEW_GROUP: a hidden console that claude and its shells inherit (DETACHED
        # would make Windows pop a console window for every child process)
        flags = 0x08000000 | 0x00000200 if os.name == "nt" else 0
        subprocess.Popen(args, cwd=REPO, creationflags=flags, close_fds=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[call_claude] detached: {rel or 'drain'}")
        return 0
    return drain(cfg, rel, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
