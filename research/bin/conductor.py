"""Conductor: a daemon that keeps the research campaigns moving without a human at the keyboard.

    python research/bin/conductor.py start            # detached daemon (refuses if one is alive)
    python research/bin/conductor.py run [--ticks N]  # same loop in the foreground
    python research/bin/conductor.py tick             # one tick, foreground
    python research/bin/conductor.py status | pause | resume | yield-claude [hours=4]

One tick, each step try/except-logged so a failure never kills the daemon:
  1 inbox    apply research/inbox/*.md decisions (pause/resume/kill/revive/steer/raise-cap/approve/note)
  2 workers  reap finished `campaign.py <c> --once --router` children; start new ones up to conductor.max_workers
  3 desks    `desk.py due` -> at most one `desk.py run <job>` when a slot is free (skipped if desk.py is missing)
  4 submit   `autosubmit.py --once` (skipped if missing; it decides everything itself)
  5 queue    dossier.write_queue() each tick; research/daily/<date>.md once a day after digest_hour
  6 toast    a new needs-human item pops a Windows toast
  7 commit   `git add research/` + commit, only if conductor.commit_state and the branch is conductor.state_branch

The conductor never deletes files, never resets/checks out/cleans git, never pushes, and never runs heavy compute
(Kaggle kernels are pushed by the build step). Config: research/config.yaml `conductor:`; campaigns and desks:
research/conductor/conductor.yaml. Runtime state: research/conductor/state.json, log: conductor.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, RESEARCH, load_config, read_status, write_status  # noqa: E402
import call_claude  # noqa: E402

BIN = Path(__file__).parent
NOOP_LIMIT = 3                      # consecutive --once runs that changed nothing
BACKOFF_S = 30 * 60
NEEDS_STATES = {"needs-human", "g4-timeout"}
ACTIONS = ("pause", "resume", "kill", "revive", "steer", "raise-cap", "approve", "note")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _cdir() -> Path:
    return RESEARCH / "conductor"


def stamp(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(t if t is not None else time.time()).strftime("%Y-%m-%d %H:%M")


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def cname(c) -> str:
    return Path(str(c)).name


def split_front(text: str) -> tuple[dict, str]:
    """Inbox file -> (front matter dict, body after the closing ---)."""
    m = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", text, re.S)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), m.group(2)


def join_front(fm: dict, body: str) -> str:
    return "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n" + body


def _default_runner(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=REPO, creationflags=NO_WINDOW, **kw)


def _default_toast(title: str, body: str) -> None:
    """Toast through kaggle_push.toast (PowerShell WinRT, never raises)."""
    try:
        import kaggle_push
        kaggle_push.toast(title, body)
    except Exception as e:  # noqa: BLE001
        print(f"[toast] failed: {e}", flush=True)


class Conductor:
    def __init__(self, cfg: dict | None = None, popen=subprocess.Popen, runner=None, clock=time.time,
                 toast=None, describe=None, router=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.c = self.cfg.get("conductor") or {}
        self.popen, self.clock = popen, clock
        self.runner = runner or _default_runner
        self.toast = toast or _default_toast
        self._describe = describe
        self.router = router
        self.workers: dict[str, dict] = {}          # key -> {proc, kind, ...}
        self.state = read_json(_cdir() / "state.json", {})
        for k, v in (("backoff", {}), ("noop", {}), ("seen_needs", []), ("log_lines", {})):
            self.state.setdefault(k, v)

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str) -> None:
        append(_cdir() / "conductor.log", f"{stamp(self.clock())} {msg}\n")
        print(f"[conductor] {msg}", flush=True)

    def save(self) -> None:
        _cdir().mkdir(parents=True, exist_ok=True)
        (_cdir() / "state.json").write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def settings(self) -> dict:
        return load_yaml(self.c_file())

    def c_file(self) -> Path:
        return REPO / self.c.get("campaigns_file", "research/conductor/conductor.yaml")

    def paused(self) -> bool:
        return bool(self.settings().get("paused"))

    def active_campaigns(self) -> list[Path]:
        return [REPO / p for p in (self.settings().get("campaigns") or [])]

    def max_workers(self) -> int:
        return int(self.c.get("max_workers", 2))

    def safe(self, name: str, fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001 - one failing step never kills the daemon
            self.log(f"ERROR in {name}: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------ 1. inbox
    def inbox(self) -> None:
        d = RESEARCH / "inbox"
        for f in sorted(d.glob("*.md")) if d.exists() else []:
            if f.name.lower() == "readme.md":
                continue
            text = f.read_text(encoding="utf-8")
            fm, body = split_front(text)
            if str(fm.get("status", "")).strip() != "pending":
                continue
            try:
                ok, result = self.apply_inbox(fm)
            except Exception as e:  # noqa: BLE001
                ok, result = False, f"error: {type(e).__name__}: {e}"
            fm["status"] = "applied" if ok else "rejected"
            fm["applied_at"] = stamp(self.clock())
            fm["result"] = result
            f.write_text(join_front(fm, body), encoding="utf-8")
            self.log(f"inbox {f.name}: {fm['status']} ({result})")
            if not ok:
                self.notify(f"inbox:{f.name}", "Conductor: inbox rejected", f"{f.name}: {result}"[:200])

    def apply_inbox(self, fm: dict) -> tuple[bool, str]:
        action = str(fm.get("action", "")).strip()
        camp = cname(fm.get("campaign") or "")
        idea = str(fm.get("idea") or "").strip()
        args = str(fm.get("args") or "").strip()
        reason = str(fm.get("reason") or "").strip()
        today = dt.date.fromtimestamp(self.clock()).isoformat()
        cdir = RESEARCH / "campaigns" / camp
        if action not in ACTIONS:
            return False, f"unknown action {action!r}"
        if action in ("pause", "resume"):
            if not camp:
                return False, "no campaign"
            s = self.settings()
            lst = [cname(x) for x in (s.get("paused_campaigns") or [])]
            if action == "pause" and camp not in lst:
                lst.append(camp)
            if action == "resume":
                lst = [x for x in lst if x != camp]
            s["paused_campaigns"] = lst
            dump_yaml(self.c_file(), s)
            return True, f"{camp} {action}d"
        if action == "kill":
            p = REPO / idea if idea else None
            if not p or not (p / "STATUS").exists():
                return False, f"idea not found: {idea!r}"
            write_status(p, "dead", reason=f"human (inbox): {reason}")
            return True, f"{idea} -> dead"
        if action == "revive":
            if not args:
                return False, "revive needs args (the fix)"
            src = REPO / idea if idea else None
            if not src or not src.exists():
                return False, f"idea not found: {idea!r}"
            return self.revive(src, args)
        if action == "steer":
            if not (cdir / "charter.md").exists():
                return False, f"no charter for {camp!r}"
            append(cdir / "charter.md", f"\n## Human steer ({today})\n{args}\n")
            return True, "charter appended"
        if action == "raise-cap":
            parts = args.split()
            if parts and parts[0] == "cap":
                parts = parts[1:]
            if len(parts) != 2 or not re.fullmatch(r"\d+", parts[1]):
                return False, "raise-cap args must be '<route> <n>'"
            path = _cdir() / "caps_override.yaml"
            caps = load_yaml(path)
            caps[parts[0]] = int(parts[1])
            dump_yaml(path, caps)
            self.log(f"TODO: Router does not read {path.name} yet; cap {parts[0]}={parts[1]} recorded only")
            return True, f"{parts[0]}={parts[1]} written to caps_override.yaml (Router does not read it yet)"
        if action == "approve":
            p = REPO / idea if idea else None
            if not p or not p.exists():
                return False, f"idea not found: {idea!r}"
            append(p / "questions.md", f"\n## Approved by human ({today})\n{reason or args}\n")
            return True, "questions.md appended"
        # note
        if not cdir.exists():
            return False, f"campaign not found: {camp!r}"
        append(cdir / "log.md", f"- {stamp(self.clock())} human note: {reason or args}\n")
        return True, "log.md appended"

    def revive(self, src: Path, fix: str) -> tuple[bool, str]:
        line = src.relative_to(REPO).parts[1]
        slug = re.sub(r"^\d{8}-", "", src.name) + "-rev"
        r = self.runner([sys.executable, str(BIN / "new_idea.py"), line, slug])
        if r.returncode:
            return False, f"new_idea failed: {(r.stderr or r.stdout).strip()[:200]}"
        new = REPO / r.stdout.strip().splitlines()[-1]
        cause = ""
        nec = src / "necropsy.md"
        if nec.exists():
            t = nec.read_text(encoding="utf-8")
            m = re.search(r"^.*cause.*$", t, re.I | re.M)
            cause = (m.group(0) if m else t[:300]).strip()
        rel = src.relative_to(REPO).as_posix()
        (new / "hypothesis.md").write_text(
            f"revives: {rel}\ncause: {cause}\nfix: {fix}\n", encoding="utf-8")
        return True, f"created {new.relative_to(REPO).as_posix()}"

    # ------------------------------------------------------------ 2. workers
    def lock_pid(self, camp: Path) -> int | None:
        info = read_json(camp / "campaign.lock", None)
        try:
            pid = int(info["pid"])
        except (TypeError, KeyError, ValueError):
            return None
        return pid if call_claude._pid_alive(pid) else None

    def describe(self, camp: Path) -> str:
        if self._describe:
            return self._describe(camp)
        import campaign
        return campaign.Campaign(camp, readonly=True).describe()

    def snap(self, camp: Path) -> tuple[str, str]:
        try:
            h = hashlib.md5((camp / "state.json").read_bytes()).hexdigest()
        except OSError:
            h = ""
        try:
            d = self.describe(camp)
        except Exception as e:  # noqa: BLE001
            d = f"describe failed: {type(e).__name__}"
        return h, d

    def reap(self) -> None:
        for key, w in list(self.workers.items()):
            rc = w["proc"].poll()
            if rc is None:
                continue
            del self.workers[key]
            self.log(f"worker {key} finished rc={rc}")
            if w["kind"] != "campaign" or rc == 6:
                continue
            camp = w["camp"]
            after = self.snap(camp)
            noop = self.state["noop"]
            if after == tuple(w["before"]):
                noop[key] = noop.get(key, 0) + 1
                if noop[key] >= NOOP_LIMIT:
                    self.state["backoff"][key] = self.clock() + BACKOFF_S
                    noop[key] = 0
                    self.log(f"backoff {key} 30 min (no change in {NOOP_LIMIT} runs)")
            else:
                noop[key] = 0
        self.save()

    def free_slots(self) -> int:
        return self.max_workers() - len(self.workers)

    def start_campaigns(self) -> None:
        s = self.settings()
        pausedc = {cname(x) for x in (s.get("paused_campaigns") or [])}
        for camp in self.active_campaigns():
            if self.free_slots() <= 0:
                break
            key = camp.name
            if key in pausedc or key in self.workers:
                continue
            if self.clock() < self.state["backoff"].get(key, 0):
                continue
            st = read_json(camp / "state.json", {})
            if st.get("stop"):
                continue
            if self.lock_pid(camp):
                continue
            before = self.snap(camp)
            logf = open(camp / "campaign_driver.log", "a", encoding="utf-8")
            try:
                proc = self.popen([sys.executable, str(BIN / "campaign.py"), str(camp),
                                   "--once", "--router"], cwd=REPO, stdout=logf, stderr=logf,
                                  stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            finally:
                logf.close()
            self.workers[key] = dict(proc=proc, kind="campaign", camp=camp, before=list(before))
            self.log(f"worker {key} started pid {getattr(proc, 'pid', '?')}: {before[1]}")

    def workers_step(self) -> None:
        self.reap()
        if not self.paused():
            self.start_campaigns()

    # ------------------------------------------------------------ 3. desks, 4. autosubmit
    def desks(self) -> None:
        script = BIN / "desk.py"
        if not script.exists() or self.paused() or self.free_slots() <= 0:
            return
        r = self.runner([sys.executable, str(script), "due"])
        try:
            jobs = json.loads(r.stdout or "[]")
        except ValueError:
            self.log(f"desk due: unparseable output {r.stdout[:100]!r}")
            return
        for job in jobs:
            jk = f"desk:{job}"
            if jk in self.workers:
                continue
            proc = self.popen([sys.executable, str(script), "run", str(job)], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            self.workers[jk] = dict(proc=proc, kind="desk")
            self.log(f"desk worker {job} started")
            break

    def autosubmit(self) -> None:
        script = BIN / "autosubmit.py"
        if not script.exists():
            return
        r = self.runner([sys.executable, str(script), "--once"])
        lines = (r.stdout or "").strip().splitlines()
        self.log(f"autosubmit rc={r.returncode}: {lines[-1] if lines else ''}")

    # ------------------------------------------------------------ 5. queue + digest
    def queue_digest(self) -> None:
        import dossier
        dossier.write_queue()
        now = dt.datetime.fromtimestamp(self.clock())
        today = now.date().isoformat()
        if now.hour >= int(self.c.get("digest_hour", 7)) and self.state.get("last_digest") != today:
            path = self.write_daily(today)
            self.state["last_digest"] = today
            self.save()
            self.log(f"digest {path.name}")

    def ledger_lines(self) -> list[str]:
        led = read_json(RESEARCH / "provider_ledger.json", {})
        today = dt.date.fromtimestamp(self.clock()).isoformat()
        out, paid = [], 0.0
        for route, d in sorted((led.get(today) or {}).items()):
            out.append(f"- {route}: calls {d.get('calls', 0)}, done {d.get('done', 0)}, paused {d.get('paused', 0)}, "
                       f"error {d.get('error', 0)}")
            paid += float(d.get("usd", 0))
        for route, until in sorted((led.get("paused_until") or {}).items()):
            h = (until - self.clock()) / 3600
            if h > 0:
                out.append(f"- paused: {route} for {h:.1f} h (resets {stamp(until)})")
        out.append(f"- paid usd today: {paid:.4f}")
        return out

    def autosubmit_today(self) -> int:
        n, today = 0, dt.datetime.now(dt.timezone.utc).date().isoformat()
        p = _cdir() / "autosubmit_log.jsonl"
        for ln in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
            try:
                n += str(json.loads(ln).get("time", "")).startswith(today)
            except ValueError:
                pass
        return n

    def needs_human(self) -> list[dict]:
        r = self.runner([sys.executable, str(BIN / "campaign_status.py"), "--json"])
        out = []
        for c in json.loads(r.stdout or "[]"):
            for n in c.get("needs_human") or []:
                out.append(dict(campaign=c["id"], **n))
        return out

    def write_daily(self, today: str) -> Path:
        o = [f"# Daily digest {today}", "", "## Providers", *self.ledger_lines(), "", "## Campaigns"]
        for camp in self.active_campaigns():
            key = camp.name
            o.append(f"### {key}")
            o.append(f"- next: {self.snap(camp)[1]}")
            lines = (camp / "log.md").read_text(encoding="utf-8").splitlines() if (camp / "log.md").exists() else []
            seen = self.state["log_lines"].get(key, 0)
            o += [f"  {ln}" for ln in lines[seen:][-30:]]
            self.state["log_lines"][key] = len(lines)
        o += ["", "## Submissions today", f"- auto-submitted: {self.autosubmit_today()}", "", "## Needs human"]
        nh = self.safe("needs_human", self.needs_human) or []
        o += [f"- {n['campaign']} {n['idea']} [{n['status']}] {n.get('reason') or ''}" for n in nh] or ["- none"]
        try:
            import autosubmit
            o += ["", "## Final-pick recommendation", str(autosubmit.final_pick_note())]
        except Exception:  # noqa: BLE001 - Phase 3 helper optional
            pass
        d = RESEARCH / "daily"
        d.mkdir(exist_ok=True)
        path = d / f"{today}.md"
        path.write_text("\n".join(o) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------ 6. toast
    def notify(self, key: str, title: str, body: str) -> None:
        seen = self.state["seen_needs"]
        if key in seen:
            return
        seen.append(key)
        del seen[:-500]
        self.save()
        self.toast(title, body)

    def toasts(self) -> None:
        for n in self.needs_human():
            if n["status"] in NEEDS_STATES:
                self.notify(f"idea:{n['idea']}:{n['status']}", "Conductor: needs human",
                            f"{n['idea']} is {n['status']}: {n.get('reason') or ''}"[:200])
        for camp in self.active_campaigns():
            why = read_json(camp / "state.json", {}).get("stop")
            if why:
                self.notify(f"stop:{camp.name}:{why}", "Conductor: campaign stopped", f"{camp.name}: {why}"[:200])

    # ------------------------------------------------------------ 7. commit
    def branch(self) -> str:
        return (self.runner(["git", "-C", str(REPO), "branch", "--show-current"]).stdout or "").strip()

    def commit_state(self) -> None:
        if not self.c.get("commit_state"):
            return
        if self.branch() != self.c.get("state_branch", "main"):
            return
        self.runner(["git", "-C", str(REPO), "add", "research/"])
        if self.runner(["git", "-C", str(REPO), "diff", "--cached", "--quiet"]).returncode == 0:
            return
        self.runner(["git", "-C", str(REPO), "commit", "-m", f"conductor: tick {stamp(self.clock())}"])

    # ------------------------------------------------------------ the loop
    def tick(self) -> None:
        for name, fn in (("inbox", self.inbox), ("workers", self.workers_step), ("desks", self.desks),
                         ("autosubmit", self.autosubmit), ("queue", self.queue_digest), ("toast", self.toasts),
                         ("commit", self.commit_state)):
            self.safe(name, fn)
        self.state["last_tick"] = self.clock()
        self.save()

    def run(self, ticks: int | None = None, sleep=time.sleep) -> int:
        lock = call_claude.Lock(_cdir() / "conductor.lock")
        if not lock.acquire(f"conductor pid {os.getpid()}"):
            print("conductor: refused, another daemon holds conductor.lock")
            return 6
        try:
            n = 0
            while ticks is None or n < ticks:
                self.tick()
                n += 1
                if ticks is None or n < ticks:
                    sleep(float(self.c.get("tick_s", 300)))
        finally:
            lock.release()
        return 0


# ---------------------------------------------------------------- CLI
def daemon_pid() -> int | None:
    info = read_json(_cdir() / "conductor.lock", None)
    try:
        pid = int(info["pid"])
    except (TypeError, KeyError, ValueError):
        return None
    return pid if call_claude._pid_alive(pid) else None


def set_paused(flag: bool) -> None:
    c = Conductor()
    s = c.settings()
    s["paused"] = flag
    dump_yaml(c.c_file(), s)
    print(f"conductor: paused={flag}")


def status() -> int:
    c = Conductor()
    s = c.settings()
    pid = daemon_pid()
    lt = c.state.get("last_tick")
    print(f"daemon: {'alive pid ' + str(pid) if pid else 'not running'}")
    print(f"paused: {bool(s.get('paused'))}  paused_campaigns: {s.get('paused_campaigns') or []}")
    print(f"last tick: {stamp(lt) if lt else 'never'}")
    print("campaigns:")
    for camp in c.active_campaigns():
        try:
            nxt = c.describe(camp)
        except Exception as e:  # noqa: BLE001
            nxt = f"describe failed: {type(e).__name__}: {e}"
        wp = c.lock_pid(camp)
        bo = c.state["backoff"].get(camp.name, 0)
        extra = f" backoff until {stamp(bo)}" if bo > c.clock() else ""
        print(f"  {camp.name}: next {nxt}; worker {('pid ' + str(wp)) if wp else 'none'}{extra}")
    print("providers today:")
    for ln in c.ledger_lines():
        print(f"  {ln}")
    print(f"autosubmit today: {c.autosubmit_today()}")
    inbox = RESEARCH / "inbox"
    pend = [f.name for f in sorted(inbox.glob("*.md")) if f.name.lower() != "readme.md"
            and str(split_front(f.read_text(encoding="utf-8"))[0].get("status", "")) == "pending"] if inbox.exists() else []
    print(f"inbox pending: {len(pend)} {pend}")
    lg = _cdir() / "conductor.log"
    print("log tail:")
    for ln in (lg.read_text(encoding="utf-8").splitlines()[-5:] if lg.exists() else ["(no log yet)"]):
        print(f"  {ln}")
    return 0


def start() -> int:
    pid = daemon_pid()
    if pid:
        print(f"conductor: already running (pid {pid})")
        return 6
    flags = 0x08000000 | 0x00000200 if os.name == "nt" else 0       # NO_WINDOW | NEW_PROCESS_GROUP
    _cdir().mkdir(parents=True, exist_ok=True)
    log = open(_cdir() / "conductor_daemon.log", "a")
    p = subprocess.Popen([sys.executable, __file__, "run"], cwd=REPO, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    print(f"conductor: started pid {p.pid}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "run", "status", "pause", "resume", "yield-claude", "tick"])
    ap.add_argument("hours", nargs="?", type=float, default=4.0)
    ap.add_argument("--ticks", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "start":
        return start()
    if a.cmd == "run":
        return Conductor().run(ticks=a.ticks)
    if a.cmd == "tick":
        Conductor().tick()
        return 0
    if a.cmd == "status":
        return status()
    if a.cmd in ("pause", "resume"):
        set_paused(a.cmd == "pause")
        return 0
    import providers
    providers.Router(load_config(), RESEARCH / "provider_ledger.json").yield_claude(a.hours * 3600)
    print(f"conductor: Claude yielded for {a.hours:g} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
