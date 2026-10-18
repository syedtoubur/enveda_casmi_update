"""Multi-provider model routing for campaign steps (phase 1 of the conductor).

    Router(cfg, ledger_path).run(step, prompt, log, worktree) -> (result, route)

Each step goes to the first available route of its tier's ordered list (config `routing:`), across four CLIs:
`claude` (call_claude.run_prompt), `agy`, `qoder`, `opencode`. A route that is paused, rate-limited, capped,
unhealthy, in the wrong model family or out of slots is skipped immediately. Adapters share one contract:

    run(prompt, model, log, worktree=None, web=False, paid=False) -> "done" | "paused" | "error"

`done` only means rc 0; whether a step progressed is decided by campaign.progressed() (STATUS / artefacts).
No adapter blocks on a busy slot and none ever sets ANTHROPIC_API_KEY (subscription / free-tier CLIs only).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO  # noqa: E402
import call_claude  # noqa: E402

QUOTA_RE = re.compile(r"usage limit|limit reached|rate[-_ ]?limit|429|quota|exhausted|hit your limit|"
                      r"temporarily rate-limited", re.I)
LOGIN_RE = re.compile(r"not logged in", re.I)
OPENCODE_TEMPLATE = REPO / "research" / "conductor" / "opencode.json"
FREE_OPENROUTER = {"openrouter/openrouter/free"}

PROMPT_TEMPLATE = (
    "You are a headless research agent in the repo E:/ENVEDA_casmi. Do NOT run claude, agy, qodercli, opencode, "
    "campaign.py, call_claude.py or conductor.py. Your procedure is the file .claude/skills/{step}/SKILL.md (read it "
    "first; where it names Opus/Sonnet/Haiku or says headless, that is you). Arguments: {args}. Read the brief at "
    "{brief} and only the files it lists. Write the artifacts the procedure names and set STATUS as it says.\n"
    "Heavy compute goes to Kaggle kernels, never this PC. Finish with one line: STEP-RESULT: <new STATUS or 'none'>.")


def translate(prompt: str) -> str:
    """Campaign prompts are Claude slash-commands (`/spec <idea> --brief <path>`); other CLIs get a plain procedure."""
    m = re.match(r"^/(\S+)\s*(.*)$", prompt.strip(), re.S)
    if not m:
        return prompt
    step, args = m.group(1), m.group(2).strip()
    bm = re.search(r"--brief\s+(\S+)", args)
    return PROMPT_TEMPLATE.format(step=step, args=args, brief=bm.group(1) if bm else "(none given)")


RESET_RE = re.compile(r"resets? in\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", re.I)
UNAVAILABLE_RE = re.compile(r"is not available right now; using", re.I)


def reset_seconds(text: str) -> float | None:
    """'Resets in 166h40m34s' (agy quota) -> seconds; None when the provider gave no reset time."""
    for m in RESET_RE.finditer(text):
        h, mi, se = (int(x) if x else 0 for x in m.groups())
        if h or mi or se:
            return float(h * 3600 + mi * 60 + se)
    return None


def is_paid_model(model: str) -> bool:
    """OpenRouter ids that are neither `:free` nor `openrouter/free` cost money."""
    return model.startswith("openrouter/") and not model.endswith(":free") and model not in FREE_OPENROUTER


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def _json_or_none(line: str):
    try:
        return json.loads(line)
    except ValueError:
        return None


def _error_text(lines: list[str]) -> str:
    """Text of what looks like an error: non-JSON lines, plus JSON events/results flagged as failures.
    Agent prose inside a successful JSON result never counts (a plan that mentions 'quota' is not a quota hit)."""
    out = []
    for ln in lines:
        obj = _json_or_none(ln)
        if not isinstance(obj, (dict, list)):
            out.append(ln)
        elif isinstance(obj, dict):
            status = str(obj.get("status", "")).upper()
            if (obj.get("type") == "error" or obj.get("is_error") or obj.get("error")
                    or (status and status != "SUCCESS")):
                out.append(json.dumps(obj))
    return "\n".join(out)


class Adapter:
    provider = "?"

    def __init__(self, cfg: dict, popen=subprocess.Popen):
        self.cfg = cfg
        self.rcfg = cfg.get("routing") or {}
        self.popen = popen
        self.timeout_s = float(self.rcfg.get("timeout_s", 2400))
        self.note = ""                      # why the last call paused/failed (ledger note), e.g. "login"
        self.pause_for = None               # seconds the provider said to wait ("Resets in 166h40m"), if any
        self.credits = 0.0                  # provider credits the last call reported (qodercli `total_credits`)

    # -- to override
    def build_cmd(self, prompt: str, model: str, worktree: Path | None) -> list[str]:
        raise NotImplementedError

    def classify(self, rc: int, lines: list[str]) -> str:
        raise NotImplementedError

    def prepare(self, model: str, worktree: Path | None, log: Path, paid: bool) -> str | None:
        """Pre-flight; a string (reason) means refuse with 'error' without launching."""
        return None

    # -- shared
    def run(self, prompt: str, model: str, log: Path, worktree: Path | None = None, web: bool = False,
            paid: bool = False) -> str:
        self.note = ""
        self.pause_for = None
        self.credits = 0.0
        refuse = self.prepare(model, worktree, log, paid)
        if refuse:
            self.note = "refused"
            self._log(log, {"type": "provider", "provider": self.provider, "refused": refuse})
            return "error"
        cmd = self.build_cmd(translate(prompt), model, worktree)
        lines, rc, timed_out = self._exec(cmd, log, model)
        if timed_out:
            return "error"
        result = self.classify(rc, lines)
        if result == "paused":
            self.pause_for = reset_seconds("\n".join(lines))
        return result

    @staticmethod
    def _log(log: Path, event: dict) -> None:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _exec(self, cmd: list[str], log: Path, model: str) -> tuple[list[str], int, bool]:
        lines: list[str] = []
        timed_out = False
        real = list(cmd)
        real[0] = shutil.which(cmd[0]) or cmd[0]
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "provider", "provider": self.provider, "model": model,
                                "start": dt.datetime.now().isoformat(), "cmd": cmd}) + "\n")
            try:
                p = self.popen(real, cwd=REPO, env=_clean_env(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            except OSError as e:
                f.write(json.dumps({"type": "provider", "provider": self.provider, "end": dt.datetime.now().isoformat(),
                                    "rc": -1, "error": str(e)}) + "\n")
                return [str(e)], -1, True
            fired: list[int] = []
            timer = threading.Timer(self.timeout_s, lambda: (fired.append(1), p.kill()))
            timer.daemon = True
            timer.start()
            try:
                for ln in p.stdout:
                    f.write(ln)
                    lines.append(ln.rstrip("\n"))
                rc = p.wait()
            finally:
                timer.cancel()
            timed_out = bool(fired)
            f.write(json.dumps({"type": "provider", "provider": self.provider, "end": dt.datetime.now().isoformat(),
                                "rc": rc, **({"timeout_s": self.timeout_s} if timed_out else {})}) + "\n")
        return lines, rc, timed_out


class ClaudeCLI(Adapter):
    provider = "claude"

    def run(self, prompt, model, log, worktree=None, web=False, paid=False):
        self.note = ""
        settings = self.cfg["claude"].get("web_settings") if web else None
        return call_claude.run_prompt(self.cfg, prompt, log, model, worktree, settings)   # verbatim slash-command


class AgyCLI(Adapter):
    provider = "agy"

    def build_cmd(self, prompt, model, worktree):
        exe = (self.rcfg.get("bin") or {}).get("agy", "agy")
        cmd = [exe, "-p", prompt, "--model", model, "--output-format", "json",
               "--print-timeout", f"{int(self.timeout_s)}s"]
        if worktree is not None:
            cmd += ["--add-dir", str(worktree)]
        return cmd

    def classify(self, rc, lines):
        text = "\n".join(lines)
        obj = _json_or_none(text)
        if obj is None:                                     # JSON may be the last line, after log noise
            obj = next((o for o in (_json_or_none(l) for l in reversed(lines)) if isinstance(o, dict)), None)
        if isinstance(obj, dict):
            if str(obj.get("status", "")).upper() != "SUCCESS" and QUOTA_RE.search(json.dumps(obj)):
                return "paused"
            return "error" if rc else "done"
        if QUOTA_RE.search(text):
            return "paused"
        return "error" if rc else "done"


class QoderCLI(Adapter):
    provider = "qoder"

    def build_cmd(self, prompt, model, worktree):
        exe = (self.rcfg.get("bin") or {}).get("qoder", "qodercli")
        cmd = [exe, "-p", prompt, "-m", model, "-o", "json", "--permission-mode", "dont_ask",
               "--no-session-persistence"]
        if worktree is not None:
            cmd += ["-w", str(worktree)]
        return cmd

    def classify(self, rc, lines):
        for ln in lines:                    # result event: {"type":"result",...,"total_credits":0.25}
            obj = _json_or_none(ln)
            if isinstance(obj, dict) and obj.get("type") == "result":
                self.credits += float(obj.get("total_credits") or 0)
        err = _error_text(lines)
        if LOGIN_RE.search(err):
            self.note = "login"
            return "paused"
        if QUOTA_RE.search(err):
            return "paused"
        if any(UNAVAILABLE_RE.search(ln) for ln in lines):     # silently fell back to the credit-costly "auto"
            self.note = "model-unavailable"
            return "error"
        return "error" if rc else "done"


class OpencodeCLI(Adapter):
    provider = "opencode"

    def build_cmd(self, prompt, model, worktree):
        # NEVER --auto: `opencode run` executes shell without asking; the permission file below is the only fence
        return ["opencode", "run", "--pure", "--format", "json", "-m", model, "--dir",
                str(worktree or REPO), prompt]

    def prepare(self, model, worktree, log, paid):
        if is_paid_model(model) and not paid:
            return f"paid model {model} refused without paid=True"
        target = Path(worktree or REPO) / "opencode.json"
        want = json.loads(OPENCODE_TEMPLATE.read_text(encoding="utf-8"))
        if target.exists():
            try:
                have = json.loads(target.read_text(encoding="utf-8"))
            except ValueError:
                have = None
            if not isinstance(have, dict) or have.get("_conductor") is not True:
                return f"foreign {target} (no _conductor marker); not overwriting"
            if have == want:
                return None
        target.write_text(json.dumps(want, indent=2) + "\n", encoding="utf-8")
        return None

    def classify(self, rc, lines):
        if QUOTA_RE.search(_error_text(lines)):
            return "paused"
        return "error" if rc else "done"


ADAPTERS = {"claude": ClaudeCLI, "agy": AgyCLI, "qoder": QoderCLI, "opencode": OpencodeCLI}


# --------------------------------------------------------------------------------------------- router
def split_route(route: str) -> tuple[str, str]:
    prov, _, model = route.partition(":")
    return prov, model


class Router:
    def __init__(self, cfg: dict, ledger_path: Path, clock=time.time, adapters: dict | None = None,
                 lock_dir: Path | None = None, claude_lock: Path | None = None):
        self.cfg = cfg
        self.r = cfg.get("routing") or {}
        self.ledger_path = Path(ledger_path)
        self.clock = clock
        self.adapters = adapters if adapters is not None else {p: c(cfg) for p, c in ADAPTERS.items()}
        self.lock_dir = Path(lock_dir) if lock_dir else REPO / "research" / "queue"
        self.claude_lock = Path(claude_lock) if claude_lock else REPO / (cfg.get("claude") or {}).get(
            "lock_file", "research/queue/claude.lock")
        self.pause_s = float(self.r.get("pause_s", 1800))
        self.last_note = ""

    # ------------------------------------------------------------------ ledger
    def _today(self) -> str:
        return dt.datetime.fromtimestamp(self.clock()).strftime("%Y-%m-%d")

    def ledger(self) -> dict:
        try:
            led = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            led = {}
        led.setdefault("paused_until", {})
        led.setdefault("unhealthy", {})
        led.setdefault("yield_claude_until", 0)
        return led

    def _save(self, led: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(led, indent=1), encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    def _day(self, led: dict, route: str) -> dict:
        d = led.setdefault(self._today(), {}).setdefault(route, {})
        for k in ("calls", "done", "paused", "error"):
            d.setdefault(k, 0)
        d.setdefault("usd", 0.0)
        d.setdefault("credits", 0.0)
        return d

    def _record(self, route: str, result: str, usd: float = 0.0, note: str = "", pause_for: float | None = None,
                credits: float = 0.0) -> None:
        led = self.ledger()
        d = self._day(led, route)
        d[result] = d.get(result, 0) + 1
        d["usd"] = round(d["usd"] + usd, 6)
        d["credits"] = round(d["credits"] + credits, 4)
        if note:
            d["note"] = note
        if result == "paused":
            led["paused_until"][route] = self.clock() + max(self.pause_s, pause_for or 0)
        self._save(led)

    def _count_call(self, route: str) -> None:
        led = self.ledger()
        self._day(led, route)["calls"] += 1
        self._save(led)

    def mark_unhealthy(self, route: str) -> None:
        led = self.ledger()
        lst = led["unhealthy"].setdefault(self._today(), [])
        if route not in lst:
            lst.append(route)
        self._save(led)

    def yield_claude(self, seconds: float) -> None:
        led = self.ledger()
        led["yield_claude_until"] = self.clock() + seconds
        self._save(led)

    # ------------------------------------------------------------------ routes
    @staticmethod
    def _step_key(step: str) -> str:
        s = step.split()[0] if step else step
        return "build" if s.startswith("build-") else s

    def tier(self, step: str) -> str | None:
        return (self.r.get("steps") or {}).get(self._step_key(step))

    def routes(self, step: str, exclude_family: str | None = None) -> list[str]:
        t = self.tier(step)
        rs = list((self.r.get("tiers") or {}).get(t) or [])
        return [x for x in rs if not (exclude_family and self.family(x) == exclude_family)]

    def family(self, route: str) -> str:
        fam = self.r.get("families") or {}
        prov, _ = split_route(route)
        return fam.get(route) or fam.get(prov) or prov

    def is_web_step(self, step: str) -> bool:
        return self._step_key(step) in (self.r.get("web_steps") or [])

    def _skippable(self, route: str, led: dict, skip) -> bool:
        prov, model = split_route(route)
        if route in skip or model in skip:
            return True
        cap = (self.r.get("caps_day") or {}).get(route)
        if cap is not None and (led.get(self._today(), {}).get(route) or {}).get("calls", 0) >= int(cap):
            return True
        if led["paused_until"].get(route, 0) > self.clock():
            return True
        budget = (self.r.get("credits_day") or {}).get(prov)      # e.g. qoder: 90 of the 100 free daily credits
        if budget is not None and model not in (self.r.get("credit_free") or []):
            spent = sum((v or {}).get("credits", 0) for k, v in led.get(self._today(), {}).items()
                        if isinstance(v, dict) and k.startswith(prov + ":"))
            if spent >= float(budget):
                return True
        if route in led["unhealthy"].get(self._today(), []):
            return True
        if prov == "claude" and led["yield_claude_until"] > self.clock():
            return True
        return False

    def _paid_ok(self, step: str, led: dict, exclude_family, skip) -> str | None:
        plr = self.r.get("paid_last_resort") or {}
        route = plr.get("route")
        if not route or self.tier(step) not in (plr.get("tiers") or []):
            return None
        if exclude_family and self.family(route) == exclude_family:
            return None
        if self._skippable(route, led, skip):
            return None
        today = (led.get(self._today(), {}).get(route) or {}).get("usd", 0.0)
        total = sum((day.get(route) or {}).get("usd", 0.0) for k, day in led.items()
                    if isinstance(day, dict) and re.match(r"\d{4}-\d\d-\d\d$", k))
        if today < float(plr.get("usd_day", 0)) and total < float(plr.get("usd_total", 0)):
            return route
        return None

    def candidates(self, step: str, exclude_family: str | None = None, skip=()) -> list[str]:
        led = self.ledger()
        out = [r for r in self.routes(step, exclude_family) if not self._skippable(r, led, skip)]
        paid = self._paid_ok(step, led, exclude_family, skip)
        if paid and paid not in out:
            out.append(paid)
        return out

    def pick(self, step: str, exclude_family: str | None = None, skip=()) -> str | None:
        c = self.candidates(step, exclude_family, skip)
        return c[0] if c else None

    # ------------------------------------------------------------------ locks
    def _slot_lock(self, prov: str):
        """Acquire a free slot for the provider, else None (never blocks)."""
        if prov == "claude":
            lk = call_claude.Lock(self.claude_lock)
            return lk if lk.acquire("router claude") else None
        for n in range(int((self.r.get("slots") or {}).get(prov, 1))):
            lk = call_claude.Lock(self.lock_dir / f"{prov}.{n}.lock")
            if lk.acquire(f"router {prov}"):
                return lk
        return None

    # ------------------------------------------------------------------ run
    def run(self, step: str, prompt: str, log: Path, worktree: Path | None = None,
            exclude_family: str | None = None, skip=()) -> tuple[str, str | None]:
        tried: set[str] = set()
        web = self.is_web_step(step)
        normal = set(self.routes(step))
        plr = self.r.get("paid_last_resort") or {}
        while True:
            route = next((c for c in self.candidates(step, exclude_family, skip) if c not in tried), None)
            if route is None:
                return "paused", None
            tried.add(route)
            prov, model = split_route(route)
            adapter = self.adapters.get(prov)
            if adapter is None:
                continue
            lock = self._slot_lock(prov)
            if lock is None:
                continue                                    # all slots busy: next provider, never wait
            paid = route not in normal
            self._count_call(route)
            try:
                kw = dict(web=web)
                if paid:
                    kw["paid"] = True
                result = adapter.run(prompt, model, log, worktree, **kw)
            finally:
                lock.release()
            note = getattr(adapter, "note", "") or ""
            self._record(route, result, usd=float(plr.get("usd_per_call_est", 0)) if paid else 0.0, note=note,
                         pause_for=getattr(adapter, "pause_for", None), credits=getattr(adapter, "credits", 0.0))
            self.last_note = note
            if result == "paused":
                continue                                    # paused_until set by _record; next route immediately
            if result == "error":
                self._maybe_unhealthy(route, note)
            return result, route

    def _maybe_unhealthy(self, route: str, note: str) -> None:
        if note in ("refused", "model-unavailable"):
            self.mark_unhealthy(route)
            return
        d = (self.ledger().get(self._today()) or {}).get(route) or {}
        if d.get("error", 0) >= int(self.r.get("unhealthy_errors", 5)) and d.get("done", 0) == 0:
            self.mark_unhealthy(route)
