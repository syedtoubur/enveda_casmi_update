"""Drive one research campaign: a deterministic state machine that calls one headless model step at a time.

    python research/bin/campaign.py research/campaigns/<id> [--detach] [--once]
    python research/bin/campaign.py research/campaigns/<id> --dry-run        # print the next transition; writes nothing
    python research/bin/campaign.py research/campaigns/<id> --override <idea> --reason "<why>"
    python research/bin/campaign.py research/campaigns/<id> --resume         # clear a stop (cap, quota, stall)

The campaign-grill (interactive, Opus) writes <campaign>/charter.md. From there no human is needed:

  no current idea      -> stop rules -> /replan (Opus)          -> new idea folder, STATUS proposed (or stop)
  proposed | spec-fix  -> /spec (Opus)                          -> specced
  specced              -> preflight.py (code)                   -> preflight-ok | spec-fix (once) | dead
  preflight-ok         -> /attack (Opus)                        -> attacked
  attacked             -> /build <idea> harness (Sonnet)        -> harness-done | dead
  harness-done         -> gate.py (code, rank-only)             -> build | dead
  build                -> /build <idea> ship (Sonnet)           -> g4-pushed (the G4 watcher and poll_lb take over)
  dead | g4-pushed ... -> cycle closed: counters, branch streak, current = none, digest + submit queue (dossier.py)
  state.json `ideate`  -> /scout (Haiku, web; if research/scout/ is stale) -> /diagnose (Sonnet) -> /ideate (Opus)
                          -> /replan again; at most campaign.max_ideation_rounds per campaign
  open research/requests/*.md (any time) -> /fetch (Haiku, web): one fact or one model search per request
  after every closed cycle -> /report (Sonnet): lesson / rule drafts in research/proposals/ (promote.py applies the
                          ones the human approves)

Each model step gets a generated brief (brief.py) and nothing else. The LB is asynchronous: a pushed probe does not
block the next cycle; its later STATUS and LB score flow back into state.json history on every step (replan reads them).

Safety (loop audit 2026-10-12):
- one driver per campaign (`<campaign>/campaign.lock`), and every model call holds the global Claude lock
  (`claude.lock_file`) so it never races a G4 fix agent in the same worktree (W14);
- a step that leaves STATUS unchanged (or exits non-zero) is retried at most `campaign.step_retries` times, then the
  idea goes to needs-human (or, for /replan, the campaign stops) (LOOP);
- model calls are capped per model per campaign (`campaign.call_caps`); hitting a cap stops the campaign (`--resume`
  after raising the cap);
- `--dry-run` reads only (W15).
Stop rules (config campaign.*, overridable in charter `## Stop rules` as `key: value`): max_cycles, max_lb_probes
(probes pushed to G4), dead_streak (consecutive harness deaths on one branch close that branch), quota_pause_h.
On stop the campaign writes report.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, load_config, read_status, write_status  # noqa: E402
import brief as brief_mod  # noqa: E402
import call_claude  # noqa: E402
import dossier  # noqa: E402
import gate  # noqa: E402
import preflight  # noqa: E402
import providers  # noqa: E402

DEFAULTS = dict(max_cycles=5, max_lb_probes=4, dead_streak=2, quota_pause_h=12)
MODELS = dict(replan="opus", spec="opus", attack="opus", build="sonnet",
              scout="haiku", fetch="haiku", diagnose="sonnet", ideate="opus", report="sonnet",
              necropsy="sonnet", **{"necropsy-synth": "opus"})
NECROPSY_PER_ROUND = 5                # dead ideas autopsied per ideation round in a `Mode: necropsy` campaign
WEB_STEPS = {"scout", "fetch"}        # run with claude.web_settings (WebSearch / WebFetch allowed)
TERMINAL = {"dead", "g4-pushed", "g4-failed", "g4-fix", "needs-human", "ready-to-submit", "instrument-ready",
            "scored", "scored-unpaired", "keep-pending", "keep", "revert", "submitted"}
PUSHED = {"g4-pushed", "g4-fix", "ready-to-submit", "scored", "scored-unpaired", "keep-pending", "keep", "revert",
          "submitted"}
SUBMITTED = {"submitted", "scored", "scored-unpaired", "keep-pending", "keep", "revert"}
GLOBAL_LOCK_WAIT_S = 4 * 3600


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


class Busy(Exception):
    """Another driver owns this campaign."""


class Campaign:
    def __init__(self, path: Path, runner=None, sleep=time.sleep, readonly: bool = False,
                 router=None):
        self.dir = path.resolve()
        self.router = router            # providers.Router: per-step multi-provider routing (--router)
        self.cfg = load_config()
        self.runner = runner or self._claude
        self.sleep = sleep
        self.readonly = readonly
        sf = self.dir / "state.json"
        self.state = json.loads(sf.read_text()) if sf.exists() else dict(
            cycle=0, lb_probes=0, branch_streak={}, closed_branches=[], current=None, stop=None, history=[])
        self.state.setdefault("calls", {})
        self.state.setdefault("stalls", {})
        self.state.setdefault("ideation_rounds", 0)
        ccfg = self.cfg.get("campaign") or {}
        self.limits = dict(DEFAULTS, **{k: v for k, v in ccfg.items() if k in DEFAULTS})
        charter = (self.dir / "charter.md").read_text(encoding="utf-8")
        for k in DEFAULTS:
            m = re.search(rf"^\s*-?\s*{k}\s*:\s*(\d+)", brief_mod.section(charter, "Stop rules"), re.M)
            if m:
                self.limits[k] = int(m.group(1))
        m = re.search(r"^\s*-?\s*Line\s*:\s*([\w-]+)", charter, re.M | re.I)
        self.line = m.group(1).lower() if m and m.group(1).lower() in self.cfg["lines"] else "exploit"
        self.retries = int(ccfg.get("step_retries", 2))
        self.caps = dict(ccfg.get("call_caps") or {})
        self.models = dict(MODELS, **((self.cfg.get("claude") or {}).get("models") or {}))
        self.web_settings = (self.cfg.get("claude") or {}).get("web_settings")
        self.max_ideation = int(ccfg.get("max_ideation_rounds", 2))
        self.scout_max_age_d = float(ccfg.get("scout_max_age_days", 7))
        if not sf.exists() and not readonly:
            self.save()                     # /replan reads it

    # ---------------------------------------------------------------- plumbing
    def save(self):
        if not self.readonly:
            (self.dir / "state.json").write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def log(self, msg: str):
        if not self.readonly:
            with open(self.dir / "log.md", "a", encoding="utf-8") as f:
                f.write(f"- {now()} {msg}\n")
        print(f"[campaign] {msg}", flush=True)

    def worktree(self, idea: Path | None) -> Path | None:
        line = read_status(idea)[1].get("line", self.line) if idea and (idea / "STATUS").exists() else self.line
        wt = Path(self.cfg["lines"].get(line, {}).get("worktree", ""))
        return wt if str(wt) and wt.exists() else None

    def _claude(self, prompt: str, model: str, log: Path, idea: Path | None = None,
                settings: str | None = None) -> str:
        """The real runner: waits for the global Claude lock (G4 fix agents, /run-idea drains) first."""
        lock = call_claude.Lock(REPO / self.cfg["claude"]["lock_file"])
        t0 = time.time()
        while not lock.acquire(f"campaign {self.dir.name}"):
            if time.time() - t0 > GLOBAL_LOCK_WAIT_S:
                return "error"
            self.log("Claude lock busy (another agent); retry in 2 min")
            self.sleep(120)
        try:
            return call_claude.run_prompt(self.cfg, prompt, log, model=model, worktree=self.worktree(idea),
                                          settings=settings)
        finally:
            lock.release()

    def call(self, step: str, prompt: str, idea: Path | None) -> str:
        """Brief + one model step; a usage-limit pause is retried until quota_pause_h, then the campaign stops."""
        bstep = step if step in brief_mod.STEPS else step.split()[0]
        if self.router is not None:
            return self.call_routed(bstep, prompt, idea)
        model = self.models.get(bstep) or self.models[bstep.split("-")[0]]
        cap = self.caps.get(model)
        if cap is not None and self.state["calls"].get(model, 0) >= int(cap):
            self.state["stop"] = f"call cap: {model} {cap} calls used (raise campaign.call_caps.{model}, then --resume)"
            self.save()
            return "stopped"
        path, n = brief_mod.brief(self.dir, bstep, idea)
        self.log(f"{bstep} ({model}) brief {n} chars{'  OVER 6000' if n > 6000 else ''}")
        t0 = time.time()
        while True:
            self.state["calls"][model] = self.state["calls"].get(model, 0) + 1
            self.save()
            args = (f"{prompt} --brief {path.relative_to(REPO).as_posix()}", model,
                    (idea or self.dir) / "claude_run.jsonl")
            if self.runner == self._claude:
                out = self._claude(*args, idea=idea, settings=self.web_settings if bstep in WEB_STEPS else None)
            else:
                out = self.runner(*args)
            if out != "paused":
                return out
            if time.time() - t0 > self.limits["quota_pause_h"] * 3600:
                self.state["stop"] = f"Claude usage limit for > {self.limits['quota_pause_h']} h"
                self.save()
                return "stopped"
            self.log("usage limit: retry in 30 min")
            self.sleep(1800)

    def call_routed(self, bstep: str, prompt: str, idea: Path | None) -> str:
        """call() through providers.Router: first available route of the step's tier; attack is cross-family."""
        routes = self.router.routes(bstep)
        capped = {r for r in routes if self.caps.get(r) is not None
                  and self.state["calls"].get(r, 0) >= int(self.caps[r])}
        if routes and len(capped) == len(routes):
            self.state["stop"] = f"call cap: every route of {bstep} is at its cap (raise campaign.call_caps, then --resume)"
            self.save()
            return "stopped"
        exclude = None
        if bstep == "attack" and idea is not None and (idea / "STATUS").exists():
            exclude = read_status(idea)[1].get("spec_by") or None
        path, n = brief_mod.brief(self.dir, bstep, idea)
        self.log(f"{bstep} (router) brief {n} chars{'  OVER 6000' if n > 6000 else ''}")
        t0 = time.time()
        while True:
            out, route = self.router.run(bstep, f"{prompt} --brief {path.relative_to(REPO).as_posix()}",
                                         (idea or self.dir) / "claude_run.jsonl", self.worktree(idea),
                                         exclude_family=exclude, skip=capped)
            if route is not None:
                self.state["calls"][route] = self.state["calls"].get(route, 0) + 1
                self.save()
                self.log(f"{bstep} -> {route}: {out}")
            if out == "done" and bstep == "spec" and idea is not None and route is not None:
                write_status(idea, None, spec_by=self.router.family(route))   # attack must come from another family
            if out != "paused":
                return out
            if time.time() - t0 > self.limits["quota_pause_h"] * 3600:
                self.state["stop"] = f"no model route available for > {self.limits['quota_pause_h']} h"
                self.save()
                return "stopped"
            self.log("every route paused: retry in 30 min")
            self.sleep(1800)

    def progressed(self, key: str, before: str, after: str, out: str) -> bool:
        """Count a model step that changed nothing (or failed); True while it may still be retried."""
        if out != "error" and after != before:
            self.state["stalls"].pop(key, None)
            self.save()
            return True
        n = self.state["stalls"][key] = self.state["stalls"].get(key, 0) + 1
        self.save()
        self.log(f"{key}: no progress ({'exit error' if out == 'error' else 'STATUS still ' + after}), "
                 f"attempt {n}/{self.retries}")
        return n < self.retries

    # ---------------------------------------------------------------- steps
    def stop_reason(self) -> str | None:
        s, L = self.state, self.limits
        if s.get("stop"):
            return s["stop"]
        if s["cycle"] >= L["max_cycles"]:
            return f"max_cycles {L['max_cycles']} reached"
        if s["lb_probes"] >= L["max_lb_probes"]:
            return f"max_lb_probes {L['max_lb_probes']} reached"
        return None

    def refresh_history(self):
        """Closed cycles keep moving (G4, submit, LB): copy each idea's current STATUS and LB back into history."""
        changed = False
        for h in self.state["history"]:
            d = REPO / h["idea"]
            if not (d / "STATUS").exists():
                continue
            st, meta = read_status(d)
            new = dict(now=st, lb=meta.get("lb", ""), decision=meta.get("decision", ""))
            if any(h.get(k) != v for k, v in new.items()):
                h.update(new)
                changed = True
        sub = sum(1 for h in self.state["history"] if h.get("now", h["status"]) in SUBMITTED or h.get("lb"))
        if self.state.get("submitted") != sub:
            self.state["submitted"] = sub
            changed = True
        if changed:
            self.save()

    def close_cycle(self, idea: Path, st: str):
        meta = read_status(idea)[1]
        branch = meta.get("branch", "?")
        streak = self.state["branch_streak"]
        if st == "dead":
            streak[branch] = streak.get(branch, 0) + 1
            if streak[branch] >= self.limits["dead_streak"] and branch not in self.state["closed_branches"]:
                self.state["closed_branches"].append(branch)
                self.log(f"branch {branch} closed after {streak[branch]} deaths in a row")
        else:
            streak[branch] = 0
        if st in PUSHED:
            self.state["lb_probes"] += 1
        self.state["history"].append(dict(idea=idea.relative_to(REPO).as_posix(), status=st, branch=branch,
                                          reason=meta.get("reason", ""), override=meta.get("override", ""), t=now()))
        self.state["cycle"] += 1
        self.state["current"] = None
        self.state["report_pending"] = idea.relative_to(REPO).as_posix()
        self.save()
        self.log(f"cycle {self.state['cycle']} closed: {idea.name} -> {st}")
        try:                                   # the human's reading: digest + submit queue (never blocks a cycle)
            dossier.write_digest(self.dir)
            dossier.write_queue()
        except Exception as e:  # noqa: BLE001
            self.log(f"dossier failed: {e}")

    def harness_gate(self, idea: Path) -> tuple[bool, str]:
        r = gate.evaluate(idea, self.cfg.get("harness") or {}, read_status(idea)[1])
        return r["ok"], r["why"]

    def describe(self) -> str:
        """The next transition, read-only (--dry-run)."""
        if self.open_requests():
            return f"/fetch research/requests ({len(self.open_requests())} open)"
        if not self.state.get("current"):
            why = self.stop_reason()
            if why:
                return f"stop: {why}"
            return (f"ideation: scout? diagnose, ideate ({self.state['ideate']})" if self.state.get("ideate")
                    else f"/replan {self.dir.relative_to(REPO).as_posix()}")
        idea = REPO / self.state["current"]
        st, _ = read_status(idea)
        rel = idea.relative_to(REPO).as_posix()
        return {"proposed": f"/spec {rel}", "spec-fix": f"/spec {rel}", "specced": f"preflight {rel}",
                "preflight-ok": f"/attack {rel}", "attacked": f"/build {rel} harness",
                "harness-done": f"gate {rel}: {self.harness_gate(idea)[1]}",
                "build": f"/build {rel} ship"}.get(st, f"close cycle {rel} ({st})" if st in TERMINAL
                                                     else f"stop: unexpected STATUS {st}")

    def model_step(self, key: str, step: str, prompt: str, idea: Path) -> bool:
        before = read_status(idea)[0]
        out = self.call(step, prompt, idea)
        if out == "stopped":
            return False
        if not self.progressed(key, before, read_status(idea)[0], out):
            write_status(idea, "needs-human", reason=f"{step} made no progress {self.retries} times")
            with open(idea / "questions.md", "a", encoding="utf-8") as f:
                f.write(f"\n## {step} stalled ({now()})\n- STATUS stayed `{before}` after {self.retries} "
                        f"attempts; see claude_run.jsonl.\n")
        return True

    # ---------------------------------------------------------------- ideation engine (grill 2026-10-13)
    def web_allowed(self) -> bool:
        """Web steps: a router sends them to browsing-capable routes; otherwise Claude's web settings must exist."""
        return self.router is not None or self.web_ok() or self.runner != self._claude

    def web_ok(self) -> bool:
        return bool(self.web_settings) and (REPO / self.web_settings).exists()

    def open_requests(self) -> list[Path]:
        d = REPO / "research" / "requests"
        return [p for p in sorted(d.glob("*.md")) if re.search(r"^status:\s*open\s*$", p.read_text(encoding="utf-8"),
                                                                 re.M)] if d.exists() else []

    def fetch(self) -> bool:
        """Answer the open research/requests/ with the cheap web tier before anything else runs."""
        reqs = self.open_requests()
        before = ",".join(p.name for p in reqs)
        out = self.call("fetch", "/fetch research/requests", None)
        if out == "stopped":
            return False
        after = ",".join(p.name for p in self.open_requests())
        if not self.progressed("fetch", before, after, out):
            for p in self.open_requests():          # stop asking: park them for the digest / a human
                p.write_text(re.sub(r"^status:\s*open\s*$", "status: unanswered", p.read_text(encoding="utf-8"),
                                    flags=re.M), encoding="utf-8")
            self.log(f"fetch made no progress; parked {len(reqs)} request(s) as unanswered")
        return True

    def newest_age_days(self, folder: str, pattern: str) -> float:
        files = sorted((REPO / folder).glob(pattern), key=lambda p: p.stat().st_mtime) if (REPO / folder).exists() else []
        return (time.time() - files[-1].stat().st_mtime) / 86400 if files else float("inf")

    def ideate(self) -> bool:
        """Portfolio and idea pool are dry: scout (if stale) -> diagnose -> ideate, then /replan again."""
        if self.state["ideation_rounds"] >= self.max_ideation:
            return self.finish(f"ideation ran {self.state['ideation_rounds']} times without a live idea "
                               f"({self.state.get('ideate')})")
        self.state["ideation_rounds"] += 1
        self.save()
        rel = self.dir.relative_to(REPO).as_posix()
        if self.necropsy_mode():
            return self.ideate_necropsy(rel)
        if self.newest_age_days("research/scout", "*.md") > self.scout_max_age_d:
            if self.web_allowed():
                if self.call("scout", f"/scout {rel}", None) == "stopped":
                    return False
            else:
                self.log("scout skipped: claude.web_settings file missing (the human creates it); ideating from "
                         "evidence only")
        for step in ("diagnose", "ideate"):
            if self.call(step, f"/{step} {rel}", None) == "stopped":
                return False
        self.state["ideate"] = None
        self.save()
        self.log(f"ideation round {self.state['ideation_rounds']} done; back to /replan")
        return True

    def necropsy_mode(self) -> bool:
        charter = (self.dir / "charter.md").read_text(encoding="utf-8")
        return bool(re.search(r"^\s*-?\s*Mode\s*:\s*necropsy\b", charter, re.M | re.I))

    def ideate_necropsy(self, rel: str) -> bool:
        """Necropsy campaigns ideate from dead ideas: autopsy up to NECROPSY_PER_ROUND, then /necropsy-synth."""
        import necropsy
        todo = [REPO / d["path"] for d in necropsy.inventory() if not necropsy.is_filled(REPO / d["path"])]
        for idea in todo[:NECROPSY_PER_ROUND]:
            necropsy.skeleton(idea)
            if self.call("necropsy", f"/necropsy {idea.relative_to(REPO).as_posix()}", idea) == "stopped":
                return False
        if self.call("necropsy-synth", f"/necropsy-synth {rel}", None) == "stopped":
            return False
        self.state["ideate"] = None
        self.save()
        self.log(f"necropsy round {self.state['ideation_rounds']}: {min(len(todo), NECROPSY_PER_ROUND)} autopsied, "
                 f"{max(len(todo) - NECROPSY_PER_ROUND, 0)} left; back to /replan")
        return True

    def step(self) -> bool:
        """Advance one transition. False = nothing more to do (stopped or waiting)."""
        self.refresh_history()
        if self.open_requests() and self.web_allowed():
            return self.fetch()
        if self.state.get("report_pending"):     # learn from the closed cycle once; drafts only (promote.py)
            idea = REPO / self.state["report_pending"]
            self.state["report_pending"] = None
            self.save()
            return self.call("report", f"/report {idea.relative_to(REPO).as_posix()}", idea) != "stopped"
        cur = self.state.get("current")
        if not cur:
            why = self.stop_reason()
            if why:
                return self.finish(why)
            if self.state.get("ideate"):
                return self.ideate()
            out = self.call("replan", f"/replan {self.dir.relative_to(REPO).as_posix()}", None)
            disk = json.loads((self.dir / "state.json").read_text())   # replan writes current, ideate or stop
            for k in ("current", "stop", "ideate"):
                self.state[k] = disk.get(k)
            self.save()
            if out == "stopped" or self.state.get("stop"):
                return self.finish(self.state.get("stop") or "stopped")
            if self.state.get("ideate"):
                self.log(f"replan asks for new ideas: {self.state['ideate']}")
                return True
            if not self.state.get("current"):
                if self.progressed("replan", "", "", out):
                    return True
                return self.finish(f"replan picked no idea {self.retries} times")
            self.state["stalls"].pop("replan", None)
            return True
        idea = REPO / cur
        rel = idea.relative_to(REPO).as_posix()
        st, meta = read_status(idea)
        if st in TERMINAL:
            self.close_cycle(idea, st)
            return True
        if st in ("proposed", "spec-fix"):
            return self.model_step(f"{idea.name}:spec", "spec", f"/spec {rel}", idea)
        if st == "specced":
            found = preflight.run(idea, preflight.load_rules())
            if not found:
                write_status(idea, "preflight-ok")
            else:
                (idea / "preflight.md").write_text("# preflight findings\n" + "".join(f"- {x}\n" for x in found))
                if meta.get("spec_fixes", "0") == "0":
                    write_status(idea, "spec-fix", spec_fixes=1)
                else:
                    write_status(idea, "dead", reason=f"preflight still failing after one fix: {found[0]}")
            self.log(f"preflight {idea.name}: {len(found)} finding(s)")
            return True
        if st == "preflight-ok":
            return self.model_step(f"{idea.name}:attack", "attack", f"/attack {rel}", idea)
        if st == "attacked":
            return self.model_step(f"{idea.name}:build-harness", "build-harness", f"/build {rel} harness", idea)
        if st == "harness-done":
            ok, why = self.harness_gate(idea)
            write_status(idea, "build" if ok else "dead", **({"gate": why[:300]} if ok else {"reason": why[:300]}))
            self.log(f"harness gate {idea.name}: {'queued for build' if ok else 'dead'} ({why})")
            return True
        if st == "build":
            return self.model_step(f"{idea.name}:build-ship", "build-ship", f"/build {rel} ship", idea)
        self.log(f"{idea.name}: unexpected STATUS {st}; stopping for a human")
        return self.finish(f"idea {idea.name} in unexpected STATUS {st}")

    def override(self, idea: Path, reason: str) -> None:
        """Human override of a dead gate (W3): recorded in STATUS, history and log, then re-gated (passes)."""
        st, _ = read_status(idea)
        write_status(idea, "harness-done" if st in ("dead", "harness-done") else st, override=reason)
        if not self.state.get("current"):
            self.state["current"] = idea.relative_to(REPO).as_posix()
        self.save()
        self.log(f"OVERRIDE {idea.name} (was {st}): {reason}")

    def finish(self, why: str) -> bool:
        self.state["stop"] = why
        self.save()
        h = self.state["history"]
        rep = [f"# Campaign report: {self.dir.name}", "", f"Stopped {now()}: {why}", "",
               f"Cycles {self.state['cycle']}, probes pushed {self.state['lb_probes']}, submitted "
               f"{self.state.get('submitted', 0)}, model calls {self.state.get('calls', {})}, closed branches "
               f"{', '.join(self.state['closed_branches']) or 'none'}.", "",
               "| idea | closed as | now | LB | branch | reason | override |", "|---|---|---|---|---|---|---|",
               *[f"| {x['idea']} | {x['status']} | {x.get('now', '')} | {x.get('lb', '')} | {x['branch']} | "
                 f"{x['reason']} | {x.get('override', '')} |" for x in h], "",
               "## Agenda for the next grill",
               "- New preflight rules since the charter: `python research/bin/preflight.py --selftest` (veto any).",
               "- Open questions: every questions.md of this campaign's ideas.",
               "- LB results: `python research/bin/poll_lb.py` DECIDE lines for the pushed probes."]
        if not self.readonly:
            (self.dir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
        self.log(f"stopped: {why}")
        return False

    def run(self, once: bool = False):
        lock = call_claude.Lock(self.dir / "campaign.lock")
        if not lock.acquire(f"campaign pid {os.getpid()}"):
            raise Busy(f"another driver owns {self.dir.name} ({self.dir / 'campaign.lock'})")
        try:
            while self.step() and not once:
                pass
        finally:
            lock.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the next transition; read-only")
    ap.add_argument("--override", metavar="IDEA", help="pass a dead harness gate for this idea (needs --reason)")
    ap.add_argument("--reason", default="")
    ap.add_argument("--router", action="store_true",
                    help="route each step across claude/agy/qoder/opencode (config routing:) instead of claude only")
    ap.add_argument("--resume", action="store_true", help="clear state.json `stop` (after raising a cap)")
    a = ap.parse_args(argv)
    path = Path(a.campaign) if Path(a.campaign).is_absolute() else REPO / a.campaign
    if a.dry_run:
        print("next:", Campaign(path, readonly=True).describe())
        return 0
    if a.override or a.resume:
        c = Campaign(path)
        if a.override:
            if not a.reason.strip():
                ap.error("--override needs --reason")
            idea = Path(a.override) if Path(a.override).is_absolute() else REPO / a.override
            c.override(idea, a.reason.strip())
        if a.resume:
            c.log(f"resumed (was stopped: {c.state.get('stop')})")
            c.state["stop"] = None
            c.state["stalls"] = {}
            c.save()
        return 0
    if a.detach:
        flags = 0x08000000 | 0x00000200 if sys.platform == "win32" else 0   # NO_WINDOW | NEW_PROCESS_GROUP
        log = open(path / "campaign_driver.log", "a")
        subprocess.Popen([sys.executable, __file__, str(path)] + (["--router"] if a.router else []), cwd=REPO, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        print(f"[campaign] detached; log {path / 'campaign_driver.log'}")
        return 0
    try:
        router = providers.Router(load_config(), REPO / "research" / "provider_ledger.json") if a.router else None
        Campaign(path, router=router).run(once=a.once)
    except Busy as e:
        print(f"[campaign] refused: {e}")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
