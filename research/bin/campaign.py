"""Drive one research campaign: a deterministic state machine that calls one headless model step at a time.

    python research/bin/campaign.py research/campaigns/<id> [--detach] [--dry-run] [--once]

The campaign-grill (interactive, Opus) writes <campaign>/charter.md. From there no human is needed:

  no current idea      -> stop rules -> /replan (Opus)          -> new idea folder, STATUS proposed (or stop)
  proposed | spec-fix  -> /spec (Opus)                          -> specced
  specced              -> preflight.py (code)                   -> preflight-ok | spec-fix (once) | dead
  preflight-ok         -> /attack (Opus)                        -> attacked
  attacked             -> /build <idea> harness (Sonnet)        -> harness-done | dead
  harness-done         -> harness_result.json gate (code)       -> build | dead
  build                -> /build <idea> ship (Sonnet)           -> g4-pushed (the G4 watcher and poll_lb take over)
  dead | g4-pushed ... -> cycle closed: counters, branch streak, current = none

Each model step gets a generated brief (brief.py) and nothing else. The LB is asynchronous: a pushed probe does not
block the next cycle. Stop rules (config campaign.*, overridable in charter `## Stop rules` as `key: value`):
max_cycles, max_lb_probes, dead_streak (consecutive harness deaths on one branch close that branch; no open
branch left = stop), quota_pause_h (a Claude usage-limit pause is retried every 30 min up to this long, then stop).
On stop the campaign writes report.md, the agenda for the next grill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO, RESEARCH, load_config, read_status, write_status  # noqa: E402
import brief as brief_mod  # noqa: E402
import call_claude  # noqa: E402
import preflight  # noqa: E402

DEFAULTS = dict(max_cycles=5, max_lb_probes=4, dead_streak=2, quota_pause_h=12)
MODELS = dict(replan="opus", spec="opus", attack="opus", build="sonnet")
TERMINAL = {"dead", "g4-pushed", "g4-failed", "g4-fix", "needs-human", "ready-to-submit", "instrument-ready",
            "scored", "scored-unpaired", "keep-pending", "keep", "revert", "submitted"}
PUSHED = {"g4-pushed", "g4-fix", "ready-to-submit", "scored", "scored-unpaired", "keep-pending", "keep", "revert",
          "submitted"}


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


class Campaign:
    def __init__(self, path: Path, runner=None, sleep=time.sleep):
        self.dir = path.resolve()
        self.cfg = load_config()
        self.runner = runner or self._claude
        self.sleep = sleep
        sf = self.dir / "state.json"
        self.state = json.loads(sf.read_text()) if sf.exists() else dict(
            cycle=0, lb_probes=0, branch_streak={}, closed_branches=[], current=None, stop=None, history=[])
        self.limits = dict(DEFAULTS, **(self.cfg.get("campaign") or {}))
        charter = (self.dir / "charter.md").read_text(encoding="utf-8")
        for k in DEFAULTS:
            m = re.search(rf"^\s*-?\s*{k}\s*:\s*(\d+)", brief_mod.section(charter, "Stop rules"), re.M)
            if m:
                self.limits[k] = int(m.group(1))
        self.models = dict(MODELS, **((self.cfg.get("claude") or {}).get("models") or {}))
        if not sf.exists():
            self.save()                     # /replan reads it

    # ---------------------------------------------------------------- plumbing
    def save(self):
        (self.dir / "state.json").write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def log(self, msg: str):
        with open(self.dir / "log.md", "a", encoding="utf-8") as f:
            f.write(f"- {now()} {msg}\n")
        print(f"[campaign] {msg}", flush=True)

    def _claude(self, prompt: str, model: str, log: Path) -> str:
        wt = Path(self.cfg["lines"]["exploit"]["worktree"])
        return call_claude.run_prompt(self.cfg, prompt, log, model=model, worktree=wt if wt.exists() else None)

    def call(self, step: str, prompt: str, idea: Path | None) -> str:
        """Brief + one model step; a usage-limit pause is retried until quota_pause_h, then the campaign stops."""
        bstep = step if step in brief_mod.STEPS else step.split()[0]
        path, n = brief_mod.brief(self.dir, bstep, idea)
        self.log(f"{bstep} ({self.models[bstep.split('-')[0]]}) brief {n} chars{'  OVER 6000' if n > 6000 else ''}")
        t0 = time.time()
        while True:
            out = self.runner(f"{prompt} --brief {path.relative_to(REPO).as_posix()}",
                              self.models[bstep.split("-")[0]], (idea or self.dir) / "claude_run.jsonl")
            if out != "paused":
                return out
            if time.time() - t0 > self.limits["quota_pause_h"] * 3600:
                self.state["stop"] = f"Claude usage limit for > {self.limits['quota_pause_h']} h"
                self.save()
                return "stopped"
            self.log("usage limit: retry in 30 min")
            self.sleep(1800)

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
                                          reason=meta.get("reason", ""), t=now()))
        self.state["cycle"] += 1
        self.state["current"] = None
        self.save()
        self.log(f"cycle {self.state['cycle']} closed: {idea.name} -> {st}")

    def harness_gate(self, idea: Path) -> tuple[bool, str]:
        f = idea / "harness_result.json"
        if not f.exists():
            return False, "no harness_result.json"
        r = json.loads(f.read_text())
        v = r.get("verdict")
        if v in ("BLIND", "BASE-COUPLED"):         # research/harness/capability_map.md: the harness can't judge it
            if (idea / "prior.md").exists():
                return True, f"family {r.get('family')} {v} on the harness, written prior present (LB decides)"
            return False, f"family {r.get('family')} {v} on the harness and no prior.md"
        if r.get("lb_slot"):
            return True, f"harness delta {r['delta']:+.4f} CI {r['ci90']}"
        if r.get("leak_sensitive") and (idea / "prior.md").exists():
            return True, "leak-sensitive, written prior present (LB decides)"
        return False, f"harness delta {r.get('delta', float('nan')):+.4f} CI {r.get('ci90')} below the LB-slot bar"

    def step(self) -> bool:
        """Advance one transition. False = nothing more to do (stopped or waiting)."""
        cur = self.state.get("current")
        if not cur:
            why = self.stop_reason()
            if why:
                return self.finish(why)
            out = self.call("replan", f"/replan {self.dir.relative_to(REPO).as_posix()}", None)
            self.state = json.loads((self.dir / "state.json").read_text())   # replan writes current or stop
            if out == "stopped" or self.state.get("stop") or not self.state.get("current"):
                return self.finish(self.state.get("stop") or "replan picked no idea")
            return True
        idea = REPO / cur
        rel = idea.relative_to(REPO).as_posix()
        st, meta = read_status(idea)
        if st in TERMINAL:
            self.close_cycle(idea, st)
            return True
        if st in ("proposed", "spec-fix"):
            return self.call("spec", f"/spec {rel}", idea) != "stopped"
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
            return self.call("attack", f"/attack {rel}", idea) != "stopped"
        if st == "attacked":
            return self.call("build-harness", f"/build {rel} harness", idea) != "stopped"
        if st == "harness-done":
            ok, why = self.harness_gate(idea)
            write_status(idea, "build" if ok else "dead", **({} if ok else {"reason": why}))
            self.log(f"harness gate {idea.name}: {'LB slot' if ok else 'dead'} ({why})")
            return True
        if st == "build":
            return self.call("build-ship", f"/build {rel} ship", idea) != "stopped"
        self.log(f"{idea.name}: unexpected STATUS {st}; stopping for a human")
        return self.finish(f"idea {idea.name} in unexpected STATUS {st}")

    def finish(self, why: str) -> bool:
        self.state["stop"] = why
        self.save()
        h = self.state["history"]
        rep = [f"# Campaign report: {self.dir.name}", "", f"Stopped {now()}: {why}", "",
               f"Cycles {self.state['cycle']}, LB probes pushed {self.state['lb_probes']}, closed branches "
               f"{', '.join(self.state['closed_branches']) or 'none'}.", "", "| idea | status | branch | reason |",
               "|---|---|---|---|", *[f"| {x['idea']} | {x['status']} | {x['branch']} | {x['reason']} |" for x in h], "",
               "## Agenda for the next grill",
               "- New preflight rules since the charter: `python research/bin/preflight.py --selftest` (veto any).",
               "- Open questions: every questions.md of this campaign's ideas.",
               "- LB results: `python research/bin/poll_lb.py` DECIDE lines for the pushed probes."]
        (self.dir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
        self.log(f"stopped: {why}")
        return False

    def run(self, once: bool = False):
        while self.step() and not once:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the next transition's command; no model call")
    a = ap.parse_args(argv)
    path = Path(a.campaign) if Path(a.campaign).is_absolute() else REPO / a.campaign
    if a.detach:
        flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        log = open(path / "campaign_driver.log", "a")
        subprocess.Popen([sys.executable, __file__, str(path)], cwd=REPO, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        print(f"[campaign] detached; log {path / 'campaign_driver.log'}")
        return 0
    runner = (lambda prompt, model, log: print(f"DRY {model}: {prompt}") or "done") if a.dry_run else None
    c = Campaign(path, runner=runner)
    c.run(once=a.once or a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
