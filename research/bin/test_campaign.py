"""Dry campaign: stubbed model steps walk fake ideas through every state (no Claude, no Kaggle).

    python research/bin/test_campaign.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import campaign  # noqa: E402
from _common import REPO, RESEARCH, read_status, write_status  # noqa: E402

TAG = "zz-test-campaign"
CHARTER = """# Charter
## Scope
- exploit line, post-scoring knobs only (L10, L13)
## Bars
- harness delta >= +0.005, CI > 0
## Stop rules
- max_cycles: 3
- max_lb_probes: 1
- dead_streak: 2
## Out of scope
- ranker retraining
"""


def stub(outcomes: list[str], calls: list[str]):
    """outcomes: per idea, 'ship' (harness passes) or 'die' (harness below bar)."""
    def run(prompt: str, model: str, log: Path) -> str:
        calls.append(f"{model} {prompt.split(' --brief')[0]}")
        assert " --brief " in prompt and Path(REPO / prompt.split(" --brief ")[1]).exists()
        verb, target = prompt.split()[0], prompt.split()[1]
        d = REPO / target
        if verb == "/replan":
            st = json.loads((d / "state.json").read_text())
            if not outcomes:
                st["stop"] = "portfolio empty"
            else:
                n = len(st["history"]) + 1
                idea = RESEARCH / "exploit" / f"{TAG}-{n}"
                (idea).mkdir(parents=True, exist_ok=True)
                (idea / "STATUS").write_text(f"proposed\nline: exploit\nbranch: b{n}\noutcome: {outcomes.pop(0)}\n")
                st["current"] = idea.relative_to(REPO).as_posix()
            (d / "state.json").write_text(json.dumps(st))
        elif verb == "/spec":
            (d / "spec.md").write_text("# Spec\nknob X default off\n")
            write_status(d, "specced")
        elif verb == "/attack":
            (d / "attack.md").write_text("| attack | check | dies if |\n")
            write_status(d, "attacked")
        elif verb == "/build" and prompt.split()[2] == "harness":
            ship = read_status(d)[1]["outcome"] == "ship"
            (d / "harness_result.json").write_text(json.dumps(dict(delta=0.01 if ship else -0.002, ci90=[0.002, 0.02],
                                                                   lb_slot=ship, leak_sensitive=False)))
            write_status(d, "harness-done")
        elif verb == "/build":
            write_status(d, "g4-pushed")
        return "done"
    return run


def _idea(cdir: Path, name: str, result=None, stress=None, status="harness-done\nline: exploit\n") -> Path:
    d = cdir / name
    shutil.rmtree(d, ignore_errors=True)   # OneDrive can leave a folder behind a failed cleanup
    d.mkdir()
    (d / "STATUS").write_text(status)
    if result is not None:
        (d / "harness_result.json").write_text(result if isinstance(result, str) else json.dumps(result))
    if stress is not None:
        (d / "stress_result.json").write_text(json.dumps(stress))
    return d


def rank_only_gate(c: "campaign.Campaign", cdir: Path) -> None:
    """gate.py (grill 2026-10-13): dead on no gain or real harm only; prior.md never passes by itself."""
    sc = lambda **cohorts: dict(schema="scorecard-1", target="C2", n_target=400, displaced=0,
                                cohorts={k: dict(delta=v[0], ci90=[v[1], v[2]], advisory=len(v) > 3)
                                         for k, v in cohorts.items()})
    gate = lambda d: c.harness_gate(d)
    # BASE-COUPLED + prior.md used to pass on the prior alone (10-12 promo-soften); now the numbers decide
    d = _idea(cdir, "coupled_neg", dict(delta=-0.001, ci90=[-0.004, 0.002], verdict="BASE-COUPLED", shard="C2"))
    (d / "prior.md").write_text("prior")
    assert not gate(d)[0]
    ok, why = gate(_idea(cdir, "small_gain", sc(C2=(0.002, -0.001, 0.005), C1_npx=(0.0, -0.003, 0.003))))
    assert ok and "queued" in why and "CI lower bound" in why, why        # small gain: queued, flagged
    ok, why = gate(_idea(cdir, "harm", sc(C2=(0.02, 0.01, 0.03), C1_npx=(-0.012, -0.02, -0.005))))
    assert not ok and "harm on C1_npx" in why, why
    ok, why = gate(_idea(cdir, "harm_advisory", sc(C2=(0.02, 0.01, 0.03), C2_np=(-0.012, -0.02, -0.005, "adv"))))
    assert ok, why                                                          # advisory cohorts never veto
    hijack = sc(C2=(0.02, 0.01, 0.03)); hijack["displaced"] = 9
    assert "displaced" in gate(_idea(cdir, "hijack", hijack))[1]
    assert not gate(_idea(cdir, "heavy_none", dict(heavy_path=True, ceiling=None)))[0]
    assert gate(_idea(cdir, "heavy_ok", dict(heavy_path=True, ceiling=0.021)))[0]
    ok, why = gate(_idea(cdir, "broken", "{not json"))
    assert not ok and "unreadable" in why, why                              # never crashes the driver
    # attacks: a death or a blocking UNTESTED kills; a proxy deviation is a flag
    stress = [{"id": "A1", "died": False}, {"id": "A3", "status": "UNTESTED", "blocking": True, "why": "no data"}]
    assert "A3 UNTESTED" in gate(_idea(cdir, "untested", sc(C2=(0.02, 0.01, 0.03)), stress))[1]
    stress = [{"id": "A3", "died": False, "deviation": {"original": "val", "proxy": "row MRR", "accepted_by": "human"}}]
    ok, why = gate(_idea(cdir, "proxy", sc(C2=(0.02, 0.01, 0.03)), stress))
    assert ok and "proxy" in why, why
    # a human override passes a dead idea and says so
    d = _idea(cdir, "overridden", sc(C2=(-0.003, -0.006, 0.0)))
    c.override(d, "LB slot as the harness-vs-LB test")
    ok, why = gate(d)
    assert ok and why.startswith("OVERRIDE"), why
    print("rank-only gate ok")


def ideation(cdir: Path) -> None:
    """Pool dry -> replan sets `ideate` -> scout, diagnose, ideate -> replan picks; capped per campaign."""
    calls: list[str] = []
    pool: list[str] = []

    def run(prompt: str, model: str, log: Path) -> str:
        verb = prompt.split()[0]
        calls.append(f"{model} {verb}")
        st = json.loads((cdir / "state.json").read_text())
        if verb == "/replan":
            if pool:
                idea = RESEARCH / "exploit" / f"{TAG}-ideated"
                idea.mkdir(parents=True, exist_ok=True)
                (idea / "STATUS").write_text("proposed\nline: exploit\nbranch: new\n")
                st["current"] = idea.relative_to(REPO).as_posix()
            else:
                st["ideate"] = "portfolio exhausted"
        elif verb == "/ideate":
            pool.append("x")
            st["ideate"] = None
        (cdir / "state.json").write_text(json.dumps(st))
        return "done"
    (cdir / "state.json").unlink(missing_ok=True)
    c = campaign.Campaign(cdir, runner=run, sleep=lambda s: None)
    c.newest_age_days = lambda folder, pattern: float("inf")     # no scout file yet (independent of the real repo)
    for _ in range(3):
        c.step()
    assert [x.split()[1] for x in calls] == ["/replan", "/scout", "/diagnose", "/ideate", "/replan"], calls
    assert calls[1].startswith("haiku") and calls[2].startswith("sonnet") and calls[3].startswith("opus"), calls
    assert c.state["current"] and c.state["ideation_rounds"] == 1
    # the cap: a pool that never fills stops the campaign instead of ideating forever
    c.state.update(current=None, ideate="still dry", ideation_rounds=c.max_ideation)
    assert c.step() is False and "ideation ran" in c.state["stop"]
    print("ideation ok")


def safety(cdir: Path) -> None:
    """Lock, stall cap, call cap, read-only dry run (loop audit W14, LOOP, W15)."""
    # a second driver on the same campaign is refused
    c = campaign.Campaign(cdir, runner=lambda *a: "done", sleep=lambda s: None)
    lock = campaign.call_claude.Lock(cdir / "campaign.lock")
    assert lock.acquire("other driver")
    try:
        campaign.Campaign(cdir, runner=lambda *a: "done").run(once=True)
        raise AssertionError("second driver was not refused")
    except campaign.Busy:
        pass
    finally:
        lock.release()
    # a spec step that never moves STATUS: retried step_retries times, then needs-human (no endless re-runs)
    idea = RESEARCH / "exploit" / f"{TAG}-stall"
    idea.mkdir(parents=True)
    (idea / "STATUS").write_text("proposed\nline: exploit\nbranch: s\n")
    c.state.update(current=idea.relative_to(REPO).as_posix(), stop=None, cycle=0, lb_probes=0)
    c.save()
    calls = []
    c.runner = lambda *a: calls.append(a) or "done"
    while read_status(idea)[0] == "proposed":
        assert c.step() and len(calls) <= c.retries
    assert read_status(idea)[0] == "needs-human" and len(calls) == c.retries
    # the dry run writes nothing
    before = {p.name: p.stat().st_mtime_ns for p in cdir.iterdir() if p.is_file()}
    assert campaign.main([str(cdir), "--dry-run"]) == 0
    assert before == {p.name: p.stat().st_mtime_ns for p in cdir.iterdir() if p.is_file()}
    # a call cap stops the campaign; --resume clears the stop
    c.state.update(current=None, stop=None)
    c.caps = {"opus": 1}
    c.state["calls"] = {"opus": 1}
    assert c.step() is False and "call cap" in c.state["stop"]
    assert campaign.main([str(cdir), "--resume"]) == 0
    assert json.loads((cdir / "state.json").read_text())["stop"] is None
    print("safety ok")


def routed_replan(cdir: Path) -> None:
    """--router: the call counter save must not clobber the current/stop that /replan wrote into state.json."""
    class Router:
        def routes(self, step):
            return ["claude:opus"]

        def family(self, route):
            return "claude"

        def run(self, step, prompt, log, wt, exclude_family=None, skip=None):
            st = json.loads((cdir / "state.json").read_text())
            st["stop"] = "pool dry"
            (cdir / "state.json").write_text(json.dumps(st))
            return "done", "claude:opus"

    (cdir / "state.json").unlink(missing_ok=True)
    c = campaign.Campaign(cdir, runner=lambda *a: "done", sleep=lambda s: None, router=Router())
    assert c.step() is False and c.state["stop"] == "pool dry", c.state["stop"]
    assert c.state["calls"] == {"claude:opus": 1}, c.state["calls"]
    assert not c.state.get("stalls", {}).get("replan"), c.state["stalls"]
    print("routed replan ok")


def main() -> int:
    cdir = RESEARCH / "campaigns" / TAG
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "charter.md").write_text(CHARTER)
        calls: list[str] = []
        c = campaign.Campaign(cdir, runner=stub(["die", "ship", "ship"], calls), sleep=lambda s: None)
        assert c.limits["max_cycles"] == 3 and c.limits["max_lb_probes"] == 1
        c.run()
        st = json.loads((cdir / "state.json").read_text())
        hist = [h["status"] for h in st["history"]]
        assert hist == ["dead", "g4-pushed"], hist                 # 2nd ship hits max_lb_probes = 1 -> stop
        assert st["stop"] == "max_lb_probes 1 reached" and st["lb_probes"] == 1
        assert [x.split()[1] for x in calls][:6] == ["/replan", "/spec", "/attack", "/build", "/report", "/replan"], calls
        assert all(x.startswith("sonnet") for x in calls if x.split()[1] == "/report")
        assert all(x.startswith("opus") for x in calls if x.split()[1] in ("/replan", "/spec", "/attack"))
        assert all(x.startswith("sonnet") for x in calls if x.split()[1] == "/build")
        assert (cdir / "report.md").exists() and "max_lb_probes" in (cdir / "report.md").read_text()
        sizes = [int(ln.split("brief ")[1].split()[0]) for ln in (cdir / "log.md").read_text().splitlines() if " brief " in ln]
        assert sizes and max(sizes) <= 6000, sizes
        print("campaign dry run ok:", hist, "| brief sizes", sizes)
        rank_only_gate(c, cdir)
        safety(cdir)
        ideation(cdir)
        routed_replan(cdir)
        return 0
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
        for p in (RESEARCH / "exploit").glob(f"{TAG}-*"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
