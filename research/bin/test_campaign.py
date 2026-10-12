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


def blind_gate(cdir: Path) -> None:
    """A BLIND family never passes on lb_slot; with prior.md it goes to the LB (capability map, R08)."""
    d = cdir / "blind_idea"
    d.mkdir()
    (d / "harness_result.json").write_text(json.dumps(dict(delta=0.02, ci90=[0.01, 0.03], lb_slot=True,
                                                           leak_sensitive=False, family="fusion", verdict="BLIND")))
    ok, why = campaign.Campaign.harness_gate(None, d)
    assert not ok and "BLIND" in why, why
    (d / "prior.md").write_text("prior")
    ok, why = campaign.Campaign.harness_gate(None, d)
    assert ok and "LB decides" in why, why
    # BASE-COUPLED (gate_report 2026-10-12, Option 2) is handled exactly like BLIND
    c = cdir / "coupled_idea"
    c.mkdir()
    (c / "harness_result.json").write_text(json.dumps(dict(delta=0.02, ci90=[0.01, 0.03], lb_slot=True,
                                                           leak_sensitive=False, family="promotion",
                                                           verdict="BASE-COUPLED")))
    ok, why = campaign.Campaign.harness_gate(None, c)
    assert not ok and "BASE-COUPLED" in why, why
    (c / "prior.md").write_text("prior")
    ok, why = campaign.Campaign.harness_gate(None, c)
    assert ok and "LB decides" in why, why
    print("blind gate ok")


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
        assert [x.split()[1] for x in calls][:5] == ["/replan", "/spec", "/attack", "/build", "/replan"]
        assert all(x.startswith("opus") for x in calls if x.split()[1] in ("/replan", "/spec", "/attack"))
        assert all(x.startswith("sonnet") for x in calls if x.split()[1] == "/build")
        assert (cdir / "report.md").exists() and "max_lb_probes" in (cdir / "report.md").read_text()
        sizes = [int(ln.split("brief ")[1].split()[0]) for ln in (cdir / "log.md").read_text().splitlines() if " brief " in ln]
        assert sizes and max(sizes) <= 6000, sizes
        print("campaign dry run ok:", hist, "| brief sizes", sizes)
        blind_gate(cdir)
        return 0
    finally:
        shutil.rmtree(cdir, ignore_errors=True)
        for p in (RESEARCH / "exploit").glob(f"{TAG}-*"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
