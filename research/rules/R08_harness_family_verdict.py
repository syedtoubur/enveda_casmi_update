ID = "R08"
TAGS = ["measurement", "decision"]
LESSON = ("A harness result counts only for families the capability map does not call BLIND or BASE-COUPLED; it must "
          "carry its family + verdict, and such a family's idea goes to the LB on a written prior.md (LB decides).")
SRC = ("H2 back-test FAIL 2026-10-11: on C2 the leaked BASE FPNet made USE_ENG=False read +0.257 (LB: fusion helps); "
       "C2_np 2026-10-12: BASE leaks on every train-derived val shard (FPNet or np-fold ranker) -> BASE-COUPLED; "
       "research/harness/capability_map.md, gate_report.md 'back-test suite', 'Option 2 BASE-coupling rule'")
FLAGS = ["rules/fixtures/R08_blind_family_no_prior", "rules/fixtures/R08_unmapped_knob",
         "rules/fixtures/R08_base_coupled_no_prior"]
NOT_CITABLE = ("BLIND", "BASE-COUPLED")

import json  # noqa: E402


def check(idea):
    f = idea / "harness_result.json"
    if not f.exists():
        return []
    r = json.loads(f.read_text(encoding="utf-8"))
    v = r.get("verdict")
    if v is None:
        return ["harness_result.json: no family/verdict (made before the capability map); re-run harness.py compare"]
    if v == "unmapped" or "unknown:" in str(r.get("family", "")):
        return [f"harness_result.json: knobs {sorted(r.get('b', {}))} have no family in research/harness/backtests.yaml"]
    if v in NOT_CITABLE and not (idea / "prior.md").exists():
        return [f"harness_result.json: family {r.get('family')} is {v} on the harness; write prior.md (LB decides)"]
    return []
