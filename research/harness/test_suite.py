"""CPU tests for the back-test suite (harness.py suite / verdict rules).  python research/harness/test_suite.py"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402
from test_harness import make_dump  # noqa: E402


def test_spec_maps_every_knob():
    spec = harness.load_spec()
    cp = harness.load_module(harness.WORKTREE / "build" / "casmi_post.py", "cp_spec_test")
    fam = harness.knob_families(spec)
    missing = [k for k in cp.CFG if k not in fam]
    assert not missing, f"knobs without a family in backtests.yaml: {missing}"
    assert harness.family_of({"ICE_FALLBACK": True, "METRIC_FUSION": True}, spec) == "e006_overlay"
    assert harness.family_of({"ICE_FALLBACK": True}, spec) == "ice_fallback"
    assert harness.family_of({"USE_ENG": False}, spec) == "fusion"
    for sw in spec["sweeps"]:   # PyYAML reads 1.0e9 (unsigned exponent) as a string
        for v in sw["values"]:
            assert all(isinstance(x, (int, float)) for x in (v if isinstance(v, list) else [v])), (sw["knob"], v)


def test_flip_status():
    assert harness.flip_status([0.01, 0.02], 0.007) == "match"
    assert harness.flip_status([-0.01, -0.02], 0.007) == "mismatch"
    assert harness.flip_status([0.01, -0.02], 0.007) == "regime-dependent"
    assert harness.flip_status([0.0, 0.02], 0.007) == "zero"
    assert harness.flip_status([0.01], None) == "no key"


def test_verdict_rule():
    v = harness.family_verdict
    m, x, r = "match", "mismatch", "regime-dependent"
    # rule 1: mismatches outnumber matches -> BLIND (B flips count here)
    assert v([("A", x), ("B", x), ("S", m)], lam_ok=True, sanity_ok=True) == "BLIND"
    # rule 2: >= 3 S/A all match -> TRUSTED
    assert v([("S", m), ("A", m), ("A", m)], lam_ok=False, sanity_ok=False) == "TRUSTED"
    # rule 3: one S/A match, none mismatch/regime, lam + sanity -> PROVISIONAL
    assert v([("S", m)], lam_ok=True, sanity_ok=True) == "TRUSTED-PROVISIONAL"
    assert v([("S", m), ("B", x)], lam_ok=True, sanity_ok=True) == "TRUSTED-PROVISIONAL"
    assert v([("S", m)], lam_ok=False, sanity_ok=True) == "ADVISORY"
    assert v([("S", m), ("A", r)], lam_ok=True, sanity_ok=True) == "ADVISORY"
    # rule 4: no keyed flip -> ADVISORY
    assert v([], lam_ok=True, sanity_ok=True) == "ADVISORY"
    assert v([("none", "no key")], lam_ok=True, sanity_ok=True) == "ADVISORY"


def test_coupling_rule():
    # gate_report 2026-10-12 "Option 2 BASE-coupling rule": share of leak-hit molecules whose truth rank moves
    import numpy as np
    r0 = np.array([1, 1, 1, 1, 3]); leak = np.array([True, True, True, True, False])
    assert harness.coupling_rate(np.array([1, 1, 1, 1, 9]), r0, leak) == 0.0          # non-leak moves don't count
    assert harness.coupling_rate(np.array([2, 1, 1, 1, 3]), r0, leak) == 0.25
    assert harness.coupling_rate(r0, r0, np.zeros(5, bool)) == 0.0                     # no leak-hit molecules
    cv = harness.coupled_verdict
    assert cv("ADVISORY", 0.06) == "BASE-COUPLED"
    assert cv("ADVISORY", 0.05) == "ADVISORY"                                          # strictly greater than 0.05
    assert cv("TRUSTED", 0.2) == "BASE-COUPLED"
    assert cv("BLIND", 0.2) == "BLIND"                                                 # BLIND stays the worst
    assert harness.ORDER.index("BLIND") < harness.ORDER.index("BASE-COUPLED") < harness.ORDER.index("ADVISORY")


def test_sweep_ok():
    assert harness.sweep_ok([0.25, 0.5, 1.0, 1.5, 2.0], [0.1, 0.2, 0.3, 0.29, 0.1], 1.0)
    assert harness.sweep_ok([0.25, 0.5, 1.0, 1.5, 2.0], [0.1, 0.2, 0.29, 0.3, 0.1], 1.0)   # one grid step
    assert not harness.sweep_ok([0.25, 0.5, 1.0, 1.5, 2.0], [0.1, 0.2, 0.29, 0.28, 0.3], 1.0)


def test_suite_on_synthetic_dump():
    td = Path(tempfile.mkdtemp())
    try:
        tr = make_dump(td)
        harness.DUMPS = td / "dumps"
        harness.truths = lambda: tr
        harness._CACHES.clear()
        spec = dict(families={"fusion": ["USE_ENG"], "tail": ["FINAL_PASS"]},
                    flips=[dict(id="eng", family="fusion", set={"USE_ENG": False}, tier="S", lb=-0.01, source="t"),
                           dict(id="tail", family="tail", set={"FINAL_PASS": False}, tier="none", lb=None, source="t")],
                    sweeps=[dict(knob="KRR", values=[1.0, 3.0], evidence=False)])
        out = td / "map"
        res = harness.suite(["T"], spec=spec, out_dir=out, report=None)
        san = res["sanity"]["T"]
        assert san["identity"] == 0.0
        assert abs(san["oracle"] - (1 - res["mrr"]["T"])) < 1e-9
        assert "shuffle" in san
        assert set(res["families"]) == {"fusion", "tail"}
        assert res["families"]["tail"]["verdict"] == "ADVISORY"
        assert res["flips"]["eng"]["status"] in {"match", "mismatch", "zero"}
        assert (out / "capability_map.md").exists()
        j = json.loads((out / "capability_map.json").read_text())
        assert j["knobs"]["USE_ENG"] == "fusion" and j["verdicts"]["tail"] == "ADVISORY"
        # compare() reads the map: family + verdict, and a BLIND family never gets an LB slot
        harness.CAPMAP = out / "capability_map.json"
        j["verdicts"]["fusion"] = "BLIND"
        (out / "capability_map.json").write_text(json.dumps(j))
        r = harness.compare("T", {}, {"USE_ENG": False})
        assert r["family"] == "fusion" and r["verdict"] == "BLIND" and r["lb_slot"] is False
        j["verdicts"]["fusion"] = "BASE-COUPLED"
        (out / "capability_map.json").write_text(json.dumps(j))
        r = harness.compare("T", {}, {"USE_ENG": False})
        assert r["verdict"] == "BASE-COUPLED" and r["lb_slot"] is False
        assert harness.capmap_verdict({"USE_ENG": False, "FINAL_PASS": False})[1] == "BASE-COUPLED"   # worst wins
        # coupling recorded per flip / sweep point / family; verdict consistent with the threshold.
        # v2's truth = BASE rank 1 here, so v2 is a leak-hit molecule; KRR sweep measures the sweep-only family.
        tr2 = dict(tr, v2=dict(tr["v2"], smiles=tr["v2"]["smiles"]))
        tr2["v2"]["smiles"] = harness.load_dump("T")[0]["base"]["v2"][0][0]
        harness.truths = lambda: tr2
        spec["families"]["merge_gate"] = ["KRR"]
        res = harness.suite(["T"], spec=spec, out_dir=out, report=None)
        assert res["strata"]["T"]["base_top1_truth"] == 1
        for fid in ("eng", "tail"):
            assert 0.0 <= res["flips"][fid]["per_shard"]["T"]["coupling"] <= 1.0
        assert len(res["sweeps"]["KRR"]["per_shard"]["T"]["coupling"]) == 2
        for fam, f in res["families"].items():
            if f["verdict"] != "BLIND":
                assert (f["verdict"] == "BASE-COUPLED") == (f["coupling"] > harness.COUPLING_TH), (fam, f)
        assert "coupling" in (out / "capability_map.md").read_text()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
