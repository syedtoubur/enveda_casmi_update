"""Tests for build/c3_gate.py. Run: python build/test_c3_gate.py

The gate replaces a rank gate that always fired on 30% of the field.  The properties that matter
are therefore as much about *declining* as about firing, so most of these assert a veto -- above
all the one the old gate could not express: an all-weak-signals molecule, which is an
uninformative spectrum rather than a novel structure, must be declined.
"""
import collections
import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c3_gate as G

FAIL = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAIL.append(msg)


# a molecule sitting comfortably in the "sure" corner, used as the base for one-term perturbations
SURE = dict(lib_max=0.30, exact_max=0.20, ana_max=0.85, delta_steps=1)


def at(**kw):
    d = dict(SURE)
    d.update(kw)
    return G.classify(**d)


def test_tiers():
    print("test_tiers")
    check(at() == G.SURE, "strong analog, weak exact match, 1 step -> sure")
    check(at(ana_max=0.65, exact_max=0.45) == G.LIKELY, "moderate analog and margin -> likely")
    check(at(ana_max=G.ANA_SURE, exact_max=G.ANA_SURE - G.MARGIN_SURE) == G.SURE,
          "exactly on both SURE thresholds -> sure (inclusive)")
    check(at(ana_max=G.ANA_SURE - 1e-6) == G.LIKELY, "a hair below ANA_SURE -> likely, not sure")
    check(G.SLOTS[G.SURE][0] < G.SLOTS[G.LIKELY][0],
          "the sure tier is offered a better first slot than likely")
    check(1 not in G.SLOTS[G.LIKELY],
          "the likely tier never takes rank 1, where the displacement risk is")


def test_class1_veto():
    print("test_class1_veto")
    check(at(lib_max=G.LIB_TAU) is None, "lib_max exactly at LIB_TAU is a library match -> declined")
    check(at(lib_max=0.999) is None, "a near-perfect library match is never routed to the mutator")
    check(at(lib_max=G.LIB_TAU - 1e-6) == G.SURE, "a hair below LIB_TAU still fires")
    check(at(lib_max=None) == G.SURE, "unknown lib_max does not veto")


def test_bad_spectrum_is_declined():
    print("test_bad_spectrum_is_declined")
    # the defect in the old gate: every f.z weak looks identical for a novel molecule and for an
    # uninformative spectrum.  Weak analog evidence is what separates them.
    check(G.classify(lib_max=0.05, exact_max=0.05, ana_max=0.05, delta_steps=1) is None,
          "all signals weak (an uninformative spectrum) -> declined, not fired")
    check(at(ana_max=G.ANA_MIN - 1e-6) is None, "a hair below ANA_MIN -> declined")
    check(at(ana_max=G.ANA_MIN, exact_max=0.0) in (G.SURE, G.LIKELY), "exactly at ANA_MIN fires")
    check(at(ana_max=None) is None, "no analog at all -> declined")


def test_exact_match_veto():
    print("test_exact_match_veto")
    check(at(exact_max=0.85) is None,
          "a strong same-mass match explains the spectrum -> Class 1/2, declined")
    check(at(ana_max=0.85, exact_max=0.85 - G.MARGIN_MIN) in (G.SURE, G.LIKELY),
          "exactly at MARGIN_MIN fires")
    check(at(ana_max=0.85, exact_max=0.85 - G.MARGIN_MIN + 1e-6) is None,
          "a hair under MARGIN_MIN -> declined")
    check(at(exact_max=None) == G.SURE, "unknown exact_max is treated as 0.0, not as a veto")


def test_depth_veto():
    print("test_depth_veto")
    for k in range(0, G.MAX_STEPS + 1):
        check(at(delta_steps=k) in (G.SURE, G.LIKELY), f"depth {k} is inside the sweet spot")
    check(at(delta_steps=G.MAX_STEPS + 1) is None,
          f"depth {G.MAX_STEPS + 1} -> declined (measured 3.7% at >=3, 0% at >=5)")
    check(at(delta_steps=None) is None, "an unreachable delta -> declined")


def test_ice_veto_is_optional():
    print("test_ice_veto_is_optional")
    check(at(ice_max=None) == G.SURE, "ice_max=None never vetoes (the mutator cell has no ICE yet)")
    check(at(ice_max=0.0) == G.SURE, "a poorly simulated candidate pool does not veto")
    check(at(ice_max=G.ICE_MAX) is None, "exactly at ICE_MAX: a candidate reproduces the spectrum")
    check(at(ice_max=G.ICE_MAX - 1e-6) == G.SURE, "a hair below ICE_MAX still fires")


def test_reasons_funnel():
    print("test_reasons_funnel")
    r = collections.Counter()
    G.classify(0.95, 0.1, 0.9, 1, reasons=r)
    check(sum(r.values()) == 1, "one call records exactly one reason, so the funnel does not double-count")
    check(r["class1_library_match"] == 1, "and it names the term that actually vetoed")
    r2 = collections.Counter()
    # lib_max vetoes first even though ana_max would also fail -- cheapest-first ordering
    G.classify(0.95, 0.1, 0.0, None, reasons=r2)
    check(r2["class1_library_match"] == 1 and r2["no_analog_evidence"] == 0,
          "the first (cheapest) failing term wins, so reasons read as a funnel")
    r3 = collections.Counter()
    for _ in range(3):
        G.classify(**SURE, reasons=r3)
    check(r3["fired_sure"] == 3, "firing is counted too, so fire rate comes from the same counter")


def test_delta_helpers():
    print("test_delta_helpers")
    EL = ("C", "H", "N", "O")
    check(G.delta_of((6, 6, 0, 1), (6, 6, 0, 0)) == (0, 0, 0, 1), "delta_of subtracts elementwise")
    check(G.delta_of(None, (1, 1, 1, 1)) is None, "delta_of tolerates an unknown side")
    check(G.heavy_negative((0, -2, 0, 0), EL) is False, "-2H is legal: it is in the palette")
    check(G.heavy_negative((-1, 0, 0, 0), EL) is True, "a lost carbon is not reachable additively")
    reach = {(0, 0, 0, 1): (("+O",),), (0, 2, 0, 1): (("+O", "+H2"), ("+H2O",))}
    check(G.min_steps((0, 0, 0, 1), reach) == 1, "min_steps reads the shortest decomposition")
    check(G.min_steps((0, 2, 0, 1), reach) == 1, "and prefers the 1-step route over the 2-step one")
    check(G.min_steps((9, 9, 9, 9), reach) is None, "an absent delta is unreachable")
    check(G.min_steps(None, reach) is None, "a None delta is unreachable")
    check(G.min_steps((-1, 0, 0, 0), reach, EL) is None,
          "a heavy-negative delta is unreachable even if the table happened to hold it")


def test_thresholds_are_env_overridable():
    print("test_thresholds_are_env_overridable")
    # the notebook moves these without a code edit; confirm the module reads the environment
    os.environ["C3_MAX_STEPS"] = "4"
    try:
        H = importlib.reload(G)
        check(H.MAX_STEPS == 4, "C3_MAX_STEPS is read from the environment")
        check(H.classify(0.3, 0.2, 0.85, 4) in (H.SURE, H.LIKELY),
              "and the looser depth actually takes effect")
    finally:
        del os.environ["C3_MAX_STEPS"]
        importlib.reload(G)
    check(G.MAX_STEPS == 1, "and the default is restored for later tests")


if __name__ == "__main__":
    for fn in [test_tiers, test_class1_veto, test_bad_spectrum_is_declined, test_exact_match_veto,
               test_depth_veto, test_ice_veto_is_optional, test_reasons_funnel, test_delta_helpers,
               test_thresholds_are_env_overridable]:
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
