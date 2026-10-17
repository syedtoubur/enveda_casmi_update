"""c3-detector (campaign 20261015-c3-gen-slot2, K2 v0) tests.

    python build/test_c3_detector.py

Decision math on hand-built arrays; features and the mut-first top-1 on REAL pseudo_c3 dump rows and the real K1
`p4_raw.pkl` (no toy SMILES: an invalid toy would silently drop out, L: casmi-toy-smiles-vacuous-tests).
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

import c3_detector as d  # noqa: E402

RAW = "E:/casmi/c3_gen/p4_raw.pkl"


def test_prior_shift():
    # fitted at 600:133, a p equal to the training base rate maps to pi exactly
    base = 600 / 733
    assert abs(d.prior_shift([base], 600, 133, 0.25)[0] - 0.25) < 1e-9
    p = d.prior_shift([0.1, 0.5, 0.9], 600, 133, 0.25)
    assert np.all(np.diff(p) > 0) and np.all(p < [0.1, 0.5, 0.9])


def test_slot2_cost():
    assert d.slot2_cost(1) == 0 and d.slot2_cost(0) == 0 and d.slot2_cost(26) == 0
    assert abs(d.slot2_cost(2) - (1 / 2 - 1 / 3)) < 1e-12


def test_choose_tau():
    # positives: high p ones hit; other negatives: low p, cost 0.1
    p_pos = np.array([0.2, 0.6, 0.9]); hit = np.array([0, 1, 1])
    p_oth = np.array([0.1, 0.3]); cost = np.array([0.1, 0.1])
    t = d.choose_tau(p_pos, hit, p_oth, cost)
    # at 0.2: h 2/3, L 0.1 (one other fired) -> 0.2*0.333 = 0.067 < 0.8*0.1 -> no
    # at 0.3: h 1.0 (0.6, 0.9 fired), L 0.1 -> 0.3*0.5 = 0.15 > 0.07 -> yes
    assert t["tau"] == 0.3 and t["h_fired"] == 1.0 and t["n_other_fired"] == 1 and not t["L_from_prior"]
    # no positive ever hits -> no tau
    assert d.choose_tau(p_pos, np.zeros(3), p_oth, cost) is None
    # nothing "other" fires at the chosen tau -> L from the pooled prior
    t2 = d.choose_tau(np.array([0.9]), np.array([1]), np.array([0.1]), np.array([0.2]))
    assert t2["L_from_prior"] and t2["L_fired"] == d.L_PRIOR


def test_verdict():
    assert d.verdict(None, 0, 0)[0] == "DEAD"
    assert d.verdict({"tau": 0.5}, 0.005, 0.01)[0] == "PASS"
    assert d.verdict({"tau": 0.5}, 0.011, 0.0)[0] == "DEAD"
    assert d.verdict({"tau": 0.5}, 0.0, 0.021)[0] == "DEAD"


def test_oof_never_trains_on_leak_hit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)); y = np.r_[np.ones(30), np.zeros(30)].astype(int)
    train = np.ones(60, bool); train[50:] = False          # last 10 negatives = leak-hit, scored only
    X[50:] = 1e6                                            # would dominate any fit that saw them
    oof = d.oof_scores(X, y, train, np.arange(60) % 7, "logit")
    assert not np.isnan(oof).any()


def test_real_features_and_mutfirst():
    import harness as H
    state, deps, cp = H.load_dump("pseudo_c3", d.POST)
    key = cp.canonicalize_smiles_to_ik14
    raw = pickle.load(open(RAW, "rb"))["raw"]
    mids = [m for m in state["te_mids"] if m in raw and not raw[m].get("error")][:20]
    assert mids
    n_mut = 0
    for m in mids:
        f = d.features(state, m)
        assert set(d.CLEAN) | set(d.LEAKY) <= set(f)
        assert all(np.isfinite(f[c]) for c in ("lib_max", "margin", "ice_cov", "mass", "n_base"))
        g = d.gen_top1(raw[m], key)
        assert g["mutfirst"] is not None and len(g["mutfirst"][1]) == 14
        if g["mut"]:
            n_mut += 1
            assert g["mutfirst"] == g["mut"]
        else:
            assert g["mutfirst"] == g["dec"]
    assert n_mut > 0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    raise SystemExit(1 if fails else 0)
