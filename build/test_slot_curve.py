"""Plain-function tests: python build/test_slot_curve.py  (real-row tests need `slot_curve.py rows` caches)."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import slot_curve as sc

ROW = [f"K{i:02d}" for i in range(25)]


def test_insert_len_dedupe_order():
    for s in sc.SLOTS:
        for x in ("NEW", "K03", "K24", sc.PLACEHOLDER):
            out = sc.insert_at(ROW, x, s)
            assert len(out) <= 25 and len(set(out)) == len(out)
            assert out[s - 1] == x
            others = [r for r in out if r != x]
            assert others == [r for r in ROW if r != x][:len(others)]


def test_short_row():
    assert sc.insert_at(["a", "b"], "x", 10) == ["a", "b", "x"]


def test_noop_s26_and_none():
    assert sc.insert_at(ROW, "K03", 26) == ROW
    assert sc.insert_at(ROW, None, 5) == ROW


def test_truth_displacement():
    assert sc.rank_in(sc.insert_at(ROW, "NEW", 2), "K01") == 3
    assert sc.rank_in(sc.insert_at(ROW, "NEW", 2), "K00") == 1
    assert sc.rank_in(sc.insert_at(ROW, "NEW", 2), "K24") == 0   # pushed out
    assert sc.rank_in(sc.insert_at(ROW, "K05", 2), "K05") == 2   # generator hit


def test_keep_higher_variant():
    assert sc.insert_at(ROW, "K01", 5, keep_higher=True) == ROW
    assert sc.insert_at(ROW, "K10", 5, keep_higher=True) == sc.insert_at(ROW, "K10", 5)


def _real(shard):
    p = sc.CACHE.format(shard)
    assert os.path.exists(p), f"run `slot_curve.py rows --shard {shard}` first"
    return pickle.load(open(p, "rb"))


def test_real_pseudo_c3():
    r = _real("pseudo_c3")
    n_x = sum(x is not None for x in r["x"])
    assert n_x > 500, n_x
    assert abs(r["mrr"] - 0.0562) < 5e-5
    assert sum(len(x) for x in r["rows"]) > 5000
    for row in r["rows"]:
        assert len(row) <= 25 and len(set(row)) == len(row), "V22 row has duplicate IK14"
    assert np.all(sc.delta_for_slot(r["rows"], r["truth"], r["x"], 26) == 0)
    assert np.any(sc.delta_for_slot(r["rows"], r["truth"], r["x"], 2) != 0)
    for s in (2, 7, 25):
        for row, x in zip(r["rows"], r["x"]):
            o = sc.insert_at(row, x, s)
            assert len(o) <= 25 and len(set(o)) == len(o)


def test_real_other_cohorts_noop_and_cost_only():
    for sh in ("C2", "C1_npx"):
        r = _real(sh)
        ph = [sc.PLACEHOLDER] * len(r["rows"])
        assert len(r["rows"]) > 100
        assert np.all(sc.delta_for_slot(r["rows"], r["truth"], ph, 26) == 0)
        d = sc.delta_for_slot(r["rows"], r["truth"], ph, 2)
        assert np.all(d <= 0) and np.any(d < 0)


def test_no_fpnet_in_choice():
    src = open(sc.__file__, encoding="utf-8").read()
    body = src.split("def mutfirst_ik")[1].split("def delta_for_slot")[0]
    assert "fz" not in body and "gen_rank" not in body


if __name__ == "__main__":
    fails = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            try:
                v()
                print("PASS", k)
            except Exception as e:
                fails += 1
                print("FAIL", k, repr(e))
    sys.exit(1 if fails else 0)
