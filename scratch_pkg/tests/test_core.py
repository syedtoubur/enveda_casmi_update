"""casmi_scratch offline checks (no Kaggle, no GPU, no ms-pred). Ported from build/test_simple_sim.py.

Run: python scratch_pkg/tests/run.py   (or pytest, if installed)
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi_scratch import chem, pipeline, rank, sim, spectra, submit  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402
from casmi_scratch.library import build_library, lib_sim  # noqa: E402
from casmi_scratch.pool import CandidatePool  # noqa: E402

CFG = Config()


def no_frag(smis, specs, mode, cfg):
    return np.zeros(len(smis), np.float32)


def toy_pool(target, ppms, keys=None):
    mass = [target * (1 + p * 1e-6) for p in ppms]
    keys = keys or [f"K{i:013d}" for i in range(len(mass))]
    return CandidatePool(mass, keys, [f"S{k}" for k in keys])


def spectrum_row(adduct, prec, ces, mode="positive"):
    return dict(adduct=adduct, precursor_mz=prec, collision_energy_ev=np.array(ces, float), ionization_mode=mode,
                ms2_mzs=np.array([100.0, 150.0]), ms2_normalized_intensities=np.array([1.0, 0.5]))


def test_chem():
    nm = chem.neutral_mass(np.array([195.0877, 193.0732, 1.0]), np.array(["[M+H]+", "[M-H]-", "[X]"], dtype=object))
    assert abs(nm[0] - (195.0877 - chem.PROTON)) < 1e-9 and abs(nm[1] - (193.0732 + chem.PROTON)) < 1e-9
    assert np.isnan(nm[2])
    assert abs(chem.precursor_of(194.0804, "[M+H]+") - (194.0804 + chem.PROTON)) < 1e-12
    assert abs(chem.formula_mass("C8H10N4O2") - 194.08037557) < 1e-6
    assert np.isnan(chem.formula_mass("C8H10Xx")) and np.isnan(chem.formula_mass(None))
    t = chem.query_target([500.0, 500.0, np.nan], 1.45)
    assert abs(t - 500.0 / (1 + 1.45e-6)) < 1e-12


def test_ces_priority_requests():
    assert spectra.spectrum_ces((20.0, 40.0, 60.0), "max", 40.0) == ([20.0, 40.0, 60.0], False)
    assert spectra.spectrum_ces((0.0, 20.0, 40.0), "sum", 40.0) == ([20.0, 40.0], False)
    assert spectra.spectrum_ces((20.0, 50.0), "mid", 40.0) == ([35.0], False)
    assert spectra.spectrum_ces((), "max", 40.0) == ([40.0], True)
    assert spectra.ce_tuple(None) == () and spectra.ce_tuple(np.array([20.04])) == (20.0,)
    assert list(sim.sim_priority([0.2, 0.5, 0.2, 0.2], [0.1, 0.0, 0.3, 0.1], [1.0, 9.0, 5.0, 0.5])) == [3, 0, 1, 2]
    parts = [dict(key=["A" * 14, "B" * 14], smiles=["C", "CC"], mass=[300.0, 300.001], prio=[0, 1],
                  conds=[("[M+H]+", 20.0), ("[M+H]+", 40.0)]),
             dict(key=["B" * 14], smiles=["CC"], mass=[300.001], prio=[0], conds=[("[M+H]+", 40.0)]),
             dict(key=[], smiles=[], mass=[], prio=[], conds=[("[M+H]+", 40.0)])]
    req = sim.build_requests(parts)
    assert list(req.columns) == sim.REQ_COLS and len(req) == 4
    b40 = req[(req.key == "B" * 14) & (req.ce_ev == 40.0)]
    assert len(b40) == 1 and int(b40.prio.iloc[0]) == 0 and list(req.prio) == sorted(req.prio)
    assert len(sim.build_requests([])) == 0
    assert sim.rows_that_fit(110.0, 10.0, 10.0) == 1000 and sim.rows_that_fit(5.0, 10.0, 10.0) == 0


def test_rank_two_tier_and_slots():
    lv = [0.9, 0.2, 0.75, 0.0, 0.0, 0.0, 0.74]
    sv = [0.0, 0.5, 0.0, 0.5, np.nan, 0.0, 0.1]
    fr = [0.0, 0.1, 0.0, 0.1, 0.9, 0.9, 0.0]
    ppm = [1.0, 3.0, 1.0, 2.0, 1.0, np.nan, 1.0]
    assert list(rank.rank_two_tier(lv, sv, fr, ppm, 0.75)) == [0, 2, 3, 1, 6, 4, 5]
    assert len(rank.rank_two_tier([], [], [], [], 0.75)) == 0
    s = rank.Slots(3)
    assert s.add("ABCDEFGHIJKLMN-XX", "a") and not s.add("ABCDEFGHIJKLMN", "b") and not s.add("short", "c")
    assert s.padded() == ["a", "CCO", "CCO"]


def test_max_peaks_reaches_sv():
    """The R0 fix: Config.max_peaks drives the ICEBERG-side cleaning too (was hard-coded 64)."""
    mz = np.arange(100, 300, dtype=np.float32)
    it = np.linspace(1.0, 0.5, len(mz)).astype(np.float32)
    assert len(spectra.prep(mz, it, 400.0, CFG.with_(max_peaks=64))[0]) == 64
    assert len(spectra.prep(mz, it, 400.0, CFG.with_(max_peaks=100))[0]) == 100
    assert len(spectra.prep(mz, it, 150.0, CFG)[0]) == 53          # precursor + 2 Da cut


def test_spectrum_sv():
    q = spectra.prep(np.array([100.0, 150.0, 200.0]), np.array([1.0, 0.5, 0.2]), 300.0, CFG)
    a = (np.array([100.0, 150.0, 200.0], np.float32), np.array([1.0, 0.5, 0.2], np.float32), 300.0)
    b = (np.array([120.0, 170.0], np.float32), np.array([1.0, 1.0], np.float32), 300.0)
    assert abs(spectra.spectrum_sv(q, [a, b], CFG) - 1.0) < 1e-5
    assert spectra.spectrum_sv(q, [None, b], CFG) < 1e-5
    s = spectra.spectrum_sv(q, [a, b], CFG.with_(merge_rule="sum"))
    assert 0.1 < s < 0.99
    assert spectra.spectrum_sv(q, [None, None], CFG) == 0.0


def test_library_lv_and_exclude():
    # two spectra of key X (one identical to the query), one of key Y at another mass
    peaks = [([100.0, 150.0], [1.0, 0.5]), ([100.0, 151.0], [1.0, 0.5]), ([100.0, 150.0], [1.0, 0.5])]
    off = np.cumsum([0] + [len(p[0]) for p in peaks])
    lib = build_library(off, np.concatenate([p[0] for p in peaks]), np.concatenate([p[1] for p in peaks]),
                        prec=np.array([501.007276, 501.007276, 601.007276]),
                        adduct=np.array(["[M+H]+"] * 3, dtype=object),
                        ik=np.array(["X" * 14, "X" * 14, "Y" * 14], dtype=object), struct_mass=False)
    target = 501.007276 - chem.PROTON
    hits = lib_sim(lib, [(np.array([100.0, 150.0]), np.array([1.0, 0.5]))], target, CFG)
    assert set(hits) == {"X" * 14} and hits["X" * 14] > 0.99
    excl = np.array([True, False, False])
    hits2 = lib_sim(lib, [(np.array([100.0, 150.0]), np.array([1.0, 0.5]))], target, CFG, exclude=excl)
    assert 0 < hits2["X" * 14] < hits["X" * 14]
    assert lib_sim(lib, [(np.array([100.0]), np.array([1.0]))], target, CFG, exclude=np.array([1, 1, 0], bool)) == {}


def test_scored_range():
    pool = toy_pool(500.0, [-3, -1, 0, 30, 60])
    assert pipeline.scored_range(pool, 500.0, CFG) == (0, 5)          # < 25 -> widened to every tier
    pool2 = toy_pool(500.0, [200, 300])
    assert pipeline.scored_range(pool2, 500.0, CFG) == (0, 0)         # nothing within 80 ppm


def test_pass2_slot_filling():
    t = 500.0
    ppms = [-3, -1, 0, 2, 4] + [-18, -15, -12, -10, 10, 12, 15, 18] + \
           [-40, -35, -30, -25, -22, 22, 25, 30, 35, 40] + [-110, -90, -70, -60, -50, 50, 60, 70, 90, 110]
    keys = [f"K{i:013d}" for i in range(len(ppms))]
    keys[4] = keys[3]                                               # a duplicate ik14 in stage A
    pool = toy_pool(t, ppms, keys)
    a = pool.window(t, 8.5)
    assert len(a) == 5
    k_t1, k_sv = pool.keys[a[0]], pool.keys[a[2]]
    q = spectra.prep(np.array([100.0, 150.0]), np.array([1.0, 0.5]), 510.0, CFG)
    plan = dict(target=t, lib_hits={k_t1: 0.95}, ion_mode=1.0, specs=[], fr={}, qs=[("[M+H]+", [40.0], q)],
                conds=[("[M+H]+", 40.0)])
    SIM = {(k_sv, "[M+H]+", 40.0): (np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32), 510.0)}
    slots, diag, _ = pipeline.rank_molecule(plan, pool, SIM, CFG, no_frag)
    got = slots.padded()
    assert len(got) == 25 and len(set(got)) == 25
    assert got[0] == f"S{k_t1}" and got[1] == f"S{k_sv}"
    assert diag["top_is_tier1"] and diag["n_cand"] == 5


def fake_worker(calls, fail_first=True):
    def call(req, tag, batch_size=None):
        calls.append((tag, len(req), batch_size))
        n = len(req)
        ok = np.ones(n, bool)
        if fail_first:
            ok[0] = False
        err = np.array(["predict:X" if not o else "" for o in ok])
        sp = [(np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32))] * n
        return sp, ok, err, dict(mol_per_s=40.0, t_pred=1.0), 2.0
    return call


def toy_run_inputs():
    t = 500.0
    pool = toy_pool(t, [-3, -1, 0, 2, 4, 30, 60])
    sub0 = pd.DataFrame([spectrum_row("[M+H]+", chem.precursor_of(t, "[M+H]+"), [0.0, 20.0, 40.0]),
                         spectrum_row("[M+K]+", chem.precursor_of(t, "[M+K]+"), [20.0])])
    sub1 = sub0.iloc[[1]].reset_index(drop=True)
    lib = build_library(np.array([0, 2]), np.array([100.0, 150.0]), np.array([1.0, 0.5]),
                        prec=np.array([chem.precursor_of(pool.mass[0], "[M+H]+")]),
                        adduct=np.array(["[M+H]+"], dtype=object), ik=np.array([pool.keys[0]], dtype=object),
                        struct_mass=False)
    return pool, lib, [("m0", sub0), ("m1", sub1)]


def test_run_end_to_end():
    pool, lib, mols = toy_run_inputs()
    cfg = CFG.with_(ppm_center=0.0, chunk=4)
    calls = []
    ranked, rep, aux = pipeline.run(mols, lib, pool, cfg, call=fake_worker(calls), frag_fn=no_frag, log=lambda s: None)
    p0, p1 = aux["plans"]["m0"], aux["plans"]["m1"]
    assert p0["conds"] == [("[M+H]+", 20.0), ("[M+H]+", 40.0)] and p1["conds"] == []
    assert p0["n_cand"] == 7 and len(p0["sim_idx"]) == 6             # the tier-1 library match is not simulated
    assert rep["n_requests"] == 12 and [n for _, n, _ in calls] == [4, 4, 4]
    assert rep["n_predicted"] == 9 and rep["sim_errors"] == {"predict": 3} and rep["capped_rows"] == 0
    assert math.isclose(rep["sim_ok_share"], 0.75)
    assert ranked["m0"][0] == f"S{pool.keys[0]}" and len(ranked["m0"]) == 25
    assert rep["tier1_rate"] == 1.0                                   # both match the library spectrum


def test_run_watchdog_crash_and_nosim():
    pool, lib, mols = toy_run_inputs()
    cfg = CFG.with_(ppm_center=0.0, chunk=4)
    calls = []
    _, rep, _ = pipeline.run(mols, lib, pool, cfg.with_(hard_budget_h=0.0), call=fake_worker(calls),
                             frag_fn=no_frag, log=lambda s: None)
    assert calls == [] and rep["capped_rows"] == 12

    def crash(req, tag, batch_size=None):
        calls.append((tag, batch_size))
        raise RuntimeError("boom")
    _, rep, _ = pipeline.run(mols, lib, pool, cfg, call=crash, frag_fn=no_frag, log=lambda s: None)
    assert rep["failed_rows"] == 12 and rep["n_predicted"] == 0 and calls[1] == ("c0r", 32)
    try:
        pipeline.run(mols, lib, pool, cfg, call=crash, strict=True, frag_fn=no_frag, log=lambda s: None)
        raise AssertionError("strict (fast commit) must raise on a worker crash")
    except RuntimeError:
        pass
    calls.clear()
    _, rep, aux = pipeline.run(mols, lib, pool, cfg.with_(use_sim=False), call=fake_worker(calls),
                               frag_fn=no_frag, log=lambda s: None)
    assert calls == [] and rep["n_requests"] == 0 and len(aux["plans"]["m0"]["sim_idx"]) == 0


def test_run_fast_mode_fallback():
    pool, lib, mols = toy_run_inputs()
    ranked, rep, aux = pipeline.run(mols, lib, pool, CFG.with_(ppm_center=0.0), call=None, deep={"m0"},
                                    frag_fn=no_frag, log=lambda s: None)
    assert rep["n_deep"] == 1 and not aux["plans"]["m1"]["deep"]
    assert ranked["m1"][0] == f"S{pool.keys[2]}"                     # mass-nearest (0 ppm) first


def test_submission_and_mode():
    fast, info = submit.run_mode(400, 1213, CFG, env={})
    assert fast and info["mode"] == "fast"
    assert not submit.run_mode(400, 1213, CFG, env={"KAGGLE_IS_COMPETITION_RERUN": "1"})[0]
    assert not submit.run_mode(401, 1500, CFG, env={})[0]
    assert not submit.run_mode(400, 1213, CFG.with_(force_full=True), env={})[0]
    sample = pd.DataFrame({"molecule_id": ["b", "a"], "smiles": ["", ""]})
    sub = submit.make_submission({"a": ["C"] * 30}, sample)
    assert list(sub.molecule_id) == ["b", "a"] and sub.smiles.iloc[0] == ";".join(["CCO"] * 25)
    assert sub.smiles.iloc[1].count(";") == 24


def test_metfrag_rdkit():
    from casmi_scratch.metfrag import fragment_masses, frag_scores
    caffeine = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    fm = fragment_masses(caffeine)
    assert abs(fm.max() - 194.0803756) < 1e-3 and len(fm) > 5
    spec = [(np.array([195.0877, 138.0662]), np.array([1.0, 0.6]))]            # [M+H]+ and a known fragment
    fr = frag_scores([caffeine, "CCCCCCCCCCCC", "not a smiles"], spec, 1.0, CFG.with_(frag_workers=1))
    assert fr[0] > 0.5 and fr[0] > fr[1] and fr[2] == 0.0


def test_worker_resource_matches_build():
    root = Path(__file__).resolve().parents[2]
    assert (root / "build" / "iceberg_worker.py").read_text(encoding="utf-8") == \
        sim.WORKER_SRC.read_text(encoding="utf-8"), "resources/iceberg_worker.py drifted from build/"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    t0 = time.time()
    for f in TESTS:
        f()
    print(f"test_core: {len(TESTS)} OK ({time.time() - t0:.1f}s)")
