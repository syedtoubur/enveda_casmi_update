"""Offline checks for the Phase-4 submission notebook (no Kaggle, no GPU, no ms-pred, no RDKit needed).

1. the generated notebook is up to date; every code cell parses; the phase1 kernel / library / pool /
   evidence cells, simsim_common and the worker are each embedded exactly once and verbatim
2. phase1's vectorised neutral_mass survives simsim_common being embedded after it
3. the submission helpers: spectrum_ces, sim_priority, build_requests, rows_that_fit, rank_two_tier
4. with numba: spectrum_sv (max / sum / mid) through the real phase1 kernels, and the notebook's pass-2
   cell executed on a toy pool (stage A -> B -> C slot filling, ik14 dedup, tier order, exactly 25)
"""
import ast
import importlib.util
import json
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mk_simple_sim as mk  # noqa: E402
import simsim_common as sc  # noqa: E402


def code_cells():
    nb = json.loads(mk.DST.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def check_build():
    before = {d: d.read_text(encoding="utf-8") if d.exists() else None for d in (mk.DST, mk.DST_NOSIM)}
    mk.build()
    for d, b in before.items():
        assert b is None or b == d.read_text(encoding="utf-8"), \
            f"{d.name} was stale -- it has now been regenerated; re-run the test"
    # the control differs from the submission in exactly one line: CFG.USE_SIM
    a, b = mk.render().split("\n"), mk.render_nosim().split("\n")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(diff) == 1 and "USE_SIM = False" in diff[0][1], diff
    code = code_cells()
    for i, c in enumerate(code):
        ast.parse(c, filename=f"{mk.DST.name}[code {i}]")
    bodies = {"kernels": mk.phase1_kernels(), "common": mk.COMMON.read_text(encoding="utf-8"),
              **{marker: mk.phase1_cell(marker) for marker in mk.PHASE1_CELLS.values()}}
    for name, body in bodies.items():
        assert sum(c.strip("\n") == body.strip("\n") for c in code) == 1, f"{name} not embedded verbatim once"
    emb = [c for c in code if "WORKER_SRC = " in c]
    assert len(emb) == 1, "expected one worker cell"
    line = next(ln for ln in emb[0].split("\n") if ln.startswith("WORKER_SRC = "))
    assert ast.literal_eval(line[len("WORKER_SRC = "):]) == mk.WORKER.read_text(encoding="utf-8")
    # order matters: kernels -> library -> pool -> evidence -> common
    pos = [next(i for i, c in enumerate(code) if c.strip("\n") == bodies[k].strip("\n"))
           for k in ("kernels", "def load_library(", "def build_candidate_pool(", "def frag_scores(", "common")]
    assert pos == sorted(pos), f"embedded cells out of order: {pos}"
    return code


def check_no_shadowing():
    ns = {}
    exec(mk.phase1_cell("def load_library("), ns)        # phase1: vectorised neutral_mass
    exec(mk.COMMON.read_text(encoding="utf-8"), ns)      # embedded after it in the notebook
    nm = ns["neutral_mass"](np.array([195.0877, 193.0732]), np.array(["[M+H]+", "[M-H]-"], dtype=object))
    assert nm.shape == (2,) and abs(nm[0] - (195.0877 - ns["PROTON"])) < 1e-9, nm
    assert abs(ns["neutral_mass1"](195.0877, "[M+H]+") - nm[0]) < 1e-9
    assert ns["formula_mass"]("C8H10N4O2") == sc.formula_mass("C8H10N4O2")


def check_helpers():
    # CE lists: 0 eV dropped, merged lists expanded (max/sum) or averaged (mid), empty -> default
    assert sc.spectrum_ces((20.0, 40.0, 60.0), "max", 40.0) == ([20.0, 40.0, 60.0], False)
    assert sc.spectrum_ces((0.0, 20.0, 40.0, 60.0, 80.0), "sum", 40.0) == ([20.0, 40.0, 60.0, 80.0], False)
    assert sc.spectrum_ces((20.0, 50.0), "mid", 40.0) == ([35.0], False)
    assert sc.spectrum_ces((), "max", 40.0) == ([40.0], True)
    assert sc.spectrum_ces((0.0,), "mid", 30.0) == ([30.0], True)
    assert sc.spectrum_ces((35.0, 35.0), "max", 40.0) == ([35.0], False)

    # priority: lv desc, fr desc, |ppm| asc
    assert list(sc.sim_priority([0.2, 0.5, 0.2, 0.2], [0.1, 0.0, 0.3, 0.1], [1.0, 9.0, 5.0, 0.5])) == [3, 0, 1, 2]

    # requests: theoretical precursor, one row per (key, adduct, ce), shared rows keep the best priority
    parts = [dict(key=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"], smiles=["C", "CC"], mass=[300.0, 300.001], prio=[0, 1],
                  conds=[("[M+H]+", 20.0), ("[M+H]+", 40.0)]),
             dict(key=["BBBBBBBBBBBBBB"], smiles=["CC"], mass=[300.001], prio=[0], conds=[("[M+H]+", 40.0)]),
             dict(key=[], smiles=[], mass=[], prio=[], conds=[("[M+H]+", 40.0)]),
             dict(key=["CCCCCCCCCCCCCC"], smiles=["CCC"], mass=[301.0], prio=[0], conds=[])]
    req = sc.build_requests(parts)
    assert list(req.columns) == sc.REQ_COLS and len(req) == 4, req
    assert req.duplicated(["key", "adduct", "ce_ev"]).sum() == 0
    b40 = req[(req.key == "BBBBBBBBBBBBBB") & (req.ce_ev == 40.0)]
    assert len(b40) == 1 and int(b40.prio.iloc[0]) == 0
    assert list(req.prio) == sorted(req.prio)
    assert np.allclose(req.precursor_mz, [sc.precursor_of(m, "[M+H]+") for m in
                                          np.where(req.key == "AAAAAAAAAAAAAA", 300.0, 300.001)])
    assert (req.ce == req.ce_ev).all() and (req.instrument == "QTOF").all()
    assert len(sc.build_requests([])) == 0 and list(sc.build_requests([]).columns) == sc.REQ_COLS

    # watchdog arithmetic
    assert sc.rows_that_fit(110.0, 10.0, 10.0) == 1000
    assert sc.rows_that_fit(5.0, 10.0, 10.0) == 0

    # two-tier order: tier 1 by lv; tier 2 by sv, fr, |ppm|; NaN sv/fr = 0; NaN ppm last
    lv = [0.9, 0.2, 0.75, 0.0, 0.0, 0.0, 0.74]
    sv = [0.0, 0.5, 0.0, 0.5, np.nan, 0.0, 0.1]
    fr = [0.0, 0.1, 0.0, 0.1, 0.9, 0.9, 0.0]
    ppm = [1.0, 3.0, 1.0, 2.0, 1.0, np.nan, 1.0]
    assert list(sc.rank_two_tier(lv, sv, fr, ppm, 0.75)) == [0, 2, 3, 1, 6, 4, 5]
    assert list(sc.rank_two_tier([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 1.0], 0.75)) == [1, 0]
    assert len(sc.rank_two_tier([], [], [], [], 0.75)) == 0


def load_kernels():
    try:
        import numba  # noqa: F401
    except ImportError:
        return None
    tmp = Path(tempfile.mkdtemp()) / "phase1_kernels.py"      # numba cache=True needs a real file
    tmp.write_text(mk.phase1_kernels(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("phase1_kernels", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sc._clean, sc.entropy_sim = mod._clean, mod.entropy_sim
    return mod


def check_spectrum_sv():
    q = sc.prep(np.array([100.0, 150.0, 200.0], np.float32), np.array([1.0, 0.5, 0.2], np.float32), 300.0)
    a = (np.array([100.0, 150.0, 200.0], np.float32), np.array([1.0, 0.5, 0.2], np.float32), 300.0)
    b = (np.array([120.0, 170.0], np.float32), np.array([1.0, 1.0], np.float32), 300.0)
    assert abs(sc.spectrum_sv(q, [a, b], "max") - 1.0) < 1e-5
    assert sc.spectrum_sv(q, [None, b], "max") < 1e-5
    s = sc.spectrum_sv(q, [a, b], "sum")
    assert 0.1 < s < 0.99, s                           # b's peaks dilute the summed spectrum
    assert abs(sc.spectrum_sv(q, [a, None], "sum") - 1.0) < 1e-5
    assert sc.spectrum_sv(q, [None, None], "max") == 0.0
    assert sc.spectrum_sv((np.empty(0), np.empty(0)), [a], "max") == 0.0
    assert abs(sc.spectrum_sv(q, [a], "mid") - 1.0) < 1e-5


class ToyPool:
    """CandidatePool's interface (window / keys / smiles / mass) on a hand-made mass list."""

    def __init__(self, mass, keys):
        o = np.argsort(mass)
        self.mass = np.asarray(mass, np.float64)[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.array([f"S{k}" for k in self.keys], dtype=object)

    def window(self, t, ppm):
        a = np.searchsorted(self.mass, t * (1 - ppm / 1e6), "left")
        b = np.searchsorted(self.mass, t * (1 + ppm / 1e6), "right")
        return np.arange(a, b)


def check_pass2(code):
    """Execute the notebook's pass-2 cell on a toy pool: stage A (5 candidates, one duplicate key), B, C."""
    cell = next(c for c in code if "def take_slots(" in c and "rank_two_tier" in c)
    t = 500.0
    ppms = [-3, -1, 0, 2, 4] + [-18, -15, -12, -10, 10, 12, 15, 18] + \
           [-40, -35, -30, -25, -22, 22, 25, 30, 35, 40] + [-110, -90, -70, -60, -50, 50, 60, 70, 90, 110]
    mass = [t * (1 + p * 1e-6) for p in ppms]
    keys = [f"K{i:013d}" for i in range(len(mass))]
    keys[4] = keys[3]                                            # a duplicate ik14 inside stage A
    pool = ToyPool(mass, keys)
    a_idx = pool.window(t, 8.5)
    assert len(a_idx) == 5
    k_t1, k_sv = pool.keys[a_idx[0]], pool.keys[a_idx[2]]
    CFG = types.SimpleNamespace(PPM_WIN=8.5, PPM_FALLBACK=30.0, PPM_EXPAND=(8.5, 20.0, 45.0, 80.0), TOPN=25,
                                EXPAND_BACKFILL=True, BACKFILL_PPM=120.0, TAU=0.75, MERGE_RULE="max")
    q = sc.prep(np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32), 510.0)
    plan = dict(target=t, deep=True, lib_hits={k_t1: 0.95}, ion_mode=1.0, specs=[],
                fr={int(c): 0.0 for c in range(len(mass))}, qs=[("[M+H]+", [40.0], q)], conds=[("[M+H]+", 40.0)])
    SIM = {(k_sv, "[M+H]+", 40.0): (np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32), 510.0)}
    ns = dict(np=np, time=time, CFG=CFG, pool=pool, SIM=SIM, FAST_MODE=True, DEEP=[0],
              mols=[("m0", None), ("m1", None)], PLANS={0: plan},
              rank_two_tier=sc.rank_two_tier, spectrum_sv=sc.spectrum_sv, TIMES={},
              frag_scores=lambda smis, specs, mode: np.zeros(len(smis), np.float32))
    exec(cell, ns)
    rows = dict(ns["rows"])
    got = rows["m0"].split(";")
    assert len(got) == 25 and len(set(got)) == 25, got
    assert got[0] == f"S{k_t1}" and got[1] == f"S{k_sv}", got[:3]     # tier 1, then the best sv
    assert got[:4].count(f"S{keys[3]}") <= 1                           # duplicate ik14 emitted once
    assert rows["m1"] == ";".join(["CCO"] * 25)                          # no plan -> padded
    assert len(ns["diag"]) == 1 and ns["diag"][0][-1] is True and 0 in ns["STAGE_A"]


def check_pass1_and_sim(code):
    """Execute the notebook's pass-1 and simulate cells on a toy pool with a fake worker.

    Molecule m0: covered [M+H]+ with a merged [0,20,40] list, plus an uncovered [M+K]+ spectrum.
    Molecule m1: only uncovered spectra -> no requests. A candidate with lv >= tau is not simulated
    (SIM_TIER1 False). The fake worker fails one row; the deadline check runs on the real clock.
    """
    p1 = next(c for c in code if "PLANS = {}" in c)
    simc = next(c for c in code if "SIM = {}" in c)
    t = 500.0
    pool = ToyPool([t * (1 + p * 1e-6) for p in (-3, -1, 0, 2, 4, 30, 60)], [f"K{i:013d}" for i in range(7)])
    k_t1 = pool.keys[0]
    CFG = types.SimpleNamespace(PPM_CENTER=0.0, TAU=0.75, MERGE_RULE="max", COVERED=("[M+H]+", "[M-H]-"), CE_DEFAULT=40.0, INSTRUMENT="QTOF", CE_MODE="ev",
                                FAST_DEV_MOLECULES=30, CHUNK=4, BATCH_RETRY=32, T4_RATE=36.6, STARTUP_S=10.4,
                                HARD_BUDGET_H=7.5, PASS2_S_PER_MOL=1.0, USE_SIM=True)
    spec = dict(ms2_mzs=np.array([100.0, 150.0]), ms2_normalized_intensities=np.array([1.0, 0.5]))
    sub0 = pd.DataFrame([dict(adduct="[M+H]+", precursor_mz=t + 1.007, collision_energy_ev=np.array([0.0, 20.0, 40.0]),
                              ionization_mode="positive", nm=t, **spec),
                         dict(adduct="[M+K]+", precursor_mz=t + 38.96, collision_energy_ev=np.array([20.0]),
                              ionization_mode="positive", nm=t, **spec)])
    sub1 = sub0.iloc[[1]].copy()
    calls = []

    def fake_sim_call(req, tag, bs=None):
        calls.append((tag, len(req)))
        n = len(req)
        spectra = [(np.array([100.0], np.float32), np.array([1.0], np.float32))] * n
        ok = np.ones(n, bool)
        ok[0] = False
        err = np.array(["predict:X"] + [""] * (n - 1))
        return spectra, ok, err, dict(mol_per_s=40.0, t_pred=1.0), 2.0

    work = Path(tempfile.mkdtemp())
    ns = dict(np=np, pd=pd, time=time, math=__import__("math"), CFG=CFG, pool=pool, T0=time.time(),
              mols=[("m0", sub0), ("m1", sub1)], IS_FULL_RUN=True, FAST_MODE=False, SIM_TIER1=False,
              lib_sim=lambda L, specs, target: {k_t1: 0.9}, L=None,
              frag_scores=lambda smis, specs, mode: np.linspace(0.5, 0.1, len(smis)).astype(np.float32),
              sim_call=fake_sim_call, WORK=work, TIMES={},
              **{k: getattr(sc, k) for k in ("scored_range", "spectrum_ces", "ce_tuple", "prep", "sim_priority",
                                             "build_requests", "rows_that_fit", "err_counts")})
    exec(p1, ns)
    plans, req = ns["PLANS"], ns["REQ"]
    assert plans[0]["conds"] == [("[M+H]+", 20.0), ("[M+H]+", 40.0)], plans[0]["conds"]
    assert plans[1]["conds"] == [] and plans[1]["n_sim"] == plans[1]["n_cand"] - 1
    assert plans[0]["n_cand"] == 7 and plans[0]["n_sim"] == 6          # scored_range expands; tier-1 skipped
    assert k_t1 not in set(req.key) and len(req) == 12 and set(req.ce_ev) == {20.0, 40.0}
    exec(simc, ns)
    assert [n for _, n in calls] == [4, 4, 4], calls
    assert len(ns["SIM"]) == 12 - 3 and ns["ERRS"] == {"predict": 3} and ns["CAPPED_ROWS"] == 0
    assert abs(ns["SIM_OK"] - 9 / 12) < 1e-9 and ns["rate"] == 40.0

    # watchdog: no time left -> nothing simulated, everything counted as capped
    calls.clear()
    exec(simc, dict(ns, CFG=types.SimpleNamespace(**{**vars(CFG), "HARD_BUDGET_H": 0.0})))
    assert calls == []

    # worker crash twice: full run degrades (rows counted as failed), fast commit raises
    def crash(req, tag, bs=None):
        calls.append((tag, bs))
        raise RuntimeError("boom")
    calls.clear()
    ns2 = dict(ns, sim_call=crash)
    exec(simc, ns2)
    assert ns2["FAILED_ROWS"] == 12 and len(ns2["SIM"]) == 0 and calls[1] == ("c0r", 32), calls
    try:
        exec(simc, dict(ns, sim_call=crash, FAST_MODE=True))
        raise AssertionError("a fast commit must raise on a worker crash")
    except RuntimeError:
        pass

    # the no-simulation control: no requests, the worker is never called, sv stays 0
    calls.clear()
    ns3 = dict(ns, sim_call=fake_sim_call, CFG=types.SimpleNamespace(**{**vars(CFG), "USE_SIM": False}))
    exec(p1, ns3)
    assert len(ns3["REQ"]) == 0 and ns3["PLANS"][0]["n_sim"] == 0 and ns3["PLANS"][0]["n_cand"] == 7
    exec(simc, ns3)
    assert calls == [] and ns3["SIM"] == {} and np.isnan(ns3["SIM_OK"])


def main():
    code = check_build()
    check_no_shadowing()
    check_helpers()
    ran = load_kernels() is not None
    if ran:
        check_spectrum_sv()
        check_pass2(code)
        check_pass1_and_sim(code)
    print("test_simple_sim: OK" + ("" if ran else " (numba checks SKIPPED -- numba not installed)"))


if __name__ == "__main__":
    main()
