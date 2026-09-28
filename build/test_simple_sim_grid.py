"""Offline checks for the Phase-1 grid build (no Colab, no Kaggle, no GPU, no ms-pred needed).

1. the generated notebooks are up to date with their sources (regenerate and compare)
2. every code cell parses; simsim_common is embedded verbatim in both, the phase1 kernels only in the
   Colab notebook, and the embedded worker is iceberg_worker.py
3. simsim_common on hand-made inputs: adduct physics, CE snapping and conditions, tolerance and base-CE
   rules, sum-merge, tie-aware truth rank, windows (stage A/B), load arithmetic, decision rule
4. prep / esim / sum_merge through the real phase1 kernels (numba, CPU), when numba is installed
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mk_simple_sim_grid as mk  # noqa: E402
import simsim_common as sc  # noqa: E402


def cells(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def check_build():
    common = mk.COMMON.read_text(encoding="utf-8").strip("\n")
    kernels = mk.phase1_kernels().strip("\n")
    worker = mk.WORKER.read_text(encoding="utf-8")
    for src, dst, needs_kernels in mk.BUILDS:
        before = dst.read_text(encoding="utf-8") if dst.exists() else None
        mk.build(src, dst, needs_kernels)
        assert before is None or before == dst.read_text(encoding="utf-8"), \
            f"{dst.name} was stale -- it has now been regenerated; re-run the test"
        code = cells(dst)
        for i, c in enumerate(code):
            ast.parse(c, filename=f"{dst.name}[code {i}]")
        assert sum(c.strip("\n") == common for c in code) == 1, f"{dst.name}: simsim_common not embedded verbatim"
        assert sum(c.strip("\n") == kernels for c in code) == (1 if needs_kernels else 0), f"{dst.name}: kernels"
        emb = [c for c in code if "WORKER_SRC = " in c]
        assert len(emb) == 1, f"{dst.name}: expected one worker cell"
        line = next(ln for ln in emb[0].split("\n") if ln.startswith("WORKER_SRC = "))
        assert ast.literal_eval(line[len("WORKER_SRC = "):]) == worker, "embedded worker differs from iceberg_worker.py"


def check_common():
    # adduct physics: phase1 values, and precursor_of inverts neutral_mass
    assert abs(sc.precursor_of(100.0, "[M+H]+") - 101.00727646) < 1e-6
    assert abs(sc.precursor_of(100.0, "[M-H]-") - 98.99272354) < 1e-6
    for ad in sc.TEST_ADDUCTS:
        assert abs(sc.neutral_mass1(sc.precursor_of(350.123, ad), ad) - 350.123) < 1e-9, ad
    assert np.isnan(sc.neutral_mass1(300.0, "[2M+H]+"))
    assert set(sc.ADDUCT_DELTA) == set(sc.TEST_ADDUCTS)
    assert abs(sc.formula_mass("C6H12O6") - 180.0633881) < 1e-6 and np.isnan(sc.formula_mass("C6H12Xx"))
    assert abs(sc.to_model_ce(40.0, 400.0, "nce") - 50.0) < 1e-12 and sc.to_model_ce(40.0, 400.0, "ev") == 40.0

    # CE tuples, snapping and per-spectrum conditions
    assert sc.ce_tuple(np.array([20.0, 40.04])) == (20.0, 40.0) and sc.ce_tuple(None) == ()
    grid = [20.0, 40.0, 60.0]
    assert sc.snap_ce(22.5, grid, 2.5) == 20.0 and sc.snap_ce(22.6, grid, 2.5) is None
    assert sc.snap_ce(30.0, grid, 10.0) == 20.0          # tie -> lower CE
    assert sc.snap_ce(35.0, grid, 0.0) is None and sc.snap_ce(40.0, grid, 0.0) == 40.0
    cov = ["[M+H]+", "[M-H]-"]
    assert sc.spectrum_conditions("[M+H]+", (20.0, 40.0, 60.0), grid, 2.5, cov) == \
        {("[M+H]+", 20.0, True), ("[M+H]+", 40.0, True), ("[M+H]+", 60.0, True)}
    assert sc.spectrum_conditions("[M+H]+", (0.0, 35.0), grid, 2.5, cov) == {("[M+H]+", 35.0, False)}
    assert sc.spectrum_conditions("[M+K]+", (20.0,), grid, 2.5, cov) == set()

    # tolerance and base-CE rules
    assert sc.choose_tolerance({2.5: 0.97, 5.0: 0.93, 10.0: 0.85, 15.0: 0.7}, 0.90) == 5.0
    assert sc.choose_tolerance({2.5: 0.8}, 0.90) == 0.0
    tuples = [(20.0,)] * 50 + [(40.0,)] * 45 + [(35.0,)] * 3 + [(10.0,)] * 2 + [(20.0, 40.0, 60.0)] * 10
    assert sc.base_ce_set(tuples, 0.95) == [20.0, 40.0, 60.0]
    assert sc.base_ce_set(tuples, 0.97) == [20.0, 35.0, 40.0, 60.0]
    assert sc.base_ce_set(tuples, 0.99) == [10.0, 20.0, 35.0, 40.0, 60.0]

    # sum-merge: shared fragment summed, renormalised to base 1, sorted; empties ignored
    a = (np.array([100.0, 150.0], np.float32), np.array([1.0, 0.5], np.float32))
    b = (np.array([100.0, 120.0], np.float32), np.array([0.2, 1.0], np.float32))
    mz, it = sc.sum_merge([a, b, (np.empty(0), np.empty(0))])
    assert list(mz) == [100.0, 120.0, 150.0], mz
    assert np.allclose(it, [1.0, 1.0 / 1.2, 0.5 / 1.2]), it
    assert len(sc.sum_merge([(np.empty(0), np.empty(0))])[0]) == 0

    # truth rank: ties count half, a tied truth is not top-1
    assert sc.rank_of_truth([0.9, 0.5, 0.1]) == (1, True)
    assert sc.rank_of_truth([0.5, 0.9, 0.5]) == (2.5, False)
    assert sc.rank_of_truth([0.0, 0.0, 0.0]) == (2.0, False)
    assert abs(sc.pb_sf([0.5, 0.5], 1) - 0.75) < 1e-12

    # windows: stage A 8.5 ppm; expands through the tiers until >= topn; fallback when A is empty
    t = 500.0
    pm = np.sort(np.concatenate([t * (1 + np.array([-5, 0, 5]) * 1e-6),       # 3 inside 8.5 ppm
                                 t * (1 + np.linspace(-19, 19, 10) * 1e-6),   # +10 inside 20 ppm
                                 t * (1 + np.linspace(-40, 40, 20) * 1e-6)]))  # +20 inside 45 ppm
    lo, hi = sc.window(pm, t, 8.5)
    assert hi - lo == 3 + 2 * 0 + int(np.sum(np.abs(np.linspace(-40, 40, 20)) <= 8.5)) + \
        int(np.sum(np.abs(np.linspace(-19, 19, 10)) <= 8.5))
    lo, hi = sc.scored_range(pm, t, topn=5)
    assert (lo, hi) == sc.window(pm, t, 8.5)                  # enough already: no expansion
    lo, hi = sc.scored_range(pm, t, topn=25)
    assert (lo, hi) == sc.window(pm, t, 45.0) and hi - lo == len(pm)
    far = np.array([t * (1 + 25e-6), t * (1 - 25e-6)])
    far.sort()
    assert sc.scored_range(far, t, topn=1) == sc.window(far, t, 30.0) == (0, 2)   # PPM_FALLBACK
    tq = sc.query_target([sc.precursor_of(t, "[M+H]+"), sc.precursor_of(t, "[M-H]-")], ["[M+H]+", "[M-H]-"])
    assert abs(tq - t / (1 + 1.45e-6)) < 1e-9

    # load arithmetic and the decision rule
    conds = {("[M+H]+", 20.0, True), ("[M+H]+", 40.0, True), ("[M+Na]+", 20.0, True), ("[M+H]+", 35.0, False)}
    assert len(sc.on_the_fly(conds, set())) == 4
    assert sorted(sc.on_the_fly(conds, {"[M+H]+"})) == [("[M+H]+", 35.0, False), ("[M+Na]+", 20.0, True)]
    assert sc.on_the_fly(conds, {"[M+H]+", "[M+Na]+"}) == [("[M+H]+", 35.0, False)]
    bt = sc.bootstrap_total([10, 10, 10], n_mol=400)
    assert bt["mean"] == bt["p50"] == bt["p95"] == 4000.0
    assert sc.bootstrap_total([], 400)["p95"] == 0.0
    h = sc.kaggle_hours(load=36000, rate=10.0, n_mol=400, t_base_s=9.0, fixed_s=0.0, simlib_gb=1.0,
                        load_s_per_gb=3600.0, worker_calls=0, startup_s=5.0)
    assert abs(h - (3600 + 3600 + 3600) / 3600) < 1e-12        # base 1 h + simlib 1 h + sims 1 h
    assert sc.kaggle_hours(0, 10.0, 0, 0.0, 0.0, 0.0, 0.0, worker_calls=20, startup_s=60.0) == 0.0
    opts = [dict(name="A", colab_h=0.0, kaggle_h=8.0), dict(name="B", colab_h=17.0, kaggle_h=5.0),
            dict(name="C", colab_h=52.0, kaggle_h=3.0)]
    assert sc.decide(opts, 6.3)["name"] == "B"
    assert sc.decide(opts, 9.0)["name"] == "A"
    assert sc.decide(opts, 2.0) is None                        # none fits -> STOP

    # read_response must share ONE array across all row slices (per-row z[key] access OOM'd Colab)
    import tempfile
    n = 50
    p = Path(tempfile.mkdtemp()) / "resp.npz"
    mzf = np.arange(n * 3, dtype=np.float32)
    np.savez(p, off=np.arange(0, n * 3 + 1, 3), mz=mzf, it=mzf, ok=np.ones(n, bool),
             err=np.array([""] * n), timing=json.dumps({"mol_per_s": 1.0}))
    spectra, ok, err, timing = sc.read_response(p, n)
    assert all(s[0].base is spectra[0][0].base for s in spectra), "row slices hold separate array copies"
    assert list(spectra[7][0]) == [21.0, 22.0, 23.0] and timing["mol_per_s"] == 1.0


def check_kernels():
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    import importlib.util
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "phase1_kernels.py"      # numba cache=True needs a real file
    tmp.write_text(mk.phase1_kernels(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("phase1_kernels", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sc._clean, sc.entropy_sim = mod._clean, mod.entropy_sim
    mz = np.array([100.0, 150.0, 200.0, 305.0], np.float32)
    it = np.array([1.0, 0.5, 0.25, 0.9], np.float32)
    a = sc.prep(mz, it, 300.0)
    assert 305.0 not in a[0], "precursor + 2 Da cut not applied"
    assert abs(sc.esim(a, a) - 1.0) < 1e-5 and sc.esim(a, (np.empty(0), np.empty(0))) == 0.0
    assert abs(sc.esim(a, sc.prep(mz + 30.0, it, 400.0))) < 1e-5
    # a sum-merge of identical spectra is the spectrum itself
    m2 = sc.sum_merge([(mz[:3], it[:3]), (mz[:3], it[:3])])
    assert abs(sc.esim(sc.prep(*m2, 300.0), a) - 1.0) < 1e-5
    return True


def main():
    check_build()
    check_common()
    ran = check_kernels()
    print("test_simple_sim_grid: OK" + ("" if ran else " (kernel numeric check SKIPPED -- numba not installed)"))


if __name__ == "__main__":
    main()
