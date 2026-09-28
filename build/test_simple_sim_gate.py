"""Offline checks for the Phase-0 gate build (no Colab, no GPU, no ms-pred needed).

1. the generated notebooks are up to date with their sources (regenerate and compare)
2. every code cell parses, apart from Colab/IPython-only syntax (none is used)
3. the embedded kernels are phase1's, byte for byte, and the embedded worker is iceberg_worker.py
4. the worker parses and knows the CASMI -> ms-pred spellings for the two adducts that differ
5. the worker's precursor cut (to_peaks) and G2b's Poisson-binomial p-value, on hand-made inputs
6. the phase1 kernels score identical spectra 1.0 and disjoint ones 0.0 (numba, CPU)
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mk_simple_sim_gate as mk  # noqa: E402


def load_func(tree, name, ns):
    """Compile one top-level function out of a module's AST (its imports -- torch, ms_pred -- are not needed)."""
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), ns)
    return ns[name]


def cells(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def main():
    kernels = mk.phase1_kernels()
    worker = mk.WORKER.read_text(encoding="utf-8")
    for src, dst, needs_worker in mk.BUILDS:
        before = dst.read_text(encoding="utf-8") if dst.exists() else None
        mk.build(src, dst, needs_worker)
        assert before is None or before == dst.read_text(encoding="utf-8"), \
            f"{dst.name} was stale -- it has now been regenerated; re-run the test"
        code = cells(dst)
        for i, c in enumerate(code):
            ast.parse(c, filename=f"{dst.name}[code {i}]")
        # py2nb trims blank lines at cell edges; the body must match exactly
        assert sum(c.strip("\n") == kernels.strip("\n") for c in code) == 1, \
            f"{dst.name}: phase1 kernels not embedded verbatim"
        if needs_worker:
            emb = [c for c in code if c.startswith("WORKER_SRC = ")]
            assert len(emb) == 1
            src_lit = emb[0].split("\n", 1)[0][len("WORKER_SRC = "):]
            assert ast.literal_eval(src_lit) == worker, "embedded worker differs from iceberg_worker.py"

    tree = ast.parse(worker)
    amap = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "ADDUCT_MAP")
    amap = ast.literal_eval(amap.value)
    assert amap == {"[M-2H2O+H]+": "[M+H-2H2O]+", "[M+CH2O2-H]-": "[M+CHO2]-"}, amap

    import numpy as np
    to_peaks = load_func(tree, "to_peaks", {"np": np})
    # [M-H2O+H]+ at 300: the unshifted root at +17 Da must go; top-k then renormalises the rest
    raw = np.array([[100.0, 5.0], [200.0, 10.0], [301.5, 2.0], [317.0, 20.0], [150.0, np.nan]])
    mz, it, n_ab, share, off = to_peaks(raw, 64, 300.0, 2.0)
    assert list(mz) == [100.0, 200.0, 301.5] and abs(it.max() - 1.0) < 1e-6 and it[1] == 1.0, (mz, it)
    assert n_ab == 1 and abs(share - 20 / 37) < 1e-6 and abs(off - 17.0) < 1e-6, (n_ab, share, off)
    mz, it, n_ab, share, off = to_peaks(np.array([[320.0, 1.0]]), 64, 300.0, 2.0)
    assert len(mz) == 0 and n_ab == 1 and share == 1.0
    mz, it, n_ab, share, off = to_peaks(raw[:2], 1, 300.0, 2.0)
    assert list(mz) == [200.0] and n_ab == 0 and np.isnan(off)

    gate_tree = ast.parse((HERE / "simple_sim_gate.py").read_text(encoding="utf-8"))
    pb_sf = load_func(gate_tree, "pb_sf", {"np": np})
    assert abs(pb_sf([0.5, 0.5], 1) - 0.75) < 1e-12 and abs(pb_sf([0.5, 0.5], 2) - 0.25) < 1e-12
    assert abs(pb_sf([0.1] * 10, 0) - 1.0) < 1e-12 and abs(pb_sf([0.2, 0.3], 3)) < 1e-12
    try:
        import numba  # noqa: F401
    except ImportError:
        print("test_simple_sim_gate: OK (kernel numeric check SKIPPED -- numba not installed)")
        return
    import importlib.util
    import tempfile
    # numba's cache=True needs a real source file, so load the kernels as a module
    tmp = Path(tempfile.mkdtemp()) / "phase1_kernels.py"
    tmp.write_text(kernels, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("phase1_kernels", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = vars(mod)
    mz = np.array([100.0, 150.0, 200.0], np.float32)
    it = np.array([1.0, 0.5, 0.25], np.float32)
    a = ns["_clean"](mz, it, 0.002, 64, 1.0, True)
    b = ns["_clean"](mz + 30.0, it, 0.002, 64, 1.0, True)
    same = ns["entropy_sim"](a[0], a[1], a[0], a[1], 0.01)
    diff = ns["entropy_sim"](a[0], a[1], b[0], b[1], 0.01)
    assert abs(same - 1.0) < 1e-5 and abs(diff) < 1e-5, (same, diff)
    print("test_simple_sim_gate: OK")


if __name__ == "__main__":
    main()
