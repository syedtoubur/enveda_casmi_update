"""Checks for build/patch_isdb_cfm_fallback.py (ISDB CFM-ID 4.0 fallback for negative-mode spectra).
Verifies:
- Gate G1: notebook structure, AST compilability, clean outputs, control preservation
- Gate G2: cosine similarity math, loader fallbacks, replay engine invariants (K3), activation (K2),
  and Cell 32 rerank_row factorial verification.
Local execution, no Kaggle environment needed.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_isdb_cfm_fallback as P  # noqa: E402
import casmi_post as CP  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    N += 1


def code_cells(nb):
    return [P.src_of(c) for c in nb["cells"] if c.get("cell_type") == "code"]


def compiles(src):
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(body)


# ==============================================================================
# Gate G1: Notebook Structure and Surgical Integrity
# ==============================================================================

def test_g1_notebook_delta_and_compilation():
    out = os.path.join(tempfile.mkdtemp(), "probe.ipynb")
    P.patch_notebook(P.SRC, out)
    nb = json.load(open(out, encoding="utf-8"))
    base_nb = json.load(open(P.SRC, encoding="utf-8"))
    
    base_cs = code_cells(base_nb)
    var_cs = code_cells(nb)
    
    # 1. Base notebook had 32 cells (all code cells). Patched has +1 ISDB loader cell = 33 cells.
    ok(len(var_cs) == len(base_cs) + 1, f"cell count delta: {len(base_cs)} -> {len(var_cs)}")
    
    # 2. All cells compile with AST
    for idx, c in enumerate(var_cs):
        try:
            compiles(c)
        except Exception as e:
            raise AssertionError(f"Cell {idx} failed AST compilation: {e}")
            
    # 3. Outputs and execution counts are cleared
    for idx, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            ok(c.get("outputs") == [], f"cell {idx} output not cleared")
            ok(c.get("execution_count") is None, f"cell {idx} execution_count not None")
            
    # 4. Invariance of all prior cells (cells 0..30 in base must match cells 0..30 in var)
    for i in range(len(base_cs) - 1):
        ok(base_cs[i] == var_cs[i], f"cell {i} modified unexpectedly")
        
    # 5. Check ISDB loader cell
    loader_cells = [c for c in var_cs if "load_isdb_scores" in c]
    ok(len(loader_cells) == 1, "exactly one ISDB loader cell inserted")
    loader_cell = loader_cells[0]
    ok("load_isdb_scores" in loader_cell, "ISDB loader cell missing load_isdb_scores")
    ok("ISDB_FALLBACK = True" in loader_cell, "ISDB_FALLBACK switch missing")
    ok("ISDB_LAM = 0.20" in loader_cell, "ISDB_LAM missing")
    
    # 6. Check modified Cell 32 (_factorial_entry)
    factorial_cells = [c for c in var_cs if "def _factorial_entry(scope):" in c]
    ok(len(factorial_cells) == 1, "exactly one _factorial_entry cell found")
    c32 = factorial_cells[0]
    ok("isdb_fallback=False" in c32, "rerank_row default missing")
    ok("isdb_fallback=isdb_fallback" in c32, "rerank_row treatment call missing")
    ok('assert ";".join(rebuilt) == controls[mid]' in c32, "E004 control reconstruction assertion missing")
    ok("ISDB_FALLBACK" in c32, "ISDB_FALLBACK counter instrument missing")


# ==============================================================================
# Gate G2: Pure Function & Loader Logic
# ==============================================================================

def test_g2_spectral_cosine():
    ns = {}
    exec(P.ISDB_LOADER_CODE, ns)
    cosine_fn = ns["_compute_spectral_cosine"]
    
    # 1. Empty or zero intensity cases
    ok(cosine_fn([], [], [100.0], [1.0]) == 0.0, "empty query -> 0.0")
    ok(cosine_fn([100.0], [1.0], [], []) == 0.0, "empty ref -> 0.0")
    ok(cosine_fn([100.0], [0.0], [100.0], [1.0]) == 0.0, "zero norm -> 0.0")
    
    # 2. Perfect match
    q_mzs = [100.0, 200.0, 300.0]
    q_its = [1.0, 2.0, 3.0]
    ok(abs(cosine_fn(q_mzs, q_its, q_mzs, q_its) - 1.0) < 1e-6, "identical spectra -> 1.0")
    
    # 3. Disjoint peaks (outside tolerance 0.02)
    r_mzs = [105.0, 205.0, 305.0]
    r_its = [1.0, 2.0, 3.0]
    ok(cosine_fn(q_mzs, q_its, r_mzs, r_its) == 0.0, "disjoint peaks -> 0.0")
    
    # 4. Matching within tolerance 0.02
    r_mzs_close = [100.015, 200.019, 300.010]
    ok(abs(cosine_fn(q_mzs, q_its, r_mzs_close, q_its) - 1.0) < 1e-6, "peaks within tolerance match")


def test_g2_load_isdb_scores_mocked(monkeypatch=None):
    ns = {}
    exec(P.ISDB_LOADER_CODE, ns)
    load_fn = ns["load_isdb_scores"]
    
    # When index file does not exist
    import pandas as pd
    te_dummy = pd.DataFrame([{"molecule_id": "1", "ms2_mzs": [100.0], "ms2_normalized_intensities": [1.0]}])
    cands_dummy = {"1": ["SMILES_1"]}
    res = load_fn(te_dummy, cands_dummy)
    ok(res == {}, "missing file returns empty dict")
    ok("NOT FOUND" in ns["ISDB_HEALTH"], "health indicates not found")
    
    # Mocking index file with pickle
    with tempfile.NamedTemporaryFile(suffix="isdb_index.pkl", delete=False) as tf:
        import pickle
        mock_idx = {
            "SMILES_1": ([100.0], [1.0]),
            "SMILES_2": ([200.0], [1.0]),
            "SMILES_SCALAR": 0.85
        }
        pickle.dump(mock_idx, tf)
        tmp_idx_path = tf.name
        
    try:
        import glob
        orig_glob = glob.glob
        glob.glob = lambda pat, recursive=True: [tmp_idx_path]
        
        # Test molecule with GLACIER present: should be skipped
        ns["gl_of"] = lambda mid: {"SMILES_1": 0.5} if mid == "1" else {}
        te_df = pd.DataFrame([
            {"molecule_id": "1", "ms2_mzs": [100.0], "ms2_normalized_intensities": [1.0]},
            {"molecule_id": "2", "ms2_mzs": [200.0], "ms2_normalized_intensities": [1.0]},
            {"molecule_id": "3", "ms2_mzs": [300.0], "ms2_normalized_intensities": [1.0]},
        ])
        cands_map = {
            "1": ["SMILES_1"],
            "2": ["SMILES_2", "SMILES_SCALAR"],
            "3": ["UNKNOWN_SMILES"]
        }
        scores = load_fn(te_df, cands_map)
        
        ok("1" not in scores, "molecule 1 has GLACIER -> skipped by ISDB loader")
        ok("2" in scores, "molecule 2 has no GLACIER -> scored by ISDB loader")
        ok(abs(scores["2"]["SMILES_2"] - 1.0) < 1e-6, "molecule 2 SMILES_2 cosine match")
        ok(scores["2"]["SMILES_SCALAR"] == 0.85, "molecule 2 scalar score preserved")
        ok("3" not in scores, "molecule 3 has no matching candidates in index -> omitted")
        glob.glob = orig_glob
    finally:
        if os.path.exists(tmp_idx_path):
            os.remove(tmp_idx_path)


# ==============================================================================
# Gate G2: Replay Engine & Invariants (K2, K3) via casmi_post._metric_overlay
# ==============================================================================

def test_g2_metric_overlay_invariance_k3():
    """K3: Positive-mode GLACIER-covered queries must have 100% byte-for-byte identical candidate order."""
    out = {"m_pos": ["CC", "CCC", "CCCC", "CCCCC"]}
    prefusion = {"m_pos": "CC;CCC;CCCC;CCCCC"}
    active = {"m_pos"}
    ENG = {"m_pos": {"smiles": ["CC", "CCC", "CCCC", "CCCCC"]}}
    ICE = {"m_pos": {"CC": 0.2, "CCC": 0.8, "CCCC": 0.3, "CCCCC": 0.1}}
    gl_of = lambda m: {"CC": 0.1, "CCC": 0.9, "CCCC": 0.4, "CCCCC": 0.2}
    
    score_key = lambda s: s
    form = lambda s: "C10H20"
    
    def fake_ice_rerank(cands, keys, scs, forms, ice_map, lam=1.0, top_n=25):
        return sorted(range(len(cands)), key=lambda i: ice_map.get(keys[i], 0.0), reverse=True)
        
    def fake_gl_multi(cands, keys, scs, forms, score_maps, lams, top_n=25):
        gl_map = score_maps[1]
        return sorted(range(len(cands)), key=lambda i: gl_map.get(keys[i], 0.0), reverse=True)

    c_base = {
        "ALPHA": 0.5, "KRR": 0, "FUSED_GL": True, "ICE_FALLBACK": True,
        "ICE_LAM": 1.0, "GL_LAM": 1.0, "ISDB_FALLBACK": False, "ISDB_LAM": 0.20
    }
    stats_base = {}
    res_base = CP._metric_overlay(
        copy.deepcopy(out), prefusion, active, ENG, ICE, gl_of, c_base,
        window=40, score_key=score_key, form=form,
        ice_rerank=fake_ice_rerank, gl_multi=fake_gl_multi, stats=stats_base, ISDB=None
    )

    c_isdb = dict(c_base, ISDB_FALLBACK=True)
    ISDB = {"m_pos": {"CC": 0.99, "CCC": 0.01, "CCCC": 0.02, "CCCCC": 0.03}}
    stats_isdb = {}
    res_isdb = CP._metric_overlay(
        copy.deepcopy(out), prefusion, active, ENG, ICE, gl_of, c_isdb,
        window=40, score_key=score_key, form=form,
        ice_rerank=fake_ice_rerank, gl_multi=fake_gl_multi, stats=stats_isdb, ISDB=ISDB
    )

    ok(res_base["m_pos"] == res_isdb["m_pos"], "K3 INVARIANCE: GLACIER-covered query ordering is 100% byte-for-byte identical")
    ok(stats_isdb.get("isdb_fallback", 0) == 0, "no ISDB fallback invoked for GLACIER covered query")


def test_g2_metric_overlay_activation_k2():
    """K2: Negative-mode / evidence gap queries (not gl) must activate ISDB fallback."""
    out = {"m_neg": ["CC", "CCC", "CCCC", "CCCCC"]}
    prefusion = {"m_neg": "CC;CCC;CCCC;CCCCC"}
    active = {"m_neg"}
    ENG = {"m_neg": {"smiles": ["CC", "CCC", "CCCC", "CCCCC"]}}
    ICE = {"m_neg": {}}
    gl_of = lambda m: {}
    
    score_key = lambda s: s
    form = lambda s: "C10H20"
    
    def fake_ice_rerank(cands, keys, scs, forms, score_map, lam=1.0, top_n=25):
        return sorted(range(len(cands)), key=lambda i: score_map.get(keys[i], 0.0), reverse=True)
        
    def fake_gl_multi(cands, keys, scs, forms, score_maps, lams, top_n=25):
        return list(range(len(cands)))

    c_off = {
        "ALPHA": 0.5, "KRR": 0, "FUSED_GL": True, "ICE_FALLBACK": True,
        "ICE_LAM": 1.0, "GL_LAM": 1.0, "ISDB_FALLBACK": False, "ISDB_LAM": 0.20
    }
    stats_off = {}
    res_off = CP._metric_overlay(
        copy.deepcopy(out), prefusion, active, ENG, ICE, gl_of, c_off,
        window=40, score_key=score_key, form=form,
        ice_rerank=fake_ice_rerank, gl_multi=fake_gl_multi, stats=stats_off, ISDB=None
    )

    c_on = dict(c_off, ISDB_FALLBACK=True)
    ISDB = {"m_neg": {"CC": 0.1, "CCC": 0.2, "CCCC": 0.95, "CCCCC": 0.05}}
    stats_on = {}
    res_on = CP._metric_overlay(
        copy.deepcopy(out), prefusion, active, ENG, ICE, gl_of, c_on,
        window=40, score_key=score_key, form=form,
        ice_rerank=fake_ice_rerank, gl_multi=fake_gl_multi, stats=stats_on, ISDB=ISDB
    )

    ok(stats_on.get("isdb_fallback", 0) == 1, "K2 ACTIVATION: isdb_fallback counter incremented")
    ok(res_on["m_neg"][0] == "CCCC", f"candidate CCCC boosted to top-1 by ISDB fallback: {res_on['m_neg']}")
    ok(res_on["m_neg"] != res_off["m_neg"], "candidate ranking changed by ISDB fallback")


def test_g2_metric_overlay_graceful_degradation():
    """Missing or non-finite ISDB scores safely degrade to original ranking without exception."""
    out = {"m_deg": ["CC", "CCC", "CCCC"]}
    prefusion = {"m_deg": "CC;CCC;CCCC"}
    active = {"m_deg"}
    ENG = {"m_deg": {"smiles": ["CC", "CCC", "CCCC"]}}
    ICE = {"m_deg": {}}
    gl_of = lambda m: {}
    
    score_key = lambda s: s
    form = lambda s: "C10H20"
    
    def fake_ice_rerank(cands, keys, scs, forms, score_map, lam=1.0, top_n=25):
        if any(v is None for v in score_map.values()):
            raise ValueError("Non-finite value encountered in test")
        return list(range(len(cands)))

    c = {
        "ALPHA": 0.5, "KRR": 0, "FUSED_GL": True, "ICE_FALLBACK": True,
        "ICE_LAM": 1.0, "GL_LAM": 1.0, "ISDB_FALLBACK": True, "ISDB_LAM": 0.20
    }
    stats = {}
    res = CP._metric_overlay(
        copy.deepcopy(out), prefusion, active, ENG, ICE, gl_of, c,
        window=40, score_key=score_key, form=form,
        ice_rerank=fake_ice_rerank, gl_multi=None, stats=stats, ISDB={"m_deg": {}}
    )
    ok(res["m_deg"] == ["CC", "CCC", "CCCC"], "empty ISDB gracefully preserves order")


# ==============================================================================
# Gate G2: Cell 32 (_factorial_entry / rerank_row) Execution
# ==============================================================================

def test_g2_factorial_entry_rerank_row():
    """Executes Cell 32 logic from the patched notebook to verify control reconstruction & fallback branching."""
    nb = json.load(open(P.OUT, encoding="utf-8"))
    c32_cells = [P.src_of(c) for c in nb["cells"] if "def _factorial_entry(scope):" in P.src_of(c)]
    ok(len(c32_cells) == 1, "found _factorial_entry cell in patched notebook")
    c32_src = c32_cells[0]
    
    ns = {}
    
    def fake_chem():
        return types.SimpleNamespace(score_key=lambda s: s)
        
    def fake_ice_fuse():
        def rerank(cands, keys, scs, forms, score_map, lam=1.0, top_n=25):
            return sorted(range(len(cands)), key=lambda i: score_map.get(keys[i], 0.0), reverse=True)
        return types.SimpleNamespace(rerank=rerank)
        
    def fake_gl_fuse():
        def rerank_multi(cands, keys, scs, forms, score_maps, lams, top_n=25):
            gl_map = score_maps[1]
            return sorted(range(len(cands)), key=lambda i: gl_map.get(keys[i], 0.0), reverse=True)
        return types.SimpleNamespace(rerank_multi=rerank_multi)
        
    class MockScope(dict):
        pass

    scope = MockScope({
        "chem": fake_chem(),
        "ice_fuse": fake_ice_fuse(),
        "gl_fuse": fake_gl_fuse(),
        "_form2": lambda s: "C10H20",
        "safe_extend": lambda a, b: a + [x for x in b if x not in a],
        "ICE_LAM": 1.0,
        "GL_LAM": 1.0,
        "ISDB_LAM": 0.20,
        "ICE_SCORES": {"m1": {"A": 0.1, "B": 0.8}, "m2": {"A": 0.1, "B": 0.8}},
        "ISDB_SCORES": {"m2": {"A": 0.05, "B": 0.95}},
        "gl_of": lambda mid: {"A": 0.2, "B": 0.9} if mid == "m1" else {},
        "C12": {"m1": ["A", "B"], "m2": ["A", "B"]},
        "PREFUSION": {"m1": "A;B", "m2": "A;B"},
        "PREFUSION_LISTS": {"m1": ["A", "B"], "m2": ["A", "B"]},
        "ENG": {"m1": {"smiles": ["A", "B"]}, "m2": {"smiles": ["A", "B"]}},
        "ALPHA": 0.5,
        "KRR": 0,
        "controls": {"m1": "B;A", "m2": "A;B"},
        "metric_identity": lambda s: s,
        "metric_fusion": lambda f, e, ident, alpha, offset, limit: (["A", "B"], [0.5, 0.4], 0),
        "active": {"m1", "m2"},
    })

    # Extract rerank_row from within _factorial_entry:
    rerank_row_code = c32_src[c32_src.index("    def rerank_row"):c32_src.index("    def run_overlay")]
    dedented_lines = [line[4:] if line.startswith("    ") else line for line in rerank_row_code.splitlines()]
    rerank_ns = dict(ns)
    rerank_ns["_finite_coverage"] = lambda d, smis, ks: sum(math.isfinite(d[k]) for k in ks if k in d and d[k] is not None)
    rerank_ns["metric_fusion"] = lambda f, e, ident, alpha, offset, limit: (["A", "B"], [0.5, 0.4], 0)
    rerank_ns["metric_identity"] = lambda s: s
    exec("\n".join(dedented_lines), rerank_ns)
    rerank_row = rerank_ns["rerank_row"]
    
    # 1. Test GLACIER branch on m1
    res1, audit1 = rerank_row(scope, "m1", ["A", "B"], window=40, ice_fallback=True, isdb_fallback=True, processed=1)
    ok(audit1["branch"] == "glacier", "m1 routes to glacier")
    ok(audit1["isdb_finite_count"] == 0, "m1 isdb finite count 0")
    
    # 2. Test ISDB fallback branch on m2 (not gl, isdb_fallback=True)
    res2, audit2 = rerank_row(scope, "m2", ["A", "B"], window=40, ice_fallback=True, isdb_fallback=True, processed=1)
    ok(audit2["branch"] == "isdb_fallback", "m2 routes to isdb_fallback")
    ok(audit2["isdb_finite_count"] == 2, "m2 has isdb finite coverage")
    ok(audit2["fallback_changed"] is True, "m2 fallback changed order")
    ok(res2[0] == "B", "m2 top-1 candidate is B")
    
    # 3. Test Control reconstruction (isdb_fallback=False, ice_fallback=False)
    res_ctrl, audit_ctrl = rerank_row(scope, "m2", ["A", "B"], window=40, ice_fallback=False, isdb_fallback=False, processed=2)
    ok(audit_ctrl["branch"] == "unrescored", "m2 with fallbacks off routes to unrescored")
    ok(";".join(res_ctrl) == scope["controls"]["m2"], "E004 control exact reconstruction")


if __name__ == "__main__":
    test_g1_notebook_delta_and_compilation()
    test_g2_spectral_cosine()
    test_g2_load_isdb_scores_mocked()
    test_g2_metric_overlay_invariance_k3()
    test_g2_metric_overlay_activation_k2()
    test_g2_metric_overlay_graceful_degradation()
    test_g2_factorial_entry_rerank_row()
    print(f"\nALL G1 & G2 TESTS PASSED! (total checks: {N})")
