"""Generator script to write E:/casmi/lines/exploit/build/patch_isdb_cfm_fallback.py and test_patch_isdb_cfm_fallback.py."""
import os

patch_script_content = '''"""Exploit probe generator: CFM-ID 4.0 ISDB fallback for negative-mode and evidence-gap spectra.

Idea: research/exploit/20261015-isdb-cfm-fallback (spec.md, attack.md).
Base: casmi26-e006-window-40-ice-fallback-true.ipynb (LB 0.440).

GLACIER is hardcoded to [M+H]+ positive-mode spectra only. For negative-mode ([M-H]-)
and evidence-gap molecules (where GLACIER is blind), this patch uses precomputed CFM-ID 4.0
in-silico spectral database (ISDB) matching to re-rank same-formula candidate clusters.
Positive-mode GLACIER-covered queries remain 100% byte-for-byte identical.

Usage:
    python build/patch_isdb_cfm_fallback.py [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-isdb-cfm-fallback.ipynb")


def src_of(cell):
    return "".join(cell.get("source", []))


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def must_replace(text, old, new, what):
    if old not in text:
        raise SystemExit(f"PATCH FAILED: anchor not found for {what}:\\n  {old[:120]}")
    if text.count(old) != 1:
        raise SystemExit(f"PATCH FAILED: anchor for {what} occurs {text.count(old)} times")
    return text.replace(old, new)


# ---------------------------------------------------------------- ISDB loader cell
ISDB_LOADER_CODE = \'\'\'# === ISDB CFM-ID 4.0 Precomputed Spectral Fallback ===
# Provides orthogonal Metlin-trained forward fragmentation for [M-H]- and evidence gaps
import glob
import math
import os
import pickle
import time

ISDB_LAM = 0.20
ISDB_FALLBACK = True
ISDB_SCORES = {}
ISDB_HEALTH = "ISDB NOT RUN"

def _compute_spectral_cosine(q_mzs, q_its, r_mzs, r_its, mz_tol=0.02):
    """Fast spectral cosine similarity between query and reference peak lists."""
    if not q_mzs or not r_mzs or not q_its or not r_its:
        return 0.0
    q_norm = math.sqrt(sum(it * it for it in q_its))
    r_norm = math.sqrt(sum(it * it for it in r_its))
    if q_norm <= 0.0 or r_norm <= 0.0:
        return 0.0
    j = 0
    ref_len = len(r_mzs)
    dot = 0.0
    for q_mz, q_it in zip(q_mzs, q_its):
        while j < ref_len and r_mzs[j] < q_mz - mz_tol:
            j += 1
        best_match = 0.0
        k = j
        while k < ref_len and abs(r_mzs[k] - q_mz) <= mz_tol:
            prod = q_it * r_its[k]
            if prod > best_match:
                best_match = prod
            k += 1
        dot += best_match
    return float(dot / (q_norm * r_norm))


def load_isdb_scores(te_df, cands_by_mid):
    """Query ISDB index for candidates of molecules where GLACIER is blind."""
    global ISDB_HEALTH
    idx_candidates = glob.glob("/kaggle/input/**/isdb_index.pkl", recursive=True)
    if not idx_candidates:
        ISDB_HEALTH = "ISDB INDEX NOT FOUND (offline / mock mode)"
        print(f"[ISDB] {ISDB_HEALTH}", flush=True)
        return {}
    
    idx_path = idx_candidates[0]
    t0 = time.time()
    try:
        with open(idx_path, "rb") as f:
            isdb_idx = pickle.load(f)
        elapsed = time.time() - t0
        ISDB_HEALTH = f"ISDB INDEX LOADED ({len(isdb_idx)} records in {elapsed:.1f}s)"
        print(f"[ISDB] {ISDB_HEALTH}", flush=True)
    except Exception as e:
        ISDB_HEALTH = f"ISDB INDEX LOAD FAILED: {repr(e)}"
        print(f"[ISDB] {ISDB_HEALTH}", flush=True)
        return {}

    # Query spectra map from te_df
    te_by_mid = {}
    for r in te_df.itertuples(index=False):
        te_by_mid.setdefault(str(r.molecule_id), []).append(r)

    scores_out = {}
    covered = 0
    for mid, cands in cands_by_mid.items():
        mid_str = str(mid)
        # Only compute for molecules where GLACIER is blind
        gl_cur = gl_of(mid_str) if 'gl_of' in globals() else {}
        if gl_cur and any(v is not None for v in gl_cur.values()):
            continue
        
        specs = te_by_mid.get(mid_str, [])
        if not specs:
            continue
        # Take the most intense or first spectrum
        spec = specs[0]
        q_mzs = getattr(spec, 'ms2_mzs', []) or []
        q_its = getattr(spec, 'ms2_normalized_intensities', []) or []
        if not q_mzs or not q_its:
            continue

        mol_scores = {}
        for c in cands:
            c_str = str(c)
            # Try SMILES, then canonical InChIKey14
            rec = isdb_idx.get(c_str)
            if rec is None and 'canonicalize_smiles_to_ik14' in globals():
                ik14 = canonicalize_smiles_to_ik14(c_str)
                if ik14:
                    rec = isdb_idx.get(ik14)
            if rec is not None:
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    r_mzs, r_its = rec[0], rec[1]
                elif isinstance(rec, dict):
                    r_mzs, r_its = rec.get('mzs', []), rec.get('intensities', [])
                elif isinstance(rec, (int, float)):
                    mol_scores[c_str] = float(rec)
                    continue
                else:
                    continue
                sim = _compute_spectral_cosine(q_mzs, q_its, r_mzs, r_its)
                mol_scores[c_str] = sim
        if mol_scores:
            scores_out[mid_str] = mol_scores
            covered += 1

    print(f"[ISDB] Scored {len(scores_out)} molecules across blind/negative-mode queries", flush=True)
    return scores_out

try:
    if 'te' in globals() and 'PREFUSION_LISTS' in globals():
        _cands_map = {m: PREFUSION_LISTS[m] for m in te.molecule_id.astype(str) if m in PREFUSION_LISTS}
        ISDB_SCORES = load_isdb_scores(te, _cands_map)
except Exception as _e:
    print(f"[ISDB] Optional loader exception (graceful degradation): {repr(_e)}", flush=True)
\'\'\'


def patch_notebook(src_path: str, out_path: str) -> None:
    nb = json.load(open(src_path, encoding="utf-8"))
    
    # 1. Clear outputs for clean git storage and commit
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None

    # 2. Patch Cell 30 (_factorial_entry)
    c30_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and "def _factorial_entry(scope):" in src_of(cell):
            c30_idx = i
            break
    if c30_idx is None:
        raise SystemExit("PATCH FAILED: Cell 30 (_factorial_entry) not found")

    c30_src = src_of(nb["cells"][c30_idx])
    
    # Update rerank_row signature
    c30_src = must_replace(
        c30_src,
        "def rerank_row(scope, mid, original, *, window, ice_fallback, processed):",
        "def rerank_row(scope, mid, original, *, window, ice_fallback, isdb_fallback=False, processed):",
        "rerank_row signature"
    )
    
    # Update audit dictionary initialization
    audit_init_old = \'\'\'            "gl_finite_count": 0,
            "fallback_changed": False,
            "fallback_changed_top1": False,\'\'\'
    audit_init_new = \'\'\'            "gl_finite_count": 0,
            "isdb_finite_count": 0,
            "fallback_changed": False,
            "fallback_changed_top1": False,\'\'\'
    c30_src = must_replace(c30_src, audit_init_old, audit_init_new, "audit dict init")

    # Update ISDB scores extraction and coverage audit
    audit_cov_old = \'\'\'        keys = [scope["chem"].score_key(s) or s for s in candidates]
        ice_scores = scope["ICE_SCORES"]
        ice = ice_scores.get(mid, {}) if isinstance(ice_scores, dict) else {}
        gl = scope["gl_of"](mid)
        audit.update(
            branch="unrescored",
            candidate_count=len(candidates),
            collapsed_source_votes=collapsed,
            ice_finite_count=_finite_coverage(ice, candidates, keys),
            gl_finite_count=_finite_coverage(gl, candidates, keys),
        )\'\'\'
    audit_cov_new = \'\'\'        keys = [scope["chem"].score_key(s) or s for s in candidates]
        ice_scores = scope["ICE_SCORES"]
        ice = ice_scores.get(mid, {}) if isinstance(ice_scores, dict) else {}
        isdb_scores = scope.get("ISDB_SCORES", {})
        isdb = isdb_scores.get(mid, {}) if isinstance(isdb_scores, dict) else {}
        gl = scope["gl_of"](mid)
        audit.update(
            branch="unrescored",
            candidate_count=len(candidates),
            collapsed_source_votes=collapsed,
            ice_finite_count=_finite_coverage(ice, candidates, keys),
            gl_finite_count=_finite_coverage(gl, candidates, keys),
            isdb_finite_count=_finite_coverage(isdb, candidates, keys),
        )\'\'\'
    c30_src = must_replace(c30_src, audit_cov_old, audit_cov_new, "audit coverage extraction")

    # Insert ISDB fallback branch
    branch_old = \'\'\'        if ice and gl:
            # Preserve v80\\\'s exact GL-present gate and GL-only score arguments.
            order = scope["gl_fuse"].rerank_multi(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                [{}, gl],
                [scope["ICE_LAM"], scope["GL_LAM"]],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "glacier"
        elif ice_fallback and ice and not gl:
            # Public v1 / script 356785434: ICE is used only where gl_of(mid) is empty.
            # Keep its truthy-dict gate: all-None ICE is an observable harmless no-op.
            order = scope["ice_fuse"].rerank(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                ice,
                lam=scope["ICE_LAM"],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "ice_fallback"\'\'\'

    branch_new = \'\'\'        if ice and gl:
            # Preserve v80\\\'s exact GL-present gate and GL-only score arguments.
            order = scope["gl_fuse"].rerank_multi(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                [{}, gl],
                [scope["ICE_LAM"], scope["GL_LAM"]],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "glacier"
        elif isdb_fallback and isdb and not gl:
            # CFM-ID 4.0 ISDB fallback for negative-mode and evidence-gap spectra
            order = scope["ice_fuse"].rerank(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                isdb,
                lam=scope.get("ISDB_LAM", 0.20),
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "isdb_fallback"
        elif ice_fallback and ice and not gl:
            # Public v1 / script 356785434: ICE is used only where gl_of(mid) is empty.
            # Keep its truthy-dict gate: all-None ICE is an observable harmless no-op.
            order = scope["ice_fuse"].rerank(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                ice,
                lam=scope["ICE_LAM"],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "ice_fallback"\'\'\'
    c30_src = must_replace(c30_src, branch_old, branch_new, "isdb fallback branch")

    # Update fallback_changed evaluation for both ice_fallback and isdb_fallback
    eval_old = \'\'\'        if audit["branch"] == "ice_fallback":
            without_ice = (
                scope["safe_extend"](before[:25], before[25:] + original) if before else original
            )
            audit["fallback_changed"] = result != without_ice
            audit["fallback_changed_top1"] = result[:1] != without_ice[:1]\'\'\'
    eval_new = \'\'\'        if audit["branch"] in ("ice_fallback", "isdb_fallback"):
            without_sim = (
                scope["safe_extend"](before[:25], before[25:] + original) if before else original
            )
            audit["fallback_changed"] = result != without_sim
            audit["fallback_changed_top1"] = result[:1] != without_sim[:1]\'\'\'
    c30_src = must_replace(c30_src, eval_old, eval_new, "fallback changed evaluation")

    # Update run_overlay signature and call sites
    run_sig_old = \'\'\'    def run_overlay(scope, *, experiment, window, ice_fallback, provenance, workdir="."):\'\'\'
    run_sig_new = \'\'\'    def run_overlay(scope, *, experiment, window, ice_fallback, isdb_fallback=False, provenance, workdir="."):\'\'\'
    c30_src = must_replace(c30_src, run_sig_old, run_sig_new, "run_overlay signature")

    rebuilt_old = \'\'\'            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, processed=active
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, processed=active
            )\'\'\'
    rebuilt_new = \'\'\'            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, isdb_fallback=False, processed=active
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, isdb_fallback=isdb_fallback, processed=active
            )\'\'\'
    c30_src = must_replace(c30_src, rebuilt_old, rebuilt_new, "control reconstruction and treatment call")

    # Update report metrics and counter print
    report_old = \'\'\'            "fallback_rows": sum(a["branch"] == "ice_fallback" for a in audits),
            "fallback_finite_rows": sum(
                a["branch"] == "ice_fallback" and a["ice_finite_count"] > 0 for a in audits
            ),
            "fallback_changed_rows": sum(a["fallback_changed"] for a in audits),
            "fallback_changed_top1": sum(a["fallback_changed_top1"] for a in audits),\'\'\'
    report_new = \'\'\'            "fallback_rows": sum(a["branch"] in ("ice_fallback", "isdb_fallback") for a in audits),
            "isdb_fallback_rows": sum(a["branch"] == "isdb_fallback" for a in audits),
            "isdb_finite_rows": sum(
                a["branch"] == "isdb_fallback" and a["isdb_finite_count"] > 0 for a in audits
            ),
            "isdb_changed_rows": sum(a["branch"] == "isdb_fallback" and a["fallback_changed"] for a in audits),
            "isdb_changed_top1": sum(a["branch"] == "isdb_fallback" and a["fallback_changed_top1"] for a in audits),\'\'\'
    c30_src = must_replace(c30_src, report_old, report_new, "report fallback metrics")

    print_old = \'\'\'        print("FACTORIAL_VERIFICATION", json.dumps(report))\'\'\'
    print_new = \'\'\'        print("FACTORIAL_VERIFICATION", json.dumps(report))
        print("ISDB_FALLBACK", json.dumps({
            "processed": len(audits),
            "isdb_covered_mols": sum(a["branch"] == "isdb_fallback" for a in audits),
            "reranked_mols": sum(a["branch"] == "isdb_fallback" and a["fallback_changed"] for a in audits)
        }))\'\'\'
    c30_src = must_replace(c30_src, print_old, print_new, "ISDB_FALLBACK counter print")

    entry_old = \'\'\'    return run_overlay(scope, **{\'experiment\': \'E006\', \'window\': 40, \'ice_fallback\': True, \'provenance\': {\'base_notebook_sha256\': \'1f0f9b3026a93ddb5ad6fa6993c8ac0a4bfaddc54594eb1dd6dda7a20d8ef0d6\', \'base_source_code_sha256\': \'eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2\', \'helper_source_sha256\': \'a027a69ddda2c2e14639e3ec5196d5f1eb2ab57cba10ef929723148f31fc8f34\', \'source_script_version_id\': 356740494, \'ice_policy_source_script_version_id\': 356785434, \'ice_policy_source\': \'samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438/1\'}})\'\'\'
    entry_new = \'\'\'    return run_overlay(scope, **{\'experiment\': \'E006-ISDB\', \'window\': 40, \'ice_fallback\': True, \'isdb_fallback\': True, \'provenance\': {\'base_notebook\': \'casmi26-e006-window-40-ice-fallback-true.ipynb\', \'isdb_source\': \'n10705013/casmi26-isdb-predicted-index\', \'isdb_lam\': 0.20, \'source_script_version_id\': 356740494}})\'\'\'
    c30_src = must_replace(c30_src, entry_old, entry_new, "overlay entry call with isdb_fallback=True")

    set_src(nb["cells"][c30_idx], c30_src)

    # 3. Insert ISDB Loader Cell immediately before Cell 30
    loader_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ISDB_LOADER_CODE.splitlines(keepends=True)
    }
    nb["cells"].insert(c30_idx, loader_cell)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully wrote patched notebook: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC, help="Source notebook path")
    parser.add_argument("--out", default=OUT, help="Output notebook path")
    args = parser.parse_args()
    patch_notebook(args.src, args.out)


if __name__ == "__main__":
    main()
'''

target = "E:/casmi/lines/exploit/build/patch_isdb_cfm_fallback.py"
with open(target, "w", encoding="utf-8") as f:
    f.write(patch_script_content)
print(f"Wrote {target} ({len(patch_script_content)} bytes)")
