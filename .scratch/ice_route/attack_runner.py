"""Run attack measurements A1-A3 for ice-margin-route."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H

POST = Path("E:/casmi/lines/exploit-b/build/casmi_post.py")
MARGIN = 0.1521
E006_CFG = "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"

def run_attacks():
    out = {}
    
    # -------------------------------------------------------------
    # A1: Routing overlap with lib_max >= 0.9 on C1_npx
    # -------------------------------------------------------------
    state_c1, deps_c1, cp_c1 = H.load_dump("C1_npx", post=POST)
    T = H.truths()
    tk_c1 = {m: cp_c1.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in state_c1["te_mids"] if m in T}
    
    cfg = H.parse_set(E006_CFG)
    
    # Trace margin and lib_max for each molecule on C1_npx
    c1_records = []
    orig_ov = cp_c1._metric_overlay
    def _trace_ov(res_out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi, stats):
        for mid, original in res_out.items():
            engine = ENG.get(mid)
            if mid not in active or mid not in prefusion or not engine or not engine.get('smiles'):
                continue
            first = [s for s in prefusion[mid].split(';') if s and s != 'CCO']
            cands, scores, collapsed = cp_c1.metric_fusion(first, engine['smiles'], cp_c1.metric_identity,
                                                        alpha=c['ALPHA'], offset=c['KRR'], limit=window)
            ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
            gl = gl_of(mid)
            s1 = scores[0] if len(scores) > 0 else 0.0
            s2 = scores[1] if len(scores) > 1 else 0.0
            rel_margin = (s1 - s2) / s1 if s1 > 0 else 0.0
            b = state_c1["base"][mid]
            lib_max = float(b[2])
            
            fires_fallback = bool(c.get('ICE_FALLBACK') and ice and not gl)
            
            c1_records.append({
                'mid': mid,
                'rel_margin': rel_margin,
                'lib_max': lib_max,
                'fires_fallback': fires_fallback,
                'margin_skip': fires_fallback and (rel_margin >= MARGIN),
                'lib_skip': fires_fallback and (lib_max >= 0.9),
            })
        return orig_ov(res_out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi, stats)
    
    cp_c1._metric_overlay = _trace_ov
    _ = cp_c1.run(state_c1, deps_c1, **cfg)
    
    df_c1 = pd.DataFrame(c1_records)
    fallback_c1 = df_c1[df_c1.fires_fallback]
    
    overlap = int(((fallback_c1.margin_skip) & (fallback_c1.lib_skip)).sum())
    margin_only = int(((fallback_c1.margin_skip) & (~fallback_c1.lib_skip)).sum())
    lib_only = int(((~fallback_c1.margin_skip) & (fallback_c1.lib_skip)).sum())
    neither = int(((~fallback_c1.margin_skip) & (~fallback_c1.lib_skip)).sum())
    diff_routed = margin_only + lib_only
    
    out['A1'] = {
        'total_c1_npx': len(df_c1),
        'fallback_fired_mols': len(fallback_c1),
        'margin_skip_count': int(fallback_c1.margin_skip.sum()),
        'lib_skip_count': int(fallback_c1.lib_skip.sum()),
        'overlap': overlap,
        'margin_only': margin_only,
        'lib_only': lib_only,
        'neither': neither,
        'diff_routed': diff_routed,
        'fraction_diff': float(diff_routed / len(fallback_c1)) if len(fallback_c1) else 0.0,
    }
    print("A1 results:", out['A1'])
    
    # -------------------------------------------------------------
    # A2: Delta on the C2_np skipped set
    # -------------------------------------------------------------
    state_c2np, deps_c2np, cp_c2np = H.load_dump("C2_np", post=POST)
    tk_c2np = {m: cp_c2np.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in state_c2np["te_mids"] if m in T}
    
    # Run E006 base ranks on C2_np
    out_e006 = cp_c2np.run(state_c2np, deps_c2np, **cfg)
    r_e006 = {m: H.rank_of(out_e006.get(m, []), tk_c2np[m], cp_c2np) for m in tk_c2np}
    
    c2np_records = []
    
    orig_ov_c2np = cp_c2np._metric_overlay
    def _ov_margin(res_out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi, stats):
        res = {}
        for mid, original in res_out.items():
            original = list(original)
            engine = ENG.get(mid)
            if mid not in active or mid not in prefusion or not engine or not engine.get('smiles'):
                res[mid] = original
                continue
            first = [s for s in prefusion[mid].split(';') if s and s != 'CCO']
            cands, scores, collapsed = cp_c2np.metric_fusion(first, engine['smiles'], cp_c2np.metric_identity,
                                                          alpha=c['ALPHA'], offset=c['KRR'], limit=window)
            stats['collapsed_votes'] = stats.get('collapsed_votes', 0) + collapsed
            keys = [score_key(s) or s for s in cands]
            ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
            gl = gl_of(mid)
            
            s1 = scores[0] if len(scores) > 0 else 0.0
            s2 = scores[1] if len(scores) > 1 else 0.0
            rel_margin = (s1 - s2) / s1 if s1 > 0 else 0.0
            
            fires_fb = bool(c['ICE_FALLBACK'] and ice and not gl)
            skip = fires_fb and (rel_margin >= MARGIN)
            
            c2np_records.append({
                'mid': mid,
                'fires_fb': fires_fb,
                'skip': skip,
                'rel_margin': rel_margin,
            })
            
            try:
                if ice and gl and c['FUSED_GL']:
                    order = gl_multi(cands, keys, scores, [form(s) for s in cands], [{}, gl],
                                     [c['ICE_LAM'], c['GL_LAM']], top_n=len(cands))
                    cands = [cands[i] for i in order]
                elif fires_fb:
                    if skip:
                        stats['ice_route_skipped'] = stats.get('ice_route_skipped', 0) + 1
                    else:
                        order = ice_rerank(cands, keys, scores, [form(s) for s in cands], ice,
                                           lam=c['ICE_LAM'], top_n=len(cands))
                        cands = [cands[i] for i in order]
                        stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1
            except Exception:
                pass
            res[mid] = cp_c2np.safe_extend(cands[:25], cands[25:] + original) if cands else original
        return res
    
    cp_c2np._metric_overlay = _ov_margin
    stats_c2np = {}
    out_margin = cp_c2np.run(state_c2np, deps_c2np, stats=stats_c2np, **cfg)
    r_margin = {m: H.rank_of(out_margin.get(m, []), tk_c2np[m], cp_c2np) for m in tk_c2np}
    
    df_c2np = pd.DataFrame(c2np_records)
    skipped_mids = [m for m in df_c2np[df_c2np.skip].mid if m in tk_c2np]
    
    skipped_ranks_base = np.array([r_e006[m] for m in skipped_mids], dtype=int)
    skipped_ranks_margin = np.array([r_margin[m] for m in skipped_mids], dtype=int)
    
    skipped_rr_base = H.rr(skipped_ranks_base)
    skipped_rr_margin = H.rr(skipped_ranks_margin)
    delta_skipped = float(np.mean(skipped_rr_margin) - np.mean(skipped_rr_base))
    
    all_ranks_base = np.array([r_e006[m] for m in tk_c2np], dtype=int)
    all_ranks_margin = np.array([r_margin[m] for m in tk_c2np], dtype=int)
    delta_all_c2np = float(np.mean(H.rr(all_ranks_margin)) - np.mean(H.rr(all_ranks_base)))
    
    out['A2'] = {
        'n_c2np': len(tk_c2np),
        'n_skipped': len(skipped_mids),
        'mrr_base_skipped': float(np.mean(skipped_rr_base)) if len(skipped_mids) else 0.0,
        'mrr_margin_skipped': float(np.mean(skipped_rr_margin)) if len(skipped_mids) else 0.0,
        'delta_skipped': delta_skipped,
        'delta_all_c2np': delta_all_c2np,
        'displaced_r1': int(sum((r_e006[m] == 1) and (r_margin[m] != 1) for m in skipped_mids)),
    }
    print("A2 results:", out['A2'])
    
    # -------------------------------------------------------------
    # A3: Share of molecules with ICE per dump
    # -------------------------------------------------------------
    ice_coverage = {}
    for c_name in ["C1_npx", "C1_xlib", "C2", "C2_hard", "C2_np"]:
        try:
            st, dp, _ = H.load_dump(c_name, post=POST)
            ice_dict = st.get("ice") or {}
            mids = st.get("te_mids") or []
            has_ice_count = sum(bool(ice_dict.get(m)) for m in mids)
            ice_coverage[c_name] = {
                'total_mols': len(mids),
                'with_ice': has_ice_count,
                'pct_with_ice': float(has_ice_count / len(mids)) if mids else 0.0
            }
        except Exception as e:
            ice_coverage[c_name] = f"error: {e}"
            
    out['A3'] = ice_coverage
    print("A3 results:", out['A3'])
    
    Path(".scratch/ice_route/attack_results.json").write_text(json.dumps(out, indent=2))
    print("Wrote .scratch/ice_route/attack_results.json")

if __name__ == "__main__":
    run_attacks()
