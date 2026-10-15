"""Compute a priori ICE routing margin on C2_np dump."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
import numpy as np
import json

REPO = Path("E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H

POST = Path("E:/casmi/lines/exploit-b/build/casmi_post.py")

def main():
    cfg = H.parse_set("METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40")
    
    results = {}
    
    for cohort_name in ["C2_np", "C1_npx", "C1_xlib", "C2", "C2_hard"]:
        try:
            st, dp, cmod = H.load_dump(cohort_name, post=POST)
            records = []
            orig_ov = cmod._metric_overlay
            def _trace_ov(out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi, stats):
                for mid, original in out.items():
                    engine = ENG.get(mid)
                    if mid not in active or mid not in prefusion or not engine or not engine.get('smiles'):
                        continue
                    first = [s for s in prefusion[mid].split(';') if s and s != 'CCO']
                    cands, scores, collapsed = cmod.metric_fusion(first, engine['smiles'], cmod.metric_identity,
                                                                alpha=c['ALPHA'], offset=c['KRR'], limit=window)
                    ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
                    gl = gl_of(mid)
                    s1 = scores[0] if len(scores) > 0 else 0.0
                    s2 = scores[1] if len(scores) > 1 else 0.0
                    rel_margin = (s1 - s2) / s1 if s1 > 0 else 0.0
                    records.append({
                        'mid': mid,
                        's1': s1,
                        's2': s2,
                        'rel_margin': rel_margin,
                        'fires_fallback': bool(c.get('ICE_FALLBACK') and ice and not gl),
                        'has_ice': bool(ice),
                        'has_gl': bool(gl),
                    })
                return orig_ov(out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi, stats)
            cmod._metric_overlay = _trace_ov
            _ = cmod.run(st, dp, **cfg)
            
            c_all = [r['rel_margin'] for r in records]
            c_fb = [r['rel_margin'] for r in records if r['fires_fallback']]
            
            results[cohort_name] = {
                'n_all': len(c_all),
                'n_fallback': len(c_fb),
                'median_all': float(np.median(c_all)),
                'mean_all': float(np.mean(c_all)),
                'median_fallback': float(np.median(c_fb)) if c_fb else None,
                'mean_fallback': float(np.mean(c_fb)) if c_fb else None,
                'q25_fallback': float(np.quantile(c_fb, 0.25)) if c_fb else None,
                'q75_fallback': float(np.quantile(c_fb, 0.75)) if c_fb else None,
            }
            print(f"{cohort_name}: all_med={results[cohort_name]['median_all']:.6f}, fb_med={results[cohort_name]['median_fallback']}")
        except Exception as e:
            print(f"Error on {cohort_name}: {e}")
            
    Path(".scratch/ice_route/margin_stats.json").write_text(json.dumps(results, indent=2))
    print("Wrote .scratch/ice_route/margin_stats.json")

if __name__ == "__main__":
    main()
