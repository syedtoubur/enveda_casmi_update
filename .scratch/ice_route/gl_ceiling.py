"""ice-margin-route-gl: ceiling of margin-routing the FUSED_GL (ICE+GL) rerank, replay-free.

  python .scratch/ice_route/gl_ceiling.py margin          # truth-blind: per-molecule margins + C2_np FUSED_GL median
  python .scratch/ice_route/gl_ceiling.py ceiling <m>     # after pre-reg: Δ per cohort if FUSED_GL molecules with
                                                          # margin >= m took r_noice (no rerank) instead of r_e006

Ceiling = "skip the whole fused rerank when confident" (diag noice drops ICE and so the gl_multi branch too).
Rank columns come from .scratch/libgate_diag/per_mol_*.csv (diag.py, E006 cfg).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H  # noqa: E402

H.CFG_LINE = None
HERE = Path(__file__).resolve().parent
DIAG = REPO / ".scratch/libgate_diag"
POST = Path("E:/casmi/lines/exploit-b/build/casmi_post.py")
COHORTS = ["C2_np", "C1_npx", "C1_xlib", "C2", "C2_hard"]
E006 = "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"
rng = np.random.default_rng(0)


def margins(cohort):
    st, dp, cmod = H.load_dump(cohort, post=POST)
    recs = {}
    orig = cmod._metric_overlay

    def trace(out, prefusion, active, ENG, ICE, gl_of, c, window, *rest):
        for mid in out:
            engine = ENG.get(mid)
            if mid not in active or mid not in prefusion or not engine or not engine.get('smiles'):
                continue
            first = [s for s in prefusion[mid].split(';') if s and s != 'CCO']
            _, scores, _ = cmod.metric_fusion(first, engine['smiles'], cmod.metric_identity,
                                              alpha=c['ALPHA'], offset=c['KRR'], limit=window)
            ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
            s1 = scores[0] if scores else 0.0
            s2 = scores[1] if len(scores) > 1 else 0.0
            recs[mid] = dict(margin=(s1 - s2) / s1 if s1 > 0 else 0.0,
                             gl_path=bool(ice and gl_of(mid) and c['FUSED_GL']))
        return orig(out, prefusion, active, ENG, ICE, gl_of, c, window, *rest)

    cmod._metric_overlay = trace
    cmod.run(st, dp, **H.parse_set(E006))
    return recs


def ci90(d):
    b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]


def main():
    mode = sys.argv[1]
    mpath = HERE / "gl_margins.json"
    if mode == "margin":
        allm = {c: margins(c) for c in COHORTS}
        mpath.write_text(json.dumps(allm))
        for c, r in allm.items():
            g = [v["margin"] for v in r.values() if v["gl_path"]]
            print(c, "n_overlay", len(r), "n_gl_path", len(g), "median_gl", round(float(np.median(g)), 6))
        return
    m = float(sys.argv[2])
    allm = json.loads(mpath.read_text())
    out = {}
    for c in COHORTS:
        df = pd.read_csv(DIAG / f"per_mol_{c}.csv")
        r = allm[c]
        skip = np.array([bool(r.get(x, {}).get("gl_path")) and r[x]["margin"] >= m - 1e-9 for x in df.mid])
        r_new = np.where(skip, df.r_noice.values, df.r_e006.values)
        d = H.rr(r_new) - H.rr(df.r_e006.values)
        leak = df.leak.values.astype(bool)
        out[c] = dict(n=len(df), skipped=int(skip.sum()), delta=float(d.mean()), ci90=ci90(d),
                      up=int((d > 0).sum()), down=int((d < 0).sum()),
                      displaced=int(((df.r_e006.values == 1) & (r_new != 1)).sum()),
                      d_leak=float(d[leak].sum() / len(df)), d_other=float(d[~leak].sum() / len(df)),
                      e006=float(H.rr(df.r_e006.values).mean()))
        print(c, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) else v)
                  for k, v in out[c].items()})
    (HERE / "gl_ceiling.json").write_text(json.dumps(dict(margin=m, cohorts=out), indent=1))


if __name__ == "__main__":
    main()
