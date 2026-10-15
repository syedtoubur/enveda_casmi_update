"""Ceiling of demoting enveda-180-only structures (synthetic drug-like library) to the end of E006's top-25.
python .scratch/e180/ceiling.py  -> .scratch/e180/ceiling.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research/harness")); import harness as H
H.CFG_LINE = None
E006 = "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"
m = pd.read_parquet(REPO / ".scratch/valgen/train_meta.parquet", columns=["lib", "ik14"])
libs = m.groupby("ik14").lib.agg(frozenset)
E180 = set(libs[libs.map(lambda s: s == frozenset({"enveda-180"}))].index)
rng = np.random.default_rng(0)
out = {}
for c in ["C1_npx", "C1_xlib", "C2", "C2_hard", "C2_np"]:
    st, dp, cp = H.load_dump(c, post=Path("E:/casmi/lines/exploit-b/build/casmi_post.py"))
    T = H.truths(); mids = [x for x in st["te_mids"] if x in T]
    res = cp.run(st, dp, **H.parse_set(E006))
    r0, r1, n_e, t_e = [], [], [], 0
    for x in mids:
        tk = cp.canonicalize_smiles_to_ik14(T[x]["smiles"]); t_e += tk in E180
        rows = list(res.get(x, []))[:25]
        iks = [cp.canonicalize_smiles_to_ik14(s) for s in rows]
        keep = [s for s, k in zip(rows, iks) if k not in E180] + [s for s, k in zip(rows, iks) if k in E180]
        n_e.append(sum(k in E180 for k in iks))
        r0.append(H.rank_of(rows, tk, cp)); r1.append(H.rank_of(keep, tk, cp))
    d = H.rr(np.array(r1)) - H.rr(np.array(r0))
    b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    out[c] = dict(n=len(mids), truths_e180_only=int(t_e), mean_e180_in_top25=float(np.mean(n_e)),
                  share_mols_with_e180=float(np.mean(np.array(n_e) > 0)), e006=float(H.rr(np.array(r0)).mean()),
                  delta=float(d.mean()), ci90=[float(np.quantile(b, .05)), float(np.quantile(b, .95))],
                  up=int((d > 0).sum()), down=int((d < 0).sum()))
    print(c, out[c], flush=True)
Path(__file__).with_name("ceiling.json").write_text(json.dumps(out, indent=1))
