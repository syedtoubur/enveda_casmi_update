"""c3-iso I1b.1 prep (light, local, NO truth read): frozen top-150 + V22 fused top 10 + query spectra -> ICE runner input.

Output dir: ice_in_pseudo_c3.json (runner input, fuse.run_ice item format) and cand_meta.json ({mid: {smiles: [src, op, frag]}}).
Query spectra = train.parquet rows `train_rows` of p1_novel/pseudo_c3.parquet (peaks + CE/instrument fields only)."""
import json, pickle, sys, math
import numpy as np, pandas as pd, pyarrow.parquet as pq
FROZEN = "E:/casmi/c3_iso/i1b_pseudo_c3_full_top150.pkl"
TRAIN = "E:/casmi/data/comp_raw/train.parquet"
C3 = "E:/ENVEDA_casmi/p1_novel/pseudo_c3.parquet"
out = sys.argv[1]
fr = pickle.load(open(FROZEN, "rb"))
c3 = pd.read_parquet(C3, columns=["molecule_id", "train_rows"])
c3 = c3[c3.molecule_id.isin(set(fr["top"]))]
want = {int(r): m for m, v in zip(c3.molecule_id, c3.train_rows) for r in v}
pf = pq.ParquetFile(TRAIN)
offs = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
sel = np.array(sorted(want), dtype=np.int64)
cols = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", "ionization_mode", "instrument_type",
        "collision_energy_ev", "collision_energy_orig", "collision_energy_orig_units"]
def fl(x):
    try: return [float(v) for v in x]
    except TypeError: return None
def st(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)
spec = {m: [] for m in c3.molecule_id}
for rg in range(pf.metadata.num_row_groups):
    lo, hi = offs[rg], offs[rg + 1]
    loc = sel[(sel >= lo) & (sel < hi)] - lo
    if len(loc) == 0: continue
    t = pf.read_row_group(rg, columns=cols).take(loc).to_pandas()
    for g, r in zip(sel[(sel >= lo) & (sel < hi)], t.itertuples()):
        spec[want[int(g)]].append(dict(mz=fl(r.ms2_mzs) or [], it=fl(r.ms2_normalized_intensities) or [], prec=float(r.precursor_mz),
            adduct=st(r.adduct), mode=st(r.ionization_mode), instrument=st(r.instrument_type), ce_ev=fl(r.collision_energy_ev),
            ce_orig=st(r.collision_energy_orig), ce_units=st(r.collision_energy_orig_units)))
items, meta, ncand, ncov = [], {}, 0, 0
for m in fr["top"]:
    cm = {}
    for s in fr["fused"][m][:10]:
        cm.setdefault(s, ["fused", None, None])
    for s, ik, op, sc in fr["top"][m]["top"]:
        cm.setdefault(s, ["child", op, sc])
    sp = [s for s in spec[m] if s["mode"] == "positive" and s["adduct"] in ("[M+H]+", "[M+Na]+")]
    ncov += bool(sp); ncand += len(cm)
    items.append(dict(mid=m, spectra=sp, cands=list(cm)))
    meta[m] = cm
json.dump(items, open(f"{out}/ice_in_pseudo_c3.json", "w"))
json.dump(meta, open(f"{out}/cand_meta.json", "w"))
json.dump({"title": "casmi26-c3iso-i1b1-ice-in", "id": "syedtoubur/casmi26-c3iso-i1b1-ice-in", "licenses": [{"name": "CC0-1.0"}]}, open(f"{out}/dataset-metadata.json", "w"))
print("mols", len(items), "covered", ncov, "cands", ncand, "spectra", sum(len(i["spectra"]) for i in items))
