"""Stage A3 pseudo_c3 (Kaggle CPU): edges (parent/multiplicity) for the I1b.0 frozen top-150 + NL + parent frag + I1b.1 ICE.
NO truth is loaded (truths_pseudo_c3.pkl is never opened)."""
import glob as _g, sys as _s
_s.path.insert(0, __import__("os").path.dirname(sorted(_g.glob("/kaggle/input/**/code/kit_setup.py", recursive=True), key=len)[0]))
import json, os, pickle, sys, time
PROCS = 4
LIMIT = int(os.environ.get("A3_LIMIT", "0"))
import kit_setup as ks
ks.pip_rdkit(); CD = ks.code_dir(); R = ks.kit_root()
from pathlib import Path
from multiprocessing import Pool
import harness, a3
from common import V22
OUT = "/kaggle/working"
fr = pickle.load(open(ks.one("/kaggle/input/**/i1b_pseudo_c3_full_top150.pkl"), "rb"))
ice = json.load(open(ks.one("/kaggle/input/**/ice_out_pseudo_c3.json")))
spec = pickle.load(open(ks.one("/kaggle/input/**/pseudo_spectra.pkl"), "rb"))
state, deps, cp = harness.load_dump("pseudo_c3", Path(CD) / "casmi_post.py")
mids = [m for m in state["te_mids"] if m in fr["top"]]
if LIMIT: mids = mids[:LIMIT]
rows = cp.run(state, deps, **V22)
tasks = []
for m in mids:
    extra = list((state.get("eng") or {}).get(m, {}).get("smiles") or []) + \
        list(((state.get("pc") or {}).get(m) or {}).get("pc") or []) + list((state["base"][m][0]) or [])
    row = list(rows.get(m, []))
    tasks.append(dict(mid=m, row=row[:25], spectra=spec["spectra"].get(m, []), exclude=row + extra,
                      frozen=[tuple(k) for k in fr["top"][m]["top"]]))
del state, deps
res, t0 = {}, time.time()
with Pool(PROCS, a3.init) as pool:
    for i, r in enumerate(pool.imap_unordered(a3.process, tasks), 1):
        m = r["mid"]; r["timstof"] = bool(spec["timstof"].get(m, False))
        sc = ice.get(m, {})
        for kd in r["kids"]: kd["ice"] = sc.get(kd["smiles"])
        for p in r["parents"]: p["ice"] = sc.get(p["smiles"])
        res[m] = r
        if i % 20 == 0 or i == len(tasks): print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
pickle.dump(res, open(f"{OUT}/a3_pseudo_c3.pkl", "wb"), protocol=4)
print("done", len(res), "no-edge frozen kids", sum(r["stats"].get("frozen_without_edge", 0) for r in res.values()))
