"""Stage A3 (Kaggle CPU): C1_npx, C1_xlib, C2, C2_hard -> truth-free per-molecule tables. Shard k of NSHARD
(molecule index % NSHARD == k) so the two kernels are balanced. NO truth is loaded."""
import glob as _g, sys as _s
_s.path.insert(0, __import__("os").path.dirname(sorted(_g.glob("/kaggle/input/**/code/kit_setup.py", recursive=True), key=len)[0]))
import json, os, pickle, sys, time
SHARD, NSHARD, PROCS = __SHARD__, 2, 4
LIMIT = int(os.environ.get("A3_LIMIT", "0"))      # smoke only
COHORTS = ["C1_npx", "C1_xlib", "C2", "C2_hard"]
import kit_setup as ks
ks.pip_rdkit(); CD = ks.code_dir(); R = ks.kit_root()
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import harness, a3
from common import V22
OUT = "/kaggle/working"
val = pd.read_parquet(ks.one("/kaggle/input/**/val.parquet"))
def spectra_for(m):
    g = val[val.molecule_id == m]
    return [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                 adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1)
            for r in g.itertuples()][:16], bool((g.instrument_type == "timsTOF").any())
log = open(f"{OUT}/a3_s{SHARD}.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
for coh in COHORTS:
    state, deps, cp = harness.load_dump(coh, Path(CD) / "casmi_post.py")
    mids = [m for i, m in enumerate(state["te_mids"]) if i % NSHARD == SHARD]
    if LIMIT: mids = mids[:LIMIT]
    rows = cp.run(state, deps, **V22)
    tasks, tims = [], {}
    for m in mids:
        sp, tm = spectra_for(m); tims[m] = tm
        tasks.append(dict(mid=m, row=list(rows.get(m, []))[:25], spectra=sp, exclude=list(rows.get(m, [])[:25])))
    del state, deps
    res, t0 = {}, time.time()
    with Pool(PROCS, a3.init) as pool:
        for i, r in enumerate(pool.imap_unordered(a3.process, tasks), 1):
            r["timstof"] = tims[r["mid"]]; res[r["mid"]] = r
            if i % 20 == 0 or i == len(tasks): P(coh, f"{i}/{len(tasks)}", f"{time.time()-t0:.0f}s")
    pickle.dump(res, open(f"{OUT}/a3_{coh}_s{SHARD}.pkl", "wb"), protocol=4)
    P(coh, "done", len(res), "mols")
