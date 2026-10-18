"""Local, light (no truth, no structures): pseudo_c3 query spectra (all adducts, as I1b.0 used) -> pseudo_spectra.pkl."""
import pickle, sys
sys.path.insert(0, "E:/casmi/lines/c3-iso-rank-a/build")
import isomer_reach as ir
import pandas as pd
mids = list(pd.read_parquet("E:/ENVEDA_casmi/p1_novel/pseudo_c3.parquet", columns=["molecule_id"]).molecule_id)
spec, tims = ir.spectra_pseudo_c3(mids)
pickle.dump(dict(spectra=spec, timstof=tims), open(sys.argv[1], "wb"), protocol=4)
print(len(spec), sum(len(v) for v in spec.values()))
