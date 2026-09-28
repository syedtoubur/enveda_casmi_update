# =====================================================================================
#  PHASE 0 step 3 -- RE-RUN PASS 3 AGAINST THE CELL-C-DROPPED MATRIX.
#  Paste into the LIVE STAGE-D kernel, below the diagnostic cell.
#
#  Only the ranker changes, so pass 1 (channels 1/2/3) and pass 2 (FPNet logits) are
#  reused untouched.  Only the two arms that read the NEW matrix are rescored:
#      matrix_only = (old model, new matrix)
#      both        = (new model, new matrix)
#  `base` and `model_only` read the OLD matrix and are carried over unchanged.
#
#  Reads the same delta + ce_known breakdown as stage_d_diag_cell.py so the two outputs
#  are directly comparable line for line.
# =====================================================================================
import os
import time
import numpy as np
import pandas as pd

NOC = '/content/drive/MyDrive/casmi_2026/_stagec/rank_train_noC.npz'
assert os.path.exists(NOC), f"not found: {NOC} -- run stage_c_dropC_cell.py in the Stage-C kernel first"

# Defensively restore _z in case it was clobbered in the live kernel
def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

z_noc = np.load(NOC)
print(f"cell-C-dropped matrix: {z_noc['X'].shape[0]:,} rows x {z_noc['X'].shape[1]} features | "
      f"M=0 share {(z_noc['M'] == 0).mean():.3f}")
assert z_noc['X'].shape[1] == NFEAT, f"matrix has {z_noc['X'].shape[1]} features, rank_features makes {NFEAT}"

# --- refit only the 'new' ranker -----------------------------------------------------
RES_PREV = {k: v.copy() for k, v in RES.items()}      # keep the pre-drop results
RANKERS['new'] = fit_rankers(NOC)

for name in ('matrix_only', 'both'):
    mtag, rtag = CONFIGS[name]
    t0 = time.time()
    rows = [dict(mid=mid, cohort=COHORT[mid], rank=rank_of_truth(mid, mtag, rtag))
            for mid, _ in mols]
    d = pd.DataFrame(rows)
    d['rr'] = np.where(d['rank'] > 0, 1.0 / d['rank'], 0.0)
    RES[name] = d
    print(f"  {name:12s} MRR {d.rr.mean():.4f}  "
          f"(was {RES_PREV[name].rr.mean():.4f})  ({time.time()-t0:.0f}s)")

# --- deltas, before and after the drop, split by ce_known ----------------------------
CEK = pd.Series({r['molecule_id']: bool(r['ce_known']) for r in key})
COH = ['C1_npx', 'C1_xlib', 'C2', 'C2_hard']
b = RES['base'].set_index('mid')
ck = CEK.reindex(b.index).fillna(False).astype(bool)

print("\n" + "=" * 94)
print("DELTAS vs base  --  'before' = with cell C, 'after' = cell C dropped")
print("=" * 94)
print(f"{'config':12s} {'cohort':9s} {'ce_known':9s} {'n':>5s} "
      f"{'before':>9s} {'after':>9s} {'change':>9s} {'SE':>8s}")
for name in ('matrix_only', 'both'):
    d_new = RES[name].set_index('mid').rr - b.rr
    d_old = RES_PREV[name].set_index('mid').rr - b.rr
    for coh in COH:
        for want in (True, False):
            m = (b.cohort == coh).values & (ck.values == want)
            if m.sum() == 0: continue
            v_new, v_old = d_new[m], d_old[m]
            se = v_new.std() / np.sqrt(len(v_new))
            print(f"{name:12s} {coh:9s} {str(want):9s} {m.sum():5d} "
                  f"{v_old.mean():+9.4f} {v_new.mean():+9.4f} "
                  f"{v_new.mean()-v_old.mean():+9.4f} {se:8.4f}")
    # pooled, which is what the decision rule reads
    for coh in COH:
        m = (b.cohort == coh).values
        v_new, v_old = d_new[m], d_old[m]
        print(f"{name:12s} {coh:9s} {'POOLED':9s} {m.sum():5d} "
              f"{v_old.mean():+9.4f} {v_new.mean():+9.4f} "
              f"{v_new.mean()-v_old.mean():+9.4f} {v_new.std()/np.sqrt(len(v_new)):8.4f}")
    print()

# --- the decision rule ---------------------------------------------------------------
print("=" * 94)
print("DECISION")
print("=" * 94)
dd = RES['both'].set_index('mid').rr - b.rr
def cohort_delta(c):
    m = b.cohort == c
    return dd[m].mean(), dd[m].std() / np.sqrt(int(m.sum()))

verdict = []
for c in ('C2', 'C2_hard'):
    mu, se = cohort_delta(c)
    ok = mu >= 0.020
    verdict.append(ok)
    print(f"  {c:8s} delta {mu:+.4f} +- {se:.4f}   "
          f"{'>= +0.020' if ok else ('above 1 SE' if mu > se else 'NOT above 1 SE')}")
mu1, se1 = cohort_delta('C1_npx')
no_reg = mu1 >= -0.005
print(f"  {'C1_npx':8s} delta {mu1:+.4f} +- {se1:.4f}   "
      f"{'no regression' if no_reg else 'REGRESSION'}")

print()
if all(verdict) and no_reg:
    print("  PASS -> the cell-C drop cleared it. Proceed to Phase 1.")
else:
    print("  Not a PASS on the literal rule. Read it against the diagnostic first:")
    print("    base's C2/C2_hard scores are inflated by the OLD FPNet's memorisation")
    print("    (75.5% / 73.3% in-window top-1 on structures it trained on, vs the NEW")
    print("    model's honest 59.9% / 60.0% against its own 52.1% reference). A C2 delta")
    print("    at or near zero here is therefore consistent with a real gain on the LB.")
    print("    What would still be alarming: C1_npx regressing, or C2 moving further")
    print("    negative after the drop.")
