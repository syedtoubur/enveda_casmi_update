# =====================================================================================
#  VERIFY 2b -- run this in the LIVE kernel, in a NEW cell below the gate that failed.
#
#  The gate failed on ONE rule: held-out MRR 0.911 > 0.88. Every channel rule passed
#  (model 36.4% vs v1's 60.2%, library 8.4%, analog 6.3%) -- this is not v1's disease.
#  The 0.88 ceiling was calibrated off the FORKED matrix, whose FPNet was top-1 in only
#  42% of windows. Yours is measured at es_top1 0.7425 on scaffold-disjoint holdout, so a
#  higher MRR is expected and does not by itself mean leakage.
#
#  Section A prints the structural numbers against the forked matrix, measured here.
#  Section B re-runs the gate with the absolute-MRR proxy replaced by a direct test of the
#  leak mechanism: the FPNet-SEEN half is allowed to be easy (test Class 1 IS memorised by
#  the FPNet -- a Class-1 test molecule is in train.parquet by definition), while the
#  FPNet-UNSEEN half must stay honest. If the split is not real, B fails harder than the
#  old gate did. Section C breaks the MRR into the four design cells.
# =====================================================================================
import os
import numpy as np

if "X" not in globals():                       # kernel restarted: rank_raw.npz is on Drive
    _d = np.load(os.path.join(SC.OUT, "rank_raw.npz"))
    X, Y, M, G = _d["X"], _d["Y"], _d["M"], _d["G"]
if "REGIME" not in globals():
    REGIME = np.array([r["regime"] == "unseen" for r in records])
REG = REGIME[G]
_t  = Y == 1

print("=" * 78)
print("A. STRUCTURE OF THE MATRIX        (forked = the 0.337 submission, measured here)")
print("=" * 78)
_grp = G.astype(np.int64) * 2 + M.astype(int)
_sz  = np.bincount(np.unique(_grp, return_inverse=True)[1])
print(f"  groups {len(_sz):,} | rows {len(Y):,} | size mean {_sz.mean():.1f} "
      f"median {np.median(_sz):.0f} p90 {np.percentile(_sz, 90):.0f} max {_sz.max()} "
      f"| row-weighted {(_sz**2).sum()/_sz.sum():.1f}")
print(f"  forked                    size mean  87.2 median  52 p90 217 max 676 "
      f"| row-weighted 194.7")
_cap = (_sz >= SC.MAX_ROWS_PER_QUERY).mean() * 100
print(f"  hit the {SC.MAX_ROWS_PER_QUERY}-row cap: {_cap:.1f}% of groups"
      + ("   <- crowded windows were truncated; see the note below" if _cap > 25 else ""))

print()
print("  truth is the model's in-window top-1      (forked 42.0%, v1 90.2%)")
for _n, _s in (("FPNet-seen  ", ~REG), ("FPNet-unseen", REG)):
    _m = _t & _s
    print(f"    {_n}  {X[_m, 20].mean()*100:5.1f}%   z(raw) {X[_m, 15].mean():+.2f} "
          f"  raw-max {X[_m, 17].mean():+8.1f}      (forked +1.64 / -152.1)")

print()
print("  truth's library similarity lv             (forked M=0: mean 0.573, >0.9 35.8%)")
for _n, _s in (("FPNet-seen  ", ~REG), ("FPNet-unseen", REG)):
    for _mm in (0, 1):
        _m = _t & _s & (M == _mm)
        if not _m.any(): continue
        print(f"    {_n} M={_mm}  mean {X[_m, 0].mean():.3f}  >0.9 in "
              f"{(X[_m, 0] > 0.9).mean()*100:5.1f}%  ==0 in {(X[_m, 0] == 0).mean()*100:5.1f}%")

print()
print("=" * 78)
print("B. THE GATE, REGIME-AWARE")
print("=" * 78)

#@title VERIFY 2 — the GATE: does this matrix teach a four-channel fusion?
#  Fits the real phase1 ranker on 60% of query groups, then destroys one channel at a time on
#  the held-out 40%. v1 put 60.2% of its skill on the model channel and reduced library to
#  2.7%, analog to 0.2%, MetFrag to 0.0% -- and lost 0.05 on the leaderboard.
import warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier

CH = {'library': [0,1,2,3,4,25], 'analog': [5,6,7,8,9,10,11,12,13,26],
      'model':   [15,16,17,18,19,20], 'frag': [21,22,23,24]}

Xd, Yd, Md = X.astype(np.float64), Y.astype(int), M.astype(int)
grp = G.astype(np.int64) * 2 + Md
qs = np.unique(grp); rg = np.random.default_rng(7); rg.shuffle(qs)
tr = np.isin(grp, qs[:int(len(qs) * 0.6)])

R = []
for w1 in (0.30, 0.60):
    W = np.where(Md[tr] == 0, w1, 1.0 - w1)
    for sd in (0, 1):
        R.append(HistGradientBoostingClassifier(random_state=sd, **CFG.GBM)
                 .fit(Xd[tr], Yd[tr], sample_weight=W))
prob = lambda A: np.mean([m.predict_proba(A)[:, 1] for m in R], axis=0)

def mrr(A, Yv, gv, sel=None):
    p = prob(A); acc = []
    for q in np.unique(gv):
        m = gv == q
        if Yv[m].sum() != 1: continue
        if sel is not None and not sel[m][0]: continue
        o = np.argsort(-p[m]); acc.append(1.0 / (1 + int(np.flatnonzero(Yv[m][o])[0])))
    return (float(np.mean(acc)) if acc else float('nan')), len(acc)

rsh = np.random.default_rng(0)
def destroy(A, gv, cols):
    Z = A.copy()
    for q in np.unique(gv):
        i = np.flatnonzero(gv == q)
        Z[np.ix_(i, cols)] = Z[np.ix_(rsh.permutation(i), cols)]
    return Z

Xe, Ye, ge, Me = Xd[~tr], Yd[~tr], grp[~tr], Md[~tr]
base, nq = mrr(Xe, Ye, ge)
m0, _ = mrr(Xe, Ye, ge, Me == 0); m1, _ = mrr(Xe, Ye, ge, Me == 1)
print(f"held-out MRR {base:.4f} over {nq} groups  (forked ranker 0.713, v1 0.958)")
print(f"   M=0 class-1-like {m0:.4f} | M=1 class-2-like {m1:.4f}")

# --- the leak signature, measured where it actually shows ---------------------------
#  Absolute MRR is not a leak test, and v2 is the reason. v2 deliberately contains an
#  FPNet-SEEN half, because that is what test Class 1 is: a Class-1 test molecule is in
#  train.parquet by definition, so the FPNet trained on it there too. With a strong FPNet
#  that half is genuinely easy and drags the average up honestly. What must stay honest is
#  the FPNet-UNSEEN half -- and v1's whole disease was that no such half existed, so there
#  was nothing to compare against and a single average had to stand in for it.
#
#  0.75 is not a taste threshold: the FPNet's measured es_top1 is 0.7425 against 48 real
#  +-8.5 ppm decoys. A window here holds ~130 candidates, so it MUST do worse than that on
#  structures it never trained on. Above 0.75 means the holdout is not a holdout.
REG = REGIME[G] if "REGIME" in globals() else np.zeros(len(G), bool)
HAS_REG = bool(REG.any() and (~REG).any())
tru = Yd == 1
_t1 = lambda s: float(Xd[tru & s, 20].mean()) if (tru & s).any() else float("nan")
top1_seen, top1_unseen = _t1(~REG), _t1(REG)
mu1, nu1 = mrr(Xe, Ye, ge, REG[~tr] & (Me == 1)) if HAS_REG else (float("nan"), 0)
if HAS_REG:
    print(f"   truth is the model's top-1: FPNet-seen {top1_seen*100:.1f}% | "
          f"FPNet-unseen {top1_unseen*100:.1f}%   (forked 42.0%, v1 90.2% all-seen)")
    print(f"   MRR on the honest cell (FPNet-unseen + library banned): {mu1:.4f} "
          f"over {nu1} groups")
else:
    print("   no FPNet-unseen half in this matrix - falling back to the absolute MRR rule")
SHARE = {}
for cn, cols in CH.items():
    mm, _ = mrr(destroy(Xe, ge, cols), Ye, ge)
    SHARE[cn] = (base - mm) / base
    print(f"  {cn+' destroyed':22} MRR {mm:.4f} -> carries {SHARE[cn]*100:5.1f}% of skill")
print("  reference -- forked: library 14.5  analog 15.0  model 33.9  frag 2.3")
print("               v1    : library  2.7  analog  0.2  model 60.2  frag 0.0")

fail = []
if SHARE['model'] > 0.50:   fail.append(f"model channel carries {SHARE['model']*100:.0f}% > 50% - v1's failure mode")
if SHARE['library'] < 0.05: fail.append(f"library channel carries only {SHARE['library']*100:.0f}% - it has been switched off")
if SHARE['analog']  < 0.05: fail.append(f"analog channel carries only {SHARE['analog']*100:.0f}% - it has been switched off")
if HAS_REG:
    if top1_unseen > 0.75:
        fail.append(f"the FPNet is top-1 on {top1_unseen*100:.0f}% of FPNet-UNSEEN queries "
                    "(>75%, and its measured es_top1 is 74%) - that half is not unseen")
    if top1_seen - top1_unseen < 0.10:
        fail.append(f"seen {top1_seen*100:.0f}% vs unseen {top1_unseen*100:.0f}% - the two "
                    "halves are indistinguishable, so the split separates nothing")
    if mu1 > 0.88:
        fail.append(f"MRR {mu1:.3f} > 0.88 on FPNet-unseen + library-banned queries - the "
                    "hardest cell in the matrix should not be near-solved")
elif base > 0.88:
    fail.append(f"held-out MRR {base:.3f} > 0.88 with no FPNet-unseen half to explain it")
GATE_OK = not fail
if fail:
    print("\n*** GATE FAILED - NOT writing rank_train.npz ***")
    for f in fail: print("   -", f)
else:
    np.savez_compressed(os.path.join(SC.OUT, 'rank_train.npz'), X=X, Y=Y, M=M, G=G)
    print(f"\n*** GATE PASSED *** -> {SC.OUT}/rank_train.npz")
    if HAS_REG and not (0.60 <= mu1 <= 0.85):
        print(f"[WARN] honest-cell MRR {mu1:.3f} outside 0.60-0.85 - read Stage D carefully.")
    if SHARE['model'] > 0.40:
        print(f"[WARN] model channel {SHARE['model']*100:.0f}% is high (forked 33.9%) - "
              "expected with a stronger FPNet, but watch Stage D's C2_hard.")

print()
print("=" * 78)
print("C. THE FOUR DESIGN CELLS")
print("=" * 78)
_lab = {("FPNet-seen", 0): "A  test Class 1",
        ("FPNet-seen", 1): "C  seen, library banned",
        ("FPNet-unseen", 0): "D  cross-library Class 1",
        ("FPNet-unseen", 1): "B  TEST CLASS 2  <- the LB"}
for _n, _s in (("FPNet-seen", ~REG), ("FPNet-unseen", REG)):
    for _mm in (0, 1):
        _v, _k = mrr(Xe, Ye, ge, _s[~tr] & (Me == _mm))
        print(f"   {_n:13s} M={_mm}  {_lab[(_n, _mm)]:26s} MRR {_v:.4f}  ({_k} groups)")
print()
print("   Cell B is the one v1 had zero examples of. If it is far below the other three,")
print("   the matrix is doing its job: the ranker is being shown a regime where the model")
print("   is weak and the other channels have to carry the query.")
