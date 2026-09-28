# =====================================================================================
#  PHASE 0 step 2 -- DROP CELL C.  Paste into the LIVE STAGE-C kernel.
#
#  Cell C is (FPNet-seen structure) x (library banned).  It has no test analogue: a
#  Class-2 test molecule has no public spectra, so it cannot be in train.parquet, so the
#  FPNet cannot have trained on it.  "model-seen + library-banned" simply does not occur
#  at test.
#
#  It is 264 of the 536 M=1 groups and scores MRR 0.9646, so it teaches the ranker
#  "when lv is 0, trust the model" -- where the model is 91.7% right.  At test, lv = 0
#  means the model is ~52% right.  Cell B is the counterweight at roughly equal mass, so
#  the library-banned population is about 2x optimistic about f.z.
#
#  This writes rank_train_noC.npz and rebinds X/Y/M/G in place, so you can re-run
#  stage_c_gate2_cell.py straight afterwards and it will judge the reduced matrix.
# =====================================================================================
import os
import numpy as np

# --- recover the pieces --------------------------------------------------------------
if "X" not in globals():                       # kernel restarted: rank_raw.npz is on Drive
    _d = np.load(os.path.join(SC.OUT, "rank_raw.npz"))
    X, Y, M, G = _d["X"], _d["Y"], _d["M"], _d["G"]
    print(f"loaded rank_raw.npz -> {X.shape}")

if "REGIME" not in globals():
    assert "records" in globals(), (
        "REGIME is derived from `records` and is NOT stored in rank_raw.npz.\n"
        "Re-run the Stage-C cohort cell first -- it is deterministic (SC.SEED) and does\n"
        "not need the long extraction loop, only the library/pool already in memory."
    )
    REGIME = np.array([r["regime"] == "unseen" for r in records])

REG = REGIME[G]                                # True = FPNet-unseen
X_full, Y_full, M_full, G_full = X, Y, M, G    # keep the originals recoverable

# --- the four design cells -----------------------------------------------------------
def _cells(reg, m):
    lab = np.empty(len(reg), dtype="<U1")
    lab[(~reg) & (m == 0)] = "A"               # seen,   visible -> test Class 1
    lab[(~reg) & (m == 1)] = "C"               # seen,   banned  -> NO TEST ANALOGUE
    lab[( reg) & (m == 0)] = "D"               # unseen, visible -> cross-library Class 1
    lab[( reg) & (m == 1)] = "B"               # unseen, banned  -> test Class 2
    return lab

lab = _cells(REG, M)
grp = G.astype(np.int64) * 2 + M.astype(int)

print("=" * 78)
print("BEFORE")
print("=" * 78)
for c in "ADCB":
    s = lab == c
    print(f"  cell {c}: {len(np.unique(grp[s])):5d} groups  {int(s.sum()):7,d} rows")
print(f"  M=0 share (rows): {(M == 0).mean():.3f}")

keep = lab != "C"

# --- optional rebalance --------------------------------------------------------------
#  Dropping C removes half the M=1 population, so M=0 rises to ~2/3.  merge_shards.py
#  asserts 0.35 < (M==0).mean() < 0.65 and CFG.W1_PRIORS hedging assumes a roughly
#  balanced split.  REBALANCE subsamples cell A -- the cell C was paired with -- to
#  restore it.  Build both matrices if you can afford two pass-3 runs; they are 25 min
#  each and the comparison is informative in its own right.
REBALANCE = False

if REBALANCE:
    rng = np.random.default_rng(SC.SEED)
    gA = np.unique(grp[lab == "A"])
    n_m1 = len(np.unique(grp[keep & (M == 1)]))
    n_m0_other = len(np.unique(grp[keep & (M == 0) & (lab != "A")]))
    n_keepA = max(0, n_m1 - n_m0_other)        # target: M=0 groups == M=1 groups
    if n_keepA < len(gA):
        dropA = set(rng.permutation(gA)[n_keepA:].tolist())
        keep &= ~np.isin(grp, list(dropA))
        print(f"\n  rebalance: kept {n_keepA}/{len(gA)} cell-A groups")
    else:
        print(f"\n  rebalance: no subsampling needed")

X, Y, M, G = X_full[keep], Y_full[keep], M_full[keep], G_full[keep]
lab2, grp2 = lab[keep], grp[keep]

print("=" * 78)
print("AFTER")
print("=" * 78)
for c in "ADB":
    s = lab2 == c
    if not s.any(): continue
    print(f"  cell {c}: {len(np.unique(grp2[s])):5d} groups  {int(s.sum()):7,d} rows")
print(f"  M=0 share (rows): {(M == 0).mean():.3f}"
      f"{'   <-- outside merge_shards 0.35-0.65' if not 0.35 < (M == 0).mean() < 0.65 else ''}")
print(f"  total: {X.shape[0]:,} rows x {X.shape[1]} features "
      f"({X_full.shape[0]:,} before, -{100*(1-X.shape[0]/X_full.shape[0]):.1f}%)")

assert np.isfinite(X).all(), "non-finite values in X"
assert set(np.unique(Y)) <= {0, 1} and Y.sum() > 0, "invalid labels"

# REGIME is indexed by query, G by row -- after filtering, REGIME itself is unchanged and
# REGIME[G] still resolves correctly, so stage_c_gate2_cell.py works as-is.
out = os.path.join(SC.OUT, "rank_train_noC.npz")
np.savez_compressed(out, X=X, Y=Y, M=M, G=G)
print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
print("\nX/Y/M/G are rebound to the reduced matrix.")
print("Next: re-run stage_c_gate2_cell.py to judge it, then take rank_train_noC.npz")
print("      to Stage D and run stage_d_pass3_noC_cell.py (pass 3 only, ~25 min).")
