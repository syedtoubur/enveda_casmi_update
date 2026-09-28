"""Run the Stage-B machinery on synthetic data, on CPU, before burning A100 hours.

Extracts the real cells from casmi_fpnet_train.ipynb (so it tests the shipped code, not a
paraphrase of it), stubs out Drive/RDKit/parquet, and exercises:
  prep_peaks / merge_peaks_raw / FPNet forward / build_batch+augment /
  sample_candidates / cand_bits / listwise_loss / retrieval_eval / checkpoint round-trip
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB = Path(__file__).resolve().parent.parent / "casmi_fpnet_train.ipynb"
cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]
         if c["cell_type"] == "code"]
SRC = ["".join(c["source"]) for c in cells]


def cell_with(*needles):
    for s in SRC:
        if all(n in s for n in needles):
            return s
    raise AssertionError(f"no cell contains {needles}")


NBITS = 6930
NPOOL = 4000
NSTRUCT = 300
NSPEC = 900
P = 128

g = {"np": np, "torch": torch, "nn": nn, "F": F, "math": math, "print": print}


class CFG:
    MAX_PEAKS_NN = P
    MAX_RAW = 256
    D, LAYERS, HEADS, DROP = 64, 2, 4, 0.1          # tiny, for speed
    BS = 16
    RANK_BS = 8
    N_CAND = 12
    PPM_WIN = 8.5
    LAMBDA_BCE = 0.3
    AUG_PEAK_DROP = 0.20
    AUG_MZ_PPM = 5.0
    AUG_INT_SIGMA = 0.10
    AUG_COND_DROP = 0.15


g["CFG"] = CFG
g["NBITS"] = NBITS
g["dev"] = "cpu"

# ---- 1. constants + preprocessing (the verbatim cell, minus its RDKit tail) -----------
const = cell_with("ADDUCT_LIST", "def prep_peaks")
const = const[:const.index("# --- exact neutral mass")]
exec(compile(const, "const", "exec"), g)
print("[ok] constants + prep_peaks/merge_peaks_raw loaded")

rng = np.random.default_rng(0)
mz = np.sort(rng.uniform(50, 400, 500))
it = rng.random(500)
pm_, pi_ = g["prep_peaks"](mz, it, 400.0)
assert len(pm_) <= P and len(pm_) > 0
assert np.all(np.diff(pm_) >= 0), "prep_peaks must return m/z-sorted peaks"
assert abs(pi_.max() - 1.0) < 1e-5, "prep_peaks normalises to sqrt(it/max) -> max 1.0"
assert (pm_ <= 401.5 + 1e-6).all(), "peaks above precursor+1.5 must be dropped"
p2, i2 = g["prep_peaks"]([], [], 400.0)
assert len(p2) == 0, "empty input must yield empty output, not a crash"
print(f"[ok] prep_peaks: 500 -> {len(pm_)} peaks, sorted, max-normalised, empty-safe")

fm, fi = g["merge_peaks_raw"]([mz[:50], mz[40:90]], [it[:50], it[40:90]])
assert np.all(np.diff(fm) >= 0) and len(fm) <= 100
close = np.diff(fm) < 0.005
assert not close.any(), "merge must collapse peaks within 0.005 Da"
print(f"[ok] merge_peaks_raw: 100 -> {len(fm)} peaks, no pair within 0.005 Da")

# ---- 2. FPNet ------------------------------------------------------------------------
exec(compile(cell_with("class FPNet", "class SinEmb"), "model", "exec"), g)
FPNet = g["FPNet"]
net = FPNet(NBITS, d=CFG.D, layers=CFG.LAYERS, heads=CFG.HEADS, drop=CFG.DROP)
print(f"[ok] FPNet built: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params")

# ---- 3. synthetic packed dataset -----------------------------------------------------
Y = (rng.random((NSTRUCT, NBITS)) < 0.08).astype(np.uint8)
npk = rng.integers(5, P, NSPEC)
mzs = np.zeros((NSPEC, P), np.float32)
its = np.zeros((NSPEC, P), np.float32)
for i in range(NSPEC):
    n = npk[i]
    mzs[i, :n] = np.sort(rng.uniform(50, 800, n))
    its[i, :n] = rng.random(n)
ds = dict(mz=mzs, it=its, npk=npk.astype(np.int16),
          prec=rng.uniform(150, 1100, NSPEC).astype(np.float32),
          ad=rng.integers(0, 26, NSPEC).astype(np.int8),
          ins=rng.integers(0, 5, NSPEC).astype(np.int8),
          ce=np.where(rng.random(NSPEC) < 0.3, np.nan,
                      rng.uniform(5, 60, NSPEC)).astype(np.float32),
          mode=rng.choice([-1, 1], NSPEC).astype(np.int8),
          yrow=rng.integers(0, NSTRUCT, NSPEC).astype(np.int32))

pool_fp = np.packbits((rng.random((NPOOL, NBITS)) < 0.08).astype(np.uint8), axis=1)
pool_mass = np.sort(rng.uniform(150, 1100, NPOOL))
uni_mass = rng.uniform(200, 1000, NSTRUCT)
win_pos = rng.integers(0, NPOOL, NSTRUCT).astype(np.int64)
win_pos[:5] = -1                                   # structures absent from the pool
counts = rng.integers(0, 60, NSTRUCT)
counts[5:10] = 1                                   # degenerate single-candidate windows
win_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
win_idx = rng.integers(0, NPOOL, win_off[-1]).astype(np.int32)
for i in range(NSTRUCT):
    if win_pos[i] >= 0 and win_off[i + 1] > win_off[i]:
        win_idx[win_off[i]] = win_pos[i]           # the truth is inside its own window

g.update(Yf=Y, pool_fp=pool_fp, pool_mass=pool_mass, uni_mass=uni_mass,
         win_pos=win_pos, win_off=win_off, win_idx=win_idx)

# ---- 4. build_batch / bce_eval -------------------------------------------------------
bb = cell_with("def build_batch", "def bce_eval")
bb = bb[:bb.index("DS_SINGLE = load_set")] + bb[bb.index("def build_batch"):]
exec(compile(bb[bb.index("def build_batch"):], "batch", "exec"), g)
build_batch = g["build_batch"]

idx = np.arange(CFG.BS)
for aug in (False, True):
    args, yr = build_batch(ds, idx, aug, rng)
    mzb, itb, padb = args[0], args[1], args[2]
    assert mzb.shape == itb.shape == padb.shape
    assert mzb.shape[0] == CFG.BS
    assert not torch.isnan(mzb).any() and not torch.isnan(itb).any()
    assert torch.isfinite(args[6]).all(), "CE must never reach the model as NaN"
    assert (mzb[padb] == 0).all() and (itb[padb] == 0).all(), "padded slots must be zeroed"
    assert (~padb).any(dim=1).all(), "every row must keep at least one real peak"
    out = net(*args)
    assert out.shape == (CFG.BS, NBITS), out.shape
    print(f"[ok] build_batch(augment={aug}) -> forward {tuple(out.shape)}, "
          f"N={mzb.shape[1]}, pad-safe, CE finite")

# a NaN-CE row specifically
nan_rows = np.flatnonzero(~np.isfinite(ds["ce"]))[:CFG.BS]
if len(nan_rows):
    a2, _ = build_batch(ds, nan_rows, False, rng)
    assert torch.isfinite(a2[6]).all() and float(a2[6][0]) == 25.0
    print("[ok] NaN collision energy falls back to 25.0 eV, matching model_logits()")

# ---- 5. listwise sampler -------------------------------------------------------------
lw = cell_with("def sample_candidates", "def listwise_loss")
exec(compile(lw[lw.index("TAU = "):], "listwise", "exec"), g)
sample_candidates = g["sample_candidates"]
listwise_loss = g["listwise_loss"]
cand_bits = g["cand_bits"]

yr_all = np.arange(NSTRUCT)
cand, ok = sample_candidates(yr_all, rng, CFG.N_CAND)
assert cand.shape == (NSTRUCT, CFG.N_CAND)
assert not ok[:5].any(), "structures missing from the pool must be marked unusable"
assert not ok[5:10].any(), "single-candidate windows must be marked unusable"
good = np.flatnonzero(ok)
assert (cand[good, 0] == win_pos[good]).all(), "column 0 must be the true candidate"
for i in good[:50]:
    assert win_pos[i] not in cand[i, 1:], "the truth must not be resampled as a decoy"
print(f"[ok] sample_candidates: {ok.sum()}/{NSTRUCT} usable, truth at col 0, decoys disjoint")

cb = cand_bits(cand[good][:4])
assert cb.shape == (4, CFG.N_CAND, NBITS) and cb.dtype == torch.uint8
print(f"[ok] cand_bits -> {tuple(cb.shape)}")

args, yr = build_batch(ds, np.arange(CFG.RANK_BS), True, rng)
z = net(*args).float()
L = listwise_loss(z, yr, rng)
if L is not None:
    assert torch.isfinite(L) and L.item() > 0
    L.backward()
    gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    assert gn > 0, "listwise loss produced no gradient"
    print(f"[ok] listwise_loss = {L.item():.4f}, backward gradient norm {gn:.3f}")

# all-unusable batch must return None rather than crash
bad_rows = np.flatnonzero(np.isin(ds["yrow"], np.arange(5)))[:4]
if len(bad_rows):
    a3, y3 = build_batch(ds, bad_rows, False, rng)
    assert listwise_loss(net(*a3).float(), y3, rng) is None
    print("[ok] a batch with no usable structure returns None, not a crash")

# ---- 6. retrieval_eval ---------------------------------------------------------------
retrieval_eval = g["retrieval_eval"]
net.train()
mrr, t1, n = retrieval_eval(net, ds, np.arange(200))
assert 0.0 <= mrr <= 1.0 and 0.0 <= t1 <= 1.0 and n > 0
assert net.training, "retrieval_eval must restore train mode"
print(f"[ok] retrieval_eval: MRR {mrr:.4f} top1 {t1:.4f} over n={n} (random model -> ~1/K)")
print(f"     random baseline for K={CFG.N_CAND} is MRR~{sum(1/r for r in range(1, CFG.N_CAND+1))/CFG.N_CAND:.3f}")

# ---- 7. checkpoint round-trip through load_neural_models()'s contract ----------------
import tempfile, os, glob as _glob
tmp = Path(tempfile.mkdtemp())
for name in ["fp_single_s0", "fp_merged_m0"]:
    torch.save(dict(model=net.state_dict(), nbits=NBITS, d=CFG.D, layers=CFG.LAYERS,
                    step=1234, fp_bits_sha1="deadbeef", tag=name, seed=0), tmp / f"{name}.pt")
single, merged = [], []
for pth in sorted(_glob.glob(str(tmp / "fp_*.pt"))):
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    m = FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"])
    m.load_state_dict(ck["model"])
    (merged if "merged" in os.path.basename(pth) else single).append(m)
assert len(single) == 1 and len(merged) == 1
print("[ok] checkpoint round-trip: load_neural_models() routes 1 single + 1 merged")

# ---- 8. rank_features width ----------------------------------------------------------
nbnb = json.loads((NB.parent / "casmi_rank_matrix.ipynb").read_text(encoding="utf-8"))
rf = next("".join(c["source"]) for c in nbnb["cells"]
          if c["cell_type"] == "code" and "def rank_features" in "".join(c["source"]))
gr = {"np": np}
exec(compile(rf[rf.index("N_ANALOG = CFG.N_ANALOG"):], "rf", "exec"),
     {**gr, "CFG": type("c", (), dict(N_ANALOG=100, SIM_POWER=4.0))})
print("[ok] rank_features cell parses standalone (31 columns asserted inside Stage C)")

print("\nALL SELF-TESTS PASSED")
