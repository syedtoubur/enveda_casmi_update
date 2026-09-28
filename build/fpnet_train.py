# %% [markdown]
# # CASMI 2026 — Stage A + B: train your own FPNet (Channel 4)
#
# Produces four drop-in checkpoints for `enveda-casmi-2026-phase1-enhanced-baseline.ipynb`:
# `fp_single_s0.pt`, `fp_single_s1.pt`, `fp_merged_m0.pt`, `fp_merged_m1.pt`.
#
# **Run on Colab with an A100.** Stage A is CPU-only — if you want to save units, run
# Stage A on a CPU runtime first (it checkpoints to Drive), then switch to A100 for Stage B.
#
# Three invariants this notebook enforces, because each one fails *silently*:
#
# 1. **`fp_bits.npy` layout.** The model's 6,930 outputs are dotted against `pool.fps()` at
#    inference. A different bit order is not an error, it is noise. The sha1 is asserted here
#    and written into every checkpoint.
# 2. **Preprocessing parity.** `prep_peaks` / `_merge_peaks` are copied verbatim from the
#    inference notebook. If they drift, the model is trained on a different input
#    distribution than it is scored on.
# 3. **Leakage.** Every structure in the union of `val_key.pkl`'s `ban_keys` is excluded from
#    training. That set is empty for C1 cohorts (real Class-1 molecules *do* have public
#    spectra, so training on them is faithful) and non-empty for C2/C2_hard (they do not).

# %%
#@title Stage 0 — config + Drive manifest
from pathlib import Path

class CFG:
    ROOT   = "/content/drive/MyDrive/casmi_2026"
    WORK   = "/content/drive/MyDrive/casmi_2026/_fpnet"
    SCRATCH= "/content/_fpnet"          # fast local disk; mirrored to WORK at checkpoints
    USE_DRIVE = True

    # ---- corpus selection -------------------------------------------------------
    TEST_ADDUCTS = ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
                    "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"]
    CAP_ENVEDA180 = 200_000   # 45% of train, ~0.000 Jaccard vs any NP library, but the
                              # largest block of timsTOF-native data. Keep some, not all.
    TIMSTOF_REPEAT = 2        # oversample factor for timsTOF rows (test is 100% timsTOF)

    # ---- packing ----------------------------------------------------------------
    MAX_PEAKS_NN = 128        # must equal the inference notebook's MAX_PEAKS_NN
    MAX_RAW      = 256        # raw peaks retained per spectrum, for merged-variant fusion
    MERGED_PER_STRUCT = 4     # pre-generated merged pseudo-spectra per structure

    # ---- model (identical to phase1 FPNet so load_neural_models() is a drop-in) ---
    D = 512
    LAYERS = 6
    HEADS = 8
    DROP = 0.1

    # ---- stage B1: BCE ----------------------------------------------------------
    BS = 512
    LR = 3e-4
    EPOCHS_BCE = 14
    ES_FRAC = 0.08            # scaffold-disjoint early-stopping split

    # ---- stage B2: listwise retrieval fine-tune ---------------------------------
    EPOCHS_RANK = 3
    RANK_BS = 48              # molecules per listwise batch
    N_CAND  = 48              # 1 positive + 47 decoys drawn from the real +-8.5 ppm window
    PPM_WIN = 8.5
    LAMBDA_BCE = 0.3          # keeps f.z on a calibrated log-odds scale for the GBDT
    LR_RANK = 5e-5

    # ---- augmentation -----------------------------------------------------------
    AUG_PEAK_DROP = 0.20      # max fraction of peaks dropped
    AUG_MZ_PPM    = 5.0
    AUG_INT_SIGMA = 0.10
    AUG_COND_DROP = 0.15      # p(zero out adduct / instrument / CE conditioning)

    SEEDS_SINGLE = (0, 1)
    SEEDS_MERGED = (0, 1)

if CFG.USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

R  = Path(CFG.ROOT)
W  = Path(CFG.WORK);    W.mkdir(parents=True, exist_ok=True)
S  = Path(CFG.SCRATCH); S.mkdir(parents=True, exist_ok=True)

COMP      = R / "enveda-CASMI26-molecule-id-mass-spectra"
TRAIN     = COMP / "train.parquet"
TEST      = COMP / "test.parquet"
COCO      = R / "coconut_dataset"
FPB       = COCO / "fp_bits.npy"
COCO_FP   = COCO / "coco_fp.npy"
COCO_MASS = COCO / "coco_mass.npy"
COCO_META = COCO / "coco_meta.pkl"
VALDIR    = R / "val_scripts"
VAL_KEY   = VALDIR / "val_key.pkl"
SCAF_PKL  = VALDIR / "scaffolds.pkl"
RDKIT_WHL = next(iter(sorted((R / "casmi26-offline-rdkit-2026033").glob("rdkit-*.whl"))), None)

REQUIRED = {"train.parquet": TRAIN, "test.parquet": TEST, "fp_bits.npy": FPB,
            "coco_fp.npy": COCO_FP, "coco_mass.npy": COCO_MASS, "coco_meta.pkl": COCO_META,
            "val_key.pkl": VAL_KEY}
OPTIONAL = {"scaffolds.pkl": SCAF_PKL, "rdkit wheel": RDKIT_WHL}

def _size(p):
    if p is None or not p.exists(): return "-"
    n = p.stat().st_size
    return f"{n/1e9:.2f} GB" if n >= 1e9 else (f"{n/1e6:.1f} MB" if n >= 1e6 else f"{n/1e3:.0f} KB")

missing = []
for tag, group in (("REQ", REQUIRED), ("opt", OPTIONAL)):
    for name, p in group.items():
        ok = p is not None and p.exists()
        if not ok and tag == "REQ": missing.append(name)
        print(f"{'ok ' if ok else ('!! ' if tag=='REQ' else ' . ')} {name:16} {_size(p):>10}   {p}")
if missing:
    raise SystemExit(f"\nMISSING REQUIRED: {missing}\nCheck CFG.ROOT (currently {CFG.ROOT}).")
print("\nall required inputs present")

import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(Stage A does not need one)")

# %%
#@title Stage 0b — install pinned RDKit + imports
import os, sys, subprocess, time, math, pickle, hashlib, json, gc
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

try:
    import rdkit
except ImportError:
    if RDKIT_WHL is not None and RDKIT_WHL.exists():
        if subprocess.run([sys.executable,"-m","pip","install","-q",str(RDKIT_WHL)]).returncode != 0:
            subprocess.run([sys.executable,"-m","pip","install","-q","rdkit"], check=True)
    else:
        subprocess.run([sys.executable,"-m","pip","install","-q","rdkit"], check=True)

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
print("rdkit", rdkit.__version__)

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# %%
#@title Stage 0c — fp_bits: the layout contract
BITS  = np.load(FPB)
NBITS = int(len(BITS))
FP_SHA1 = hashlib.sha1(np.ascontiguousarray(BITS).tobytes()).hexdigest()

assert NBITS == 6930, f"expected 6,930 bits, got {NBITS}"
assert int(BITS.max()) < 10407, "fp_bits must index into the 10,407-bit concatenation"
assert len(np.unique(BITS)) == NBITS, "fp_bits contains duplicate indices"
print(f"NBITS={NBITS}  sha1={FP_SHA1}")
print("This sha1 goes into every checkpoint and is re-asserted inside the Kaggle notebook.")

# %%
#@title Stage 0d — constants + preprocessing COPIED VERBATIM from the inference notebook
# Any edit below is a train/inference mismatch. Do not 'improve' these.

ADDUCT_LIST = ["[M+H]+","[M+NH4]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M-2H2O+H]+","[M]+",
               "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+C2H4O2-H]-","[M+Cl]-","[M]-",
               "[M+2H]2+","[M-2H]-","[2M+H]+","[2M+Na]+","[2M+NH4]+","[2M-H]-","[2M+K]+",
               "[2M+CH2O2-H]-","[2M+C2H4O2-H]-","[2M+Na-2H]-","[M+Na-2H]-","[M-H2O]+","<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
UNK_AD    = ADDUCT_IX["<unk>"]
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
OTHER_INSTR = 4

def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4

def prep_peaks(mz, inten, prec_mz, max_peaks=CFG.MAX_PEAKS_NN, floor=1e-3, win=50.0, per_win=8):
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)

def merge_peaks_raw(mz_list, it_list):
    """_merge_peaks() from the inference notebook, taking raw arrays instead of a DataFrame."""
    mz = np.concatenate([np.asarray(m, float) for m in mz_list])
    it = np.concatenate([np.asarray(v, float) / max(float(np.asarray(v, float).max()), 1e-9)
                         for v in it_list])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]

# --- exact neutral mass from the labelled formula (phase1 TIER-1) ------------------
import re as _re
_MONO = {"C":12.0,"H":1.0078250319,"N":14.0030740052,"O":15.9949146221,"S":31.97207069,
         "P":30.97376151,"F":18.99840320,"Cl":34.96885271,"Br":78.9183376,"I":126.904468,
         "Si":27.9769265327,"Se":79.9165196,"B":11.0093055,"Na":22.98976928,"K":38.96370649,
         "Fe":55.9349375,"D":2.0141017778,"As":74.9215965,"Li":7.0160034,"Sn":119.9021947}
_TOK = _re.compile(r"([A-Z][a-z]?)(\d*)")
def formula_mass(f):
    if not isinstance(f, str): return np.nan
    s, pos = 0.0, 0
    for m in _TOK.finditer(f):
        if m.start() != pos: return np.nan
        pos = m.end()
        el = m.group(1)
        if el not in _MONO: return np.nan
        s += _MONO[el] * (int(m.group(2)) if m.group(2) else 1)
    return s if pos == len(f) else np.nan

# --- fingerprint, identical concatenation to fp_and_mass() ------------------------
_g = {}
def _fp_init():
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_mass_scaf(smi):
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([
            _g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
            np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[BITS]
        try: sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        except Exception: sc = ""
        return fp, float(ExactMolWt(m)), sc
    except Exception:
        return None

def ce_scalar(v):
    try:
        a = np.atleast_1d(v)
        if len(a) == 0 or pd.isna(a).all(): return np.nan
        return float(np.nanmean(a.astype(float)))
    except Exception:
        return np.nan

print("preprocessing helpers ready (verbatim copies)")

# %% [markdown]
# ---
# ## Stage A1 — read metadata, choose the corpus
#
# Peaks are *not* read here. 2.5M list-typed rows would blow up RAM for a decision that only
# needs the scalar columns.

# %%
#@title Stage A1 — metadata + corpus selection
MDCOLS = ['inchikey14','normalized_smiles','molecular_formula','ingest_lib','adduct',
          'ionization_mode','instrument_type','precursor_mz','collision_energy_ev','num_peaks']

md = pq.read_table(TRAIN, columns=MDCOLS).to_pandas()
md['row'] = np.arange(len(md), dtype=np.int64)
log(f"train metadata: {len(md):,} rows / {md.inchikey14.nunique():,} structures")

md['instr_ix'] = [instr_family(s) for s in md.instrument_type]
md['ce'] = [ce_scalar(v) for v in md.collision_energy_ev]
md['mode'] = np.where(md.ionization_mode.astype(str).str.lower().str.startswith('pos'), 1.0, -1.0)

sel = (md.adduct.isin(CFG.TEST_ADDUCTS).values
       & md.inchikey14.notna().values
       & md.normalized_smiles.notna().values
       & md.precursor_mz.notna().values
       & (md.num_peaks.values >= 3))
log(f"after adduct/label filter: {sel.sum():,} ({sel.mean()*100:.1f}% of train)")

# cap enveda-180 — wrong chemical space, right instrument
rng = np.random.default_rng(20260922)
is180 = (md.ingest_lib.values == 'enveda-180') & sel
n180 = int(is180.sum())
if n180 > CFG.CAP_ENVEDA180:
    idx180 = np.flatnonzero(is180)
    drop = rng.choice(idx180, size=n180 - CFG.CAP_ENVEDA180, replace=False)
    sel[drop] = False
    log(f"enveda-180 capped: {n180:,} -> {CFG.CAP_ENVEDA180:,}")

SEL = np.flatnonzero(sel)
log(f"corpus: {len(SEL):,} spectra / {md.loc[SEL,'inchikey14'].nunique():,} structures")
print(md.loc[SEL].ingest_lib.value_counts().to_string())
print("\ninstrument mix of the corpus:")
print(pd.Series([INSTR_LIST[i] for i in md.loc[SEL,'instr_ix']]).value_counts().to_string())

# spectra-per-molecule distribution of the REAL test set — reused for merged sampling
SPEC_N = pq.read_table(TEST, columns=['molecule_id']).to_pandas().groupby('molecule_id').size().values
log(f"test spectra-per-molecule: median {np.median(SPEC_N):.0f}, max {SPEC_N.max()}")

# %% [markdown]
# ## Stage A2 — fingerprints for **all** train structures
#
# Computed over every unique `inchikey14` in train, not just the selected corpus, because the
# same matrix serves two purposes: training targets (`Y`) and the train half of the
# **candidate pool** that `build_candidate_pool()` assembles at inference. Computing it once
# guarantees the two agree.

# %%
#@title Stage A2 — Y.npy, masses, scaffolds
from multiprocessing import Pool as MPool

UNI_NPZ = S / "uni.npz"
if UNI_NPZ.exists():
    d = np.load(UNI_NPZ, allow_pickle=True)
    uni_ik, uni_smi, Y, uni_mass, uni_scaf = d['ik'], d['smi'], d['Y'], d['mass'], d['scaf']
    log(f"uni cached: {Y.shape}")
else:
    u = md[['inchikey14','normalized_smiles']].dropna().drop_duplicates('inchikey14').reset_index(drop=True)
    log(f"computing fingerprints for {len(u):,} unique structures...")
    with MPool(os.cpu_count()) as mp:
        res = mp.map(fp_mass_scaf, list(u.normalized_smiles), chunksize=400)
    ok = [i for i, r in enumerate(res) if r is not None]
    Y        = np.stack([res[i][0] for i in ok]).astype(np.uint8)
    uni_mass = np.array([res[i][1] for i in ok], dtype=np.float64)
    uni_scaf = np.array([res[i][2] for i in ok], dtype=object)
    uni_ik   = u.inchikey14.values[ok].astype(object)
    uni_smi  = u.normalized_smiles.values[ok].astype(object)
    np.savez(UNI_NPZ, ik=uni_ik, smi=uni_smi, Y=Y, mass=uni_mass, scaf=uni_scaf)
    log(f"Y: {Y.shape}  ({len(u)-len(ok):,} SMILES failed to parse)")

assert Y.shape[1] == NBITS
ik2row = {k: i for i, k in enumerate(uni_ik)}
log(f"mean bits on per structure: {Y.mean()*NBITS:.0f} / {NBITS}")

# %% [markdown]
# ## Stage A3 — pack the peaks into flat memmaps
#
# Streamed row-group by row-group so peak memory stays bounded. Two arrays per spectrum:
#
# * **prepped** (`mz`,`it` at 128 peaks) — what the model consumes, via the verbatim `prep_peaks`
# * **raw** (top-256 by intensity) — needed later, because merged-variant fusion must run
#   `_merge_peaks` on *raw* lists and only then `prep_peaks`, exactly as inference does

# %%
#@title Stage A3 — pack (~10-20 min)
NP_ = len(SEL)
P = CFG.MAX_PEAKS_NN
RW = CFG.MAX_RAW

paths = {k: S / f"{k}.npy" for k in
         ["mz","it","npk","rmz","rit","rnpk","prec","ad","ins","ce","mode","yrow"]}

if all(p.exists() for p in paths.values()):
    log("packed arrays cached")
else:
    A = {
        "mz":   np.lib.format.open_memmap(paths["mz"],   mode='w+', dtype=np.float32, shape=(NP_, P)),
        "it":   np.lib.format.open_memmap(paths["it"],   mode='w+', dtype=np.float32, shape=(NP_, P)),
        "npk":  np.lib.format.open_memmap(paths["npk"],  mode='w+', dtype=np.int16,   shape=(NP_,)),
        "rmz":  np.lib.format.open_memmap(paths["rmz"],  mode='w+', dtype=np.float32, shape=(NP_, RW)),
        "rit":  np.lib.format.open_memmap(paths["rit"],  mode='w+', dtype=np.float32, shape=(NP_, RW)),
        "rnpk": np.lib.format.open_memmap(paths["rnpk"], mode='w+', dtype=np.int16,   shape=(NP_,)),
    }
    pos_of_row = np.full(len(md), -1, dtype=np.int64)
    pos_of_row[SEL] = np.arange(NP_)

    pf = pq.ParquetFile(TRAIN)
    offs = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    prec_all = md.precursor_mz.values.astype(np.float64)
    done = 0
    for rg in range(pf.metadata.num_row_groups):
        lo, hi = offs[rg], offs[rg+1]
        loc = SEL[(SEL >= lo) & (SEL < hi)]
        if len(loc) == 0: continue
        t = pf.read_row_group(rg, columns=['ms2_mzs','ms2_normalized_intensities'])
        a = t.column('ms2_mzs').combine_chunks()
        b = t.column('ms2_normalized_intensities').combine_chunks()
        ao, av = a.offsets.to_numpy(), a.values.to_numpy(zero_copy_only=False)
        bo, bv = b.offsets.to_numpy(), b.values.to_numpy(zero_copy_only=False)
        for gr in loc:
            k = int(gr - lo); j = int(pos_of_row[gr])
            mzr = av[ao[k]:ao[k+1]].astype(np.float64)
            itr = bv[bo[k]:bo[k+1]].astype(np.float64)
            pm  = float(prec_all[gr])
            # raw top-RW by intensity, kept m/z-sorted
            if len(mzr) > RW:
                keep = np.argsort(-itr)[:RW]; keep.sort()
                mzr, itr = mzr[keep], itr[keep]
            A["rnpk"][j] = len(mzr)
            A["rmz"][j, :len(mzr)] = mzr
            A["rit"][j, :len(itr)] = itr
            pm_, pi_ = prep_peaks(mzr, itr, pm)
            n = len(pm_)
            A["npk"][j] = n
            if n:
                A["mz"][j, :n] = pm_
                A["it"][j, :n] = pi_
        done += len(loc)
        if rg % 5 == 0 or done == NP_:
            log(f"  packed {done:,}/{NP_:,} spectra (row-group {rg+1}/{pf.metadata.num_row_groups})")
        del t, a, b, av, bv; gc.collect()
    for v in A.values(): v.flush()
    del A; gc.collect()

    sub = md.loc[SEL]
    np.save(paths["prec"], sub.precursor_mz.values.astype(np.float32))
    np.save(paths["ad"],   np.array([ADDUCT_IX.get(a, UNK_AD) for a in sub.adduct], np.int8))
    np.save(paths["ins"],  sub.instr_ix.values.astype(np.int8))
    np.save(paths["ce"],   sub.ce.values.astype(np.float32))
    np.save(paths["mode"], sub['mode'].values.astype(np.int8))
    np.save(paths["yrow"], np.array([ik2row.get(k, -1) for k in sub.inchikey14], np.int32))
    log("packed arrays written")

yrow = np.load(paths["yrow"])
assert (yrow >= 0).all(), "a selected spectrum has no fingerprint row — drop those rows first"
log(f"packed: {NP_:,} spectra x {P} peaks")

# %% [markdown]
# ## Stage A4 — splits: the leakage rule
#
# `ban_keys` from `val_key.pkl` is exactly the right exclusion set, and it is asymmetric
# on purpose:
#
# * **C1 cohorts → empty.** Real Class-1 test molecules *do* have public spectra, so an FPNet
#   that trained on them is faithful to the test condition. Excluding them would make the C1
#   number pessimistic and wrong.
# * **C2 → the structure. C2_hard → the whole Murcko scaffold family.** Real Class-2
#   molecules have no public spectra, so an FPNet that trained on them makes the C2 number a
#   fantasy.

# %%
#@title Stage A4 — splits.npz
val_key = pickle.load(open(VAL_KEY, 'rb'))
BAN = set()
for r in val_key:
    BAN |= set(r.get('ban_keys') or [])
log(f"val_key: {len(val_key):,} molecules -> {len(BAN):,} structures banned from FPNet training")

cohort_n = pd.Series([r['cohort'] for r in val_key]).value_counts().to_dict()
print("  cohorts:", cohort_n)

struct_banned = np.array([k in BAN for k in uni_ik])
is_banned = struct_banned[yrow]
log(f"spectra excluded by ban: {is_banned.sum():,} ({is_banned.mean()*100:.2f}%)")

# scaffold-disjoint early-stopping split over the surviving structures
scaf = uni_scaf.astype(str)
rng2 = np.random.default_rng(7)
free_scaf = np.array(sorted({s for s, b in zip(scaf, struct_banned) if not b and s}))
rng2.shuffle(free_scaf)
n_es = int(len(free_scaf) * CFG.ES_FRAC)
es_scaf = set(free_scaf[:n_es].tolist())

struct_es = np.array([(s in es_scaf) for s in scaf])
is_es    = struct_es[yrow] & ~is_banned
is_train = ~struct_es[yrow] & ~is_banned

assert not (set(scaf[struct_es & ~struct_banned]) & set(scaf[~struct_es & ~struct_banned])), "scaffold leakage"
assert not (BAN & set(uni_ik[~struct_banned])), "ban set inconsistency"
log(f"train {is_train.sum():,} spectra | early-stop {is_es.sum():,} | banned {is_banned.sum():,}")
log(f"scaffolds: train {len(free_scaf)-n_es:,} / es {n_es:,}")

np.savez(S/"splits.npz", is_train=is_train, is_es=is_es, is_banned=is_banned,
         struct_banned=struct_banned, struct_es=struct_es)

# timsTOF oversampling index for the training sampler
ins_arr = np.load(paths["ins"])
tr_idx = np.flatnonzero(is_train)
tims   = tr_idx[ins_arr[tr_idx] == 0]
TRAIN_IDX = np.concatenate([tr_idx] + [tims] * (CFG.TIMSTOF_REPEAT - 1))
ES_IDX = np.flatnonzero(is_es)
log(f"sampler index: {len(TRAIN_IDX):,} (timsTOF {len(tims):,} x{CFG.TIMSTOF_REPEAT})")

# %% [markdown]
# ## Stage A5 — candidate pool + per-structure ±8.5 ppm windows
#
# Rebuilt exactly as `build_candidate_pool()` does at inference: COCONUT 2.0 ∪ (train
# structures not already in COCONUT), sorted by `ExactMolWt`. The windows precomputed here
# are the decoy sets Stage B2 trains against — the same decoys the LB will present.

# %%
#@title Stage A5 — pool + windows
POOL_NPZ = S / "pool.npz"
if POOL_NPZ.exists():
    d = np.load(POOL_NPZ, allow_pickle=True)
    pool_fp, pool_mass, pool_keys = d['fp'], d['mass'], d['keys']
    log(f"pool cached: {len(pool_mass):,}")
else:
    cm = pickle.load(open(COCO_META, 'rb'))
    co_fp   = np.load(COCO_FP)
    co_mass = np.load(COCO_MASS)
    co_keys = np.asarray(cm['keys'], dtype=object)
    assert cm['nbits'] == NBITS, f"pool nbits {cm['nbits']} != {NBITS}"
    log(f"COCONUT: {len(co_mass):,} structures")

    _co = set(co_keys.tolist())
    extra = np.array([k not in _co for k in uni_ik])
    tr_fp = np.packbits(Y[extra], axis=1)
    log(f"train-only structures added to pool: {extra.sum():,}")

    fp   = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, uni_mass[extra]])
    keys = np.concatenate([co_keys, uni_ik[extra]])
    good = np.isfinite(mass)
    o = np.argsort(mass[good])
    pool_fp, pool_mass, pool_keys = fp[good][o], mass[good][o], keys[good][o]
    np.savez(POOL_NPZ, fp=pool_fp, mass=pool_mass, keys=pool_keys)
    log(f"pool: {len(pool_mass):,} structures, packed fp {pool_fp.shape}")

pool_k2i = {k: i for i, k in enumerate(pool_keys)}

# ragged +-PPM_WIN windows, one per train structure
WIN_NPZ = S / "windows.npz"
if WIN_NPZ.exists():
    d = np.load(WIN_NPZ)
    win_off, win_idx, win_pos = d['off'], d['idx'], d['pos']
    log("windows cached")
else:
    ppm = CFG.PPM_WIN / 1e6
    lo = np.searchsorted(pool_mass, uni_mass * (1 - ppm), 'left')
    hi = np.searchsorted(pool_mass, uni_mass * (1 + ppm), 'right')
    win_pos = np.array([pool_k2i.get(k, -1) for k in uni_ik], dtype=np.int64)
    counts = np.maximum(hi - lo, 0)
    win_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    win_idx = np.empty(win_off[-1], np.int32)
    for i in range(len(uni_ik)):
        win_idx[win_off[i]:win_off[i+1]] = np.arange(lo[i], hi[i], dtype=np.int32)
    np.savez(WIN_NPZ, off=win_off, idx=win_idx, pos=win_pos)
    log(f"windows: median {np.median(counts):.0f} candidates, p95 {np.percentile(counts,95):.0f}, "
        f"{(win_pos<0).sum():,} structures missing from pool")

usable_rank = (win_pos >= 0) & ((win_off[1:] - win_off[:-1]) >= 2)
log(f"structures usable for the listwise stage: {usable_rank.sum():,}")

# %% [markdown]
# ## Stage A6 — pre-generate merged pseudo-spectra
#
# `model_logits()` averages a per-spectrum family and a **merged** family. The merged path
# fuses all of a molecule's peak lists with `_merge_peaks` and calls the model once, with
# `prec = median(precursor_mz)`, adduct/instrument from the first row and — note —
# **`ce = 25.0` hard-coded**. Training the merged variant under exactly those conditions is
# what makes it a drop-in rather than a near-miss.

# %%
#@title Stage A6 — merged set
MG = {k: S / f"mg_{k}.npy" for k in ["mz","it","npk","prec","ad","ins","ce","mode","yrow"]}

if all(p.exists() for p in MG.values()):
    log("merged set cached")
else:
    rmz  = np.load(paths["rmz"],  mmap_mode='r'); rit = np.load(paths["rit"], mmap_mode='r')
    rnpk = np.load(paths["rnpk"], mmap_mode='r')
    prec_a = np.load(paths["prec"]); ad_a = np.load(paths["ad"])
    ins_a  = np.load(paths["ins"]);  md_a = np.load(paths["mode"])

    order = np.argsort(yrow, kind='stable')
    ys = yrow[order]
    bnd = np.flatnonzero(np.diff(ys)) + 1
    groups = np.split(order, bnd)
    log(f"{len(groups):,} structure groups")

    rg = np.random.default_rng(11)
    M_mz, M_it, M_np, M_pr, M_ad, M_in, M_md, M_y = [], [], [], [], [], [], [], []
    for gi, g in enumerate(groups):
        sr = int(yrow[g[0]])
        if struct_banned[sr]: continue
        for _ in range(CFG.MERGED_PER_STRUCT):
            k = int(min(len(g), rg.choice(SPEC_N)))
            pick = g if k >= len(g) else rg.choice(g, size=k, replace=False)
            mzs = [rmz[i, :rnpk[i]] for i in pick]
            its = [rit[i, :rnpk[i]] for i in pick]
            if not any(len(m) for m in mzs): continue
            fm, fi = merge_peaks_raw(mzs, its)
            pm = float(np.median(prec_a[pick]))
            pmz, pit = prep_peaks(fm, fi, pm)
            if len(pmz) == 0: continue
            M_mz.append(pmz); M_it.append(pit); M_np.append(len(pmz))
            M_pr.append(pm); M_ad.append(ad_a[pick[0]]); M_in.append(ins_a[pick[0]])
            M_md.append(float(np.mean(md_a[pick]))); M_y.append(sr)
        if gi % 20000 == 0: log(f"  merged {gi:,}/{len(groups):,} groups")

    NM = len(M_np)
    mm = np.lib.format.open_memmap(MG["mz"], mode='w+', dtype=np.float32, shape=(NM, P))
    mi = np.lib.format.open_memmap(MG["it"], mode='w+', dtype=np.float32, shape=(NM, P))
    for j, (a, b) in enumerate(zip(M_mz, M_it)):
        mm[j, :len(a)] = a; mi[j, :len(b)] = b
    mm.flush(); mi.flush(); del mm, mi
    np.save(MG["npk"],  np.array(M_np, np.int16))
    np.save(MG["prec"], np.array(M_pr, np.float32))
    np.save(MG["ad"],   np.array(M_ad, np.int8))
    np.save(MG["ins"],  np.array(M_in, np.int8))
    np.save(MG["ce"],   np.full(NM, 25.0, np.float32))   # matches model_logits()'s hard-coded 25.0
    np.save(MG["mode"], np.array(M_md, np.float32))
    np.save(MG["yrow"], np.array(M_y, np.int32))
    log(f"merged set: {NM:,} pseudo-spectra")

mg_yrow = np.load(MG["yrow"])
mg_es    = struct_es[mg_yrow]
MG_TRAIN_IDX = np.flatnonzero(~mg_es)
MG_ES_IDX    = np.flatnonzero(mg_es)
log(f"merged: train {len(MG_TRAIN_IDX):,} | early-stop {len(MG_ES_IDX):,}")

# %%
#@title Stage A7 — mirror Stage A to Drive (resume insurance)
import shutil
(W/"pack").mkdir(parents=True, exist_ok=True)
for p in sorted(S.glob("*.np*")):
    dst = W/"pack"/p.name
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        log(f"  -> {p.name} ({p.stat().st_size/1e6:.0f} MB)")
        shutil.copy2(p, dst)
log("Stage A mirrored to Drive. If the session drops, copy this back and skip to Stage B.")

# %% [markdown]
# ---
# # Stage B — train
#
# Architecture **unchanged** from the inference notebook: `load_neural_models()` reads `d`,
# `layers` and `nbits` straight from the checkpoint, so an identical module is a guaranteed
# drop-in and the compute goes into steps and objective instead of debugging a new block.

# %%
#@title Stage B0 — model (verbatim FPNet)
import torch, torch.nn as nn, torch.nn.functional as F

class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))

class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d); self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(2048, nbits))
    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec),
                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev)
if dev == 'cpu':
    print("!! Stage B on CPU is not viable. Switch the runtime to an A100 and re-run from Stage A7.")

# %%
#@title Stage B0b — in-RAM arrays + batch builder
def load_set(pmap):
    return dict(
        mz=np.load(pmap["mz"], mmap_mode='r'), it=np.load(pmap["it"], mmap_mode='r'),
        npk=np.load(pmap["npk"]), prec=np.load(pmap["prec"]), ad=np.load(pmap["ad"]),
        ins=np.load(pmap["ins"]), ce=np.load(pmap["ce"]), mode=np.load(pmap["mode"]),
        yrow=np.load(pmap["yrow"]))

DS_SINGLE = load_set(paths)
DS_MERGED = load_set(MG)
Yf = Y  # uint8 [n_struct, NBITS]

def build_batch(ds, idx, augment, rng):
    n = len(idx)
    npk = ds["npk"][idx].astype(np.int64)
    N = max(int(npk.max()), 1)
    mz = np.asarray(ds["mz"][idx][:, :N], np.float32).copy()
    it = np.asarray(ds["it"][idx][:, :N], np.float32).copy()
    pad = np.arange(N)[None, :] >= npk[:, None]

    prec = ds["prec"][idx].astype(np.float32)
    ad   = ds["ad"][idx].astype(np.int64)
    ins  = ds["ins"][idx].astype(np.int64)
    ce   = ds["ce"][idx].astype(np.float32)
    ce   = np.where(np.isfinite(ce), ce, 25.0)
    mode = ds["mode"][idx].astype(np.float32)

    if augment:
        # peak dropout — test spectra are sparser and ~44% lack the precursor peak
        frac = rng.uniform(0.0, CFG.AUG_PEAK_DROP, size=(n, 1))
        drop = (rng.random((n, N)) < frac) & (~pad)
        keep_one = drop.all(axis=1) & (npk > 0)
        drop[keep_one, 0] = False
        pad = pad | drop
        # m/z jitter — libraries differ in calibration; riken reports m/z to 2 dp
        mz *= (1.0 + rng.normal(0.0, CFG.AUG_MZ_PPM * 1e-6, size=mz.shape)).astype(np.float32)
        # intensity jitter — collision-energy / detector-response variation
        it *= np.exp(rng.normal(0.0, CFG.AUG_INT_SIGMA, size=it.shape)).astype(np.float32)
        np.clip(it, 0.0, 4.0, out=it)
        # conditioning dropout — CE is missing for much of train but always eV in test
        cd = rng.random(n) < CFG.AUG_COND_DROP
        ad[cd] = UNK_AD; ins[cd] = OTHER_INSTR; ce[cd] = 25.0

    mz[pad] = 0.0; it[pad] = 0.0
    T = lambda x, d=None: torch.as_tensor(x, device=dev) if d is None else torch.as_tensor(x, dtype=d, device=dev)
    return (T(mz), T(it), T(pad), T(prec), T(ad), T(ins), T(ce), T(mode)), ds["yrow"][idx]

def bce_eval(net, ds, idx, bs=1024, cap=60_000):
    net.eval(); tot, n = 0.0, 0
    sub = idx[:cap]
    rng = np.random.default_rng(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev=='cuda')):
        for i in range(0, len(sub), bs):
            b = sub[i:i+bs]
            args, yr = build_batch(ds, b, False, rng)
            yt = torch.as_tensor(Yf[yr].astype(np.float32), device=dev)
            tot += F.binary_cross_entropy_with_logits(net(*args).float(), yt).item() * len(b)
            n += len(b)
    net.train(); return tot / max(n, 1)
print("batch builder ready")

# %%
#@title Stage B0c — listwise decoy sampler (the retrieval-aligned objective)
POOL_FP_T = torch.from_numpy(np.ascontiguousarray(pool_fp))   # packed uint8, CPU
TAU = math.sqrt(NBITS)

def sample_candidates(yr, rng, k=CFG.N_CAND):
    """For each structure row, return [truth, 47 decoys] drawn from its real +-8.5 ppm window."""
    out = np.empty((len(yr), k), np.int64)
    ok  = np.ones(len(yr), bool)
    for i, s in enumerate(yr):
        t = win_pos[s]
        a, b = win_off[s], win_off[s+1]
        if t < 0 or b - a < 2:
            ok[i] = False; out[i] = t if t >= 0 else 0; continue
        w = win_idx[a:b]
        w = w[w != t]
        if len(w) >= k - 1:
            d = rng.choice(w, size=k - 1, replace=False)
        else:
            d = np.concatenate([w, rng.choice(w, size=k - 1 - len(w), replace=True)]) if len(w) \
                else np.full(k - 1, t)
        out[i, 0] = t; out[i, 1:] = d
    return out, ok

def cand_bits(cand_idx):
    flat = cand_idx.reshape(-1)
    bits = np.unpackbits(pool_fp[flat], axis=1)[:, :NBITS]
    return torch.as_tensor(bits.reshape(*cand_idx.shape, NBITS), device=dev)

def listwise_loss(z, yr, rng):
    """CE over f.z for the true candidate against its own mass-window decoys.

    f.z is *literally* the inference score, and raw / _z(raw) / _rank_norm(raw) /
    (raw - raw.max()) are all order-preserving transforms of it, so improving this ordering
    improves four of the six model features the GBDT consumes.
    """
    cand, ok = sample_candidates(yr, rng)
    if not ok.any(): return None
    cb = cand_bits(cand[ok]).to(z.dtype)
    logits = torch.einsum('bkd,bd->bk', cb, z[torch.as_tensor(np.flatnonzero(ok), device=dev)]) / TAU
    tgt = torch.zeros(logits.shape[0], dtype=torch.long, device=dev)
    return F.cross_entropy(logits, tgt)

@torch.no_grad()
def retrieval_eval(net, ds, idx, cap=4000, k=CFG.N_CAND):
    """Top-1 and MRR of f.z against real mass-window decoys. The number that matters."""
    net.eval(); rng = np.random.default_rng(3)
    rr, top1, n = 0.0, 0, 0
    for i in range(0, min(len(idx), cap), 256):
        b = idx[i:i+256]
        args, yr = build_batch(ds, b, False, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev=='cuda')):
            z = net(*args).float()
        cand, ok = sample_candidates(yr, rng, k)
        if not ok.any(): continue
        sel = torch.as_tensor(np.flatnonzero(ok), device=dev)
        cb = cand_bits(cand[ok]).float()
        sc = torch.einsum('bkd,bd->bk', cb, z[sel])
        rank = (sc > sc[:, :1]).sum(1) + 1
        rr += (1.0 / rank.float()).sum().item(); top1 += (rank == 1).sum().item(); n += len(rank)
    net.train()
    return (rr / max(n, 1)), (top1 / max(n, 1)), n

# %%
#@title Stage B1+B2 — training driver
def train_one(tag, seed, ds, train_idx, es_idx, epochs_bce=CFG.EPOCHS_BCE,
              epochs_rank=CFG.EPOCHS_RANK, out_dir=W/"ckpt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    net = FPNet(NBITS, d=CFG.D, layers=CFG.LAYERS, heads=CFG.HEADS, drop=CFG.DROP).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    log(f"=== {tag} (seed {seed}) — {nparam/1e6:.1f}M params, {len(train_idx):,} train samples")

    spe   = max(1, len(train_idx) // CFG.BS)
    steps = epochs_bce * spe
    opt   = torch.optim.AdamW(net.parameters(), lr=CFG.LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.LR, total_steps=steps, pct_start=0.1)
    best, bad, gstep = math.inf, 0, 0
    ckpt = out_dir / f"{tag}.pt"

    def save(step):
        torch.save(dict(model=net.state_dict(), nbits=NBITS, d=CFG.D, layers=CFG.LAYERS,
                        step=step, fp_bits_sha1=FP_SHA1, tag=tag, seed=seed), ckpt)

    # ---- B1: BCE. Keeps f.z a calibrated log-odds dot product, which the f.z ranking
    #          formulation and the GBDT's _z(raw)/_rank_norm(raw) features both assume.
    for ep in range(epochs_bce):
        perm = rng.permutation(len(train_idx)); run, nb = 0.0, 0
        for i in range(0, len(train_idx) - CFG.BS + 1, CFG.BS):
            b = train_idx[perm[i:i+CFG.BS]]
            args, yr = build_batch(ds, b, True, rng)
            yt = torch.as_tensor(Yf[yr].astype(np.float32), device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev=='cuda')):
                loss = F.binary_cross_entropy_with_logits(net(*args).float(), yt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1: sched.step()
            run += loss.item(); nb += 1; gstep += 1
        v = bce_eval(net, ds, es_idx)
        mrr, t1, n = retrieval_eval(net, ds, es_idx)
        flag = ""
        if v < best - 1e-5:
            best, bad = v, 0; save(gstep); flag = " *"
        else:
            bad += 1
        log(f"  [B1 {tag}] ep {ep:02d}  train {run/max(nb,1):.5f}  es_bce {v:.5f}  "
            f"es_MRR {mrr:.4f} top1 {t1:.4f} (n={n}){flag}")
        if bad >= 4: log("  early stop"); break

    net.load_state_dict(torch.load(ckpt, map_location=dev)['model'])
    mrr0, t10, _ = retrieval_eval(net, ds, es_idx)
    log(f"  [B1 {tag}] done. es_MRR {mrr0:.4f} top1 {t10:.4f}")

    # ---- B2: listwise retrieval fine-tune against the real +-8.5 ppm decoys.
    if epochs_rank > 0:
        opt2 = torch.optim.AdamW(net.parameters(), lr=CFG.LR_RANK, weight_decay=1e-2)
        best_mrr = mrr0
        rk = out_dir / f"{tag}_rank.pt"
        save_rank = lambda s: torch.save(
            dict(model=net.state_dict(), nbits=NBITS, d=CFG.D, layers=CFG.LAYERS, step=s,
                 fp_bits_sha1=FP_SHA1, tag=tag + "_rank", seed=seed), rk)
        save_rank(gstep)
        for ep in range(epochs_rank):
            perm = rng.permutation(len(train_idx)); run, nb = 0.0, 0
            for i in range(0, len(train_idx) - CFG.RANK_BS + 1, CFG.RANK_BS):
                b = train_idx[perm[i:i+CFG.RANK_BS]]
                args, yr = build_batch(ds, b, True, rng)
                yt = torch.as_tensor(Yf[yr].astype(np.float32), device=dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev=='cuda')):
                    z = net(*args).float()
                lb = F.binary_cross_entropy_with_logits(z, yt)
                lr_ = listwise_loss(z, yr, rng)
                loss = lb * CFG.LAMBDA_BCE if lr_ is None else lr_ + CFG.LAMBDA_BCE * lb
                opt2.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt2.step()
                run += float(loss); nb += 1; gstep += 1
                if nb % 400 == 0:
                    log(f"    [B2 {tag}] ep{ep} step {nb}  loss {run/nb:.4f}")
            mrr, t1, n = retrieval_eval(net, ds, es_idx)
            flag = ""
            if mrr > best_mrr:
                best_mrr = mrr; save_rank(gstep); flag = " *"
            log(f"  [B2 {tag}] ep {ep:02d}  loss {run/max(nb,1):.4f}  es_MRR {mrr:.4f} "
                f"top1 {t1:.4f} (n={n}){flag}")
        # keep the fine-tuned weights only if they actually improved retrieval
        if best_mrr > mrr0:
            net.load_state_dict(torch.load(rk, map_location=dev)['model'])
            log(f"  [{tag}] listwise stage kept: MRR {mrr0:.4f} -> {best_mrr:.4f}")
            save(gstep)
        else:
            log(f"  [{tag}] listwise stage DISCARDED (MRR {mrr0:.4f} -> {best_mrr:.4f}); keeping B1")
    return ckpt, best

# %%
#@title Stage B — run (single variants, then merged). ~8 h on an A100.
RESULTS = {}
for sd in CFG.SEEDS_SINGLE:
    RESULTS[f"fp_single_s{sd}"] = train_one(f"fp_single_s{sd}", sd,
                                            DS_SINGLE, TRAIN_IDX, ES_IDX)

for sd in CFG.SEEDS_MERGED:
    RESULTS[f"fp_merged_m{sd}"] = train_one(f"fp_merged_m{sd}", 100 + sd,
                                            DS_MERGED, MG_TRAIN_IDX, MG_ES_IDX)

log("all checkpoints trained")
for k, (p, v) in RESULTS.items():
    print(f"  {k:18s} es_bce {v:.5f}  {p}")

# %%
#@title Stage B3 — export + verify the drop-in contract
EXPORT = W / "casmi26-fpnet-v3"
EXPORT.mkdir(parents=True, exist_ok=True)
import shutil

for name in list(CFG.SEEDS_SINGLE and [f"fp_single_s{s}" for s in CFG.SEEDS_SINGLE]) + \
            [f"fp_merged_m{s}" for s in CFG.SEEDS_MERGED]:
    src = W/"ckpt"/f"{name}.pt"
    shutil.copy2(src, EXPORT/f"{name}.pt")

(EXPORT/"fp_bits_sha1.txt").write_text(FP_SHA1)

# replay load_neural_models() exactly as the Kaggle notebook will
import glob as _glob
paths_pt = sorted(_glob.glob(str(EXPORT/"fp_*.pt")))
single, merged = [], []
for pth in paths_pt:
    ck = torch.load(pth, map_location='cpu', weights_only=False)
    assert ck['nbits'] == NBITS, f"{pth}: nbits mismatch"
    assert ck['fp_bits_sha1'] == FP_SHA1, f"{pth}: fp_bits layout mismatch"
    net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'])
    net.load_state_dict(ck['model'])
    (merged if 'merged' in os.path.basename(pth) else single).append(pth)
    print(f"  ok {os.path.basename(pth)}: d={ck['d']} layers={ck['layers']} step={ck['step']}")

assert len(single) >= 1 and len(merged) >= 1, \
    "load_neural_models() routes on 'merged' in the basename — need at least one of each"
print(f"\nrouting check: {len(single)} single + {len(merged)} merged")
print(f"sha1: {FP_SHA1}")
print(f"\nEXPORT -> {EXPORT}")
print("Download this folder and upload it as the Kaggle dataset 'casmi26-fpnet-v3',")
print("then run the Stage C notebook to regenerate rank_train.npz AGAINST THESE CHECKPOINTS.")
print("\nDo NOT submit with these checkpoints and the old rank_train.npz — 12 of the 31")
print("ranker features are functions of these logits and the old GBDTs are calibrated to")
print("a different model's distribution.")
