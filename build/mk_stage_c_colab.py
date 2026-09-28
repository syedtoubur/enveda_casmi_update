"""Build the Colab version of Stage C v2 (single notebook, no Kaggle sharding).

Same logic as the Kaggle build, three environment changes:
  * paths resolve under Google Drive instead of /kaggle/input
  * one process, N_SHARDS=1, checkpointed to Drive so a disconnect costs minutes
  * worker pools sized from os.cpu_count() instead of a hard-coded 4

Plus the thing the user actually asked for: real pre-flight verification and a single
paste-back report block at the end.

phase1 cells 1-8 remain verbatim except for cells 2 and 7, where the literal string
'/kaggle/input' is substituted for the Drive root. That substitution is printed at runtime so
it stays auditable.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "enveda-casmi-2026-phase1-enhanced-baseline.ipynb"
OUT = HERE.parent / "casmi_rank_matrix_colab.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
P = ["".join(c["source"]) for c in nb["cells"]]

split = P[8].index("# Fit Calibrated Rankers")
RANK_FEATURES = P[8][:split].rstrip()
assert "def rank_features" in RANK_FEATURES and "rank_train.npz" not in RANK_FEATURES

DRIVE = "/content/casmi_stage"          # local mirror of the Drive inputs
# cells 2 and 7 are the only ones that hard-code the Kaggle input root
assert P[2].count("/kaggle/input") == 3 and P[7].count("/kaggle/input") == 1
assert all("/kaggle" not in P[i] for i in (1, 3, 4, 5, 6)), "unexpected Kaggle path"
P2 = P[2].replace("/kaggle/input", DRIVE)
P7 = P[7].replace("/kaggle/input", DRIVE)

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(True)})


md("""# CASMI 2026 — Stage C v2 (Colab)

Rebuilds `rank_train.npz` against your new FPNet, without v1's leak.

## Which runtime

Stage C is **CPU-bound, not GPU-bound**. The GPU runs `model_logits` on a handful of spectra
per query — microseconds. The wall clock goes to numba entropy search over 2.54M library
spectra (Channels 1 and 2) and RDKit bond-breaking (Channel 3), both of which scale with
**vCPU count**.

| runtime | vCPU | RAM | units/hr | cores per unit |
|---|---|---|---|---|
| T4 + High-RAM | 4 | 51 GB | ~2.0 | 2.0 |
| **L4 + High-RAM** | **8** | **53 GB** | **~4.8** | **1.7** |
| A100 | 12 | 83 GB | ~13 | 0.9 |

**Use L4 with High-RAM on.** It is ~2× faster than T4 at almost the same cost-efficiency, and
A100 costs ~3× an L4 to give you 1.5× the cores on work the GPU barely touches. High-RAM is not
optional — `load_library` holds all 2.54M peak lists in memory.

> Runtime → Change runtime type → **L4 GPU**, and Runtime shape → **High-RAM**.

## Run order

Cells top to bottom. Two checkpoints matter:

* **Cell "VERIFY 1 — pre-flight"** runs a 3-query smoke test and prints a measured ETA. If the
  ETA is too long, lower `SC.N_SEEN` / `SC.N_UNSEEN` there and re-run — nothing after it has
  started yet.
* **Cell "VERIFY 2 — report"** prints one paste-back block. Send me that block.

The generation loop checkpoints to Drive every 50 queries and resumes automatically, so a
dropped session costs minutes.

## What v2 fixes

v1 sampled ranker queries from `train.parquet` excluding only the *validation* structures — but
the FPNet had trained on everything else, so every query was memorised and `f·z` was a
near-oracle (truth top-1 in **90.2%** of windows vs **42.0%** in the forked matrix). The GBDT
leaned **60.2%** of its skill on the model channel and reduced library to 2.7%, analog to 0.2%,
MetFrag to 0.0%. On the leaderboard, where the model has not seen the Class-2/3 structures,
there was no fallback. 0.337 → 0.287.

v2 draws queries from **both** model-strength regimes, scores each under **both** ban settings,
keeps **full** candidate windows, and refuses to write the matrix if the channel balance looks
like v1 again.""")

code('''#@title Colab setup — mount Drive, check the runtime, stage inputs locally
import os, shutil, subprocess, sys, time, glob as _glob
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/casmi_2026"
STAGE      = "''' + DRIVE + '''"     # local mirror; Drive I/O is far too slow for a 2.5M-row read

from google.colab import drive
drive.mount("/content/drive")

# ---- runtime sanity ---------------------------------------------------------------
import multiprocessing, psutil, torch
NCPU = multiprocessing.cpu_count()
RAM  = psutil.virtual_memory().total / 1e9
print(f"vCPU      : {NCPU}")
print(f"RAM       : {RAM:.0f} GB")
print(f"GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"free disk : {shutil.disk_usage('/content').free/1e9:.0f} GB")

if RAM < 40:
    raise SystemExit(
        f"Only {RAM:.0f} GB RAM. load_library holds all 2.54M peak lists in memory.\\n"
        "Runtime -> Change runtime type -> Runtime shape: High-RAM, then re-run.")
if NCPU <= 2:
    print("\\n!! 2 vCPU. Stage C is CPU-bound; this will take many hours.")
    print("   Switch to an L4 + High-RAM runtime (8 vCPU) before continuing.")

# ---- stage the inputs onto local disk ----------------------------------------------
NEED = {
    "enveda-CASMI26-molecule-id-mass-spectra": ["train.parquet", "test.parquet",
                                                "sample_submission.csv"],
    "coconut_dataset":        ["fp_bits.npy", "coco_fp.npy", "coco_mass.npy", "coco_meta.pkl"],
    "casmi26-fpnet-v3":       None,     # whole folder: checkpoints + fpnet_split_keys.npz
    "val_scripts":            ["val_key.pkl"],
    "casmi26-offline-rdkit-2026033": None,
}
os.makedirs(STAGE, exist_ok=True)
t0 = time.time()
for sub, files in NEED.items():
    src = Path(DRIVE_ROOT) / sub
    if not src.exists():
        raise SystemExit(f"missing on Drive: {src}")
    dst = Path(STAGE) / sub; dst.mkdir(parents=True, exist_ok=True)
    for f in (files if files is not None else [p.name for p in src.iterdir() if p.is_file()]):
        s, d = src / f, dst / f
        if not s.exists():
            raise SystemExit(f"missing on Drive: {s}")
        if d.exists() and d.stat().st_size == s.stat().st_size:
            print(f"  cached  {sub}/{f}  ({s.stat().st_size/1e6:.0f} MB)"); continue
        print(f"  copying {sub}/{f}  ({s.stat().st_size/1e6:.0f} MB) ...", flush=True)
        shutil.copy2(s, d)
print(f"\\ninputs staged in {time.time()-t0:.0f}s -> {STAGE}")

# results and resume checkpoints live on Drive so a disconnect costs minutes, not hours
WORK = Path(DRIVE_ROOT) / "_stagec"
WORK.mkdir(parents=True, exist_ok=True)
print(f"outputs  -> {WORK}")''')

md(f"""## phase1 cells 1–8 — verbatim

Cells **2** and **7** have one substitution: the literal `/kaggle/input` becomes `{DRIVE}`
(3 occurrences in cell 2, 1 in cell 7). Everything else — `CFG`, the numba kernels, the adduct
physics, `build_candidate_pool`, the evidence channels, `FPNet`/`model_logits` and
`rank_features` — is byte-identical to `enveda-casmi-2026-phase1-enhanced-baseline.ipynb`.

That matters: the feature matrix the GBDT trains on must be produced by the same code that
produces it at inference.""")

code(P[1])
code("# NOTE: '/kaggle/input' -> '" + DRIVE + "' (3 occurrences). Otherwise verbatim.\n" + P2)
code(P[3])
code(P[4])
code(P[5])
code(P[6])
code("# NOTE: '/kaggle/input' -> '" + DRIVE + "' (1 occurrence). Otherwise verbatim.\n" + P7)
code(RANK_FEATURES)

code('''#@title Stage-C v2 config
class SC:
    SHARD     = 0
    N_SHARDS  = 1          # single Colab process

    # Query structures per model-strength regime. Each is scored under TWO ban settings,
    # so this is ~2,800 row-groups -- comfortably more than the forked matrix's 1,638.
    # The pre-flight cell measures the real rate; lower these there if the ETA is too long.
    N_SEEN    = 700        # FPNet trained on these -> cells A (M=0) and C (M=1)
    N_UNSEEN  = 700        # FPNet never saw these  -> cells D (M=0) and B (M=1)

    NP_LIBS = ['riken','mona','msdial','massbank','gnps','enveda-np-examples','masaryk']
    MASS_LO, MASS_HI = 157.0, 1159.0
    MIN_PEAKS = 5

    # A query builds its mass window from its OWN reported precursor m/z, but both search
    # structures are indexed on the label's exact mass (CFG.STRUCT_MASS for the library,
    # ExactMolWt for the pool). A query whose reported m/z cannot place its own label mass
    # inside the window can never retrieve itself -- it is guaranteed lv=0 and, worse, a
    # guaranteed all-negative row group. Keep this equal to CFG.PPM_WIN.
    QUERY_PPM_MAX = CFG.PPM_WIN

    # Full windows. The cap only guards pathological cases and sits above the forked
    # matrix's p90 (344). When it fires it keeps every positive plus a RANDOM negative
    # sample -- never hard-mined; censoring easy negatives was part of what broke v1.
    MAX_ROWS_PER_QUERY = 400

    SEED = 20260923
    OUT  = str(WORK)
    WORKERS = max(2, min(NCPU, 8))

NFEAT = N_FEAT   # 31 -- the natural width of rank_features
print(f"NFEAT={NFEAT} | workers={SC.WORKERS} | queries={SC.N_SEEN + SC.N_UNSEEN:,} "
      f"-> {2*(SC.N_SEEN + SC.N_UNSEEN):,} row-groups")
print(f"outputs -> {SC.OUT}")''')

code('''#@title Verify the fp_bits contract, then load YOUR checkpoints
import hashlib
FP_SHA1 = hashlib.sha1(np.ascontiguousarray(BITS).tobytes()).hexdigest()
print(f"fp_bits sha1: {FP_SHA1}")

found = sorted(glob.glob(f'{STAGE}/**/fp_*.pt', recursive=True))
print("checkpoints visible to load_neural_models():")
for p in found: print("   ", os.path.basename(p))
assert found, "no fp_*.pt found - is casmi26-fpnet-v3 on Drive?"

for p in found:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    got = ck.get('fp_bits_sha1')
    assert got is not None, (
        f"{os.path.basename(p)} carries no fp_bits_sha1 - that is an OLD checkpoint. "
        "Only casmi26-fpnet-v3 may be staged.")
    assert got == FP_SHA1, (
        f"{os.path.basename(p)} was trained against a different fp_bits layout "
        f"({got[:12]}... vs {FP_SHA1[:12]}...). f.z would be noise.")
print("all checkpoints agree on the fp_bits layout")

load_neural_models()
assert _MODEL is not None, "FPNet failed to load"
single, merged, dev_, nb_ = _MODEL
assert nb_ == len(BITS), f"checkpoint nbits {nb_} != {len(BITS)}"
assert len(single) >= 1 and len(merged) >= 1, "need at least one single and one merged model"
print(f"ready: {len(single)} single + {len(merged)} merged on {dev_}")''')

code('''#@title Library, pool, representative analog set (same construction as inference)
t0 = time.time()
pool = build_candidate_pool()
L = load_library(TRAIN)
rep, rep_key, rep_nm, rep_pos = build_rep(L)
print(f"[INFO] pool {len(pool.mass):,} | library {len(L['off'])-1:,} spectra | rep {len(rep):,}")
print(f"[INFO] engine ready in {time.time()-t0:.0f}s")''')

code('''#@title Ban-aware channels + a persistent MetFrag pool
def _drop(cand, ban_rows, ban_keys, ik_arr):
    if cand.size == 0 or (not len(ban_rows) and not len(ban_keys)): return cand
    m = np.ones(len(cand), bool)
    if len(ban_rows): m &= ~np.isin(cand, ban_rows)
    if len(ban_keys): m &= ~np.isin(ik_arr[cand], ban_keys)
    return cand[m]

def lib_sim_ban(L, specs, target, ban_rows=(), ban_keys=()):
    cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6)
    cand = _drop(cand, np.asarray(ban_rows, np.int64),
                 np.asarray(ban_keys, dtype=object), L['ik'])
    if len(cand) == 0: return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it)
        if len(qm) == 0: continue
        sc = search(qm, qp, cand, L['off'], L['mz'], L['it'],
                    CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return agg

def analog_sim_ban(L, specs, target, rep, rep_key, rep_nm, ban_rows=(), ban_keys=()):
    lo = np.searchsorted(rep_nm, target - CFG.ANALOG_WIN, 'left')
    hi = np.searchsorted(rep_nm, target + CFG.ANALOG_WIN, 'right')
    if hi <= lo: return []
    cand  = rep[lo:hi]
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey  = rep_key[lo:hi]
    if len(ban_rows) or len(ban_keys):
        m = np.ones(len(cand), bool)
        if len(ban_rows): m &= ~np.isin(cand, np.asarray(ban_rows, np.int64))
        if len(ban_keys): m &= ~np.isin(ckey, np.asarray(ban_keys, dtype=object))
        cand, shift, ckey = cand[m], shift[m], ckey[m]
    if len(cand) == 0: return []
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it)
        if len(qm) == 0: continue
        sc = search_shift(qm, qp, cand, L['off'], L['mz'], L['it'],
                          CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                          CFG.INT_POWER, CFG.ENT_WEIGHT, shift)
        for c, k, s in zip(cand, ckey, sc):
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return sorted(agg.items(), key=lambda x: -x[1])[:CFG.N_ANALOG]

_MP = None
def frag_scores_fast(cand_smiles, specs, mode):
    """phase1's frag_scores with one persistent pool instead of a fork per molecule.
    Numerically identical (fragment_masses is a pure function of the SMILES); the pre-flight
    cell asserts equality against frag_scores on a real query before anything relies on it."""
    global _MP
    if not HAVE_RDKIT: return np.zeros(len(cand_smiles), np.float32)
    if _MP is None: _MP = MPool(SC.WORKERS)
    frags = _MP.map(_frag_masses_wrapper, cand_smiles, chunksize=8)
    peaks = []
    for mz, it in specs:
        m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=CFG.MZ_TOL) for a, b in peaks),
                     default=0.0)
    return out

print(f"ban-aware channels ready ({SC.WORKERS} MetFrag workers)")''')

code('''#@title Query cohort — BOTH model-strength regimes, disjoint from the validation set
MDCOLS = ['inchikey14','normalized_smiles','ingest_lib','adduct','ionization_mode',
          'instrument_type','precursor_mz','num_peaks','collision_energy_ev',
          'molecular_formula']
md = pq.read_table(TRAIN, columns=MDCOLS).to_pandas()
md['row'] = np.arange(len(md), dtype=np.int64)
md['nm']  = neutral_mass(md.precursor_mz.values.astype(np.float64), md.adduct.values)

# --- the query must be able to RETRIEVE ITSELF -------------------------------------
# Both search structures are indexed on the LABEL's exact mass -- the library through
# CFG.STRUCT_MASS, the candidate pool through ExactMolWt(normalized_smiles). A query's
# target, exactly as at test time, comes from its own reported precursor m/z. Several NP
# source libraries report m/z to 2 decimal places (riken above all: CFG.STRUCT_MASS's own
# note says 30% of its 341k spectra miss a +-8.5 ppm window for that reason alone). At
# 400 Da, 0.005 Da is 12 ppm -- such a query sits outside its own window and cannot see
# its own structure in EITHER channel: lv is 0 by construction and the row group has no
# positive at all. The test set is high-precision timsTOF and contains no such rows, so
# these are not a regime worth training on; they are rows that teach the ranker the
# library channel is dead. That is the v1 failure mode wearing a different hat.
_fm = {f: formula_mass(f) for f in set(md.molecular_formula.dropna().tolist())}
md['nm_exact'] = [_fm.get(f, np.nan) for f in md.molecular_formula]
md['ppm_off']  = ((md.nm.values / (1.0 + CFG.PPM_CENTER / 1e6) - md.nm_exact.values)
                  / md.nm_exact.values * 1e6)
mass_ok = np.isfinite(md.ppm_off.values) & (np.abs(md.ppm_off.values) <= SC.QUERY_PPM_MAX)
_isnp = md.ingest_lib.isin(SC.NP_LIBS).values
print(f"[INFO] self-retrievable within +-{SC.QUERY_PPM_MAX} ppm: "
      f"{mass_ok[_isnp].sum():,}/{_isnp.sum():,} NP-library spectra")
for _lib in SC.NP_LIBS:
    _m = _isnp & (md.ingest_lib.values == _lib)
    if _m.sum():
        print(f"       {_lib:22s} {mass_ok[_m].mean()*100:5.1f}% usable   "
              f"(median |off| {np.nanmedian(np.abs(md.ppm_off.values[_m])):5.2f} ppm, "
              f"{_m.sum():,} spectra)")

SPLIT = np.load(find_file('fpnet_split_keys.npz'))
SEEN_K   = set(SPLIT['seen_keys'].tolist())     # FPNet gradient-updated on these
ES_K     = set(SPLIT['es_keys'].tolist())       # scaffold-disjoint holdout: never trained
BANNED_K = set(SPLIT['banned_keys'].tolist())   # val ban_keys: Stage D only
print(f"[INFO] FPNet split: seen {len(SEEN_K):,} | unseen(es) {len(ES_K):,} | "
      f"val-reserved {len(BANNED_K):,}")

val_key = pickle.load(open(find_file('val_key.pkl'), 'rb'))
VAL_RESERVED = set()
for r in val_key:
    VAL_RESERVED.add(r['inchikey14'])
    VAL_RESERVED |= set(r.get('ban_keys') or [])
VAL_QUERY = {r['inchikey14'] for r in val_key}
print(f"[INFO] validation reserves {len(VAL_RESERVED):,} structures "
      f"({len(VAL_QUERY):,} of them are Stage-D queries)")

TEST_ADDUCTS = ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
                "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"]
elig_base = (md.ingest_lib.isin(SC.NP_LIBS).values
             & md.adduct.isin(TEST_ADDUCTS).values
             & np.isfinite(md.nm.values)
             & (md.nm.values >= SC.MASS_LO) & (md.nm.values <= SC.MASS_HI)
             & (md.num_peaks.values >= SC.MIN_PEAKS)
             & mass_ok
             & md.inchikey14.notna().values
             & md.normalized_smiles.notna().values)

def ce_tag(v):
    try:
        a = np.atleast_1d(v)
        return float(np.nanmean(a.astype(float))) if len(a) and not pd.isna(a).all() else np.nan
    except Exception:
        return np.nan

def build_E(mask):
    e = md[mask].copy()
    e['ce'] = [ce_tag(v) for v in e.collision_energy_ev]
    e['ce_known'] = e.ce.notna()
    return e

E = build_E(elig_base & ~md.inchikey14.isin(VAL_RESERVED).values)
print(f"[INFO] eligible query spectra: {len(E):,} / {E.inchikey14.nunique():,} structures")

elig_keys   = set(E.inchikey14)
seen_pool   = np.array(sorted(elig_keys & SEEN_K))
unseen_pool = np.array(sorted(elig_keys & ES_K))
print(f"[INFO] eligible & FPNet-seen  : {len(seen_pool):,}")
print(f"[INFO] eligible & FPNet-unseen: {len(unseen_pool):,}   <- the Class-2 regime")

USED_FALLBACK = False
if len(unseen_pool) < SC.N_UNSEEN:
    USED_FALLBACK = True
    print(f"[WARN] only {len(unseen_pool):,} FPNet-unseen structures survive the filters, "
          f"need {SC.N_UNSEEN:,}.")
    print( "       Widening to val-banned SCAFFOLD RELATIVES (never the val queries themselves).")
    print( "       NOTE: this makes Stage D's C2_hard cohort mildly optimistic.")
    E = build_E(elig_base & ~md.inchikey14.isin(VAL_QUERY).values)
    elig_keys = set(E.inchikey14)
    unseen_pool = np.array(sorted(elig_keys & (ES_K | (BANNED_K - VAL_QUERY))))
    seen_pool   = np.array(sorted(elig_keys & SEEN_K))
    print(f"       unseen pool now {len(unseen_pool):,}")
    assert not (set(unseen_pool) & VAL_QUERY), "a Stage-D query leaked into the Stage-C cohort"

assert len(unseen_pool) > 0, "no FPNet-unseen structures available - Stage C cannot fix v1"
assert not (set(seen_pool) & set(unseen_pool)), "seen/unseen cohorts overlap"

rng = np.random.default_rng(SC.SEED)
q_seen   = rng.choice(seen_pool,   size=min(SC.N_SEEN,   len(seen_pool)),   replace=False)
q_unseen = rng.choice(unseen_pool, size=min(SC.N_UNSEEN, len(unseen_pool)), replace=False)
QUERIES = [(k, 'seen') for k in q_seen] + [(k, 'unseen') for k in q_unseen]
rng.shuffle(QUERIES)
QUERIES = QUERIES[SC.SHARD::SC.N_SHARDS]
print(f"\\n[INFO] {len(QUERIES):,} query structures "
      f"({sum(1 for _,t in QUERIES if t=='seen')} seen / "
      f"{sum(1 for _,t in QUERIES if t=='unseen')} unseen), each under 2 ban regimes")

SPEC_N = pq.read_table(TEST, columns=['molecule_id']).to_pandas().groupby('molecule_id').size().values''')

code('''#@title Assemble each query: pick spectra like the test set, compute both ban sets
rows_of_key = md.dropna(subset=['inchikey14']).groupby('inchikey14').row.apply(list).to_dict()
Eg = {k: g for k, g in E.groupby('inchikey14')}

records = []
for ik, regime in QUERIES:
    g = Eg.get(ik)
    if g is None or len(g) == 0: continue
    src = g.ingest_lib.value_counts().idxmax()
    g = g[g.ingest_lib == src]
    if len(g) == 0: continue

    # Prefer CE-known rows: the test set always reports collision energy in eV, so a query
    # built from a CE-null row would exercise FPNet's 25.0 eV fallback, not the real path.
    gk = g[g.ce_known]
    gp = gk if len(gk) else g

    n = int(min(len(gp), rng.choice(SPEC_N)))
    pick, seen = [], set()
    for r in gp.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).itertuples():
        ce_key = round(r.ce, 1) if not np.isnan(r.ce) else f"unk_{r.row}"
        t = (r.adduct, ce_key)
        if t not in seen:
            seen.add(t); pick.append(r.row)
        if len(pick) == n: break
    if not pick: continue

    # M=0 "library visible": ban ONLY the query's own spectra, so it cannot match itself but
    #     other measurements of the compound stay findable. That is the real Class-1 condition
    #     -- a fresh timsTOF run of a compound that public libraries already contain.
    # M=1 "library banned": every spectrum of the structure is hidden and the key is banned
    #     from the analog channel too. That is Class 2.
    records.append(dict(ik=ik, regime=regime, rows=pick,
                        ban_vis=sorted(pick), ban_all=sorted(rows_of_key.get(ik, []))))

print(f"[INFO] {len(records):,} query molecules -> {2*len(records):,} row-groups, "
      f"{sum(len(r['rows']) for r in records):,} spectra")

sel_rows = np.array(sorted({x for r in records for x in r['rows']}), dtype=np.int64)
pf   = pq.ParquetFile(TRAIN)
offs = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
PK = {}
for rg in range(pf.metadata.num_row_groups):
    lo, hi = offs[rg], offs[rg+1]
    loc = sel_rows[(sel_rows >= lo) & (sel_rows < hi)] - lo
    if len(loc) == 0: continue
    t = pf.read_row_group(rg, columns=['ms2_mzs','ms2_normalized_intensities'])
    a = t.column('ms2_mzs').combine_chunks(); b = t.column('ms2_normalized_intensities').combine_chunks()
    ao, av = a.offsets.to_numpy(), a.values.to_numpy(zero_copy_only=False)
    bo, bv = b.offsets.to_numpy(), b.values.to_numpy(zero_copy_only=False)
    for k in loc:
        PK[int(lo + k)] = (av[ao[k]:ao[k+1]].astype(np.float32),
                           bv[bo[k]:bo[k+1]].astype(np.float32))
    del t, a, b, av, bv
print(f"[INFO] peak lists loaded for {len(PK):,} spectra")

_PREC = md['precursor_mz'].values
_AD   = md['adduct'].values
_ION  = md['ionization_mode'].values
_INS  = md['instrument_type'].values
_CE   = md['collision_energy_ev'].values''')

code('''#@title The per-query worker — ONE definition, used by both the smoke test and the main loop
#  Features are computed over the FULL candidate window and only THEN are rows subsampled.
#  _rank_norm and _z are window-wide statistics; subsampling first would compute them over a
#  different population than inference does.
#
#  The row selection is shared between the two ban regimes, so a query's M=0 and M=1 halves are
#  aligned candidate-for-candidate -- the property that makes columns 14-24 identical between
#  the forked matrix's halves.

def build_sub(rows):
    return pd.DataFrame([dict(
        ms2_mzs=PK[g][0], ms2_normalized_intensities=PK[g][1],
        precursor_mz=float(_PREC[g]), adduct=_AD[g], ionization_mode=_ION[g],
        instrument_type=_INS[g], collision_energy_ev=_CE[g]) for g in rows])

def process_query(r, qi, timing=None):
    """-> (X, Y, M, G) stacked over both ban regimes, or None if the query is unusable."""
    tk = lambda k, t: timing.__setitem__(k, timing.get(k, 0.0) + time.time() - t) if timing is not None else None

    sub = build_sub(r['rows'])
    nms = neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
    nms = nms[np.isfinite(nms)]
    if len(nms) == 0: return None
    target = float(np.median(nms)) / (1.0 + CFG.PPM_CENTER / 1e6)
    specs  = [(a, b) for a, b in zip(sub.ms2_mzs, sub.ms2_normalized_intensities)]

    # ---- expensive, computed ONCE and reused by both regimes ------------------------
    t = time.time()
    cand = pool.window(target, CFG.PPM_WIN)
    if len(cand) == 0: cand = pool.window(target, CFG.PPM_FALLBACK)
    if len(cand) == 0: return None
    cfp = pool.fps(cand); tk('window', t)

    ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
    t = time.time(); fsc = frag_scores_fast([pool.smiles[c] for c in cand], specs, ion_mode); tk('metfrag', t)
    t = time.time(); zlog = model_logits(sub); tk('fpnet', t)
    y = (pool.keys[cand] == r['ik']).astype(np.int8)

    if len(cand) > SC.MAX_ROWS_PER_QUERY:
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        take = rng.choice(neg, size=SC.MAX_ROWS_PER_QUERY - len(pos), replace=False)
        keep = np.sort(np.concatenate([pos, take]))
    else:
        keep = np.arange(len(cand))

    Xs, Ys, Ms, Gs = [], [], [], []
    for M, ban_rows, ban_keys in ((0, r['ban_vis'], []), (1, r['ban_all'], [r['ik']])):
        t = time.time(); lib_hits = lib_sim_ban(L, specs, target, ban_rows, ban_keys); tk('library', t)
        t = time.time(); analogs = analog_sim_ban(L, specs, target, rep, rep_key, rep_nm,
                                                  ban_rows, ban_keys); tk('analog', t)
        lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
        ids, sims = [], []
        for k, s in analogs:
            i = pool.k2i.get(k, -1)
            if i >= 0: ids.append(i); sims.append(s)
        afp = pool.fps(np.array(ids)) if ids else None
        t = time.time()
        X = rank_features(cfp, lv, afp, np.array(sims, np.float32), zlog, fsc)[:, :NFEAT]
        tk('features', t)
        Xs.append(X[keep]); Ys.append(y[keep])
        Ms.append(np.full(len(keep), M, np.int8)); Gs.append(np.full(len(keep), qi, np.int32))
    return np.vstack(Xs), np.concatenate(Ys), np.concatenate(Ms), np.concatenate(Gs)

print("process_query ready")''')

code('''#@title VERIFY 1 — pre-flight: leakage, equivalence, smoke test, measured ETA
#  Nothing below this cell has started yet. If the ETA is too long, lower SC.N_SEEN /
#  SC.N_UNSEEN in the config cell, re-run the cohort + assembly cells, and come back.
print("="*74); print("VERIFY 1 - PRE-FLIGHT"); print("="*74)

def _query_target(r):
    sub = build_sub(r['rows'])
    nm  = neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
    nm  = nm[np.isfinite(nm)]
    specs = [(a, b) for a, b in zip(sub.ms2_mzs, sub.ms2_normalized_intensities)]
    return sub, specs, float(np.median(nm)) / (1.0 + CFG.PPM_CENTER / 1e6)

# --- T1: self-retrieval, then the ban filter, over 8 queries not one ----------------
#  One query is not a test -- a single low-precision source library was enough to make
#  the old single-query version fail while the pipeline was fine, and would equally have
#  let a real failure through on a lucky draw.
probe1 = [r for r in records if r['regime'] == 'unseen'][:8]
t1 = []
for r in probe1:
    sub, specs, tgt = _query_target(r)
    cw    = lib_window(L, tgt, tgt * CFG.PPM_WIN / 1e6)
    nself = sum(1 for c in cw if L['ik'][c] == r['ik'])
    ub    = lib_sim_ban(L, specs, tgt, [], []).get(r['ik'], 0.0)
    bn    = lib_sim_ban(L, specs, tgt, r['ban_all'], [r['ik']]).get(r['ik'], 0.0)
    off   = (tgt - float(md.nm_exact.values[r['rows'][0]])) / tgt * 1e6
    t1.append((nself, ub, bn))
    print(f"   {r['ik']}  off {off:+6.2f} ppm | window {len(cw):5,d} spectra, "
          f"{nself:3d} are the truth | no-ban {ub:.3f} | M=1 {bn:.3f}")
found  = sum(1 for n, ub, bn in t1 if ub > 0.5)
worst  = max(x[2] for x in t1)
print(f"T1 self-retrieval  : {found}/{len(t1)} unbanned queries find themselves "
      f"(median sim {np.median([x[1] for x in t1]):.3f})")
print(f"T1 ban filter      : worst M=1 self-match {worst:.4f} (must be ~0)")
assert worst < 0.2, "BAN FILTER BROKEN - every number after this would be meaningless"
assert found >= 0.8 * len(t1), (
    "queries cannot retrieve their own structure even unbanned. The library is indexed on "
    "the label's exact mass (CFG.STRUCT_MASS) while the target comes from the reported "
    "precursor m/z - read the per-library 'usable %' printed by the cohort cell and check "
    "SC.QUERY_PPM_MAX.")
print("   -> PASS")

sub0, specs0, tgt0 = _query_target(probe1[0])

# --- T2: frag_scores_fast is numerically identical to phase1's frag_scores ----------
cand0 = pool.window(tgt0, CFG.PPM_WIN)
if len(cand0) == 0: cand0 = pool.window(tgt0, CFG.PPM_FALLBACK)
ion0 = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub0.ionization_mode]))
smi0 = [pool.smiles[c] for c in cand0]
fa = frag_scores_fast(smi0, specs0, ion0)
fb = frag_scores(smi0, specs0, ion0)
print(f"T2 MetFrag pool    : max|fast-ref| = {np.abs(fa-fb).max():.2e} over {len(fa)} candidates")
assert np.allclose(fa, fb), "frag_scores_fast diverged from frag_scores"
print("   -> PASS")

# --- T3: the truth is not merely in the pool, it is INSIDE the query's window -------
#  Pool membership alone is not enough: the pool is indexed on exact mass, so a query
#  whose target is off by more than PPM_WIN has its own structure in the pool and still
#  outside its window. Such a row group is all-negative and silently poisons the matrix.
n3 = min(60, len(records))
inpool = inwin = 0
for r in records[:n3]:
    if r['ik'] not in pool.k2i: continue
    inpool += 1
    _, _, tg = _query_target(r)
    w = pool.window(tg, CFG.PPM_WIN)
    if len(w) and r['ik'] in set(pool.keys[w].tolist()): inwin += 1
print(f"T3 truth reachable : in pool {inpool}/{n3} | inside the +-{CFG.PPM_WIN} ppm "
      f"window {inwin}/{n3}")
assert inpool >= 0.95 * n3, "queries whose truth is not in the candidate pool can never be positives"
assert inwin >= 0.90 * n3, ("the truth is in the pool but outside the query's own mass window - "
                            "the row group would have no positive row at all")
print("   -> PASS")

# --- T4: smoke test, 3 queries of each regime, with a timing breakdown --------------
probe = ([r for r in records if r['regime'] == 'seen'][:3] +
         [r for r in records if r['regime'] == 'unseen'][:3])
timing = {}
t0 = time.time(); got = 0; rowsum = 0
for j, r in enumerate(probe):
    out = process_query(r, -1, timing)
    if out is not None: got += 1; rowsum += len(out[1])
el = time.time() - t0
per_q = el / max(len(probe), 1)
print(f"\\nT4 smoke test      : {got}/{len(probe)} queries produced rows, "
      f"{rowsum/max(got,1):.0f} rows/query")
print(f"   {per_q:.2f} s/query   (first call includes numba JIT, so the real rate is faster)")
for k, v in sorted(timing.items(), key=lambda x: -x[1]):
    print(f"     {k:9s} {v:7.1f}s  {v/el*100:4.1f}%")
assert got == len(probe), "some probe queries produced nothing - investigate before the full run"

eta_h = per_q * len(records) / 3600
print(f"\\n   ETA for {len(records):,} queries: ~{eta_h:.1f} h on {NCPU} vCPU")
if eta_h > 6:
    print(f"   !! That is long. Lower SC.N_SEEN / SC.N_UNSEEN to about "
          f"{int(len(records)*5/eta_h/2)} each and re-run the cohort + assembly cells.")
elif eta_h < 2:
    print(f"   You have headroom - raising N_SEEN/N_UNSEEN to ~{int(len(records)*4/eta_h/2)} "
          "each would give the ranker more data.")
print("\\nPRE-FLIGHT PASSED - safe to run the generation loop.")''')

code('''#@title Generate the matrix (resumable; checkpoints to Drive every 50 queries)
import traceback
CKPT = os.path.join(SC.OUT, 'rank_partial.npz')
Xs, Ys, Ms, Gs, done = [], [], [], [], 0
if os.path.exists(CKPT):
    d = np.load(CKPT)
    Xs, Ys, Ms, Gs = [d['X']], [d['Y']], [d['M']], [d['G']]
    done = int(d['done'])
    print(f"[RESUME] {done} queries done, {d['X'].shape[0]:,} rows")

REGIME = np.array([r['regime'] == 'unseen' for r in records])
t_loop = time.time(); fails = 0
for qi, r in enumerate(records):
    if qi < done: continue
    try:
        out = process_query(r, qi)
        if out is not None:
            Xs.append(out[0]); Ys.append(out[1]); Ms.append(out[2]); Gs.append(out[3])
    except Exception:
        fails += 1
        print(f"[WARN] query {qi} ({r['ik']}) failed:"); traceback.print_exc()

    if (qi + 1) % 25 == 0 or qi == len(records) - 1:
        nrow = sum(len(x) for x in Xs); el = time.time() - t_loop
        rate = (qi + 1 - done) / max(el, 1e-9)
        print(f"  {qi+1}/{len(records)} | {nrow:,} rows | {el/60:.0f} min | "
              f"{rate:.2f} q/s | eta {(len(records)-qi-1)/max(rate,1e-9)/60:.0f} min", flush=True)
    if (qi + 1) % 50 == 0:
        np.savez_compressed(CKPT, X=np.vstack(Xs), Y=np.concatenate(Ys),
                            M=np.concatenate(Ms), G=np.concatenate(Gs), done=qi + 1)

X = np.vstack(Xs).astype(np.float32)
Y = np.concatenate(Ys).astype(np.int8)
M = np.concatenate(Ms).astype(np.int8)
G = np.concatenate(Gs).astype(np.int32)
print(f"\\n[DONE] X{X.shape} | positives {int(Y.sum()):,} ({Y.mean()*100:.2f}%) | "
      f"M=0 {int((M==0).sum()):,} | M=1 {int((M==1).sum()):,} | {fails} failed queries")
np.savez_compressed(os.path.join(SC.OUT, 'rank_raw.npz'), X=X, Y=Y, M=M, G=G)''')

code('''#@title VERIFY 2 — the GATE: does this matrix teach a four-channel fusion?
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
    print("\\n*** GATE FAILED - NOT writing rank_train.npz ***")
    for f in fail: print("   -", f)
else:
    np.savez_compressed(os.path.join(SC.OUT, 'rank_train.npz'), X=X, Y=Y, M=M, G=G)
    print(f"\\n*** GATE PASSED *** -> {SC.OUT}/rank_train.npz")
    if HAS_REG and not (0.60 <= mu1 <= 0.85):
        print(f"[WARN] honest-cell MRR {mu1:.3f} outside 0.60-0.85 - read Stage D carefully.")
    if SHARE['model'] > 0.40:
        print(f"[WARN] model channel {SHARE['model']*100:.0f}% is high (forked 33.9%) - "
              "expected with a stronger FPNet, but watch Stage D's C2_hard.")''')

code('''#@title VERIFY 3 — full report (copy the block between the markers and send it back)
OLDP = f"{DRIVE_ROOT}/casmi26-ranker-features/rank_train.npz"
old = np.load(OLDP) if os.path.exists(OLDP) else None
if old is not None:
    # Identify which matrix this actually is, so the comparison column is not mislabelled.
    _oam = old['X'][old['Y'] == 1, 20].mean() * 100
    _tag = ("forked (0.337)" if _oam < 60 else "v1 LEAKED (0.287)")
    print(f"reference matrix: {OLDP}")
    print(f"  X{old['X'].shape}  model-top-1 {_oam:.1f}%  ->  looks like the {_tag} matrix")
    if _oam >= 60:
        print("  !! that is the leaked v1 matrix, not the forked one - the 'OLD' columns below")
        print("     are NOT the 0.337 baseline. Say so when you send the report.")
else:
    print(f"reference matrix not found at {OLDP} - 'OLD' columns will read n/a")

NAMES = ['lv','rk(lv)','lvmax','lv-lvmax','lv>0','ap','rk(ap)','apmax','ap-apmax','a1',
         'best_tan','top_tan','mean_tan','top_sim','log(nc)','z(raw)','rk(raw)','raw-max',
         'z(nrm)','rk(nrm)','argmax','frag','rk(frag)','frag-max','z(frag)',
         'lv*(1-mr)','ap*(1-mr)','agree','agree_a','agree*lvmax','corr']
pos = Y == 1
seen_g = np.isin(G, [qi for qi, r in enumerate(records) if r['regime'] == 'seen'])

print("\\n===== PASTE FROM HERE =====")
print(f"runtime        : {NCPU} vCPU / {RAM:.0f} GB / "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
print(f"fp_bits sha1   : {FP_SHA1}")
print(f"fallback used  : {USED_FALLBACK}")
print(f"queries        : {len(records):,}  (seen {sum(1 for r in records if r['regime']=='seen')}"
      f" / unseen {sum(1 for r in records if r['regime']=='unseen')})")
print(f"matrix         : X{X.shape}  positives {int(Y.sum()):,} ({Y.mean()*100:.2f}%)  "
      f"M=0 {(M==0).mean():.2f}")
print(f"row-wtd window : {np.exp(X[:,14]).mean():.1f}   (forked 194.7 | v1 106.8)")
print("")
print("--- the leak indicator: is the truth the model's top-1 in its window? ---")
print(f"  all queries        : {X[pos,20].mean()*100:5.1f}%     (v1 90.2% | forked 42.0%)")
print(f"  FPNet-SEEN  queries: {X[pos & seen_g,20].mean()*100:5.1f}%     (memorised - expect high)")
print(f"  FPNet-UNSEEN       : {X[pos & ~seen_g,20].mean()*100:5.1f}%     (the honest number)")
print("")
print("--- channel reliance (held-out, channel destroyed) ---")
print(f"  held-out MRR : {base:.4f}   M=0 {m0:.4f} | M=1 {m1:.4f}")
for cn in ('library','analog','model','frag'):
    print(f"  {cn:8s}: {SHARE[cn]*100:5.1f}%")
print(f"  GATE         : {'PASS' if GATE_OK else 'FAIL'}")
print("")
print("--- positives: NEW vs forked (mean+-sd) ---")
print(f"{'col':>3} {'name':11} {'NEW M=0':>15} {'OLD M=0':>15} {'NEW M=1':>15} {'OLD M=1':>15}")
def _stat(Xm, sel, c):
    v = Xm[sel, c]
    return f"{v.mean():6.3f}+-{v.std():5.2f}" if len(v) else "      n/a     "

for c in range(31):
    row = []
    for mv in (0, 1):
        row.append(_stat(X, pos & (M == mv), c))
        row.append(_stat(old['X'], (old['Y'] == 1) & (old['M'] == mv), c)
                   if old is not None else "      n/a     ")
    print(f"{c:>3} {NAMES[c]:11} {row[0]:>15} {row[1]:>15} {row[2]:>15} {row[3]:>15}")
print("===== PASTE TO HERE =====\\n")

if GATE_OK:
    print("NEXT: copy rank_train.npz from")
    print(f"  {SC.OUT}/rank_train.npz")
    print("into the casmi26-fpnet-v3 Kaggle dataset, NEXT TO the checkpoints it was built from.")
    print("Then run casmi_validate_ab.ipynb (Stage D). Do NOT submit before Stage D.")
else:
    print("Gate failed - send the block above before changing anything.")''')

out = {"cells": cells,
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"},
                    "accelerator": "GPU"},
       "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT.name}: {len(cells)} cells")
