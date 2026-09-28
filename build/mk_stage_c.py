"""Build the Stage-C v2 Kaggle notebook: regenerate rank_train.npz WITHOUT the leak.

v1 shipped a matrix in which every query structure had been memorised by the FPNet, so f.z was
a near-oracle (truth top-1 in 90.2% of windows vs 42.0% in the forked matrix). The GBDT
responded rationally: it leaned 60.2% of its skill on the model channel and reduced library to
2.7%, analog to 0.2% and MetFrag to 0.0%. On the leaderboard, where the model has not seen the
Class-2/3 structures, there was no fallback left. 0.337 -> 0.287.

v2 changes three things:
  1. queries are drawn from BOTH model-strength regimes (FPNet-seen and FPNet-unseen)
  2. each query is scored TWICE -- library visible and library banned -- reusing the same
     candidates, logits and MetFrag scores, as the forked builder did
  3. full candidate windows, no hard-negative mining, and a `G` group column

Cells 1-8 of the phase1 inference notebook are still embedded VERBATIM.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "enveda-casmi-2026-phase1-enhanced-baseline.ipynb"
OUT = HERE.parent / "casmi_rank_matrix.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
P = ["".join(c["source"]) for c in nb["cells"]]

split = P[8].index("# Fit Calibrated Rankers")
RANK_FEATURES = P[8][:split].rstrip()
assert "def rank_features" in RANK_FEATURES and "rank_train.npz" not in RANK_FEATURES

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(True)})


md("""# CASMI 2026 - Stage C **v2**: rebuild `rank_train.npz` without the leak

### What went wrong in v1

v1 excluded the *validation* structures from its query sampling but not the structures the
*FPNet trained on*. The FPNet trained on essentially all of `train.parquet`, so **every v1
query was a molecule the model had memorised** and `f.z` behaved as a near-oracle:

| | forked matrix | v1 matrix |
|---|---|---|
| truth is the model's top-1 in its window | 42.0% | **90.2%** |
| `_z(raw)` for the truth | 1.64 sigma | **2.64 sigma** |

The GBDT did the rational thing with a feature that alone solves 90% of its training queries -
it used that feature and discarded the others. Channel contribution, measured by destroying
one channel at a time on held-out queries:

| channel | forked ranker | v1 ranker |
|---|---|---|
| library | 14.5% of skill | **2.7%** |
| analog | 15.0% | **0.2%** |
| model `f.z` | 33.9% | **60.2%** |
| MetFrag | 2.3% | **0.0%** |
| held-out MRR | 0.713 | 0.958 |

On the leaderboard the model has *not* seen the Class-2/3 structures, `f.z` degrades, and
there is no fallback. 0.337 -> 0.287.

### What v2 does

**1. Queries from both model-strength regimes.** `fpnet_split_keys.npz` says which structures
the FPNet actually gradient-updated on. Four cells:

| cell | query structure | library | `M` | mirrors |
|---|---|---|---|---|
| A | FPNet-**seen** | visible | 0 | test Class 1 - model strong, `lv` high |
| C | FPNet-**seen** | banned | 1 | a Class-2 molecule the model happens to nail |
| D | FPNet-**unseen** | visible | 0 | cross-library Class 1 - model weak, `lv` strong |
| B | FPNet-**unseen** | banned | 1 | **test Class 2** - absent from v1 entirely |

**2. Paired two-regime scoring.** Each query runs the expensive work once - window,
`model_logits`, `frag_scores` - then re-runs only the library and analog channels under both
ban settings. Same design as the forked builder, where columns 14-24 are byte-identical
between the `M=0` and `M=1` halves. Halves the runtime and gives a controlled contrast.

**3. Full windows and a `G` column.** No hard-negative mining: v1's flat ~37-row cap
under-weighted crowded windows 2x (row-weighted mean window 106.8 vs the forked 194.7), and
those are exactly where MRR is lost.

### Attach
competition data, `casmi26-offline-rdkit-2026033`, `coconut_dataset`, `val_scripts`,
**`casmi26-fpnet-v3`** (checkpoints + `fpnet_split_keys.npz`).
**Not** `casmi26-fp-models-v2`, **not** `casmi26-ranker-features` - `load_neural_models()`
globs `fp_*.pt` across all of `/kaggle/input`.

### Sharding
`SC.SHARD` = 0, 1, 2 in three concurrent notebooks, then run the merge cell once. The final
cell is a **hard gate**: it refuses to write `rank_train.npz` if the channel balance looks
like v1 again.""")

md("## Cells 1-8 - verbatim from the inference notebook")
for i in range(1, 8):
    code(P[i])
code(RANK_FEATURES)

code('''# ===================================================================================
#  Stage-C v2 config
# ===================================================================================
class SC:
    SHARD     = 0          # <- 0, 1, 2 in three concurrent notebooks
    N_SHARDS  = 3

    # Query structures per model-strength regime. Each yields TWO row-groups
    # (library visible + library banned), so this is ~2,800 groups total.
    N_SEEN    = 700        # FPNet trained on these  -> cells A (M=0) and C (M=1)
    N_UNSEEN  = 700        # FPNet never saw these   -> cells D (M=0) and B (M=1)

    # Cohort definition, mirroring the validation-set builder.
    NP_LIBS = ['riken','mona','msdial','massbank','gnps','enveda-np-examples','masaryk']
    MASS_LO, MASS_HI = 157.0, 1159.0
    MIN_PEAKS = 5

    # A query builds its mass window from its OWN reported precursor m/z, but both search
    # structures are indexed on the label's exact mass (CFG.STRUCT_MASS for the library,
    # ExactMolWt for the pool). A query whose reported m/z cannot place its own label mass
    # inside the window can never retrieve itself -- it is guaranteed lv=0 and, worse, a
    # guaranteed all-negative row group. Keep this equal to CFG.PPM_WIN.
    QUERY_PPM_MAX = CFG.PPM_WIN

    # Full windows. The cap only guards pathological windows and sits above the forked
    # matrix's p90 (344), so in practice almost nothing is dropped. When it does fire it
    # keeps every positive plus a RANDOM negative sample - never hard-mined, because
    # censoring the easy negatives is part of what broke v1.
    MAX_ROWS_PER_QUERY = 400

    SEED = 20260923
    OUT = '/kaggle/working'

NFEAT = N_FEAT   # 31 -- the natural width of rank_features
print(f"shard {SC.SHARD+1}/{SC.N_SHARDS}  |  NFEAT={NFEAT}")''')

code('''# ===================================================================================
#  Verify the fp_bits contract, then load YOUR checkpoints
# ===================================================================================
import hashlib
FP_SHA1 = hashlib.sha1(np.ascontiguousarray(BITS).tobytes()).hexdigest()
print(f"fp_bits sha1 (this notebook): {FP_SHA1}")

found = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
print("checkpoints visible to load_neural_models():")
for p in found: print("   ", p)
assert found, "no fp_*.pt found - attach casmi26-fpnet-v3"

for p in found:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    got = ck.get('fp_bits_sha1')
    if got is None:
        print(f"  !! {os.path.basename(p)} carries no fp_bits_sha1 - this is an OLD checkpoint.")
        print("     Detach casmi26-fp-models-v2, or Stage C bakes the old model into the features.")
        raise SystemExit("old checkpoint attached")
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

code('''# ===================================================================================
#  Library, pool, representative analog set  (same construction as inference)
# ===================================================================================
pool = build_candidate_pool()
L = load_library(TRAIN)
rep, rep_key, rep_nm, rep_pos = build_rep(L)
print(f"[INFO] pool {len(pool.mass):,} | library {len(L['off'])-1:,} spectra | rep {len(rep):,}")''')

code('''# ===================================================================================
#  Ban-aware channels - the leakage machinery, same as the validation notebook
# ===================================================================================
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
    """phase1's frag_scores with one persistent worker pool instead of one fork per molecule.
    Numerically identical - fragment_masses is a pure function of the SMILES - and the
    generation loop asserts equivalence on the first real query before using it."""
    global _MP
    if not HAVE_RDKIT: return np.zeros(len(cand_smiles), np.float32)
    if _MP is None: _MP = MPool(4)
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

print("ban-aware channels ready")''')

code('''# ===================================================================================
#  Query cohort: BOTH model-strength regimes, disjoint from the validation set
# ===================================================================================
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

# --- which structures did the FPNet actually gradient-update on? --------------------
SPLIT = np.load(find_file('fpnet_split_keys.npz'))
SEEN_K   = set(SPLIT['seen_keys'].tolist())     # FPNet trained on these
ES_K     = set(SPLIT['es_keys'].tolist())       # scaffold-disjoint holdout: never trained
BANNED_K = set(SPLIT['banned_keys'].tolist())   # val ban_keys: reserved for Stage D
print(f"[INFO] FPNet split: seen {len(SEEN_K):,} | unseen(es) {len(ES_K):,} | "
      f"val-reserved {len(BANNED_K):,}")

# Stage D's holdout is off limits as a Stage-C query, in BOTH regimes.
val_key = pickle.load(open(find_file('val_key.pkl'), 'rb'))
VAL_RESERVED = set()
for r in val_key:
    VAL_RESERVED.add(r['inchikey14'])
    VAL_RESERVED |= set(r.get('ban_keys') or [])
print(f"[INFO] validation reserves {len(VAL_RESERVED):,} structures")

TEST_ADDUCTS = ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
                "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"]

# Spectrum-quality / chemistry filter, independent of who is allowed to be a query.
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

VAL_QUERY = {r['inchikey14'] for r in val_key}     # the 1,145 Stage-D molecules themselves

E = build_E(elig_base & ~md.inchikey14.isin(VAL_RESERVED).values)
print(f"[INFO] eligible query spectra: {len(E):,} / {E.inchikey14.nunique():,} structures")

elig_keys   = set(E.inchikey14)
seen_pool   = np.array(sorted(elig_keys & SEEN_K))
unseen_pool = np.array(sorted(elig_keys & ES_K))
print(f"[INFO] eligible & FPNet-seen  : {len(seen_pool):,}")
print(f"[INFO] eligible & FPNet-unseen: {len(unseen_pool):,}   <- the Class-2 regime")

# Fallback: the val-banned set is ~15k structures but only 1,145 are actual Stage-D queries.
# The rest are scaffold relatives swept in by C2_hard, and they are also FPNet-unseen. Using
# them makes Stage D's C2_hard reading mildly optimistic (shared scaffolds), so only reach for
# them if the es pool genuinely cannot fill the cohort.
if len(unseen_pool) < SC.N_UNSEEN:
    print(f"[WARN] only {len(unseen_pool):,} FPNet-unseen structures survive the filters, "
          f"need {SC.N_UNSEEN:,}.")
    print( "       Widening to val-banned SCAFFOLD RELATIVES (never the val queries themselves).")
    print( "       NOTE: this makes Stage D's C2_hard cohort mildly optimistic - record it.")
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
print(f"\\n[INFO] shard {SC.SHARD}: {len(QUERIES):,} query structures "
      f"({sum(1 for _,t in QUERIES if t=='seen')} seen / "
      f"{sum(1 for _,t in QUERIES if t=='unseen')} unseen), each scored under 2 ban regimes")

SPEC_N = pq.read_table(TEST, columns=['molecule_id']).to_pandas().groupby('molecule_id').size().values''')

code('''# ===================================================================================
#  Assemble each query: pick spectra like the test set, then both ban settings
# ===================================================================================
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

    # M=0 "library visible": ban ONLY the query's own spectra, so the pipeline cannot match a
    #     spectrum against itself but other measurements of the compound stay findable. That is
    #     exactly the real Class-1 condition -- a fresh timsTOF run of a compound that is in
    #     public libraries, measured elsewhere.
    # M=1 "library banned": every spectrum of the structure is hidden and the key is banned
    #     from the analog channel too. That is Class 2.
    all_rows = sorted(rows_of_key.get(ik, []))
    records.append(dict(ik=ik, regime=regime, rows=pick,
                        ban_vis=sorted(pick), ban_all=all_rows))

print(f"[INFO] {len(records):,} query molecules -> {2*len(records):,} row-groups, "
      f"{sum(len(r['rows']) for r in records):,} spectra")

# read the peak lists for exactly those rows, row-group by row-group
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

# Positional views. `.at[row, col]` on an object column holding numpy arrays is fragile
# across pandas versions; raw .values indexing is not.
_PREC = md['precursor_mz'].values
_AD   = md['adduct'].values
_ION  = md['ionization_mode'].values
_INS  = md['instrument_type'].values
_CE   = md['collision_energy_ev'].values''')

code('''# ===================================================================================
#  Generate the matrix - one expensive pass per query, two cheap ban regimes
# ===================================================================================
#  Features are computed over the FULL candidate window and only THEN are rows subsampled.
#  _rank_norm and _z are window-wide statistics; subsampling first would compute them over a
#  different population than inference does.
#
#  The row selection is shared between the two regimes, so the M=0 and M=1 halves of a query
#  are aligned candidate-for-candidate - the same property that makes columns 14-24 identical
#  between the forked matrix's halves.
import traceback

CKPT = os.path.join(SC.OUT, f'rank_shard_{SC.SHARD}_partial.npz')
Xs, Ys, Ms, Gs, done = [], [], [], [], 0
if os.path.exists(CKPT):
    d = np.load(CKPT)
    Xs, Ys, Ms, Gs = [d['X']], [d['Y']], [d['M']], [d['G']]
    done = int(d['done'])
    print(f"[RESUME] {done} queries already processed, {d['X'].shape[0]:,} rows")

t_loop = time.time()
checked = False
regime_rows = {'seen': 0, 'unseen': 0}

for qi, r in enumerate(records):
    if qi < done: continue
    try:
        sub = pd.DataFrame([dict(
            ms2_mzs=PK[g][0], ms2_normalized_intensities=PK[g][1],
            precursor_mz=float(_PREC[g]), adduct=_AD[g],
            ionization_mode=_ION[g], instrument_type=_INS[g],
            collision_energy_ev=_CE[g]) for g in r['rows']])

        nms = neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
        nms = nms[np.isfinite(nms)]
        if len(nms) == 0: continue
        target = float(np.median(nms)) / (1.0 + CFG.PPM_CENTER / 1e6)
        specs  = [(a, b) for a, b in zip(sub.ms2_mzs, sub.ms2_normalized_intensities)]

        # ---- expensive, computed ONCE and reused by both regimes --------------------
        cand = pool.window(target, CFG.PPM_WIN)
        if len(cand) == 0: cand = pool.window(target, CFG.PPM_FALLBACK)
        if len(cand) == 0: continue

        cfp = pool.fps(cand)
        ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
        fsc = frag_scores_fast([pool.smiles[c] for c in cand], specs, ion_mode)
        if not checked:
            ref = frag_scores([pool.smiles[c] for c in cand], specs, ion_mode)
            assert np.allclose(fsc, ref), "frag_scores_fast diverged from frag_scores"
            print("[CHECK] frag_scores_fast == frag_scores on a real query"); checked = True
        zlog = model_logits(sub)
        y = (pool.keys[cand] == r['ik']).astype(np.int8)

        # ---- shared row selection: full window, capped only for pathological cases ---
        if len(cand) > SC.MAX_ROWS_PER_QUERY:
            pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
            take = rng.choice(neg, size=SC.MAX_ROWS_PER_QUERY - len(pos), replace=False)
            keep = np.sort(np.concatenate([pos, take]))
        else:
            keep = np.arange(len(cand))

        # ---- two ban regimes, cheap ------------------------------------------------
        for M, ban_rows, ban_keys in ((0, r['ban_vis'], []),
                                      (1, r['ban_all'], [r['ik']])):
            lib_hits = lib_sim_ban(L, specs, target, ban_rows, ban_keys)
            analogs  = analog_sim_ban(L, specs, target, rep, rep_key, rep_nm,
                                      ban_rows, ban_keys)
            lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
            ids, sims = [], []
            for k, s in analogs:
                i = pool.k2i.get(k, -1)
                if i >= 0: ids.append(i); sims.append(s)
            afp = pool.fps(np.array(ids)) if ids else None

            X = rank_features(cfp, lv, afp, np.array(sims, np.float32), zlog, fsc)[:, :NFEAT]
            Xs.append(X[keep]); Ys.append(y[keep])
            Ms.append(np.full(len(keep), M, np.int8))
            Gs.append(np.full(len(keep), qi, np.int32))
            regime_rows[r['regime']] += len(keep)
    except Exception:
        print(f"[WARN] query {qi} ({r['ik']}) failed:"); traceback.print_exc()

    if (qi + 1) % 25 == 0 or qi == len(records) - 1:
        nrow = sum(len(x) for x in Xs)
        el = time.time() - t_loop
        rate = (qi + 1 - done) / max(el, 1e-9)
        print(f"  {qi+1}/{len(records)} queries | {nrow:,} rows | {el:.0f}s | "
              f"{rate:.2f} q/s | eta {(len(records)-qi-1)/max(rate,1e-9)/60:.0f} min", flush=True)
    if (qi + 1) % 50 == 0:
        np.savez_compressed(CKPT, X=np.vstack(Xs), Y=np.concatenate(Ys),
                            M=np.concatenate(Ms), G=np.concatenate(Gs), done=qi + 1)

X = np.vstack(Xs).astype(np.float32)
Y = np.concatenate(Ys).astype(np.int8)
M = np.concatenate(Ms).astype(np.int8)
G = np.concatenate(Gs).astype(np.int32)
out = os.path.join(SC.OUT, f'rank_shard_{SC.SHARD}.npz')
np.savez_compressed(out, X=X, Y=Y, M=M, G=G)
print(f"\\n[SUCCESS] {out}: X{X.shape}")
print(f"  positives {int(Y.sum()):,} ({Y.mean()*100:.2f}%) | M=0 {int((M==0).sum()):,} | M=1 {int((M==1).sum()):,}")
print(f"  rows from FPNet-seen queries   : {regime_rows['seen']:,}")
print(f"  rows from FPNet-unseen queries : {regime_rows['unseen']:,}")

# The single number that says whether v1's leak is gone.
argmax_seen = X[(Y == 1), 20]
print(f"\\n  truth is the model's top-1 in {argmax_seen.mean()*100:.1f}% of windows "
      f"(v1 was 90.2%, forked matrix 42.0%)")''')

code('''# ===================================================================================
#  Merge the shards, then GATE on channel balance before writing rank_train.npz
# ===================================================================================
shards = sorted(glob.glob('/kaggle/input/**/rank_shard_[0-9].npz', recursive=True)) or \\
         sorted(glob.glob(os.path.join(SC.OUT, 'rank_shard_[0-9].npz')))
print("shards:", [os.path.basename(s) for s in shards])
assert shards, "no shards found"

Xl, Yl, Ml, Gl, off = [], [], [], [], 0
for s in shards:
    d = np.load(s)
    Xl.append(d['X']); Yl.append(d['Y']); Ml.append(d['M'])
    Gl.append(d['G'].astype(np.int64) + off)      # keep group ids unique across shards
    off += int(d['G'].max()) + 1
X = np.vstack(Xl).astype(np.float32)
Y = np.concatenate(Yl).astype(np.int8)
M = np.concatenate(Ml).astype(np.int8)
G = np.concatenate(Gl).astype(np.int64)

assert X.shape[1] == 31, f"expected 31 features, got {X.shape[1]}"
assert np.isfinite(X).all(), "non-finite feature values"
assert set(np.unique(Y)) <= {0, 1} and Y.sum() > 0
frac0 = (M == 0).mean()
print(f"\\nX{X.shape}  positives {int(Y.sum()):,} ({Y.mean()*100:.2f}%)  M=0 fraction {frac0:.2f}")
assert 0.35 < frac0 < 0.65, (
    f"M split is {frac0:.2f}; CFG.W1_PRIORS=(0.30,0.60) hedging assumes a balanced split")

nc = np.exp(X[:, 14])
print(f"row-weighted mean window {nc.mean():.1f} (forked 194.7, v1 106.8)")
print(f"truth is model top-1 in {X[Y==1, 20].mean()*100:.1f}% of windows (v1 90.2%, forked 42.0%)")''')

code('''# ===================================================================================
#  THE GATE - the test that would have caught v1
# ===================================================================================
#  Fit the real phase1 ranker on 60% of query groups, then destroy one channel at a time on
#  the held-out 40% and measure how much MRR each channel was carrying. A healthy fusion
#  spreads its skill; v1 put 60.2% on the model channel and reduced the other three to noise.
import warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier

CH = {'library': [0,1,2,3,4,25], 'analog': [5,6,7,8,9,10,11,12,13,26],
      'model':   [15,16,17,18,19,20], 'frag': [21,22,23,24]}

Xd, Yd, Md = X.astype(np.float64), Y.astype(int), M.astype(int)
grp = G * 2 + Md                                   # one group per (query, ban regime)
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
    return float(np.mean(acc)) if acc else float('nan'), len(acc)

rsh = np.random.default_rng(0)
def destroy(A, gv, cols):
    Z = A.copy()
    for q in np.unique(gv):
        i = np.flatnonzero(gv == q)
        Z[np.ix_(i, cols)] = Z[np.ix_(rsh.permutation(i), cols)]
    return Z

Xe, Ye, ge, Me = Xd[~tr], Yd[~tr], grp[~tr], Md[~tr]
base, nq = mrr(Xe, Ye, ge)
print(f"held-out MRR {base:.4f} over {nq} groups   (forked ranker 0.713, v1 0.958)")

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
share = {}
for cn, cols in CH.items():
    m2, _ = mrr(destroy(Xe, ge, cols), Ye, ge)
    share[cn] = (base - m2) / base
    print(f"  {cn+' destroyed':22} MRR {m2:.4f}  -> carries {share[cn]*100:5.1f}% of skill "
          f"(forked: library 14.5 analog 15.0 model 33.9 frag 2.3)")

fail = []
if share['model'] > 0.50:   fail.append(f"model channel carries {share['model']*100:.0f}% > 50% - v1's failure mode")
if share['library'] < 0.05: fail.append(f"library channel carries only {share['library']*100:.0f}% - it has been switched off")
if share['analog']  < 0.05: fail.append(f"analog channel carries only {share['analog']*100:.0f}% - it has been switched off")
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
if fail:
    print("\\n*** GATE FAILED - NOT writing rank_train.npz ***")
    for f in fail: print("   -", f)
    raise SystemExit("channel balance check failed; fix the query cohort before shipping")

if HAS_REG and not (0.60 <= mu1 <= 0.85):
    print(f"\\n[WARN] honest-cell MRR {mu1:.3f} is outside the expected 0.60-0.85 band. "
          "Not fatal, but read Stage D's cohort table carefully.")
if share['model'] > 0.40:
    print(f"[WARN] model channel at {share['model']*100:.0f}% is high (forked 33.9%). "
          "Expected with a stronger FPNet, but watch Stage D's C2_hard.")

np.savez_compressed('/kaggle/working/rank_train.npz', X=X, Y=Y, M=M, G=G)
print("\\n[SUCCESS] gate passed. rank_train.npz written.")
print("Add it to casmi26-fpnet-v3 ALONGSIDE the checkpoints it was built from.")
print("Then run casmi_validate_ab.ipynb (Stage D). Do NOT submit before Stage D.")''')

out = {"cells": cells,
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"}},
       "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT.name}: {len(cells)} cells (7 verbatim + rank_features + 9 new)")
