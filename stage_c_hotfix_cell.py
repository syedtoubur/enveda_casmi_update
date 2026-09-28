# =====================================================================================
#  HOTFIX for a LIVE Colab kernel -- paste this whole file into a NEW cell BELOW the one
#  that failed, and run it. It redoes the cohort, the assembly and VERIFY 1 with the
#  retrievable-target filter. It does NOT touch `pool`, `L` or `rep`, so nothing is
#  reloaded and nothing already in memory is thrown away.
#
#  Why: the library is indexed on the label's exact mass (CFG.STRUCT_MASS) and the pool on
#  ExactMolWt(smiles), while a query's window is built from its own reported precursor m/z.
#  riken reports m/z to 2 decimals -- ~12 ppm at 400 Da -- so a riken query sits outside
#  its own +-8.5 ppm window and can retrieve neither its own library spectra (lv = 0) nor
#  its own structure from the pool (no positive row at all).
#
#  After this passes, run the generation cell as usual. The regenerated notebook on disk
#  already contains all of this, so a fresh run needs no patch.
# =====================================================================================

SC.QUERY_PPM_MAX = CFG.PPM_WIN   # the live SC was built before this knob existed

#@title Query cohort — BOTH model-strength regimes, disjoint from the validation set
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
print(f"\n[INFO] {len(QUERIES):,} query structures "
      f"({sum(1 for _,t in QUERIES if t=='seen')} seen / "
      f"{sum(1 for _,t in QUERIES if t=='unseen')} unseen), each under 2 ban regimes")

SPEC_N = pq.read_table(TEST, columns=['molecule_id']).to_pandas().groupby('molecule_id').size().values

#@title Assemble each query: pick spectra like the test set, compute both ban sets
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
_CE   = md['collision_energy_ev'].values

#@title VERIFY 1 — pre-flight: leakage, equivalence, smoke test, measured ETA
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
print(f"\nT4 smoke test      : {got}/{len(probe)} queries produced rows, "
      f"{rowsum/max(got,1):.0f} rows/query")
print(f"   {per_q:.2f} s/query   (first call includes numba JIT, so the real rate is faster)")
for k, v in sorted(timing.items(), key=lambda x: -x[1]):
    print(f"     {k:9s} {v:7.1f}s  {v/el*100:4.1f}%")
assert got == len(probe), "some probe queries produced nothing - investigate before the full run"

eta_h = per_q * len(records) / 3600
print(f"\n   ETA for {len(records):,} queries: ~{eta_h:.1f} h on {NCPU} vCPU")
if eta_h > 6:
    print(f"   !! That is long. Lower SC.N_SEEN / SC.N_UNSEEN to about "
          f"{int(len(records)*5/eta_h/2)} each and re-run the cohort + assembly cells.")
elif eta_h < 2:
    print(f"   You have headroom - raising N_SEEN/N_UNSEEN to ~{int(len(records)*4/eta_h/2)} "
          "each would give the ranker more data.")
print("\nPRE-FLIGHT PASSED - safe to run the generation loop.")
