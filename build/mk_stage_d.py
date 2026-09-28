"""Build the Stage-D Kaggle notebook: 4-way A/B of {old,new} FPNet x {old,new} rank matrix.

Again, cells 1-8 of the phase1 notebook are embedded verbatim. Stage D only means something
if it scores the code that will actually run on the leaderboard.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "enveda-casmi-2026-phase1-enhanced-baseline.ipynb"
OUT = HERE.parent / "casmi_validate_ab.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
P = ["".join(c["source"]) for c in nb["cells"]]
split = P[8].index("# Fit Calibrated Rankers")
RANK_FEATURES = P[8][:split].rstrip()

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)})
def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(True)})


md("""# CASMI 2026 - Stage D: does the new stack actually win?

Scores the full phase1 ranking pipeline over the 1,145 `val_scripts` molecules in four
configurations:

| config | FPNet | rank_train.npz | what it isolates |
|---|---|---|---|
| `base`       | old | old | your current leaderboard anchor |
| `model_only` | **new** | old | the coupling trap - **expect flat or negative** |
| `matrix_only`| old | **new** | did regenerating the matrix alone help? |
| `both`       | **new** | **new** | the shipping candidate |

**Decision rule.** Ship `both` only if C2 and C2_hard each gain more than 1 SE *and* C1_npx
does not regress by more than 1 SE. At n=400 and n=200, 1 SE on MRR is roughly 0.02 and 0.03.
Anything smaller is noise and chasing it costs a day for nothing.

If `model_only` lands flat-to-negative while `both` gains, that is not a bug - it is the
coupling hypothesis confirming itself, and it is the reason Stage C exists.

### Efficiency
Channels 1, 2 and 3 do not depend on the FPNet or on the ranker, so they are computed **once**
and cached. Only the Channel-4 logits (twice: old, new) and the final feature assembly and
GBDT scoring (four times) are repeated. That turns four full passes into roughly one.

### Attach
competition data, `casmi26-offline-rdkit-2026033`, `coconut_dataset`, `val_scripts`,
`casmi26-fp-models-v2` (old), `casmi26-ranker-features` (old), `casmi26-fpnet-v3` (new).
Here you *want* both model datasets attached - this notebook selects between them explicitly
by path rather than by glob.""")

md("## Cells 1-8 - verbatim from the inference notebook")
for i in range(1, 8):
    code(P[i])
code(RANK_FEATURES)

code('''# ===================================================================================
#  Stage-D config + explicit artifact selection
# ===================================================================================
class SD:
    VAL_PARQUET = None      # auto-discovered
    TOPN = 25
    OUT = '/kaggle/working'

NFEAT = N_FEAT
SD.VAL_PARQUET = find_file('val.parquet')
VAL_KEY_PATH    = find_file('val_key.pkl')

all_pt  = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
all_npz = sorted(glob.glob('/kaggle/input/**/rank_train.npz', recursive=True))

def _is_new(p):
    try:
        return torch.load(p, map_location='cpu', weights_only=False).get('fp_bits_sha1') is not None
    except Exception:
        return False

NEW_PT = [p for p in all_pt if _is_new(p)]
OLD_PT = [p for p in all_pt if p not in NEW_PT]

# The new matrix is the one that SHIPS WITH the new checkpoints, so identify it by
# directory co-location rather than by a dataset name the user might have typed differently.
_new_dirs = {os.path.dirname(p) for p in NEW_PT}
NEW_NPZ = [p for p in all_npz if os.path.dirname(p) in _new_dirs]
if not NEW_NPZ:   # fall back to the name, then tell the user what went wrong
    NEW_NPZ = [p for p in all_npz if 'fpnet' in p.lower() and 'v3' in p.lower()]
OLD_NPZ = [p for p in all_npz if p not in NEW_NPZ]

print("OLD checkpoints:", [os.path.basename(p) for p in OLD_PT])
print("NEW checkpoints:", [os.path.basename(p) for p in NEW_PT])
print("OLD matrix     :", OLD_NPZ)
print("NEW matrix     :", NEW_NPZ)
assert OLD_PT, "no OLD checkpoints found - attach casmi26-fp-models-v2"
assert NEW_PT, "no NEW checkpoints found - attach casmi26-fpnet-v3 (they carry fp_bits_sha1)"
assert OLD_NPZ, "no OLD rank_train.npz - attach casmi26-ranker-features"
assert NEW_NPZ, (
    "no NEW rank_train.npz found next to the new checkpoints. Put the Stage-C output in the "
    "SAME Kaggle dataset as the checkpoints it was built from - they are a matched pair.")
assert len(NEW_NPZ) == 1 and len(OLD_NPZ) == 1, f"ambiguous matrices: {OLD_NPZ} / {NEW_NPZ}"''')

code('''# ===================================================================================
#  Explicit model / ranker loaders (select by path, never by glob)
# ===================================================================================
def load_models(paths):
    """load_neural_models() with an explicit file list instead of a glob over all inputs."""
    global _MODEL
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    single, merged, ck = [], [], None
    for pth in sorted(paths):
        ck = torch.load(pth, map_location='cpu', weights_only=False)
        net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(dev).eval()
        net.load_state_dict(ck['model'])
        (merged if 'merged' in os.path.basename(pth) else single).append(net)
    assert single and merged, f"need >=1 single and >=1 merged in {paths}"
    _MODEL = (single, merged, dev, ck['nbits'])
    print(f"  loaded {len(single)} single + {len(merged)} merged on {dev}")
    return _MODEL

def fit_rankers(npz_path):
    """The RANKERS fit from phase1, parameterised by which matrix it reads."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    z = np.load(npz_path)
    out = []
    for w1 in CFG.W1_PRIORS:
        W = np.where(z['M'] == 0, w1, 1.0 - w1)
        for sd in CFG.SEEDS:
            m = HistGradientBoostingClassifier(random_state=sd, **CFG.GBM)
            m.fit(z['X'], z['Y'], sample_weight=W)
            out.append(m)
    print(f"  {os.path.basename(npz_path)}: {len(out)} GBMs on {z['X'].shape[0]:,} x {z['X'].shape[1]}")
    assert z['X'].shape[1] == NFEAT, f"matrix has {z['X'].shape[1]} features, rank_features makes {NFEAT}"
    return out

def proba(rankers, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in rankers], axis=0)''')

code('''# ===================================================================================
#  Ban-aware channels (identical to Stage C)
# ===================================================================================
def _drop(cand, ban_rows, ban_keys, ik_arr):
    if cand.size == 0 or (not len(ban_rows) and not len(ban_keys)): return cand
    m = np.ones(len(cand), bool)
    if len(ban_rows): m &= ~np.isin(cand, ban_rows)
    if len(ban_keys): m &= ~np.isin(ik_arr[cand], ban_keys)
    return cand[m]

def lib_sim_ban(L, specs, target, ban_rows=(), ban_keys=()):
    cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6)
    cand = _drop(cand, np.asarray(ban_rows, np.int64), np.asarray(ban_keys, dtype=object), L['ik'])
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
        out[j] = max((explain_score(f, a, b, mode=mode, tol=CFG.MZ_TOL) for a, b in peaks), default=0.0)
    return out
print("ban-aware channels ready")''')

code('''# ===================================================================================
#  Load the engine and the validation set
# ===================================================================================
pool = build_candidate_pool()
L = load_library(TRAIN)
rep, rep_key, rep_nm, rep_pos = build_rep(L)

te = pq.read_table(SD.VAL_PARQUET).to_pandas()
te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id'))

key = pickle.load(open(VAL_KEY_PATH, 'rb'))
TRUTH  = {k['molecule_id']: k['inchikey14'] for k in key}
COHORT = {k['molecule_id']: k['cohort']     for k in key}
BAN    = {k['molecule_id']: (np.asarray(k['ban_rows'], np.int64),
                             np.asarray(k['ban_keys'], dtype=object)) for k in key}
print(f"[INFO] {len(mols):,} molecules | cohorts {pd.Series(COHORT).value_counts().to_dict()}")

# This is the blind spot to keep in view: every val truth comes from train.parquet, which is
# inside the pool, so pool presence is ~1.0 by construction. Real Class-3 test molecules are
# not in the pool at all and score 0 regardless of model quality. Val MRR will always read
# above the leaderboard; the gap is the pool miss rate, and nothing here can measure it.
in_pool = pd.Series({m: (TRUTH[m] in pool.k2i) for m, _ in mols})
print("truth-in-pool rate by cohort:")
print(pd.DataFrame({'in_pool': in_pool, 'cohort': pd.Series(COHORT)})
      .groupby('cohort').in_pool.mean().round(4).to_string())''')

code('''# ===================================================================================
#  Leakage assertion - the same check the validation-set notebook makes
# ===================================================================================
mid1 = next(m for m in COHORT if COHORT[m] == 'C2')
sub1 = te[te.molecule_id == mid1]
specs1 = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub1.itertuples()]
tgt1 = float(np.median(sub1.nm.values[np.isfinite(sub1.nm.values)])) / (1.0 + CFG.PPM_CENTER / 1e6)
br1, bk1 = BAN[mid1]
t1 = TRUTH[mid1]
banned   = lib_sim_ban(L, specs1, tgt1, br1, bk1).get(t1, 0.0)
unbanned = lib_sim_ban(L, specs1, tgt1, [], []).get(t1, 0.0)
print(f"Class-2 self-match  with ban: {banned:.4f}   (must be ~0)")
print(f"Class-2 self-match  no  ban: {unbanned:.4f}   (leakage if this were used)")
assert banned < 0.2, "ban filter is not working - every number below would be meaningless"
print("ban filter verified")''')

code('''# ===================================================================================
#  PASS 1 - channels 1/2/3, computed once and cached (config-independent)
# ===================================================================================
t0 = time.time()
CACHE = {}
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    if len(nms) == 0:
        CACHE[mid] = None; continue
    target = float(np.median(nms)) / (1.0 + CFG.PPM_CENTER / 1e6)
    specs  = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
    br, bk = BAN[mid]

    lib_hits = lib_sim_ban(L, specs, target, br, bk)
    analogs  = analog_sim_ban(L, specs, target, rep, rep_key, rep_nm, br, bk)
    ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))

    cand = pool.window(target, CFG.PPM_WIN)
    if len(cand) == 0: cand = pool.window(target, CFG.PPM_FALLBACK)

    ids, sims = [], []
    for k, s in analogs:
        i = pool.k2i.get(k, -1)
        if i >= 0: ids.append(i); sims.append(s)

    ent = dict(target=target, cand=cand, ion=ion_mode, ids=np.array(ids, np.int64),
               sims=np.array(sims, np.float32),
               lv=np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32),
               fsc=frag_scores_fast([pool.smiles[c] for c in cand], specs, ion_mode) if len(cand) else np.zeros(0, np.float32),
               lib_hits=lib_hits, tiers={})

    # Whether stage B fires depends only on the number of DISTINCT keys in the tight window,
    # which is independent of the model and the ranker -- so its inputs can be cached too.
    if getattr(CFG, 'EXPAND_BACKFILL', True) and len(set(pool.keys[cand].tolist())) < SD.TOPN:
        for tier in CFG.PPM_EXPAND[1:]:
            wide = pool.window(target, tier)
            new_cand = wide[~np.isin(wide, cand)]
            if len(new_cand) == 0: continue
            ent['tiers'][tier] = dict(
                cand=new_cand,
                lv=np.array([lib_hits.get(pool.keys[c], 0.0) for c in new_cand], np.float32),
                fsc=frag_scores_fast([pool.smiles[c] for c in new_cand], specs, ion_mode))
            if len(set(pool.keys[np.concatenate([cand, new_cand])].tolist())) >= SD.TOPN:
                break
    CACHE[mid] = ent
    if (gi + 1) % 50 == 0 or gi == len(mols) - 1:
        el = time.time() - t0
        print(f"  pass1 {gi+1}/{len(mols)} | {el:.0f}s | eta {(len(mols)-gi-1)*el/(gi+1)/60:.0f} min", flush=True)
print(f"[INFO] pass 1 done in {(time.time()-t0)/60:.1f} min")''')

code('''# ===================================================================================
#  PASS 2 - Channel-4 logits, once per model set
# ===================================================================================
ZLOG = {}
for tag, paths in (('old', OLD_PT), ('new', NEW_PT)):
    print(f"[{tag}]"); load_models(paths)
    t0 = time.time(); z = {}
    for gi, (mid, sub) in enumerate(mols):
        z[mid] = model_logits(sub) if CACHE[mid] is not None else None
        if (gi + 1) % 200 == 0:
            print(f"  logits {gi+1}/{len(mols)} | {time.time()-t0:.0f}s", flush=True)
    ZLOG[tag] = z
    print(f"  {tag} logits done in {time.time()-t0:.0f}s")''')

code('''# ===================================================================================
#  PASS 3 - assemble features and score, four ways
# ===================================================================================
RANKERS = {'old': fit_rankers(OLD_NPZ[0]), 'new': fit_rankers(NEW_NPZ[0])}

def score_set(cand, lv, fsc, ent, zlog, rankers):
    cfp = pool.fps(cand)
    afp = pool.fps(ent['ids']) if len(ent['ids']) else None
    X = rank_features(cfp, lv, afp, ent['sims'], zlog, fsc)[:, :NFEAT]
    return np.argsort(-proba(rankers, X))

def rank_of_truth(mid, mtag, rtag):
    ent = CACHE[mid]
    if ent is None: return 0
    truth = TRUTH[mid]
    zlog = ZLOG[mtag][mid]
    rk = RANKERS[rtag]
    keys_out, seen = [], set()

    def take(order, cand):
        for i in order:
            k = pool.keys[cand[i]]
            if k is not None and k not in seen and len(str(k)) >= 14:
                seen.add(k); keys_out.append(k)
                if len(keys_out) == SD.TOPN: return True
        return False

    if len(ent['cand']):
        if take(score_set(ent['cand'], ent['lv'], ent['fsc'], ent, zlog, rk), ent['cand']):
            return (keys_out.index(truth) + 1) if truth in keys_out else 0
    for tier, t in ent['tiers'].items():                       # stage B: append-only
        if len(keys_out) >= SD.TOPN: break
        if take(score_set(t['cand'], t['lv'], t['fsc'], ent, zlog, rk), t['cand']):
            break
    if len(keys_out) < SD.TOPN:                                # stage C: mass-nearest top-up
        wide = pool.window(ent['target'], getattr(CFG, 'BACKFILL_PPM', 120.0))
        if len(wide):
            for c in wide[np.argsort(np.abs(pool.mass[wide] - ent['target']))]:
                k = pool.keys[c]
                if k is not None and k not in seen and len(str(k)) >= 14:
                    seen.add(k); keys_out.append(k)
                    if len(keys_out) == SD.TOPN: break
    return (keys_out.index(truth) + 1) if truth in keys_out else 0

CONFIGS = {'base': ('old','old'), 'model_only': ('new','old'),
           'matrix_only': ('old','new'), 'both': ('new','new')}
RES = {}
for name, (mtag, rtag) in CONFIGS.items():
    t0 = time.time()
    rows = [dict(mid=mid, cohort=COHORT[mid], rank=rank_of_truth(mid, mtag, rtag))
            for mid, _ in mols]
    d = pd.DataFrame(rows)
    d['rr'] = np.where(d['rank'] > 0, 1.0 / d['rank'], 0.0)
    RES[name] = d
    print(f"  {name:12s} MRR {d.rr.mean():.4f}  ({time.time()-t0:.0f}s)")''')

code('''# ===================================================================================
#  Report + the decision rule
# ===================================================================================
def summarise(d):
    g = d.groupby('cohort').rr
    t = pd.DataFrame({'n': g.size(), 'MRR': g.mean(),
                      'top1': d.groupby('cohort')['rank'].apply(lambda r: (r == 1).mean()),
                      'hit25': d.groupby('cohort')['rank'].apply(lambda r: (r > 0).mean()),
                      'se': g.std() / np.sqrt(g.size())})
    t.loc['ALL'] = [len(d), d.rr.mean(), (d['rank'] == 1).mean(), (d['rank'] > 0).mean(),
                    d.rr.std() / np.sqrt(len(d))]
    return t.round(4)

for name, d in RES.items():
    print(f"\\n=== {name} ===")
    print(summarise(d).to_string())

base = RES['base'].set_index('mid')
print("\\n\\n=== deltas vs base (MRR, by cohort) ===")
rowsd = []
for name in ['model_only', 'matrix_only', 'both']:
    cur = RES[name].set_index('mid')
    dd = (cur.rr - base.rr)
    r = {'config': name}
    for c in sorted(base.cohort.unique()):
        m = base.cohort == c
        r[c] = round(dd[m].mean(), 4)
    r['ALL'] = round(dd.mean(), 4)
    # paired SE: the same molecules under both configs, so difference-of-paired is the
    # right error bar, and it is much tighter than two independent SEs
    r['se_paired'] = round(dd.std() / np.sqrt(len(dd)), 4)
    rowsd.append(r)
print(pd.DataFrame(rowsd).to_string(index=False))

# ---- decision rule ----------------------------------------------------------------
b = RES['both'].set_index('mid'); dd = b.rr - base.rr
def cohort_delta(c):
    m = base.cohort == c
    v = dd[m]
    return v.mean(), v.std() / np.sqrt(max(len(v), 1))

print("\\n=== SHIP / HOLD ===")
ok = True
for c in ['C2', 'C2_hard']:
    if c not in set(base.cohort): continue
    mu, se = cohort_delta(c)
    good = mu > se
    ok &= good
    print(f"  {c:9s} delta {mu:+.4f} +- {se:.4f}  -> {'gain' if good else 'NOT above 1 SE'}")
if 'C1_npx' in set(base.cohort):
    mu, se = cohort_delta('C1_npx')
    good = mu > -se
    ok &= good
    print(f"  {'C1_npx':9s} delta {mu:+.4f} +- {se:.4f}  -> {'no regression' if good else 'REGRESSED past 1 SE'}")
print(f"\\n  {'SHIP `both`' if ok else 'HOLD - the gain is not separable from noise'}")

pd.concat([d.assign(config=n) for n, d in RES.items()]).to_csv(
    os.path.join(SD.OUT, 'stage_d_results.csv'), index=False)
print(f"\\nper-molecule results -> {SD.OUT}/stage_d_results.csv")''')

out = {"cells": cells,
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12"}},
       "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT.name}: {len(cells)} cells")
