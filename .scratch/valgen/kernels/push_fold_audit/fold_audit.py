"""FPNet fold audit (Kaggle CPU): which training libraries did fpnet_0.pt train on?

There is no split file for fpnet_0. An agent-written doc claims it held out the folds np
(enveda-np-examples), nplib (gnps, riken, massbank, mona), syn (enveda-180, drug_plus), plusk (pluskal_ms2)
and twin, i.e. trained only on spectraverse / msdial / masaryk. A model ranks its own training structures far
better than unseen ones, so per-library self-retrieval of fpnet_0, relative to fpnet_full1 (trained on ALL
train folds), reveals the real map.

Per library: up to N_PER_LIB structures found ONLY in that library (enveda-np-examples has none, so its
structures are sampled anyway and flagged `shared`), one spectrum each (prefer [M+H]+). Each spectrum's truth is
ranked by f.z among every pool candidate within +-10 ppm of the truth's exact mass -- the window
casmi.traindata.window_mrr uses -- for both models.

DECISION RULE (pre-registered, fixed before any run), on MRR@25 per library:
    trained_on = yes      if MRR(fpnet_0) >= 0.8 * MRR(full1)
                 no       if MRR(fpnet_0) <= 0.5 * MRR(full1)
                 unclear  otherwise (or full1 MRR == 0, or n == 0)
Caveat: the `twin` fold (scaffold relatives) is not library-shaped and cannot be read off this table.

Reuses (casmi_v4 engine code, same package the 0.433 notebook loads as V4CODE):
  casmi.fpnet.prep_peaks / ModelBank (collate, load_model) -- the model input and logits
  casmi.spectra.clean_peaks -- the spectrum-cache cleaning casmi.build.build_spec_cache applies first
  casmi.chem.adduct_index / neutral_mass / raw_fingerprint / mol_props -- adducts, masses, missing-truth fps
  casmi.pool.Pool -- window(), fps() over the train+COCONUT pool

usage (Kaggle CPU kernel): python fold_audit.py [--n 150] [--out /kaggle/working]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

LIBS = ['enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse',
        'msdial', 'drug_plus', 'enveda-np-examples', 'masaryk']
N_PER_LIB = 150
SEED = 0
PPM = 10.0
TOPK = 25
YES_RATIO, NO_RATIO = 0.8, 0.5          # pre-registered, see module docstring
PREF_ADDUCT = '[M+H]+'
COMP_TRAIN = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
META_COLS = ['ingest_lib', 'inchikey14', 'adduct']
SPEC_COLS = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'adduct', 'ionization_mode', 'precursor_mz',
             'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities']


def log(*a):
    print(*a, flush=True)


def find(pattern, root='/kaggle/input'):
    """Shortest path matching root/**/pattern (raises if none)."""
    hits = sorted(glob.glob(f'{root}/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def import_casmi(code_dir):
    """Put the casmi_v4 engine code first on sys.path and import the modules this audit reuses."""
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from casmi import chem, fpnet, pool, spectra
    return chem, fpnet, pool, spectra


def verdict(m0, m1):
    """Pre-registered decision rule on MRR@25 (fpnet_0, full1)."""
    if not (m1 > 0) or not np.isfinite(m0):
        return 'unclear'
    if m0 >= YES_RATIO * m1:
        return 'yes'
    if m0 <= NO_RATIO * m1:
        return 'no'
    return 'unclear'


# ------------------------------------------------------------------------------------------------
# sampling
# ------------------------------------------------------------------------------------------------
def sample_structures(meta, n=N_PER_LIB, seed=SEED, libs=LIBS):
    """meta: DataFrame(row, ingest_lib, inchikey14, adduct). Returns one spectrum row per sampled structure:
    DataFrame(row, lib, inchikey14, adduct, shared). Structures are library-unique where the library has any;
    otherwise (enveda-np-examples) sampled from all its structures and flagged shared=True."""
    m = meta[meta.inchikey14.notna()]
    nlib = m.groupby('inchikey14').ingest_lib.nunique()
    out = []
    for lib in libs:
        sub = m[m.ingest_lib == lib]
        if sub.empty:
            continue
        structs = np.array(sorted(sub.inchikey14.unique()))
        uniq = structs[nlib.reindex(structs).values == 1]
        shared = len(uniq) == 0
        pool_ = structs if shared else uniq
        rng = np.random.default_rng(seed)
        pick = rng.choice(pool_, size=min(n, len(pool_)), replace=False)
        rows = sub[sub.inchikey14.isin(set(pick))].copy()
        rows['_np'] = (rows.adduct != PREF_ADDUCT).astype(np.int8)     # 0 for [M+H]+ -> sorts first
        rows = rows.sort_values(['inchikey14', '_np', 'row']).drop_duplicates('inchikey14')
        rows = rows.assign(lib=lib, shared=shared)[['row', 'lib', 'inchikey14', 'adduct', 'shared']]
        out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=['row', 'lib', 'inchikey14', 'adduct', 'shared'])


def read_meta(train_path):
    import pyarrow.parquet as pq
    t = pq.read_table(train_path, columns=META_COLS)
    df = t.to_pandas()
    df.insert(0, 'row', np.arange(len(df), dtype=np.int64))
    return df


def read_rows(train_path, rows, columns=SPEC_COLS):
    """Only the requested global rows (sorted output, column 'row' added), reading row group by row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(train_path)
    rows = np.unique(np.asarray(rows, np.int64))
    sizes = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)], np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    parts = []
    for rg in range(len(sizes)):
        sel = rows[(rows >= starts[rg]) & (rows < starts[rg + 1])]
        if len(sel) == 0:
            continue
        t = pf.read_row_group(rg, columns=columns).take(pa.array(sel - starts[rg]))
        d = t.to_pandas()
        d.insert(0, 'row', sel)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['row'] + list(columns))


# ------------------------------------------------------------------------------------------------
# model input (mirrors casmi.build.build_spec_cache + casmi.traindata.Data.item, single spectrum, no aug)
# ------------------------------------------------------------------------------------------------
def build_item(r, chem, fpnet, spectra):
    """r: mapping with precursor_mz, adduct, ionization_mode, collision_energy_ev, ms2_mzs,
    ms2_normalized_intensities. Returns the FPNet item dict."""
    prec = float(r['precursor_mz'])
    mz = np.asarray(r['ms2_mzs'], np.float64)
    it = np.asarray(r['ms2_normalized_intensities'], np.float64)
    cm, ci = spectra.clean_peaks(mz, it, prec, 0.001, 512, 2.0)          # spectrum cache cleaning
    pm, pi = fpnet.prep_peaks(cm, ci, prec)                               # Data.item
    ce_l = r['collision_energy_ev']
    ce_l = [] if ce_l is None else [float(x) for x in ce_l]
    ce = float(np.mean(ce_l)) if len(ce_l) else np.nan
    ce_n = len(ce_l)
    mode = 1 if r['ionization_mode'] == 'positive' else -1
    return dict(mz=pm, it=pi, prec=prec, adduct_ix=int(chem.adduct_index(r['adduct'])),
                ce=0.0 if np.isnan(ce) else ce, ce_known=0.0 if np.isnan(ce) else 1.0,
                n_merged=min(max(1, ce_n), 8), mode=float(mode))


# ------------------------------------------------------------------------------------------------
# truth resolution, window, ranking
# ------------------------------------------------------------------------------------------------
def truth_maps(P, train_structs=None):
    """ik14 -> pool pid. Via train_structs (inchikey14 -> sid) and pool.train_sid when available, else the
    pool score key (tautomer-canonical ik14), which equals the plain ik14 for most structures."""
    ik2pid = {}
    if train_structs is not None and {'inchikey14', 'sid'} <= set(train_structs.columns):
        sid2pid = {int(s): i for i, s in enumerate(P.train_sid) if s >= 0}
        for k, s in zip(train_structs.inchikey14.values, train_structs.sid.values):
            p = sid2pid.get(int(s), -1)
            if p >= 0:
                ik2pid[k] = p
    return ik2pid


def resolve_truth(ik14, smiles, P, ik2pid, chem, rdkit_ok):
    """Returns (pid or -1, extra_fp (nbits,) uint8 or None, exact mass or nan, how)."""
    p = ik2pid.get(ik14, -1)
    if p < 0:
        p = P.key2pid.get(ik14, -1)
    if p >= 0:
        return int(p), None, float(P.mass[p]), 'pool'
    if not rdkit_ok or not isinstance(smiles, str):
        return -1, None, np.nan, 'unresolved'
    fp = chem.raw_fingerprint(smiles)
    pr = chem.mol_props(smiles)
    if fp is None or pr is None:
        return -1, None, np.nan, 'unresolved'
    return -1, fp[P.bits].astype(np.uint8), float(pr[1]), 'added'


def window_ids(P, target, ppm=PPM):
    """Pool ids within +-ppm of target (casmi.pool.Pool.window)."""
    return P.window(target, ppm)


def candidate_set(P, cand, truth_pid, extra_fp):
    """(F (n,nbits) float32, truth_row, keys). Truth is guaranteed present: appended if it is not a pool row
    of the window."""
    cand = np.asarray(cand, np.int64)
    F = P.fps(cand).astype(np.float32) if len(cand) else np.zeros((0, P.nbits), np.float32)
    keys = list(P.key[cand]) if len(cand) else []
    if truth_pid >= 0:
        hit = np.flatnonzero(cand == truth_pid)
        if len(hit):
            return F, int(hit[0]), keys
        f = P.fps([truth_pid]).astype(np.float32)
        return np.vstack([F, f]), len(cand), keys + [P.key[truth_pid]]
    f = extra_fp[None, :].astype(np.float32)
    return np.vstack([F, f]), len(cand), keys + [None]


def rank_of_truth(scores, t, keys=None):
    """1 + number of candidates scoring strictly higher than the truth (casmi.traindata.window_mrr), ignoring
    other rows carrying the truth's own key (duplicates would only waste slots, not be wrong)."""
    s = np.asarray(scores, np.float64)
    higher = s > s[t]
    if keys is not None and keys[t] is not None:
        same = np.asarray(keys, dtype=object) == keys[t]
        same[t] = False
        higher &= ~same
    return int(higher.sum()) + 1


def rr_at(rank, k=TOPK):
    return 1.0 / rank if rank <= k else 0.0


# ------------------------------------------------------------------------------------------------
# pipeline
# ------------------------------------------------------------------------------------------------
def audit(spec, P, models, chem, fpnet, spectra, ik2pid, rdkit_ok, bs=32):
    """spec: DataFrame of sampled rows with SPEC_COLS + lib/shared. models: {name: callable(items)->(n,nbits)}.
    Returns per-query DataFrame."""
    t0 = time.time()
    items = [build_item(r, chem, fpnet, spectra) for r in spec.to_dict('records')]
    log(f'[audit] {len(items)} items built {time.time()-t0:.1f}s')
    Z = {}
    for name, fn in models.items():
        t1 = time.time()
        Z[name] = np.asarray(fn(items), np.float32)
        assert Z[name].shape == (len(items), P.nbits), (name, Z[name].shape, P.nbits)
        log(f'[audit] logits {name}: {time.time()-t1:.1f}s ({(time.time()-t1)/max(1,len(items))*1e3:.0f} ms/spec)')
    nm = chem.neutral_mass(spec.precursor_mz.values, spec.adduct.values)
    recs = []
    t2 = time.time()
    for i, r in enumerate(spec.to_dict('records')):
        pid, efp, mass, how = resolve_truth(r['inchikey14'], r.get('normalized_smiles'), P, ik2pid, chem, rdkit_ok)
        rec = dict(lib=r['lib'], shared=bool(r['shared']), row=int(r['row']), inchikey14=r['inchikey14'],
                   adduct=r['adduct'], truth=how, n_peaks=len(items[i]['mz']),
                   prec_ppm=(nm[i] - mass) / mass * 1e6 if np.isfinite(mass) and np.isfinite(nm[i]) else np.nan)
        if how == 'unresolved':
            recs.append(rec)
            continue
        cand = window_ids(P, mass, PPM)
        F, t, keys = candidate_set(P, cand, pid, efp)
        rec['n_cand'] = len(F)
        for name in models:
            rk = rank_of_truth(F @ Z[name][i], t, keys)
            rec[f'rank_{name}'] = rk
        recs.append(rec)
    log(f'[audit] ranking {time.time()-t2:.1f}s')
    return pd.DataFrame(recs)


def summarize(df, names=('fpnet_0', 'full1')):
    a, b = names
    rows = []
    for lib in [l for l in LIBS if l in set(df.lib)] + sorted(set(df.lib) - set(LIBS)):
        d = df[(df.lib == lib) & (df.truth != 'unresolved')]
        n = len(d)
        out = dict(lib=lib, n=n, shared=bool(df[df.lib == lib].shared.any()),
                   n_added=int((d.truth == 'added').sum()),
                   n_unresolved=int(((df.lib == lib) & (df.truth == 'unresolved')).sum()),
                   med_cand=float(d.n_cand.median()) if n else np.nan)
        for nm_ in names:
            rk = d[f'rank_{nm_}'].values if n else np.zeros(0)
            out[f'top1_{nm_}'] = float((rk == 1).mean()) if n else np.nan
            out[f'mrr_{nm_}'] = float(np.mean([rr_at(int(x)) for x in rk])) if n else np.nan
        out['gap'] = out[f'mrr_{b}'] - out[f'mrr_{a}']
        out['ratio'] = out[f'mrr_{a}'] / out[f'mrr_{b}'] if n and out[f'mrr_{b}'] > 0 else np.nan
        out['trained_on'] = verdict(out[f'mrr_{a}'], out[f'mrr_{b}']) if n else 'unclear'
        rows.append(out)
    return pd.DataFrame(rows)


def report(summ, names=('fpnet_0', 'full1')):
    a, b = names
    log('\n' + '=' * 110)
    log(f'{"lib":<20}{"n":>5}{"shared":>7}{"added":>6}{"medC":>7}'
        f'{"top1_"+a:>13}{"top1_"+b:>12}{"mrr_"+a:>12}{"mrr_"+b:>11}{"gap":>8}{"ratio":>7}  trained_on')
    for r in summ.to_dict('records'):
        log(f'{r["lib"]:<20}{r["n"]:>5}{str(r["shared"]):>7}{r["n_added"]:>6}{r["med_cand"]:>7.0f}'
            f'{r["top1_"+a]:>13.3f}{r["top1_"+b]:>12.3f}{r["mrr_"+a]:>12.3f}{r["mrr_"+b]:>11.3f}'
            f'{r["gap"]:>8.3f}{r["ratio"]:>7.2f}  {r["trained_on"]}')
    log('=' * 110)
    log(f'rule: yes if mrr_{a} >= {YES_RATIO} x mrr_{b}; no if <= {NO_RATIO} x; unclear otherwise')
    for r in summ.to_dict('records'):
        log(f'FOLD AUDIT {r["lib"]} trained_on={r["trained_on"]}')


def ensure_rdkit():
    """True if RDKit imports; on Kaggle installs the metric 2026.3.3 wheel offline if missing."""
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        pass
    import subprocess
    tag = 'cp%d%d' % sys.version_info[:2]
    whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
    whl.sort(key=lambda w: ('2026.3.3' not in w, w))
    if not whl:
        return False
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '-q', whl[0]])
    try:
        import rdkit  # noqa: F401
        return r.returncode == 0
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=COMP_TRAIN)
    ap.add_argument('--code', default=None, help='casmi_v4 code dir (parent of casmi/)')
    ap.add_argument('--fpnet0', default=None)
    ap.add_argument('--full1', default=None)
    ap.add_argument('--pool', default=None, help='dir holding pool_meta.parquet')
    ap.add_argument('--n', type=int, default=N_PER_LIB)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default='/kaggle/working')
    a, _ = ap.parse_known_args(argv)
    T0 = time.time()
    if not os.path.exists(a.train):
        a.train = find('train.parquet')
    code = a.code or os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/fpnet.py')))
    f0 = a.fpnet0 or find('casmi26-v4b-models/**/fpnet_0.pt')   # the 0.433 notebook's V4MOD/fpnet_0.pt
    f1 = a.full1 or find('casmi26-fpnet-full1/**/fpnet_full1.pt')
    pool_dir = a.pool or os.path.dirname(find('pool_meta.parquet'))
    log('V4CODE', code, '\nfpnet_0', f0, '\nfull1', f1, '\nPOOL', pool_dir, '\nTRAIN', a.train)
    chem, fpnet, poolmod, spectra = import_casmi(code)
    import casmi
    log('casmi from', casmi.__file__)
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f'torch {torch.__version__} threads {torch.get_num_threads()} cuda={torch.cuda.is_available()} (CPU forced)')

    t = time.time()
    P = poolmod.Pool(pool_dir, load_frag=False)
    ts_path = os.path.join(pool_dir, 'train_structs.parquet')
    ts = None
    if os.path.exists(ts_path):
        import pyarrow.parquet as pq
        cols = [c for c in ('inchikey14', 'sid') if c in pq.ParquetFile(ts_path).schema_arrow.names]
        ts = pd.read_parquet(ts_path, columns=cols)
    ik2pid = truth_maps(P, ts)
    log(f'[pool] {len(P.mass):,} structs nbits {P.nbits}; train ik14->pid {len(ik2pid):,} ({time.time()-t:.0f}s)')

    t = time.time()
    meta = read_meta(a.train)
    log(f'[train] meta {len(meta):,} rows ({time.time()-t:.0f}s)')
    samp = sample_structures(meta, a.n, SEED)
    del meta
    log('[sample] per lib:', samp.groupby('lib').size().to_dict(), '| shared libs:',
        sorted(samp[samp.shared].lib.unique()))
    t = time.time()
    spec = read_rows(a.train, samp.row.values)
    spec = spec.drop(columns=['ingest_lib']).merge(samp[['row', 'lib', 'shared']], on='row')
    assert (spec.inchikey14.values == samp.set_index('row').loc[spec.row, 'inchikey14'].values).all()
    log(f'[train] {len(spec)} spectra read ({time.time()-t:.0f}s)')

    banks = {}
    for name, path in (('fpnet_0', f0), ('full1', f1)):
        b = fpnet.ModelBank([path], device='cpu')
        assert b.nbits == P.nbits, f'{name} nbits {b.nbits} != pool nbits {P.nbits}'
        banks[name] = (lambda items, b=b: b.logits(items, bs=a.bs))
    rdkit_ok = ensure_rdkit()
    log('rdkit available for missing truths:', rdkit_ok)

    df = audit(spec, P, banks, chem, fpnet, spectra, ik2pid, rdkit_ok, bs=a.bs)
    n_add = int((df.truth == 'added').sum()); n_unr = int((df.truth == 'unresolved').sum())
    log(f'[truth] in pool {int((df.truth == "pool").sum())}, missing->added {n_add}, unresolved (dropped) {n_unr}')
    pp = df.prec_ppm.abs()
    log(f'[window] centred on truth exact mass; |precursor-derived ppm| > {PPM}: {int((pp > PPM).sum())}/{pp.notna().sum()}')
    summ = summarize(df)
    report(summ)
    os.makedirs(a.out, exist_ok=True)
    summ.to_csv(os.path.join(a.out, 'fold_audit.csv'), index=False)
    df.to_csv(os.path.join(a.out, 'fold_audit_queries.csv'), index=False)
    log(f'wrote {a.out}/fold_audit.csv; total {time.time()-T0:.0f}s')
    return summ


if __name__ == '__main__':
    main()
