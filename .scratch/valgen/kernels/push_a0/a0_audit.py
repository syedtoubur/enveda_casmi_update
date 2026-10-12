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


# ------------------------------------------------------------------------------------------------
# A0 (Option A, 2026-10-11): did fpnet_0 hold out the 250 enveda-np-examples STRUCTURES, or only their spectra?
# Query sets: 'np' = every enveda-np-examples spectrum; 'other' = the same structures' spectra in OTHER
# libraries (<= k_other per structure, distinct libraries first); 'control' = library-unique structures drawn
# from the other set's libraries in proportion. Each model is ranked by f.z in the same +-10 ppm pool window.
# PRE-REGISTERED (research/harness/gate_report.md "Option A0"), on MRR@25 ratio fpnet_0 / full1 over 'other':
#     <= 0.8 -> 'structures' held out (build C2_np);  >= 0.9 -> 'spectra' only (build C1_np_fp0);  else 'stop'.
# Any other model with ratio >= 0.9 x full1 on 'np' = declared residual leak (reported, not blocking).
# ------------------------------------------------------------------------------------------------
NP_LIB = 'enveda-np-examples'
A0_STRUCT, A0_SPEC = 0.8, 0.9
LEAK_RATIO = 0.9
QSETS = ('np', 'other', 'control')
A0_COLS = SPEC_COLS + ['instrument_type']
DEFAULT_MODE = 'a0'
PV_FP_SRC = '"""Spectrum -> fingerprint model (prvsiyan FPNet), kept apart so CPU worker processes never import torch."""\nimport math\nimport numpy as np\n# ----------------------------------------------------------------------------------------------\n# spectrum -> fingerprint model\n# ----------------------------------------------------------------------------------------------\nimport torch, torch.nn as nn, torch.nn.functional as F\n\nFP_MAX_PEAKS = 128\nADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",\n               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",\n               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",\n               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]\nADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}\nINSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]\n\n\ndef instr_family(s):\n    if s is None: return 4\n    t = str(s).lower()\n    if \'timstof\' in t: return 0\n    if \'orbitrap\' in t or \'qft\' in t or \'ftms\' in t or \'hybrid ft\' in t or \'itft\' in t or \'exactive\' in t: return 1\n    if \'tof\' in t: return 2\n    if \'trap\' in t or \'qq\' in t: return 3\n    return 4\n\n\ndef prep_peaks(mz, inten, prec_mz, max_peaks=FP_MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):\n    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = (mz <= prec_mz + 1.5)\n    mz, it = mz[keep], it[keep]\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    mx = it.max()\n    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = it >= floor * mx\n    mz, it = mz[keep], it[keep]\n    if len(mz) > max_peaks:\n        order = np.argsort(-it)\n        bucket = (mz // win).astype(np.int64)\n        cnt = {}; sel = []\n        for i in order:\n            b = bucket[i]; c = cnt.get(b, 0)\n            if c < per_win: cnt[b] = c + 1; sel.append(i)\n        sel = np.array(sel)\n        if len(sel) > max_peaks:\n            sel = sel[np.argsort(-it[sel])[:max_peaks]]\n        elif len(sel) < max_peaks:\n            rest = np.array([i for i in order if i not in set(sel.tolist())])\n            need = max_peaks - len(sel)\n            if len(rest): sel = np.concatenate([sel, rest[:need]])\n        mz, it = mz[sel], it[sel]\n    o = np.argsort(mz)\n    mz, it = mz[o], it[o]\n    v = np.sqrt(it / it.max())\n    return mz.astype(np.float32), v.astype(np.float32)\n\n\nclass SinEmb(nn.Module):\n    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):\n        super().__init__()\n        n = dim // 2\n        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)\n        self.register_buffer(\'inv\', (2 * math.pi) / wav)\n\n    def forward(self, x):\n        a = x.unsqueeze(-1) * self.inv\n        return torch.cat([torch.sin(a), torch.cos(a)], -1)\n\n\nclass Block(nn.Module):\n    def __init__(self, d, h, drop):\n        super().__init__(); self.h = h\n        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)\n        self.n2 = nn.LayerNorm(d)\n        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))\n        self.drop = nn.Dropout(drop)\n\n    def forward(self, x, pad):\n        B, N, D = x.shape; y = self.n1(x)\n        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)\n        m = (~pad)[:, None, None, :]\n        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)\n        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))\n        return x + self.drop(self.ff(self.n2(x)))\n\n\nclass FPNet(nn.Module):\n    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):\n        super().__init__()\n        self.d = d\n        self.mz_emb = SinEmb(d)\n        self.nl_emb = SinEmb(d)\n        self.pk = nn.Linear(2 * d + 1, d)\n        self.prec_emb = SinEmb(d)\n        self.ad = nn.Embedding(len(ADDUCT_LIST), d)\n        self.ins = nn.Embedding(len(INSTR_LIST), d)\n        self.gl = nn.Linear(d + 3, d)\n        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])\n        self.norm = nn.LayerNorm(d)\n        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))\n\n    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):\n        B, N = mz.shape\n        nl = (prec[:, None] - mz).clamp(min=0)\n        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))\n        g = self.gl(torch.cat([self.prec_emb(prec),\n                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),\n                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)\n        x = torch.cat([g.unsqueeze(1), p], 1)\n        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)\n        for b in self.blocks: x = b(x, pad)\n        x = self.norm(x)\n        cls = x[:, 0]\n        msk = (~pad[:, 1:]).float().unsqueeze(-1)\n        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)\n        return self.head(torch.cat([cls, mean], -1))\n\n\ndef load_fp_models(paths, dev):\n    single, merged = [], []\n    nbits = None\n    for pth in paths:\n        ck = torch.load(pth, map_location=\'cpu\', weights_only=False)\n        net = FPNet(ck[\'nbits\'], d=ck[\'d\'], layers=ck[\'layers\']).to(dev).eval()\n        net.load_state_dict(ck[\'model\'])\n        (merged if \'merged\' in str(pth).replace(\'\\\\\', \'/\').split(\'/\')[-1] else single).append(net)\n        nbits = ck[\'nbits\']\n    return single, merged, nbits\n\n\ndef merge_peaks(specs):\n    """specs: list of (mz, it). All peaks collapsed into one pseudo-spectrum."""\n    specs = [(a, b) for a, b in specs if len(a)]\n    if not specs: return np.zeros(0), np.zeros(0)\n    mz = np.concatenate([np.asarray(a, float) for a, _ in specs])\n    it = np.concatenate([np.asarray(b, float) / max(float(np.asarray(b, float).max()), 1e-9) for _, b in specs])\n    o = np.argsort(mz); mz, it = mz[o], it[o]\n    keep = np.ones(len(mz), bool)\n    for j in range(1, len(mz)):\n        if mz[j] - mz[j - 1] < 0.005:\n            if it[j] >= it[j - 1]: keep[j - 1] = False\n            else: keep[j] = False\n    return mz[keep], it[keep]\n\n\n@torch.no_grad()\ndef logits_batch(P, nets, dev, prec, adduct, instr, ce, mode):\n    """P: list of (mz, it) already prepped; per-row arrays prec/adduct(str)/instr(str)/ce/mode."""\n    keep = [i for i, (a, _) in enumerate(P) if len(a)]\n    if not keep: return None\n    P = [P[i] for i in keep]\n    B = len(P); N = max(len(a) for a, _ in P)\n    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)\n    for i, (a, b) in enumerate(P):\n        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False\n    T = lambda x: torch.as_tensor(x, device=dev)\n    args = (T(mz), T(it), T(pad),\n            T(np.asarray(prec, np.float32)[keep]),\n            T(np.array([ADDUCT_IX.get(a, ADDUCT_IX[\'<unk>\']) for a in np.asarray(adduct, object)[keep]])),\n            T(np.array([instr_family(s) for s in np.asarray(instr, object)[keep]])),\n            T(np.asarray(ce, np.float32)[keep]),\n            T(np.asarray(mode, np.float32)[keep]))\n    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)\n\n\ndef molecule_logits(models, specs, prec, adduct, instr, ce, mode):\n    """prvsiyan model_logits(): mean of (single-model per-spectrum mean) and (merged-model on merged list).\n    specs: list of (mz, it); prec/adduct/instr/ce/mode: per-spectrum."""\n    single, merged, dev = models\n    out = []\n    if single:\n        P = [prep_peaks(a, b, p) for (a, b), p in zip(specs, prec)]\n        za = logits_batch(P, single, dev, prec, adduct, instr, ce, mode)\n        if za is not None: out.append(za)\n    if merged:\n        mz, it = merge_peaks(specs)\n        pm = float(np.median(prec))\n        P = [prep_peaks(mz, it, pm)]\n        zb = logits_batch(P, merged, dev, [pm], [adduct[0]], [instr[0]], [25.0], [float(np.mean(mode))])\n        if zb is not None: out.append(zb)\n    return np.mean(out, axis=0) if out else None'   # 0.433 base notebook ENG_FILES['pv_fp.py'], staged                      # the A0 kernel staging sets this to 'a0'


def a0_decision(ratio):
    if ratio is None or not np.isfinite(ratio):
        return 'stop'
    if ratio <= A0_STRUCT:
        return 'structures'
    if ratio >= A0_SPEC:
        return 'spectra'
    return 'stop'


def sample_np_sets(meta, k_other=4, n_control=150, seed=SEED):
    """meta: DataFrame(row, ingest_lib, inchikey14, adduct) -> DataFrame(row, lib, inchikey14, adduct, qset)."""
    m = meta[meta.inchikey14.notna()].copy()
    m['_np'] = (m.adduct != PREF_ADDUCT).astype(np.int8)
    npm = m[m.ingest_lib == NP_LIB]
    nps = set(npm.inchikey14)
    parts = [npm.assign(lib=NP_LIB, qset='np')]
    o = m[m.inchikey14.isin(nps) & (m.ingest_lib != NP_LIB)].sort_values(['inchikey14', 'ingest_lib', '_np', 'row'])
    o = o.assign(_k=o.groupby(['inchikey14', 'ingest_lib']).cumcount())
    o = o.sort_values(['inchikey14', '_k', 'ingest_lib', 'row']).groupby('inchikey14').head(k_other)
    parts.append(o.assign(lib=o.ingest_lib, qset='other'))
    cnt = o.ingest_lib.value_counts().sort_index()
    if len(cnt) and n_control > 0:
        q = cnt / cnt.sum() * n_control
        quota = np.floor(q).astype(int)
        rem = (q - quota).sort_values(ascending=False, kind='mergesort')
        for lib in rem.index[:n_control - int(quota.sum())]:
            quota[lib] += 1
        nlib = m.groupby('inchikey14').ingest_lib.nunique()
        for lib, k in quota.items():
            sub = m[(m.ingest_lib == lib) & ~m.inchikey14.isin(nps)]
            structs = np.array(sorted(sub.inchikey14.unique()))
            structs = structs[nlib.reindex(structs).values == 1]
            if k <= 0 or not len(structs):
                continue
            pick = np.random.default_rng(seed).choice(structs, size=min(int(k), len(structs)), replace=False)
            rows = sub[sub.inchikey14.isin(set(pick))].sort_values(['inchikey14', '_np', 'row'])
            rows = rows.drop_duplicates('inchikey14')
            parts.append(rows.assign(lib=lib, qset='control'))
    out = pd.concat(parts, ignore_index=True)[['row', 'lib', 'inchikey14', 'adduct', 'qset']]
    return out.reset_index(drop=True)


def summarize_sets(df, names, ref='full1'):
    """Per (qset, model): n, top1, mrr@25, ratio = mrr / mrr(ref) on the same rows, leak (np only)."""
    rows = []
    for qs in [q for q in QSETS if q in set(df.qset)]:
        d = df[(df.qset == qs) & (df.truth != 'unresolved')]
        mref = np.mean([rr_at(int(x)) for x in d[f'rank_{ref}']]) if len(d) else np.nan
        for nm_ in names:
            rk = d[f'rank_{nm_}'].values
            mrr = float(np.mean([rr_at(int(x)) for x in rk])) if len(rk) else np.nan
            ratio = mrr / mref if len(rk) and mref > 0 else np.nan
            rows.append(dict(qset=qs, model=nm_, n=len(rk), top1=float((rk == 1).mean()) if len(rk) else np.nan,
                             mrr=mrr, ratio=ratio,
                             leak=bool(qs == 'np' and nm_ != ref and nm_ != 'fpnet_0' and np.isfinite(ratio)
                                       and ratio >= LEAK_RATIO)))
    return pd.DataFrame(rows)


_PVG = {}


def pv_fingerprint(smiles, bits):
    """prvsiyan's candidate fingerprint (casmi_engine.fp_and_mass of the 0.433 Engine 2): Morgan r2 4096 + Morgan
    r3 4096 + RDKit path<=6 2048 + MACCS, restricted to `bits`. None if RDKit cannot parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    if not _PVG:
        RDLogger.DisableLog('rdApp.*')
        _PVG['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _PVG['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _PVG['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        fp = np.concatenate([_PVG['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _PVG['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _PVG['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        return fp[np.asarray(bits, np.int64)]
    except Exception:
        return None


def inspect_npz(path, max_unique=50):
    """{key: {shape, dtype[, counts]}} reading only array headers; 1-D non-float arrays are loaded and value-counted
    when they have <= max_unique distinct values."""
    import zipfile
    info = {}
    with zipfile.ZipFile(path) as zf:
        for nm_ in zf.namelist():
            with zf.open(nm_) as f:
                ver = np.lib.format.read_magic(f)
                shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                   else np.lib.format.read_array_header_2_0(f))
            info[nm_[:-4] if nm_.endswith('.npy') else nm_] = dict(shape=tuple(shape), dtype=str(dtype))
    z = np.load(path, allow_pickle=True)
    for k, v in info.items():
        if len(v['shape']) == 1 and np.dtype(v['dtype']).kind in 'iubUSO':
            a = z[k]
            u, c = np.unique(a.astype(str) if a.dtype.kind == 'O' else a, return_counts=True)
            v['n_unique'] = int(len(u))
            if len(a) <= max_unique:                                   # small arrays (e.g. a code -> name table) in order
                v['values'] = [x.item() if hasattr(x, 'item') else x for x in a]
            if len(u) <= max_unique:
                v['counts'] = {(x.item() if hasattr(x, 'item') else x): int(n) for x, n in zip(u, c)}
    return info


def _pv_spec(r):
    """One train row -> (mz, it) as casmi_engine.compute_channels prepares it."""
    a = np.asarray(r['ms2_mzs'], np.float32); b = np.asarray(r['ms2_normalized_intensities'], np.float32)
    k = a <= float(r['precursor_mz']) + 2.0
    a, b = a[k], b[k]
    return a, (b / b.max() if len(b) and b.max() > 0 else b)


def _parse_ce(v):
    try:
        a = np.abs(np.atleast_1d(np.asarray(v, float)))
        return float(a.mean()) if len(a) else 25.0
    except Exception:
        return 25.0


def worker(argv):
    """Subprocess: logits of ONE model on the pickled spectra -> .npy. Own process per model so the v3 casmi
    package and prvsiyan's pv_fp never collide with the v4 casmi package."""
    import importlib
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    for k in ('--family', '--code', '--model', '--spec', '--zout', '--pvfp'):
        ap.add_argument(k, default='')
    a, _ = ap.parse_known_args(argv)
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    recs = pickle.load(open(a.spec, 'rb'))
    if a.family == 'pv':
        sys.path.insert(0, os.path.dirname(a.pvfp))
        import pv_fp
        single, merged, nbits = pv_fp.load_fp_models([a.model], 'cpu')
        Z = np.zeros((len(recs), nbits), np.float32)
        for i, r in enumerate(recs):
            mode = 1.0 if r['ionization_mode'] == 'positive' else -1.0
            z = pv_fp.molecule_logits((single, merged, 'cpu'), [_pv_spec(r)], np.array([r['precursor_mz']], np.float32),
                                      [str(r['adduct'])], [str(r.get('instrument_type'))],
                                      np.array([_parse_ce(r['collision_energy_ev'])], np.float32), np.array([mode]))
            if z is not None:
                Z[i] = z
        log(f'[worker] pv {os.path.basename(a.model)}: single {len(single)} merged {len(merged)} nbits {nbits}')
    else:
        sys.path.insert(0, a.code)
        from casmi import chem, spectra
        fp = importlib.import_module('casmi.fpnet6' if a.family == 'fpnet6' else 'casmi.fpnet')
        items = [build_item(r, chem, fp, spectra) for r in recs]
        bank = fp.ModelBank([a.model], device='cpu')
        Z = np.asarray(bank.logits(items, bs=32), np.float32)
        log(f'[worker] {a.family} {os.path.basename(a.model)} from {a.code}: nbits {bank.nbits}')
    np.save(a.zout, Z)


def subprocess_logits_factory(workdir, pvfp=''):
    import pickle
    import subprocess
    state = {}

    def fn(spec, name, path, family, code):
        if not path or not os.path.exists(path):
            return None, f'not scorable: checkpoint missing ({path})'
        if family == 'pv' and not (pvfp and os.path.exists(pvfp)):
            return None, 'not scorable: pv_fp.py source unavailable'
        if 'spec' not in state:
            state['spec'] = os.path.join(workdir, 'a0_spec.pkl')
            pickle.dump(spec[[c for c in A0_COLS if c in spec.columns]].to_dict('records'), open(state['spec'], 'wb'),
                        protocol=4)
        zout = os.path.join(workdir, f'a0_z_{name}.npy')
        t = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', '--family', family,
                            '--code', code or '', '--model', path, '--spec', state['spec'], '--zout', zout,
                            '--pvfp', pvfp], capture_output=True, text=True, timeout=3 * 3600)
        log(r.stdout[-1500:].rstrip())
        if r.returncode != 0 or not os.path.exists(zout):
            return None, 'not scorable: worker failed: ' + r.stderr[-800:].replace('\n', ' | ')
        log(f'[a0] logits {name} ({family}) {time.time()-t:.0f}s')
        return np.load(zout), None
    return fn


def run_a0(train_path, pool_dir, models, logits_fn, k_other=4, n_control=150, out='/kaggle/working', pv_bits=None,
           code=None, seed=SEED):
    """models: [(name, path, family in fpnet|fpnet6|v3|pv, code dir)]. Returns (summary, decision)."""
    code = code or next(c for _, _, f, c in models if c)
    chem, fpnet, poolmod, spectra = import_casmi(code)
    P = poolmod.Pool(pool_dir, load_frag=False)
    ts_path = os.path.join(pool_dir, 'train_structs.parquet')
    ts = pd.read_parquet(ts_path) if os.path.exists(ts_path) else None
    ik2pid = truth_maps(P, ts)
    meta = read_meta(train_path)
    samp = sample_np_sets(meta, k_other, n_control, seed)
    del meta
    log('[a0] sets:', samp.qset.value_counts().to_dict(), '| other libs:',
        samp[samp.qset == 'other'].lib.value_counts().to_dict(), '| control libs:',
        samp[samp.qset == 'control'].lib.value_counts().to_dict())
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(train_path).schema_arrow.names)
    cols = [c for c in A0_COLS if c in have]
    spec = read_rows(train_path, samp.row.values, columns=cols)
    if 'instrument_type' not in spec.columns:
        spec['instrument_type'] = None
    spec = spec.drop(columns=['ingest_lib']).merge(samp[['row', 'lib', 'qset']], on='row')
    os.makedirs(out, exist_ok=True)
    Z, skipped = {}, {}
    for name, path, family, mcode in models:
        z, err = logits_fn(spec, name, path, family, mcode)
        want = (len(pv_bits) if pv_bits is not None else None) if family == 'pv' else P.nbits
        if err is None and (want is None or z.shape != (len(spec), want)):
            err = f'not scorable: logits shape {None if z is None else z.shape} != ({len(spec)}, {want})'
        if err:
            skipped[name] = err
            log(f'A0 MODEL {name} {err}')
            continue
        Z[name] = np.asarray(z, np.float32)
    rdkit_ok = ensure_rdkit()
    pv_cache = {}

    def pv_fps(cand, extra_smiles):
        rows = []
        for c in cand:
            c = int(c)
            if c not in pv_cache:
                pv_cache[c] = pv_fingerprint(P.smiles[c], pv_bits)
            rows.append(pv_cache[c])
        if extra_smiles is not False:
            rows.append(pv_fingerprint(extra_smiles, pv_bits))
        return np.stack([r if r is not None else np.zeros(len(pv_bits), np.uint8) for r in rows]).astype(np.float32)

    nm = chem.neutral_mass(spec.precursor_mz.values, spec.adduct.values)
    recs = []
    t2 = time.time()
    for i, r in enumerate(spec.to_dict('records')):
        pid, efp, mass, how = resolve_truth(r['inchikey14'], r.get('normalized_smiles'), P, ik2pid, chem, rdkit_ok)
        rec = dict(qset=r['qset'], lib=r['lib'], row=int(r['row']), inchikey14=r['inchikey14'], adduct=r['adduct'],
                   truth=how, prec_ppm=(nm[i] - mass) / mass * 1e6 if np.isfinite(mass) and np.isfinite(nm[i]) else np.nan)
        if how == 'unresolved':
            recs.append(rec)
            continue
        cand = window_ids(P, mass, PPM)
        F, t, keys = candidate_set(P, cand, pid, efp)
        rec['n_cand'] = len(F)
        Fpv = None
        for name, z in Z.items():
            if z.shape[1] == P.nbits:
                rec[f'rank_{name}'] = rank_of_truth(F @ z[i], t, keys)
            else:
                if Fpv is None:
                    inwin = np.asarray(cand, np.int64)
                    appended = t >= len(inwin)
                    Fpv = pv_fps(inwin, (P.smiles[pid] if pid >= 0 else r.get('normalized_smiles')) if appended else False)
                rec[f'rank_{name}'] = rank_of_truth(Fpv @ z[i], t, keys)
        recs.append(rec)
    log(f'[a0] ranking {time.time()-t2:.1f}s')
    df = pd.DataFrame(recs)
    names = list(Z)
    summ = summarize_sets(df, names, ref='full1')
    df.to_csv(os.path.join(out, 'a0_queries.csv'), index=False)
    summ.to_csv(os.path.join(out, 'a0_summary.csv'), index=False)
    log('\n' + '=' * 78)
    log(f'{"qset":<9}{"model":<16}{"n":>6}{"top1":>8}{"mrr@25":>9}{"ratio":>8}  leak')
    for r in summ.to_dict('records'):
        log(f'{r["qset"]:<9}{r["model"]:<16}{r["n"]:>6}{r["top1"]:>8.3f}{r["mrr"]:>9.3f}{r["ratio"]:>8.3f}  '
            f'{"LEAK" if r["leak"] else ""}')
    log('=' * 78)
    for k, v in skipped.items():
        log(f'A0 SKIPPED {k}: {v}')
    s = summ.set_index(['qset', 'model'])
    ratio = float(s.loc[('other', 'fpnet_0'), 'ratio']) if ('other', 'fpnet_0') in s.index else np.nan
    dec = a0_decision(ratio)
    for r in summ[summ.leak].to_dict('records'):
        log(f'A0 RESIDUAL LEAK {r["model"]} np ratio {r["ratio"]:.3f}')
    log(f'A0 DECISION {dec} | fpnet_0/full1 other-library mrr ratio {ratio:.3f} '
        f'(rule: <= {A0_STRUCT} structures, >= {A0_SPEC} spectra, else stop)')
    return summ, dec


def main_a0(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=COMP_TRAIN)
    ap.add_argument('--k-other', type=int, default=4)
    ap.add_argument('--n-control', type=int, default=150)
    ap.add_argument('--out', default='/kaggle/working')
    a, _ = ap.parse_known_args(argv)
    T0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(a.train):
        a.train = find('train.parquet')
    # 1. Engine 2 ranker-row files: keys, shapes, query-set fields (for the harness ranker-row filter)
    for nm_ in ('sim_rank_rows_nofp.npz', 'rank_train.npz'):
        try:
            p = find(nm_)
            for k, v in inspect_npz(p).items():
                log(f'NPZ {nm_} {k} shape {v["shape"]} dtype {v["dtype"]} n_unique {v.get("n_unique", "-")} '
                    f'counts {v.get("counts", "-")} values {v.get("values", "-")}')
            log(f'NPZ {nm_} path {p}')
        except Exception as e:
            log(f'NPZ {nm_} inspect failed: {e!r}')
    # 2. models
    import json
    code = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/fpnet.py')))
    v4mod = os.path.dirname(find('casmi26-v4b-models/**/MANIFEST.json'))
    man = json.load(open(os.path.join(v4mod, 'MANIFEST.json')))
    fe = [os.path.join(v4mod, m['file']) for m in man.get('fe_models', [])]
    log('fe_models', [os.path.basename(p) for p in fe], 'families', man.get('families'))

    def g(pat):
        h = sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)
        return h[0] if h else ''
    v3code = os.path.dirname(os.path.dirname(g('casmi26-v3-models/**/casmi/fpnet.py') or '/x/x/x'))
    v3pts = sorted(glob.glob('/kaggle/input/**/casmi26-v3-models/**/*.pt', recursive=True))
    log('v3 .pt files', v3pts)
    v3 = {n: next((p for p in v3pts if os.path.basename(p).startswith(n)), '') for n in ('fpnet_0', 'fpnet_1')}
    models = [('full1', g('casmi26-fpnet-full1/**/fpnet_full1.pt'), 'fpnet', code),
              ('fpnet_0', os.path.join(v4mod, 'fpnet_0.pt'), 'fpnet', code)]
    models += [(f'fe_{s}', p, 'fpnet6', code) for s, p in zip('ABD', fe)]
    models += [('v3_fpnet_0', v3['fpnet_0'], 'fpnet', v3code), ('v3_fpnet_1', v3['fpnet_1'], 'fpnet', v3code)]
    models += [('fp_merged_m1', g('casmi26-fp-models-v2/**/fp_merged_m1*.pt'), 'pv', None),
               ('fp_single_s2', g('casmi26-fp-models-v2/**/fp_single_s2*.pt'), 'pv', None)]
    for m in models:
        log('A0 MODEL', m[0], m[1], m[2], m[3])
    pvfp = ''
    if 'PV_FP_SRC' in globals():
        pvfp = os.path.join(a.out, 'pvsrc', 'pv_fp.py'); os.makedirs(os.path.dirname(pvfp), exist_ok=True)
        open(pvfp, 'w').write(globals()['PV_FP_SRC'])
    try:
        pv_bits = np.load(os.path.join(os.path.dirname(find('coco_fp.npy')), 'fp_bits.npy'))
        log('pv fp_bits', len(pv_bits))
    except Exception as e:
        pv_bits = None
        log('pv fp_bits unavailable', repr(e))
    pool_dir = os.path.dirname(find('pool_meta.parquet'))
    summ, dec = run_a0(a.train, pool_dir, models, subprocess_logits_factory(a.out, pvfp), a.k_other, a.n_control,
                       a.out, pv_bits, code)
    log(f'A0 total {time.time()-T0:.0f}s')
    return summ, dec


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker(sys.argv[1:])
    elif DEFAULT_MODE == 'a0' or '--a0' in sys.argv:
        main_a0(sys.argv[1:])
    else:
        main()
