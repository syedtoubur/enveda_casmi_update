"""Track 4b NP signals: the core functions shared by the Colab build/benchmark and the local tests.

Embedded verbatim into casmi_np_signals_colab.ipynb (by build/mk_np_signals.py) and written back to a file,
so the multiprocessing workers can import it. Plan: specs/track4_priors_probes/np_signals_plan.md.

Three signals, each meant as one additive term on top of the borrowed popularity prior
pop = log1p(SID) + log1p(PMID):
  vendor_only  1 if every live SID of the CID was deposited by a "Chemical Vendors" source and PMID == 0
  np_like      Ertl NP-likeness (RDKit Contrib NP_Score), z-scored within each candidate window
  class_cons   superclass agreement between a candidate and the molecule's nearest library structures,
               similarity-weighted. A candidate with no label scores 0, never a penalty.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------- metrics

def expected_rank_metrics(scores, truth_idx, k=25, hits=(1, 5, 25)):
    """Expected reciprocal rank @k and hit@n of scores[truth_idx] under uniformly random tie-breaking.

    Ties matter here: thousands of vendor decoys share pop = log1p(1) and, under a pessimistic or optimistic
    tie rule, the ordering of equal scores would decide the metric on its own.
    Returns (rr, {n: hit}).
    """
    s = np.round(np.asarray(scores, np.float64), 9)
    t = s[truth_idx]
    g = int((s > t).sum())
    e = int((s == t).sum())               # includes the truth itself
    ranks = np.arange(g + 1, g + e + 1)
    rr = float(np.where(ranks <= k, 1.0 / ranks, 0.0).mean())
    hit = {n: float(np.clip(n - g, 0, e) / e) for n in hits}
    return rr, hit


# ---------------------------------------------------------------- vendor_only

def vendor_source_ids(source_table, source_names=None):
    """Return the set of source names/ids whose PubChem source category contains "vendor".

    source_table: DataFrame from PUG REST sourcetable/substance/CSV; must have a name column and a category
    column (case-insensitive match on 'categ'). Raises if no category column exists, because a silent fallback
    would make the vendor flag mean something else.
    source_names: optional DataFrame (id, display_name) from FTP Substance/Extras/Source-Names. SID-Map
    stores the depositor's source *id*, which for newer depositors is numeric, so ids are added for every
    vendor display name.
    """
    cols = {c.lower(): c for c in source_table.columns}
    cat = next((cols[c] for c in cols if 'categ' in c), None)
    if cat is None:
        raise KeyError(f'no category column in the source table: {list(source_table.columns)}')
    name = next((cols[c] for c in cols if c in ('source name', 'name', 'sourcename', 'source')), None)
    if name is None:
        name = next(cols[c] for c in cols if 'name' in c)
    isv = source_table[cat].astype(str).str.contains('vendor', case=False, na=False)
    vend = set(source_table.loc[isv, name].astype(str).str.strip())
    if source_names is not None:
        sid_col, disp_col = source_names.columns[:2]
        m = source_names[disp_col].astype(str).str.strip().isin(vend) | source_names[sid_col].astype(str).str.strip().isin(vend)
        vend |= set(source_names.loc[m, sid_col].astype(str).str.strip())
    return vend


def accumulate_sid_map(src, cid, vendor_set, nonvendor, anysid):
    """One SID-Map chunk: src (array of source ids, str), cid (float/int array, NaN = no standardized CID).
    Sets anysid[cid] for every SID and nonvendor[cid] for SIDs whose source is not a vendor. In place."""
    cid = np.asarray(cid, np.float64)
    ok = np.isfinite(cid) & (cid > 0) & (cid < len(nonvendor))
    if not ok.any():
        return 0
    import pandas as pd
    c = cid[ok].astype(np.int64)
    isv = pd.Series(np.asarray(src, object)[ok]).astype(str).str.strip().isin(vendor_set).values
    anysid[c] = True
    nonvendor[c[~isv]] = True
    return int(ok.sum())


def vendor_only_flags(cids, lpmid, nonvendor, anysid):
    """Per-row vendor_only for an array of representative CIDs and log1p(PMID) values."""
    cids = np.asarray(cids, np.int64)
    ok = (cids > 0) & (cids < len(nonvendor))
    c = np.where(ok, cids, 0)
    no_pmid = np.asarray(lpmid, np.float32) <= 0
    return (ok & anysid[c] & ~nonvendor[c] & no_pmid).astype(np.uint8)


# ---------------------------------------------------------------- NP-likeness

_NP = {}


def _np_model():
    if 'm' not in _NP:
        from rdkit.Chem import RDConfig
        p = os.path.join(RDConfig.RDContribDir, 'NP_Score')
        if p not in sys.path:
            sys.path.append(p)
        import npscorer
        _NP['f'] = npscorer
        _NP['m'] = npscorer.readNPModel()
    return _NP['f'], _NP['m']


def np_scores(smiles):
    """Ertl NP-likeness per SMILES; NaN where RDKit cannot parse."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    f, m = _np_model()
    out = np.full(len(smiles), np.nan, np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                out[i] = f.scoreMol(mol, m)
            except Exception:
                pass
    return out


def zscore_window(x):
    """z-score within one candidate window; NaN -> 0 (neutral), constant window -> all 0."""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    out = np.zeros(len(x))
    if ok.sum() >= 2:
        sd = x[ok].std()
        if sd > 0:
            out[ok] = (x[ok] - x[ok].mean()) / sd
    return out


# ---------------------------------------------------------------- class consistency

def morgan_fp(smiles, nbits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = _NP.setdefault('gen', rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits))
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else gen.GetFingerprint(m)


def knn_label(fp, bank_fps, bank_labels, thr=0.7):
    """Label of the nearest bank structure if its Tanimoto >= thr, else None. Returns (label, sim)."""
    if fp is None or not len(bank_fps):
        return None, 0.0
    from rdkit import DataStructs
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, bank_fps))
    j = int(sims.argmax())
    return (bank_labels[j], float(sims[j])) if sims[j] >= thr else (None, float(sims[j]))


def nearest_analogs(fp, bank_fps, k=5, exclude=None):
    """Top-k bank indices by Tanimoto (excluding indices in `exclude`). Returns (idx, sims)."""
    from rdkit import DataStructs
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, bank_fps))
    if exclude is not None and len(exclude):
        sims[np.asarray(list(exclude), np.int64)] = -1.0
    idx = np.argsort(-sims)[:k]
    idx = idx[sims[idx] > 0]
    return idx, sims[idx]


def class_consistency(cand_labels, analog_labels, analog_sims):
    """s_c = sum_k sim_k [label_c == label_k] / sum_k sim_k over labelled analogs. Unlabelled candidate -> 0;
    no labelled analog -> all 0 (the signal abstains instead of guessing)."""
    pairs = [(l, w) for l, w in zip(analog_labels, analog_sims) if l is not None and w > 0]
    out = np.zeros(len(cand_labels))
    if not pairs:
        return out
    tot = sum(w for _, w in pairs)
    wt = {}
    for l, w in pairs:
        wt[l] = wt.get(l, 0.0) + w / tot
    for i, l in enumerate(cand_labels):
        if l is not None:
            out[i] = wt.get(l, 0.0)
    return out


# ---------------------------------------------------------------- PubChem-tier windows

_FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def canon_formula(f):
    """Hill-order-insensitive formula key ('C6H12O6' == 'H12C6O6'); charges dropped."""
    f = re.sub(r'[+-]\d*$', '', str(f))
    d = {}
    for el, n in _FORMULA_RE.findall(f):
        d[el] = d.get(el, 0) + (int(n) if n else 1)
    return ''.join(f'{el}{d[el]}' for el in sorted(d))


def tier_slice(mass, off, buf, target, ppm):
    """Row range [a, b) and SMILES of the tier rows within ppm of a neutral mass."""
    tol = target * ppm * 1e-6
    a = int(np.searchsorted(mass, target - tol, 'left'))
    b = int(np.searchsorted(mass, target + tol, 'right'))
    if b <= a:
        return a, b, []
    o = np.asarray(off[a:b + 1], np.int64)
    raw = bytes(buf[o[0]:o[-1]])
    base = o[0]
    return a, b, [raw[o[i] - base:o[i + 1] - base].decode('ascii') for i in range(b - a)]


def same_formula(smiles, formula):
    """Mask of SMILES whose molecular formula equals `formula` (canonicalised)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog('rdApp.*')
    want = canon_formula(formula)
    ok = np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None and canon_formula(CalcMolFormula(m)) == want:
            ok[i] = True
    return ok


def dedupe_first(keys):
    """Indices of the first occurrence of each key, in order."""
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return np.asarray(keep, np.int64)


# ---------------------------------------------------------------- offline benchmark (one query)

G = {}                 # filled by the notebook before the worker Pool forks (mmaps, banks, label maps)
MUS = (0.25, 0.5, 1.0)
KNN_TOP = 150          # candidates (by pop) that get a kNN label when the NP table has none
KNN_THR = 0.7
K_ANALOG = 5


def _key(k):
    return k.decode('ascii', 'ignore') if isinstance(k, (bytes, np.bytes_)) else str(k)


def eval_query(task):
    """task = (qid, ik14, formula, mono_mass, smiles, cohort). Ranks the truth among its same-formula PubChem-tier
    isomers (deduplicated by IK14, keeping the most popular row) under pop and pop + mu * signal.

    G keys: mass, off, buf (tier), ik14, lsid, lpmid, vendor (row-aligned), ppm, label_of (ik14 -> superclass),
    lotus_fps, lotus_labels, train_fps, train_labels, train_rows_of (ik14 -> list of bank indices).
    """
    qid, ik14, formula, mono_mass, smiles, cohort = task
    out = dict(qid=qid, cohort=cohort, found=False, n=0)
    a, b, smis = tier_slice(G['mass'], G['off'], G['buf'], float(mono_mass), G['ppm'])
    if not smis:
        return out
    sf = np.where(same_formula(smis, formula))[0]
    if not len(sf):
        return out
    rows = a + sf
    pop = np.asarray(G['lsid'][rows], np.float32) + np.asarray(G['lpmid'][rows], np.float32)
    keys = np.array([_key(k) for k in G['ik14'][rows]], object)
    o = np.argsort(-pop, kind='stable')
    keep = o[dedupe_first(keys[o])]
    keep = keep[keys[keep] != '']
    rows, pop, keys = rows[keep], pop[keep], keys[keep]
    excl = G.get('exclude_keys') if cohort in G.get('exclude_cohorts', ()) else None
    if excl:
        # mirror supernext's PubChem channel: structures already in the candidate pool are never listed there
        m = np.array([(k == ik14) or (k not in excl) for k in keys], bool)
        rows, pop, keys = rows[m], pop[m], keys[m]
    t = np.where(keys == ik14)[0]
    out['n'] = int(len(rows))
    if not len(t):
        return out
    t = int(t[0])
    out['found'] = True
    sm = [smis[i - a] for i in rows]
    vendor = np.asarray(G['vendor'][rows], np.float64)
    npz = zscore_window(np_scores(sm))

    # class consistency: candidate labels (table, else kNN for the KNN_TOP most popular), analogs from the library
    lab = [G['label_of'].get(k) for k in keys]
    for i in np.argsort(-pop, kind='stable')[:KNN_TOP]:
        if lab[i] is None:
            lab[i], _ = knn_label(morgan_fp(sm[i]), G['lotus_fps'], G['lotus_labels'], KNN_THR)
    qfp = morgan_fp(smiles)
    if qfp is not None and len(G['train_fps']):
        ai, asim = nearest_analogs(qfp, G['train_fps'], K_ANALOG, exclude=G['train_rows_of'].get(ik14, ()))
        alab = [G['train_labels'][j] for j in ai]
    else:
        alab, asim = [], []
    cc = class_consistency(lab, alab, asim)
    mem = np.array([l is not None for l in lab], np.float64)   # membership-only ablation: "has an NP label"
    top_lab = {lab[i] for i in np.argsort(-pop, kind='stable')[:KNN_TOP] if lab[i] is not None}
    out.update(has_lab_analog=any(l is not None for l in alab), multi_class_window=len(top_lab) >= 2,
               truth_vendor=int(vendor[t]), truth_lab=lab[t] is not None, cc_truth=float(cc[t]),
               vendor_rate=float(vendor.mean()))

    variants = {'pop': pop}
    for mu in MUS:
        variants[f'vendor@{mu}'] = pop - mu * vendor
        variants[f'nplike@{mu}'] = pop + mu * npz
        variants[f'classcons@{mu}'] = pop + mu * cc
        variants[f'member@{mu}'] = pop + mu * mem
        variants[f'stack@{mu}'] = pop + mu * npz + mu * cc
    for name, sc in variants.items():
        rr, hit = expected_rank_metrics(sc, t)
        out[f'rr|{name}'] = rr
        for n, h in hit.items():
            out[f'hit{n}|{name}'] = h
    return out
