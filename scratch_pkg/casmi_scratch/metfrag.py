"""MetFrag-lite `fr`: share of (sqrt) query intensity explained by 1-2 bond-break fragments. Needs RDKit."""
from multiprocessing import Pool as MPool

import numpy as np

from .chem import ELECTRON, MASS
from .kernels import clean

H_ATOM = MASS["H"]
PROTON = H_ATOM - ELECTRON


def mol_graph(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = MASS.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_ATOM
    if (w == 0).any():
        return None
    return w, [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()], n


def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop:
            continue
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, cur = [s], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smi, max_bonds=60, max_breaks=2):
    g = mol_graph(smi)
    if g is None:
        return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds:
        return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))


def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0:
        return 0.0
    ion = np.sort(np.concatenate([frag_mass + dh * H_ATOM + (PROTON if mode > 0 else -PROTON) for dh in h_shifts]))
    w = np.sqrt(np.asarray(peak_int, float))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)


def _frag_job(args):
    smi, max_bonds = args
    try:
        return fragment_masses(smi, max_bonds)
    except Exception:
        return np.zeros(0)


def frag_scores(cand_smiles, specs, mode, cfg):
    """fr per candidate: best explained share over the molecule's spectra. mode > 0 positive, < 0 negative."""
    if not len(cand_smiles):
        return np.zeros(0, np.float32)
    jobs = [(s, cfg.max_bonds) for s in cand_smiles]
    if cfg.frag_workers > 1:
        with MPool(cfg.frag_workers) as mp:
            frags = mp.map(_frag_job, jobs, chunksize=8)
    else:
        frags = [_frag_job(j) for j in jobs]
    peaks = []
    for mz, it in specs:
        m2, i2 = clean(np.asarray(mz, np.float32), np.asarray(it, np.float32), cfg.int_floor, cfg.max_peaks, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=cfg.mz_tol) for a, b in peaks), default=0.0)
    return out
