"""Self-contained copy of nl_stoich_rda.py for iso-rank (maths and tolerances unchanged; reads no files).

Original docstring: Sub-3.5 mDa Mass-Defect Neutral-Loss Stoichiometry & Concerted 2-Bond 6-Ring RDA Cleavage Scorer.

Analyzes high-resolution Bruker timsTOF MS/MS spectra:
1. 35 exact sub-3.5 mDa diagnostic neutral losses (precursor-to-peak and top-15 peak-to-peak differences)
   matched against candidate functional-group SMARTS patterns.
2. Concerted 2-bond 6-ring Retro-Diels-Alder (RDA) ring cleavages evaluated on candidate 2D graphs
   and matched against experimental peaks within sub-4.0 mDa.
"""
from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

AMU = {
    'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196,
    'P': 30.97376163, 'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268,
    'Br': 78.9183371, 'I': 126.904473, 'Na': 22.9897692809, 'K': 38.96370668,
    'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213
}
H = AMU['H']
E = 0.00054857990
PROTON = H - E
NH4 = AMU['N'] + 4 * H
FORMATE = AMU['C'] + 2 * H + 2 * AMU['O']

CARRIERS = {
    '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,),
    '[M+NH4]+': (PROTON, NH4 - E), '[M+Na]+': (PROTON, AMU['Na'] - E), '[M+K]+': (PROTON, AMU['K'] - E),
    '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, FORMATE - PROTON),
    '[M+Cl]-': (-PROTON, AMU['Cl'] + E),
}

# 35 Curated Sub-3.5 mDa Diagnostic Neutral Losses
RAW_RULES = [
    ('H2O', 18.010565, '[OX2H]', 1.0),
    ('NH3', 17.026549, '[NX3H2,NX3H3+,NX4+]', 1.0),
    ('HCN', 27.010899, '[#6]#[#7]', 1.0),
    ('CO', 27.994915, '[CX3]=[OX1]', 1.0),
    ('C2H4', 28.031300, '[CX4][CX4]', 1.0),
    ('CH2O', 30.010565, '[OX2][CH3]', 1.0),
    ('CH4O', 32.026215, '[OX2][CH3]', 1.0),
    ('C2H2O', 42.010565, '[CX3](=[OX1])[CH3]', 1.0),
    ('C3H6', 42.046950, 'CC(C)=C', 1.0),
    ('CO2', 43.989829, '[CX3](=[OX1])[OX2H,OX1-]', 1.0),
    ('CH2O2', 46.005479, '[CX3](=[OX1])[OX2H]', 1.0),
    ('C2H6O', 46.041865, '[OX2]CC', 1.0),
    ('C4H8', 56.062600, 'CC(C)=C', 1.0),
    ('C5H8', 68.062600, 'CC(=C)C=C', 1.0),
    ('SO3', 79.956815, '[SX4](=[OX1])(=[OX1])', 1.0),
    ('HPO3', 79.966270, '[PX4](=[OX1])([OX2])', 1.0),
    ('C3H2O3', 86.000394, 'OC(=O)CC(=O)[O,N]', 1.0),
    ('C5H8O4', 132.042259, '[OX2][CX4]1[CX4][CX4][CX4][OX2]1', 1.2),
    ('C6H10O4', 146.057909, '[OX2][CX4]1[CX4][CX4][CX4][CX4][OX2]1', 1.2),
    ('C9H6O2', 146.036779, 'c1ccc(cc1)C=CC(=O)[O,N]', 1.2),
    ('C7H4O4', 152.011009, 'c1c(cc(c(c1O)O)O)C(=O)[O,N]', 1.2),
    ('C6H10O5', 162.052824, '[OX2][CX4]1[CX4][CX4][CX4][CX4][OX2]1', 1.2),
    ('C9H6O3', 162.031694, 'c1cc(c(cc1O)O)C=CC(=O)[O,N]', 1.2),
    ('C6H8O6', 176.032088, '[OX2][CX4]1[CX4][CX4][CX4][CX4]([OX2]1)C(=O)O', 1.2),
    ('C10H8O3', 176.047344, 'c1cc(c(cc1OC)O)C=CC(=O)[O,N]', 1.2),
    ('C9H8O4', 180.042259, 'c1cc(c(cc1O)O)C=CC(=O)O', 1.2),
    ('CH3NO', 45.021464, '[CX3](=[OX1])[NX3H2]', 1.0),
    ('C2H3NO', 57.021464, '[CX3](=[OX1])[NX3H][CH3]', 1.0),
    ('C2H4O2', 60.021129, '[CX3](=[OX1])[OX2][CH3]', 1.0),
    ('C3H4O', 56.026215, '[CX3]=[CX3][CX3]=O', 1.0),
    ('C3H6O', 58.041865, 'CC(=O)C', 1.0),
    ('C4H6O', 70.041865, 'C1CCOC1', 1.0),
    ('C4H4O2', 84.021129, 'C1=CC(=O)OC1=O', 1.0),
    ('C6H6O', 94.041865, 'c1ccccc1O', 1.0),
    ('C7H6O2', 122.036779, 'c1ccccc1C(=O)O', 1.0)
]

DIAG_NL = [
    (name, mass, Chem.MolFromSmarts(smarts), weight)
    for name, mass, smarts, weight in RAW_RULES
]


def _carriers(adduct, mode):
    c = CARRIERS.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)


def rda_fragments(smi: str, cache: dict | None = None) -> np.ndarray:
    """Exact neutral masses of 2-bond 6-ring Retro-Diels-Alder cleavages."""
    if cache is not None and smi in cache:
        return cache[smi]
    try:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    except Exception:
        m = None
    if m is None or m.GetNumAtoms() == 0:
        res = np.empty(0, np.float64)
        if cache is not None:
            cache[smi] = res
        return res

    pt = Chem.GetPeriodicTable()
    n = m.GetNumAtoms()
    w = np.empty(n, np.float64)
    for a in m.GetAtoms():
        s = a.GetSymbol()
        w[a.GetIdx()] = AMU.get(s, pt.GetMostCommonIsotopeMass(s)) + a.GetTotalNumHs() * H

    rings = m.GetRingInfo().AtomRings()
    # Skip fully aromatic carbocyclic rings (benzene rings do not undergo RDA cleavage)
    target_rings = [r for r in rings if len(r) == 6 and not all(m.GetAtomWithIdx(a).GetIsAromatic() and m.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in r)]
    if not target_rings:
        res = np.empty(0, np.float64)
        if cache is not None:
            cache[smi] = res
        return res

    adj = [[] for _ in range(n)]
    for b in m.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].append(v)
        adj[v].append(u)

    out = []
    bonds = m.GetBonds()
    for r in target_rings:
        r_set = set(r)
        r_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds
                   if b.GetBeginAtomIdx() in r_set and b.GetEndAtomIdx() in r_set]
        nb = len(r_bonds)
        for i in range(nb):
            u1, v1 = r_bonds[i]
            for j in range(i + 1, nb):
                u2, v2 = r_bonds[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                # cut (u1, v1) and (u2, v2)
                visited = [False] * n
                comp1 = []
                stack = [0]
                visited[0] = True
                while stack:
                    curr = stack.pop()
                    comp1.append(curr)
                    for nxt in adj[curr]:
                        if (curr == u1 and nxt == v1) or (curr == v1 and nxt == u1):
                            continue
                        if (curr == u2 and nxt == v2) or (curr == v2 and nxt == u2):
                            continue
                        if not visited[nxt]:
                            visited[nxt] = True
                            stack.append(nxt)
                if len(comp1) < n:
                    rem = [idx for idx in range(n) if not visited[idx]]
                    stack = [rem[0]]
                    visited2 = [False] * n
                    visited2[rem[0]] = True
                    comp2_cnt = 0
                    while stack:
                        curr = stack.pop()
                        comp2_cnt += 1
                        for nxt in adj[curr]:
                            if (curr == u1 and nxt == v1) or (curr == v1 and nxt == u1):
                                continue
                            if (curr == u2 and nxt == v2) or (curr == v2 and nxt == u2):
                                continue
                            if not visited[nxt] and not visited2[nxt]:
                                visited2[nxt] = True
                                stack.append(nxt)
                    if comp2_cnt == len(rem):
                        out.append(float(w[comp1].sum()))
                        out.append(float(w[rem].sum()))

    res = np.array(sorted(set(np.round(out, 5))), np.float64) if out else np.empty(0, np.float64)
    if cache is not None:
        cache[smi] = res
    return res


def _match_spectrum_nl(spec: dict, tol: float = 0.0035):
    """Extract diagnostic neutral loss matches and weights from one spectrum."""
    mz = np.asarray(spec.get('mz', ()), np.float64).ravel()
    it = np.asarray(spec.get('it', ()), np.float64).ravel()
    prec = float(spec.get('prec', 0.0))
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0 or prec <= 0:
        return []

    sqrt_it = np.sqrt(it)
    tot = sqrt_it.sum()
    if not tot > 0:
        return []
    w_peaks = sqrt_it / tot

    # Precursor losses
    losses = prec - mz
    k_nl = losses > 0.5
    nl_vals = losses[k_nl]
    nl_w = w_peaks[k_nl]

    # Top 15 peak-to-peak differences
    top_idx = np.argsort(-it)[:15]
    top_mz = mz[top_idx]
    top_w = w_peaks[top_idx]
    p2p_vals, p2p_w = [], []
    for i in range(len(top_mz)):
        for j in range(i + 1, len(top_mz)):
            diff = abs(top_mz[i] - top_mz[j])
            if diff > 0.5:
                p2p_vals.append(diff)
                p2p_w.append(math.sqrt(top_w[i] * top_w[j]))

    matched = []
    # Match against DIAG_NL
    for rule_idx, (_, rule_mass, pat, rule_w) in enumerate(DIAG_NL):
        if pat is None:
            continue
        # Check precursor losses
        diffs = np.abs(nl_vals - rule_mass)
        hits = np.where(diffs <= tol)[0]
        for h in hits:
            matched.append((rule_idx, float(nl_w[h] * rule_w)))
        # Check peak-to-peak
        if p2p_vals:
            pdiffs = np.abs(np.array(p2p_vals) - rule_mass)
            phits = np.where(pdiffs <= tol)[0]
            for ph in phits:
                matched.append((rule_idx, float(p2p_w[ph] * rule_w * 0.5)))
    return matched


_GLOBAL_CACHE = {'rda': {}, 'mol': {}, 'smarts': {}}


def score_molecule(spectra, cand_smiles, cache=None, rda_tol: float = 0.0040):
    """Computes neutral loss stoichiometry + 6-ring RDA score for candidate SMILES."""
    cache = _GLOBAL_CACHE if cache is None else cache
    rda_cache = cache.setdefault('rda', {})
    mol_cache = cache.setdefault('mol', {})
    smarts_cache = cache.setdefault('smarts', {})

    preps = []
    for s in (spectra or []):
        matched_nl = _match_spectrum_nl(s, tol=0.0035)
        mz = np.asarray(s.get('mz', ()), np.float64).ravel()
        it = np.asarray(s.get('it', ()), np.float64).ravel()
        n = min(len(mz), len(it))
        mz, it = mz[:n], it[:n]
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        sqrt_it = np.sqrt(it)
        tot = sqrt_it.sum() if len(sqrt_it) else 0.0
        w_peaks = (sqrt_it / tot) if tot > 0 else np.empty(0, np.float64)
        carriers = _carriers(s.get('adduct'), s.get('mode'))
        preps.append(dict(matched_nl=matched_nl, mz=mz, w_peaks=w_peaks, carriers=carriers))

    if not preps:
        return [0.0] * len(cand_smiles)

    out = []
    for smi in cand_smiles:
        if not isinstance(smi, str) or not smi.strip():
            out.append(None)
            continue
        mol = mol_cache.get(smi)
        if mol is None and smi not in mol_cache:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None
            mol_cache[smi] = mol
        if mol is None:
            out.append(None)
            continue

        frags = rda_fragments(smi, cache=rda_cache)

        scores_s = []
        for p in preps:
            sc = 0.0
            # 1. Neutral loss matches
            for rule_idx, wt in p['matched_nl']:
                pat = DIAG_NL[rule_idx][2]
                key = (smi, rule_idx)
                has_match = smarts_cache.get(key)
                if has_match is None:
                    has_match = mol.HasSubstructMatch(pat)
                    smarts_cache[key] = has_match
                if has_match:
                    sc += wt

            # 2. 6-ring RDA fragment matches
            if len(frags) > 0 and len(p['mz']) > 0:
                pmz = p['mz']
                pw = p['w_peaks']
                # Test carriers and H shifts
                for m_frag in frags:
                    for car in p['carriers']:
                        for dh in (-1, 0, 1):
                            pred = m_frag + car + dh * H
                            diffs = np.abs(pmz - pred)
                            hit = np.where(diffs <= rda_tol)[0]
                            if len(hit) > 0:
                                sc += float(pw[hit].max() * 1.5)
            scores_s.append(sc)

        out.append(float(np.mean(scores_s)) if scores_s else 0.0)

    return out


def nl_scores(spectra, cand_smiles, cache=None):
    """One float per candidate (higher is better); np.nan for a candidate whose SMILES does not parse.

    `spectra`: list of dicts {mz, it, adduct, prec, mode}, the format build/frag_rescore.py takes. Measured spectrum only."""
    cand_smiles = list(cand_smiles)
    res = score_molecule(spectra, cand_smiles, cache=cache)
    return np.array([np.nan if v is None else float(v) for v in res], np.float64)
