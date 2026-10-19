# Kaggle Public Notebooks: Orthogonal Ideas Analysis

**Baseline Notebook**: `casmi26-e006-window-40-ice-fallback-true.ipynb` (30 code cells, 2 datasets)

**Total Notebooks Analyzed**: 1

## 1. Executive Summary & Top Orthogonal Finds

- **Distinct / Derivative Architectures**: 1
- **Direct Clones / Hyperparameter Variations**: 0

| Notebook / Title | Match % | Classification | New Datasets / Inputs | New Packages | Score Claims |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **forensics-lab-the-apex-dossier-lb-0**<br>*🔬 Forensics Lab: The Apex Dossier — Tune* | 84.4% | Derivative / Branch | — | `frag_rescore`, `sys` | LB 0.447, score 0.433 |

---

## 3. Detailed Orthogonal Idea Breakdowns

### 💡 🔬 Forensics Lab: The Apex Dossier — Tuned Promo (rel > 150) + CC BY ChEBI + Metric-Keyed E2R2 (LB 0.447 SOTA) (`forensics-lab-the-apex-dossier-lb-0-442-sota.ipynb`)
- **Relationship to Baseline**: Derivative / Branch (84.4% shared code cells)
- **Score / Metric Notes**: LB 0.447, score 0.433, LB 0.380, LB 0.378, LB 0.354, LB 0.358, LB 0.403, LB 0.444, LB 0.443
- **New Packages**: `frag_rescore`, `sys`, `traceback`
- **New Functions**: `_e2r2_apply`, `_safe_finite`, `check`, `e2r2_make_key`, `e2r2_row`, `engine_list`, `key`, `n_dup`

**Author Markdown Guidance / Hypotheses**:
> # 🔬 Forensics Lab: The Apex Dossier — Tuned Promo (rel > 150) + CC BY ChEBI + Metric-Keyed E2R2 (LB 0.447 SOTA)
> 
> > *Welcome to the Forensics Lab. 400 molecules — 400 cold cases.*  
> > *You have fragmentation fingerprints from high-resolution mass spectrometry, but zero named suspects.*  
> > *Can you crack the case and identify the perpetrators from 105 million candidates?*
> 
> ---
> 
> ## 🕵️ The Case: Molecular Forensics
> 
> # E004: exact v80 public recipe reproduction
> Source: flexonafft/casmi26-public-baseline-adaptation-experiments v80, script356740494.
> Author submission57008151 scored0.436; this is source evidence, not our score.
> All 27 source code cells were recovered from that exact version's public rendered input cells. Their combined SHA256 is pinned and their executable AST was independently compared with the v82 source prefix (the v82 consensus overlay is excluded).
> The original code, GPU T4, image digest a

**Key Novel Code Highlights**:
```python
# [Cell 8] """Read-only E004 observer, before the source smoke switch."""

"""Read-only E004 observer, before the source smoke switch."""


def _e004_preflight(scope):
    import json
    import os
    import platform
    import subprocess
    import pandas as pd
    import rdkit

    slug = 'enveda-CASMI26-molecule-id-mass-spectra'
    comp = os.path.realpath(scope['COMP'])
    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}
    assert platform.python_version_tuple()[:2] in {('3', '12'), ('3', '13')}
    assert rdkit.__version__ == '2026.03.3'
    # Do not initialize a parent CUDA context before the source's Engine2 subprocess.
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    assert 'T4' in gpu, gpu
    ids = sorted(set(pd.read_parquet(os.path.join(comp, 'test.parquet'),
                                    columns=['molecule_id']).molecule_id.astype(str)))
    sample = pd.read_csv(os.path.join(comp, 'sample_submission.csv'), dtype=str)
    assert ids and set(sample.molecule_id) == set(ids)
    result = dict(runtime_input_path=comp, input_ids=ids, python=platform.python_version(),

# ... [truncated]

```
```python
# [Cell 23] '''gated PubChem merge (v3.6, unchanged) -> submission.csv'''

'''gated PubChem merge (v3.6, unchanged) -> submission.csv'''
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out
'''---- CLAW PROMOTION RULE (b417) --------------------------------------------------'''
'''The shipped gate only INTERLEAVES PubChem-only structures into fixed slots, so the'''
'''main list always keeps rank 1. When the best PubChem proposal is a strong outlier --'''
'''S = z(f.z over the pass-2 set) + 0.25*pop, carried out of the worker -- AND is well'''
'''documented (pop >= 5), let it lead; everything else keeps its usual slot.'''
'''NOTE: S is the Z-SCORED quantity (median ~6). The raw p['pc_fz'][0] (median -1437)'''
'''must never be compared to S_TAU; that would fire on ~71% of molecules.'''
'''EVIDENCE: b416 (promotion, ICEBERG broken) = 0.417 vs base415 (ICE broken, no'''
'''promotion) 0.374 and b415c (ICE working, no promotion) 0.393. The promotion overcame'''
'''losing ICE, so its marginal value is >= +0.024 (est. ~+0.037 using the notebook's own'''
'''ICE +0.013 / GL +0.006 measurements). b417 isolates it on the WORKING-ICE base.'''
S_TAU, POP_TAU = 6.0, 5.0

# ... [truncated]

```
```python
# [Cell 32] FRAG_GAP = {}

# v7c: MetFrag-lite fragment evidence for molecules with NO forward-model coverage (no GLACIER, no finite ICEBERG)
FRAG_GAP = {}
try:
    FRAG_SRC = '"""Untrained MetFrag-lite re-scorer: how much MS/MS intensity a candidate\'s <=2-bond-cleavage fragments explain.\n\nIndependent of pv.py (same idea as pv.fragment_masses / explain_score_adduct, new code): enumerate the neutral masses\nof the connected pieces left after removing 1 or 2 bonds of the heavy-atom graph (implicit H stay on their atom),\nremember the fewest cleavages that give each mass, then match every peak against\n`neutral fragment + dh*H + charge carrier` for dh in `h_shifts` and the carriers the precursor adduct allows.\nA peak\'s credit is the weight of its cheapest explanation (1 for 0/1 cleavages, `w2` for 2, `w3` for 3),\nand a spectrum\'s score is the credited share of its (weighted) intensity. Defaults follow the forum ablation on a\n241-structure same-formula isomer panel (+0.065 MRR over the common baseline): w2=0.6, linear intensity,\ntol 0.005 Da, H shifts -2..+3, at most 2 cleavages. `BASELINE_CFG` is the common/pv.py-style setting.\n\n    score_molecule(spectra, cand_smiles, cfg=None, cache=None) -> list[float | None]\n\n`spectra`: list of dicts {mz: ndarray, it: ndarray (base peak = 1), adduct: str, prec: float, mode: +1/-1}.\nReturns one score per candidate (mean over the spectra that have peaks; higher = better), None if the SMILES does\nnot parse. Fragment masses are cached per SMILES in `cache` (a dict; pass one in to reuse it across molecules).\nThe enumeration is numba-compiled when numba is importable (pure-Python fallback otherwise, ~100x slower).\n"""\nfrom __future__ import annotations\n\nimport numpy as np\n\n# monoisotopic masses (same values as pv.AMU); other elements come from RDKit\'s periodic table\nAMU = {\'C\': 12.0, \'H\': 1.00782503207, \'N\': 14.0030740048, \'O\': 15.9949146196, \'P\': 30.97376163,\n       \'S\': 31.97207100, \'F\': 18.99840322, \'Cl\': 34.96885268, \'Br\': 78.9183371, \'I\': 126.904473,\n       \'Na\': 22.9897692809, \'K\': 38.96370668, \'Si\': 27.9769265325, \'B\': 11.0093054, \'Se\': 79.9165213}\nH = AMU[\'H\']\nE = 0.00054857990\nPROTON = H - E\nH2O = 2 * H + AMU[\'O\']\nNH4 = AMU[\'N\'] + 4 * H\nFORMATE = AMU[\'C\'] + 2 * H + 2 * AMU[\'O\']\n\n# charge carriers a fragment ion may keep, per precursor adduct (added to the neutral fragment mass)\nCARRIERS = {\n    \'[M+H]+\': (PROTON,), \'[M-H2O+H]+\': (PROTON,), \'[M-2H2O+H]+\': (PROTON,),\n    \'[M+NH4]+\': (PROTON, NH4 - E), \'[M+Na]+\': (PROTON, AMU[\'Na\'] - E), \'[M+K]+\': (PROTON, AMU[\'K\'] - E),\n    \'[M-H]-\': (-PROTON,), \'[M-H2O-H]-\': (-PROTON,), \'[M+CH2O2-H]-\': (-PROTON, FORMATE - PROTON),\n    \'[M+Cl]-\': (-PROTON, AMU[\'Cl\'] + E),\n}\n\nDEFAULT_CFG = dict(\n    w1=1.0,                      # credit for a peak explained by the precursor or a 1-bond cleavage\n    w2=0.6,                      # credit for a peak only explained by a 2-bond cleavage\n    w3=0.3,                      # credit for 3-bond cleavages (only used when max_breaks=3)\n    intensity=\'linear\',          # \'linear\' | \'sqrt\' | a float power applied to `it`\n    mz_power=0.0,                # extra peak weight (mz / prec) ** mz_power (0 = off)\n    tol=0.005,                   # absolute m/z tolerance, Da\n    h_shifts=tuple(range(-2, 4)),  # hydrogen shifts dh applied to neutral fragments (-2..+3)\n    max_breaks=2,                # bond cleavages per fragment (1..3)\n    max_bonds_2=100,             # above this many bonds only 1-bond cleavages are enumerated (cost cap)\n    max_bonds_3=40,              # above this many bonds 3-bond cleavages are skipped\n)\n# the common baseline (pv.py-style): all cleavages equal, sqrt intensity, 0.01 Da, H shifts -2..+2\nBASELINE_CFG = dict(DEFAULT_CFG, w2=1.0, intensity=\'sqrt\', tol=0.01, h_shifts=tuple(range(-2, 3)))\n\n\ndef resolve_cfg(cfg=None):\n    """DEFAULT_CFG updated with `cfg` (unknown keys raise, so typos in variants do not pass silently)."""\n    out = dict(DEFAULT_CFG)\n    for k, v in (cfg or {}).items():\n        if k not in out:\n            raise KeyError(f\'frag_rescore: unknown cfg key {k!r}\')\n        out[k] = v\n    out[\'h_shifts\'] = tuple(int(d) for d in out[\'h_shifts\'])\n    if not 1 <= int(out[\'max_breaks\']) <= 3:\n        raise ValueError(\'frag_rescore: max_breaks must be 1, 2 or 3\')\n    return out\n\n\n# ---------------------------------------------------------------------------------------------------------------\n# fragment enumeration (numba kernel with a pure-Python fallback: same source)\n# ---------------------------------------------------------------------------------------------------------------\ndef _find(parent, a):\n    while parent[a] != a:\n        parent[a] = parent[parent[a]]\n        a = parent[a]\n    return a\n\n\ndef _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, k):\n    """Union the kept bonds; if the graph splits, append each piece\'s mass with cleavage count k."""\n    for a in range(n):\n        parent[a] = a\n        acc[a] = 0.0\n    ncomp = n\n    for e in range(ea.shape[0]):\n        if removed[e]:\n            continue\n        ra = _find(parent, ea[e])\n        rb = _find(parent, eb[e])\n        if ra != rb:\n            parent[ra] = rb\n            ncomp -= 1\n    if ncomp < 2:\n        return cnt\n    for a in range(n):\n        r = _find(parent, a)\n        acc[r] += w[a]\n    for a in range(n):\n        if parent[a] == a:\n            out_m[cnt] = acc[a]\n            out_k[cnt] = k\n            cnt += 1\n    return cnt\n\n\ndef _enumerate(n, w, ea, eb, max_breaks, base_components):\n    """All piece masses after removing 1..max_breaks bonds (k = number removed); row 0.. are the intact pieces."""\n    nb = ea.shape[0]\n    n2 = nb * (nb - 1) // 2 if max_breaks >= 2 else 0\n    n3 = nb * (nb - 1) * (nb - 2) // 6 if max_breaks >= 3 else 0\n    cap = base_components + (base_components + 1) * nb + (base_components + 2) * n2 + (base_components + 3) * n3\n    out_m = np.empty(cap, np.float64)\n    out_k = np.empty(cap, np.int8)\n    parent = np.empty(n, np.int64)\n    acc = np.empty(n, np.float64)\n    removed = np.zeros(nb, np.bool_)\n    cnt = 0\n    for i in range(nb):\n        removed[i] = True\n        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 1)\n        if max_breaks >= 2:\n            for j in range(i + 1, nb):\n                removed[j] = True\n                cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 2)\n                if max_breaks >= 3:\n                    for l in range(j + 1, nb):\n                        removed[l] = True\n                        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 3)\n                        removed[l] = False\n                removed[j] = False\n        removed[i] = False\n    return out_m[:cnt], out_k[:cnt]\n\n\ntry:  # numba is on the Kaggle image and in requirements.txt; NUMBA_CACHE_DIR is set by nb/01_setup.py\n    from numba import njit\n    _find = njit(cache=True, nogil=True)(_find)\n    _emit_components = njit(cache=True, nogil=True)(_emit_components)\n    _enumerate = njit(cache=True, nogil=True)(_enumerate)\n    HAVE_NUMBA = True\nexcept Exception:  # pragma: no cover - fallback path\n    HAVE_NUMBA = False\n\n\ndef mol_graph(smi):\n    """(atom masses incl. implicit H, bond begin idx, bond end idx, n connected pieces) or None if unparsable."""\n    from rdkit import Chem, RDLogger\n    RDLogger.DisableLog(\'rdApp.*\')\n    try:\n        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None\n    except Exception:\n        m = None\n    if m is None or m.GetNumAtoms() == 0:\n        return None\n    pt = Chem.GetPeriodicTable()\n    w = np.empty(m.GetNumAtoms(), np.float64)\n    for a in m.GetAtoms():\n        s = a.GetSymbol()\n        am = AMU.get(s)\n        if am is None:\n            am = pt.GetMostCommonIsotopeMass(s)\n        w[a.GetIdx()] = am + a.GetTotalNumHs() * H\n    ea = np.array([b.GetBeginAtomIdx() for b in m.GetBonds()], np.int64)\n    eb = np.array([b.GetEndAtomIdx() for b in m.GetBonds()], np.int64)\n    return w, ea, eb, len(Chem.GetMolFrags(m))\n\n\ndef fragment_levels(smi, max_breaks=2, max_bonds_2=100, max_bonds_3=40):\n    """Sorted neutral fragment masses grouped by the fewest cleavages that produce them: [k=0, k=1, ..., k=max].\n\n    k=0 holds the intact molecule (sum of its pieces, and each piece if the SMILES has several). None if the\n    SMILES does not parse.\n    """\n    g = mol_graph(smi)\n    if g is None:\n        return None\n    w, ea, eb, npieces = g\n    nb = len(ea)\n    mb = int(max_breaks)\n    if mb >= 3 and nb > max_bonds_3:\n        mb = 2\n    if mb >= 2 and nb > max_bonds_2:\n        mb = 1\n    levels = [np.array([w.sum()])]\n    if npieces > 1:  # salts / mixtures: each intact piece also counts as uncleaved\n        from rdkit import Chem\n        m = Chem.MolFromSmiles(smi)\n        levels = [np.unique(np.concatenate([[w.sum()], [w[list(ix)].sum() for ix in Chem.GetMolFrags(m)]]))]\n    if nb == 0:\n        return levels + [np.zeros(0) for _ in range(int(max_breaks))]\n    fm, fk = _enumerate(len(w), w, ea, eb, mb, npieces)\n    fm = np.concatenate([levels[0], fm])\n    fk = np.concatenate([np.zeros(len(levels[0]), np.int8), fk])\n    key = np.round(fm, 5)\n    order = np.lexsort((fk, key))                  # by mass, then fewest cleavages first\n    key, fm, fk = key[order], fm[order], fk[order]\n    first = np.ones(len(key), bool)\n    first[1:] = key[1:] != key[:-1]\n    fm, fk = fm[first], fk[first]\n    return [np.sort(fm[fk == k]) for k in range(int(max_breaks) + 1)]\n\n\n# ---------------------------------------------------------------------------------------------------------------\n# scoring\n# ---------------------------------------------------------------------------------------------------------------\ndef _carriers(adduct, mode):\n    c = CARRIERS.get(adduct)\n    if c is not None:\n        return c\n    if isinstance(adduct, str) and adduct.endswith(\'-\'):\n        return (-PROTON,)\n    if isinstance(adduct, str) and adduct.endswith(\'+\'):\n        return (PROTON,)\n    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)\n\n\ndef _prep_spectrum(s, c):\n    """Peak weights and the neutral masses each peak would correspond to, for every (carrier, dh) pair."""\n    mz = np.asarray(s.get(\'mz\', ()), np.float64).ravel()\n    it = np.asarray(s.get(\'it\', ()), np.float64).ravel()\n    n = min(len(mz), len(it))\n    mz, it = mz[:n], it[:n]\n    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)\n    mz, it = mz[ok], it[ok]\n    if len(mz) == 0:\n        return None\n    p = c[\'intensity\']\n    iw = it if p == \'linear\' else np.sqrt(it) if p == \'sqrt\' else it ** float(p)\n    if c[\'mz_power\']:\n        prec = float(s.get(\'prec\') or mz.max())\n        iw = iw * (mz / prec) ** float(c[\'mz_power\'])\n    tot = iw.sum()\n    if not tot > 0:\n        return None\n    shifts = np.array([car + dh * H for car in _carriers(s.get(\'adduct\'), s.get(\'mode\')) for dh in c[\'h_shifts\']])\n    q = (mz[None, :] - shifts[:, None]).ravel()  # neutral mass needed, per (shift, peak)\n    return iw / tot, q, len(shifts), len(mz)\n\n\ndef _score_spectrum(levels, weights, prep, tol):\n    iw, q, ns, npk = prep\n    best = np.zeros(npk)\n    for arr, wt in zip(levels, weights):\n        if len(arr) == 0 or wt <= 0:\n            continue\n        a = np.searchsorted(arr, q - tol, \'left\')\n        b = np.searchsorted(arr, q + tol, \'right\')\n        hit = (b > a).reshape(ns, npk).any(axis=0)\n        best = np.where(hit & (wt > best), wt, best)\n    return float(iw @ best)\n\n\ndef candidate_levels(smi, cfg=None, cache=None):\n    """Cached fragment_levels for one SMILES under `cfg` (only the enumeration keys matter)."""\n    return _levels_cached(smi, resolve_cfg(cfg), cache)\n\n\ndef _levels_cached(smi, c, cache):\n    key = (smi, int(c[\'max_breaks\']), int(c[\'max_bonds_2\']), int(c[\'max_bonds_3\']))\n    if cache is not None and key in cache:\n        return cache[key]\n    try:\n        lv = fragment_levels(smi, c[\'max_breaks\'], c[\'max_bonds_2\'], c[\'max_bonds_3\'])\n    except Exception:\n        lv = None\n    if cache is not None:\n        cache[key] = lv\n    return lv\n\n\ndef score_molecule(spectra, cand_smiles, cfg=None, cache=None):\n    """One score per candidate: mean explained-intensity share over the molecule\'s spectra (None: unparsable)."""\n    c = resolve_cfg(cfg)\n    cache = {} if cache is None else cache\n    weights = [1.0, float(c[\'w1\']), float(c[\'w2\']), float(c[\'w3\'])][: int(c[\'max_breaks\']) + 1]\n    preps = [p for p in (_prep_spectrum(s, c) for s in (spectra or [])) if p is not None]\n    tol = float(c[\'tol\'])\n    out = []\n    for smi in cand_smiles:\n        lv = _levels_cached(smi, c, cache)\n        if lv is None:\n            out.append(None)\n            continue\n        if not preps:\n            out.append(0.0)\n            continue\n        out.append(float(np.mean([_score_spectrum(lv, weights, p, tol) for p in preps])))\n    return out\n\n\ndef warmup():\n    """Compile the numba kernel once (a few seconds) so the first real molecule is not charged for it."""\n    fragment_levels(\'CCOC(=O)c1ccccc1O\')\n    return HAVE_NUMBA\n'
    open('/kaggle/working/frag_rescore.py', 'w').write(FRAG_SRC)
    if '/kaggle/working' not in sys.path:
        sys.path.insert(0, '/kaggle/working')
    import frag_rescore, math
    frag_rescore.warmup()
    _fcache = {}
    _by = {str(m): g for m, g in te.groupby('molecule_id', sort=False)}
    _t0 = time.time(); _nf = 0; _ngf = 0
    FRAG_GROUP_GAP = {}
    for _mid, _lst in PREFUSION_LISTS.items():
        _ice = ICE_SCORES.get(_mid, {}) if isinstance(ICE_SCORES, dict) else {}
        _gl = gl_of(_mid) or {}
        _has_sim = bool(_gl) or any(v is not None for v in (_ice or {}).values())
        _e = ENG.get(_mid) or {}
        _c = [s for s in dict.fromkeys(list(_lst) + list(_e.get('smiles', []))) if s and s != 'CCO']
        _g = _by.get(_mid)
        if _g is None or len(_c) < 2:
            continue
        _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                      adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                 for r in _g.itertuples()][:16]

# ... [truncated]

```

