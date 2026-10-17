"""Candidate post-scoring module for e180-drop-dreamsnp-pclite (Wave 1 B8).

Extends the V22 baseline (casmi26-v22-rel150-e2r2-clean: REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True)
with:
  - E180_DROP (default False): filters R1b enveda-180-only synthetic combinatorial decoys
    after simulator / FRAG_GAP rerank before the 25-slot cut, back-filling to 25 from fused ranks 26-40,
    C12 extras (OVER, TAIL, ENG, PC), and mass-matched DreamsNP natural products.
  - E2R2 decoy guard: prevents E2R2 from hijacking Rank 2 with an e180_only decoy from Engine 2.
  - DREAMS_FILL (default False): mass-matched DreamsNP backfill (+/- 10 ppm) with e180_keys excluded.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

CFG = dict(
    # notebook constants (consts in the state override these)
    TOPN=60, ICE_LAM=1.0, GL_LAM=1.0, LIB_TAU=0.9, REL_TH=600.0,
    SLOTS_AGG=[2, 4, 6, 8, 10], SLOTS_GENTLE=[4, 8, 12, 16, 20], S_TAU=6.0, POP_TAU=5.0, REL_PROMOTE=250.0,
    ALPHA=0.6, KRR=3.0, FUSE_N=40, BASE_ICE=False, CELL19_ICE=False, ICE_PC=True,
    # harness knobs (default = base behaviour)
    USE_ENG=True,            # C10 two-engine fusion
    FUSED_GL=True,           # C10 GLACIER rerank of the fused list
    PROMOTE=True,            # C9 promotion rule
    PC_SIM_RERANK=True,      # C8 ICE/GL rerank of the PubChem list
    FINAL_PASS=True,         # C12 append-only tail
    # E006 knobs (casmi26-e006-window-40-ice-fallback-true, cells 28 + 30; default = base behaviour)
    ICE_FALLBACK=False,      # fused list: ICEBERG rerank where GLACIER is blind (ice and not gl)
    METRIC_FUSION=False,     # E004/E006 overlay: one RRF vote per metric identity per engine, after C12
    FUSION_WINDOW=None,      # fused shortlist size before the simulator rerank (None = FUSE_N; E006 uses 40)
    # engine2-fppair knobs: read by the Engine 2 cell upstream of the dump, no-ops in run() (replay reads stored ENG)
    ENGINE2_ROWS='nofp',     # 'nofp' (base) | 'fppair': "ours" ranker on sim_rank_rows_fppair.npz + leak-free FPNets
    ENGINE2_W_PV=0.88,       # Engine 2 blend weight of the pv ranker (base 0.88; fppair probe 0.50)
    # V22 baseline knobs (casmi26-v22-rel150-e2r2-clean; default = False/250.0 for backward compatibility)
    FRAG_GAP=False,          # v7c MetFrag-lite fragment rescore where GLACIER and ICEBERG are completely blind
    E2R2=False,              # E2R2: engine-2 top-1 candidate inserted at rank 2 of every eligible row
    # e180-drop-dreamsnp-pclite knobs (default False/None = base behaviour)
    E180_DROP=False,         # remove R1b enveda-180-only structures from final rows before 25 cut and guard E2R2
    E180_KEYS=None,          # set/frozenset of ik14 or path to key file (default: E180_KEY_PATHS)
    DREAMS_FILL=False,       # mass-matched DreamsNP backfill (+/- 10 ppm) with e180_keys excluded
)

E180_KEY_PATHS = (
    '/kaggle/input/casmi26-e180-keys/e180_only_ik14.txt',
    'E:/casmi/data/e180/e180_only_ik14.txt',
)


# ---------------------------------------------------------------- C11 metric identity (copied from the notebook)
def _rd():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    return Chem, rdMolStandardize.TautomerEnumerator()


_RD = None


def canonicalize_smiles_to_ik14(smiles):
    global _RD
    if not isinstance(smiles, str):
        return None
    if _RD is None:
        _RD = _rd()
    Chem, taut = _RD
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(taut.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def heavy_atom_composition(smiles):
    global _RD
    if _RD is None:
        _RD = _rd()
    mol = _RD[0].MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


@lru_cache(maxsize=300000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None


def safe_extend(original, extra, limit=25):
    out, seen = [], set()
    for smi in list(original) + list(extra):
        smi = str(smi).strip()
        identity = metric_identity(smi)
        if identity is not None and identity not in seen:
            seen.add(identity)
            out.append(smi)
            if len(out) == limit:
                break
    return out


# ---------------------------------------------------------------- e180-drop helper functions
@lru_cache(maxsize=4)
def _read_keys(path):
    with open(path, encoding='utf-8') as f:
        return frozenset(line.strip() for line in f if line.strip())


def e180_keys(spec=None, state=None):
    """The enveda-180-only ik14 set.
    spec: a set/frozenset/list/dict of ik14, a file path, or None (checks state['e180_keys'] then E180_KEY_PATHS).
    Raises FileNotFoundError when nothing is found: a silent no-op is the L38 failure."""
    if spec is not None and not isinstance(spec, (str, os.PathLike)):
        keys = frozenset(spec)
    elif spec is not None:
        p = Path(spec)
        if not p.is_file():
            raise FileNotFoundError(f'E180_DROP=True but key file not found: {p}')
        keys = _read_keys(str(p))
    else:
        st_keys = state.get('e180_keys') if isinstance(state, dict) else None
        if st_keys is not None:
            keys = frozenset(st_keys)
        else:
            path = next((p for p in E180_KEY_PATHS if os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f'E180_DROP=True but no key file found (tried {E180_KEY_PATHS})')
            keys = _read_keys(str(path))
    if not keys:
        raise ValueError('E180_DROP=True but the key set is empty')
    return keys


def _is_keyed(smi, keys):
    ident = metric_identity(str(smi).strip())
    return ident is not None and ident[0] in keys


_DREAMS_CACHE = None


def _load_dreams(deps=None):
    global _DREAMS_CACHE
    if _DREAMS_CACHE is not None:
        return _DREAMS_CACHE
    if deps is not None:
        d = deps.get('dreams') if isinstance(deps, dict) else getattr(deps, 'dreams', None)
        if d is not None:
            return d
    candidates = [
        Path('E:/casmi/data/dreams'),
        Path('/kaggle/input/casmi26-dreams-np'),
    ]
    for p in candidates:
        mp = p / 'dreams_np_mass.npy'
        sp = p / 'dreams_np_smiles.npy'
        if mp.is_file() and sp.is_file():
            import numpy as np
            _DREAMS_CACHE = (np.load(mp), np.load(sp))
            return _DREAMS_CACHE
    try:
        import glob
        matches = glob.glob('/kaggle/input/**/dreams_np_mass.npy', recursive=True)
        if matches:
            p = Path(matches[0]).parent
            mp = p / 'dreams_np_mass.npy'
            sp = p / 'dreams_np_smiles.npy'
            if mp.is_file() and sp.is_file():
                import numpy as np
                _DREAMS_CACHE = (np.load(mp), np.load(sp))
                return _DREAMS_CACHE
    except Exception:
        pass
    return None


def _e180_final(out, keys, active, OVER, TAIL, ENG, PC, dropped, targets=None, dreams=None, dreams_fill=False, stats=None):
    """Step 2: on every active row still holding a keyed structure or left short (<25),
    drop keyed and back-fill from the C12 extras (keyed rows excluded) and optionally DreamsNP."""
    res = {}
    for mid, rows in out.items():
        rows = list(rows)
        n = sum(_is_keyed(s, keys) for s in rows[:25])
        short = len(rows) < 25 and (mid in dropped or n > 0 or (dreams_fill and dreams is not None))
        if mid not in active or not (n or short):
            res[mid] = rows
            continue
        if n:
            dropped[mid] = dropped.get(mid, 0) + n
        kept = [s for s in rows if not _is_keyed(s, keys)]
        extras = list(OVER.get(mid, [])) + list(TAIL.get(mid, []))
        extras.extend((ENG.get(mid) or {}).get('smiles', []))
        extras.extend((PC.get(mid) or {}).get('pc', []))
        non_keyed_extras = [s for s in extras if not _is_keyed(s, keys)]
        revised = safe_extend(kept, non_keyed_extras, limit=25)
        if dreams_fill and dreams is not None and len(revised) < 25 and targets:
            t = float(targets.get(mid, float('nan')))
            if math.isfinite(t) and t > 0:
                import numpy as np
                dm_mass, dm_smiles = dreams
                order_ = np.argsort(dm_mass, kind='stable'); srt = dm_mass[order_]
                lo, hi = np.searchsorted(srt, [t * (1 - 10e-6), t * (1 + 10e-6)])
                ix = order_[lo:hi]
                ix = ix[np.argsort(np.abs(dm_mass[ix] - t), kind='stable')]
                dm_cands = [x.decode() if isinstance(x, bytes) else str(x) for x in dm_smiles[ix]]
                dm_non_keyed = [s for s in dm_cands if not _is_keyed(s, keys)]
                before_len = len(revised)
                revised = safe_extend(revised, dm_non_keyed, limit=25)
                if stats is not None and len(revised) > before_len:
                    stats['dreams_added'] = stats.get('dreams_added', 0) + (len(revised) - before_len)
        res[mid] = revised or ['CCO']
    return res


# ---------------------------------------------------------------- C9 pure functions (copied from the notebook)
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


def promote(base, base_keys, pc, pc_keys, slots, n=25):
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40):
    """E006 cell 28, verbatim: RRF with one vote per metric identity per engine; rank = unique identities so far."""
    scores, representatives = {}, {}
    collapsed = 0
    for candidates, weight in ((first, 1.0), (second, alpha)):
        seen = set()
        for candidate in candidates:
            key = identity(candidate)
            if key is None:
                continue
            if key in seen:
                collapsed += 1
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
            representatives.setdefault(key, candidate)
    ordered = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed


def e2r2_make_key(identity, placeholder='CCO'):
    """SMILES -> metric identity, or None for a non-string, a blank string, the placeholder (or any SMILES with the
    placeholder's identity) and an unparseable SMILES. A token whose key is None is never inserted and never matched."""
    ph = identity(placeholder)

    def key(smiles):
        if not isinstance(smiles, str) or not smiles.strip() or smiles.strip() == placeholder:
            return None
        k = identity(smiles)
        return None if k is None or k == ph else k

    return key


def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO', e180_keys=None):
    """Put engine 2's top candidate x2 at rank 2 of one final list (best first). Pure function.

    x2 is the first entry of `eng_list` that is a non-blank string other than the placeholder
    (and not in e180_keys when provided). It is used only when key(x2) is not None; if that entry
    is unusable the row is left alone (later entries are not tried).
      empty row, or rank 1 is the placeholder  -> unchanged, 'skipped'
      no engine list                           -> unchanged, 'no_eng'
      x2 missing, unparseable or holds ';'     -> unchanged, 'invalid_x2'
      key(x2) == key(rank 1)                   -> unchanged, 'same_as_rank1'
      x2's identity already at rank 2          -> that token stays, deeper twins of it are dropped, 'already_rank2'
      otherwise                                -> every token with key(x2) is dropped, x2 goes to rank 2, the other
                                                  tokens keep their order, the list is cut to `limit`, 'inserted'
    Returns (new_toks, tag, moved_from). moved_from is 'absent' or the old 1-based rank (str) of x2's identity when
    x2 ends at rank 2, else None.
    """
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    if not eng_list:
        return toks, 'no_eng', None
    if e180_keys is not None:
        eng_list = [s for s in eng_list if not _is_keyed(s, e180_keys)]
        if not eng_list:
            return toks, 'no_eng', None
    x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', placeholder)), None)
    kx = key(x2) if x2 is not None and not any(c in x2 for c in ';\r\n') else None
    if kx is None:
        return toks, 'invalid_x2', None
    keys = [key(t) for t in toks]
    if kx == keys[0]:
        return toks, 'same_as_rank1', None
    hits = [i for i, k in enumerate(keys) if k == kx]
    if hits and hits[0] == 1:
        return [t for i, t in enumerate(toks) if i == 1 or keys[i] != kx][:limit], 'already_rank2', '2'
    rest = [t for t, k in zip(toks, keys) if k != kx]
    return (rest[:1] + [x2] + rest[1:])[:limit], 'inserted', (str(hits[0] + 1) if hits else 'absent')


_VAL_SPECTRA = None
_FRAG_LEVEL_CACHE = {}


def _load_val_spectra():
    global _VAL_SPECTRA
    if _VAL_SPECTRA is not None:
        return _VAL_SPECTRA
    _VAL_SPECTRA = {}
    try:
        from pathlib import Path
        import numpy as np
        import pandas as pd
        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / 'val_scripts' / 'val.parquet',
            Path('e:/ENVEDA_casmi/val_scripts/val.parquet'),
        ]
        vp = next((p for p in candidates if p.is_file()), None)
        if vp is not None:
            vq = pd.read_parquet(vp)
            for mid, g in vq.groupby('molecule_id', sort=False):
                _VAL_SPECTRA[str(mid)] = [
                    dict(mz=np.asarray(r.ms2_mzs, np.float64),
                         it=np.asarray(r.ms2_normalized_intensities, np.float64),
                         adduct=r.adduct, prec=float(r.precursor_mz),
                         mode=1 if r.ionization_mode == 'positive' else -1)
                    for r in g.itertuples()
                ][:16]
    except Exception:
        pass
    return _VAL_SPECTRA


def _resolve_frag_gap(state, prefusion, ENG, ICE, gl_of):
    fg = state.get('frag_gap')
    if isinstance(fg, dict) and fg:
        return {str(k): v for k, v in fg.items()}
    spectra_map = state.get('spectra')
    if not isinstance(spectra_map, dict) or not spectra_map:
        spectra_map = _load_val_spectra()
    if not spectra_map:
        return {}
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import frag_rescore
    except Exception:
        return {}
    out = {}
    for mid, s in prefusion.items():
        mid = str(mid)
        ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
        if gl_of(mid) or any(v is not None for v in (ice or {}).values()):
            continue
        spec = spectra_map.get(mid)
        if not spec:
            continue
        lst = [x for x in str(s).split(';') if x and x != 'CCO'] if isinstance(s, str) else list(s)
        e = ENG.get(mid) or {}
        cands = [x for x in dict.fromkeys(lst + list(e.get('smiles', []))) if x and x != 'CCO']
        if len(cands) < 2:
            continue
        try:
            sc = frag_rescore.score_molecule(spec, cands, cache=_FRAG_LEVEL_CACHE)
            out[mid] = {smi: float(v) for smi, v in zip(cands, sc) if v is not None}
        except Exception:
            pass
    return out


def _dep(deps, name):
    return deps[name] if isinstance(deps, dict) else getattr(deps, name)


def run(state, deps, stats=None, **overrides):
    """-> {str mid: [smiles, ...]} (<= 25 each), exactly the notebook's final submission rows."""
    c = dict(CFG)
    c.update(state.get('consts') or {})
    c.update(overrides)
    if 'REL_PROMO' in c and 'REL_PROMOTE' not in overrides:
        c['REL_PROMOTE'] = float(c['REL_PROMO'])
    elif 'REL_PROMO' in overrides:
        c['REL_PROMOTE'] = float(overrides['REL_PROMO'])
    stats = {} if stats is None else stats
    score_key, form = _dep(deps, 'score_key'), _dep(deps, 'form')
    ice_rerank, gl_multi = _dep(deps, 'ice_rerank'), _dep(deps, 'gl_rerank_multi')
    dreams = deps.get('dreams') if isinstance(deps, dict) else getattr(deps, 'dreams', None)
    if dreams is None and c.get('DREAMS_FILL'):
        dreams = _load_dreams(deps)

    BASE = {m: tuple(v) for m, v in state['base'].items()}
    PC = copy.deepcopy(state.get('pc') or {})
    ENG = state.get('eng') or {}
    ICE = state.get('ice') or {}
    GL = state.get('gl') or {}
    TAIL = {str(k): list(v[0]) for k, v in BASE.items()}          # BASE_TAIL_CANDIDATES (C7)

    def gl_of(m):
        g = GL.get(str(m), {}) if isinstance(GL, dict) else {}
        return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}

    def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n):
        if not gl:
            return order0
        try:
            return gl_multi(smis, keys, scs, forms, [ice, gl], [c['ICE_LAM'], c['GL_LAM']], top_n=top_n)
        except Exception:
            return order0

    # ---- C8 rerank loop
    for m in list(BASE):
        smis, keys, lib_max, scs, forms = BASE[m]
        ice = ICE.get(str(m), {}) if ICE else {}
        gl = gl_of(m)
        if c['BASE_ICE'] and ice and any(v is not None for v in ice.values()):
            try:
                order = ice_rerank(smis, keys, scs, forms, ice, lam=c['ICE_LAM'], top_n=c['TOPN'])
                order = gl_rerank(order, smis, keys, scs, forms, ice, gl, c['TOPN'])
                smis, keys = [smis[i] for i in order], [keys[i] for i in order]
            except Exception:
                pass
        BASE[m] = (smis[:25], keys[:25], lib_max)
        p = PC.get(m) if (c['ICE_PC'] and isinstance(PC, dict)) else None
        if c['PC_SIM_RERANK'] and p and p.get('pc_form') and ice:
            try:
                o = ice_rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=c['ICE_LAM'], top_n=25)
                o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25)
                fz0 = p['pc_fz'][0]
                p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
                p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
            except Exception:
                pass

    # ---- C9 gated PubChem merge
    rows = {}
    for mid in state['te_mids']:
        smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
        p = PC.get(mid)
        if p is None or not p['pc']:
            final = smis
        elif lib_max >= c['LIB_TAU']:
            final = smis
        else:
            bp = p['best_pool_fz']
            rel = ((p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - bp) if math.isfinite(bp) else 1e9
            slots = c['SLOTS_AGG'] if rel > c['REL_TH'] else c['SLOTS_GENTLE']
            S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
            spop = S is not None and float(S) > c['S_TAU'] and pop_top >= c['POP_TAU']
            if c['PROMOTE'] and (spop or (rel > c['REL_PROMOTE'])):
                stats['promoted'] = stats.get('promoted', 0) + 1
                final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
            else:
                if c['PROMOTE'] and not spop and 250.0 < rel <= c['REL_PROMOTE']:
                    stats['promote_rel_skipped'] = stats.get('promote_rel_skipped', 0) + 1
                final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
        if not final:
            final = ['CCO']
        rows[str(mid)] = ';'.join(final[:25])
    sub = {str(m): rows.get(str(m), 'CCO') for m in state['sample_ids']}

    # ---- C10 two-engine fusion
    def fuse2(v_smis, e_smis, e_keys, n):
        sc, smi_of = {}, {}
        for r, s in enumerate(v_smis, 1):
            k = score_key(s) or s
            sc[k] = sc.get(k, 0.0) + 1.0 / (c['KRR'] + r); smi_of.setdefault(k, s)
        for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
            if not k:
                continue
            sc[k] = sc.get(k, 0.0) + c['ALPHA'] / (c['KRR'] + r); smi_of.setdefault(k, s)
        order = sorted(sc, key=lambda k: -sc[k])[:n]
        return [smi_of[k] for k in order], order, [sc[k] for k in order]

    OVER = {}
    PREFUSION = dict(sub)                                         # E006 PREFUSION_LISTS (C9 output)
    window = c['FUSE_N'] if c['FUSION_WINDOW'] is None else int(c['FUSION_WINDOW'])
    # with METRIC_FUSION the base C10 stays untouched (it is E006's reference); the fallback lives in the overlay
    base_fallback = c['ICE_FALLBACK'] and not c['METRIC_FUSION']
    if ENG and c['USE_ENG']:
        for mid, s in list(sub.items()):
            vs = [x for x in s.split(';') if x and x != 'CCO']
            e = ENG.get(str(mid))
            if e and e['smiles']:
                fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'], c['FUSE_N'] if c['METRIC_FUSION'] else window)
                ice = ICE.get(str(mid), {}) if isinstance(ICE, dict) else {}
                if ice:
                    try:
                        fforms = [form(x) for x in fsm]
                        ice_col = ice if c['CELL19_ICE'] else {}
                        o = ice_rerank(fsm, fk, fsc, fforms, ice_col, lam=c['ICE_LAM'], top_n=len(fsm))
                        g = gl_of(mid)
                        if g and c['FUSED_GL']:
                            o = gl_multi(fsm, fk, fsc, fforms, [ice_col, g], [c['ICE_LAM'], c['GL_LAM']], top_n=len(fsm))
                        elif base_fallback and not g:
                            o = ice_rerank(fsm, fk, fsc, fforms, ice, lam=c['ICE_LAM'], top_n=len(fsm))
                            stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1
                        fsm = [fsm[i] for i in o]
                    except Exception:
                        pass
                OVER[str(mid)] = list(fsm[25:])
                vs = fsm[:25]
            sub[mid] = ';'.join(vs[:25]) if vs else 'CCO'

    active = {str(m) for m in state['te_mids']}
    if not c['FINAL_PASS']:
        out = {m: s.split(';') for m, s in sub.items()}
    else:
        out = _final_pass(sub, state, active, OVER, TAIL, ENG, PC, dreams if not c.get('E180_DROP') else None)

    e180, dropped = (e180_keys(c['E180_KEYS'], state=state) if c['E180_DROP'] else None), {}

    if c['METRIC_FUSION'] and ENG and c['USE_ENG']:
        frag_gap = _resolve_frag_gap(state, PREFUSION, ENG, ICE, gl_of) if c.get('FRAG_GAP') else {}
        out = _metric_overlay(out, PREFUSION, active, ENG, ICE, gl_of, c, window, score_key, form,
                              ice_rerank, gl_multi, stats, frag_gap=frag_gap, e180=e180, dropped=dropped)

    if e180 is not None:
        out = _e180_final(out, e180, active, OVER, TAIL, ENG, PC, dropped,
                          targets=state.get('targets'), dreams=dreams,
                          dreams_fill=c.get('DREAMS_FILL', False), stats=stats)
        stats['e180_dropped'] = sum(dropped.values())
        stats['e180_mols'] = len(dropped)

    if c.get('E2R2'):
        key = e2r2_make_key(metric_identity)
        for mid in list(out):
            if mid not in active:
                continue
            e = ENG.get(mid)
            if e is None and mid.strip().lstrip('-').isdigit():
                e = ENG.get(int(mid))
            eng_list = (e.get('smiles') or []) if isinstance(e, dict) else (list(e) if isinstance(e, (list, tuple)) else [])
            if e180 is not None:
                raw_x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', 'CCO')), None)
                if raw_x2 is not None and _is_keyed(raw_x2, e180):
                    stats['e2r2_e180_blocked'] = stats.get('e2r2_e180_blocked', 0) + 1
            new_toks, tag, _ = e2r2_row(out[mid], eng_list, key, e180_keys=e180)
            out[mid] = new_toks
            if tag == 'inserted':
                stats['e2r2_inserted'] = stats.get('e2r2_inserted', 0) + 1
            elif tag == 'already_rank2':
                stats['e2r2_already_rank2'] = stats.get('e2r2_already_rank2', 0) + 1
    return out


def _metric_overlay(out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi,
                    stats, frag_gap=None, e180=None, dropped=None):
    """E006 cell 30 rerank_row (cell 28 metric_fusion + GL-present gate + optional ICE fallback / FRAG_GAP), applied to
    the final rows: `original` is the reference E006 reads (C12 output, i.e. submission_reference.csv)."""
    frag_gap = frag_gap or {}
    res = {}
    for mid, original in out.items():
        original = list(original)
        engine = ENG.get(mid)
        if mid not in active or mid not in prefusion or not engine or not engine.get('smiles'):
            res[mid] = original
            continue
        first = [s for s in prefusion[mid].split(';') if s and s != 'CCO']
        cands, scores, collapsed = metric_fusion(first, engine['smiles'], metric_identity,
                                                 alpha=c['ALPHA'], offset=c['KRR'], limit=window)
        stats['collapsed_votes'] = stats.get('collapsed_votes', 0) + collapsed
        keys = [score_key(s) or s for s in cands]
        ice = ICE.get(mid, {}) if isinstance(ICE, dict) else {}
        gl = gl_of(mid)
        try:
            if ice and gl and c['FUSED_GL']:
                order = gl_multi(cands, keys, scores, [form(s) for s in cands], [{}, gl],
                                 [c['ICE_LAM'], c['GL_LAM']], top_n=len(cands))
                cands = [cands[i] for i in order]
            elif (c['ICE_FALLBACK'] and not gl and frag_gap.get(mid)
                  and not any(v is not None for v in (ice or {}).values())):
                order = gl_multi(cands, keys, scores, [form(s) for s in cands],
                                 [frag_gap[mid]], [0.5], top_n=len(cands))
                cands = [cands[i] for i in order]
                stats['frag_gap'] = stats.get('frag_gap', 0) + 1
            elif c['ICE_FALLBACK'] and ice and not gl:
                order = ice_rerank(cands, keys, scores, [form(s) for s in cands], ice,
                                   lam=c['ICE_LAM'], top_n=len(cands))
                cands = [cands[i] for i in order]
                stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1
        except Exception:
            pass
        row = safe_extend(cands[:25], cands[25:] + original) if cands else original
        if e180 is not None and cands:
            n = sum(_is_keyed(s, e180) for s in row[:25])
            if n:
                kept = [s for s in cands if not _is_keyed(s, e180)]
                orig = [s for s in original if not _is_keyed(s, e180)]
                row = safe_extend(kept[:25], kept[25:] + orig) or row
                if dropped is not None:
                    dropped[mid] = dropped.get(mid, 0) + n
        res[mid] = row
    return res


def _final_pass(sub, state, active, OVER, TAIL, ENG, PC, dreams):
    # ---- C12 metric-safe tail
    targets = {str(k): v for k, v in (state.get('targets') or {}).items()}
    out = {}
    for mid, s in sub.items():
        if mid not in active:
            out[mid] = s.split(';')
            continue
        original = [x.strip() for x in str(s).split(';') if x.strip()]
        clean = safe_extend(original, [])
        extras = list(OVER.get(mid, [])) + list(TAIL.get(mid, []))
        extras.extend((ENG.get(mid) or {}).get('smiles', []))
        p = PC.get(mid) or {}
        extras.extend(p.get('pc', []))
        revised = safe_extend(clean, extras)
        t = float(targets.get(mid, float('nan')))
        if len(revised) < 25 and dreams is not None and math.isfinite(t) and t > 0:
            import numpy as np
            dm_mass, dm_smiles = dreams
            order_ = np.argsort(dm_mass, kind='stable'); srt = dm_mass[order_]
            lo, hi = np.searchsorted(srt, [t * (1 - 10e-6), t * (1 + 10e-6)])
            ix = order_[lo:hi]
            ix = ix[np.argsort(np.abs(dm_mass[ix] - t), kind='stable')]
            revised = safe_extend(revised, [x.decode() if isinstance(x, bytes) else str(x) for x in dm_smiles[ix]])
        out[mid] = revised
    return out
