"""Post-scoring stage of casmi-notebook-0-433 as one importable function (the harness's replay engine).

Mirrors, line for line, what the notebook does after ICEBERG/GLACIER have scored:
  C8 rerank loop  (BASE truncation to 25, BASE_ICE rerank, PubChem-list ICE/GL rerank)
  C9              (gated PubChem merge / promotion -> per-molecule list)
  C10             (two-engine RRF fusion + fused-list GLACIER rerank, FUSION_OVERFLOW)
  C12             (metric-safe dedupe + append-only tail; optional DreaMS mass fill)
  E006 knobs      (off by default) METRIC_FUSION: E004/E006 metric-identity RRF overlay on the final rows;
                  ICE_FALLBACK: ICEBERG rerank of the fused list where GLACIER is blind; FUSION_WINDOW: shortlist size

The notebook keeps its own cells; build/patch_harness_dump.py snapshots the state just before the C8 rerank loop
and, at the end of the run, asserts run(state, deps) == the notebook's submission.csv for every molecule
(`POST EQUIV`). A probe adds a knob to CFG (default = the base behaviour), is measured by research/harness on a
val-mode dump, and ships by letting the notebook write run()'s output.

state (dict, pickled by the dump):
  base         {mid: (smis, keys, lib_max, scores, forms)}   pre-rerank BASE, top TOPN (C7 output)
  pc           {mid: {...}}   PubChem-only lists after the ICE_PC block (pc, pc_keys, pc_fz, pc_form, fz_top,
                              best_pool_fz, S, top_pop)
  eng          {str mid: {smiles, keys}}   Engine 2 top 40
  ice, gl      {str mid: {...}}            ICE_SCORES, GL_SCORES as the notebook holds them
  te_mids      molecule ids in te order (te.molecule_id.unique())
  sample_ids   sample_submission molecule ids, in file order
  targets      {mid: median neutral mass}
  consts       the notebook's constants (override CFG defaults)
deps (object or dict): score_key(smiles), ice_rerank(...) = fuse.rerank, gl_rerank_multi(...) = gl_fuse.rerank_multi,
  form(smiles) -> formula, dreams = None | (mass array, smiles array)
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache

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
    # A6: f.z common-scale merge in C9 (replaces merge + promote when the PC row carries pool_fz + pc_rawfz)
    FZ_MERGE=None,           # None = off (base); a float = the margin (raw f.z units) and turns it on
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


def fz_merge(base, base_keys, pc, pc_keys, pc_fz, pool_fz, margin, n=25):
    """A6. BASE keeps its order; PubChem-only (keys not in BASE) keeps its order. Envelope e_i = max pool f.z over
    BASE[i:] (suffix max over the keys known in pool_fz; unknown are skipped; none known -> -inf). Walk i (BASE) and
    j (PC): emit PC[j] before BASE[i] iff pc_fz[j] > e_i + margin, else emit BASE[i]; then drain. Dedup by key, cut n.
    pc_fz must be RAW f.z (same scale as pool_fz), aligned with pc/pc_keys."""
    import math
    ninf = -math.inf
    nb = len(base)
    env = [ninf] * (nb + 1)
    for i in range(nb - 1, -1, -1):
        v = pool_fz.get(base_keys[i])
        v = float(v) if v is not None and math.isfinite(float(v)) else ninf
        env[i] = max(env[i + 1], v)
    bk = set(base_keys)
    pcs, seen = [], set()
    for s, k, f in zip(pc, pc_keys, pc_fz):
        if k in bk or k in seen:
            continue
        seen.add(k)
        pcs.append((s, k, None if f is None else float(f)))
    out, used, i, j = [], set(), 0, 0

    def emit(s, k):
        if k not in used:
            used.add(k); out.append(s)
    while i < nb and j < len(pcs):
        f = pcs[j][2]
        if margin == math.inf:
            thr = math.inf
        elif env[i] == ninf:
            thr = ninf
        else:
            thr = env[i] + margin
        if f is not None and math.isfinite(f) and f > thr:
            emit(pcs[j][0], pcs[j][1]); j += 1
        else:
            emit(base[i], base_keys[i]); i += 1
    for ii in range(i, nb):
        emit(base[ii], base_keys[ii])
    for jj in range(j, len(pcs)):
        emit(pcs[jj][0], pcs[jj][1])
    return out[:n]


def fz_merge_row(p, smis, keys, margin, n=25):
    """C9 row for one molecule: the fz_merge list, or None when the PC row lacks pool_fz / raw f.z (-> base path).
    p['_fzmap'] ({key: raw f.z}) is built before any PubChem-list rerank; else zip(pc_keys, pc_rawfz) is used."""
    pool_fz = p.get('pool_fz')
    fmap = p.get('_fzmap')
    if fmap is None and p.get('pc_rawfz') is not None and len(p['pc_rawfz']) == len(p['pc_keys']):
        fmap = dict(zip(p['pc_keys'], p['pc_rawfz']))
    if not pool_fz or not fmap:
        return None
    return fz_merge(smis, keys, p['pc'], p['pc_keys'], [fmap.get(k) for k in p['pc_keys']], pool_fz, margin, n=n)


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


def _dep(deps, name):
    return deps[name] if isinstance(deps, dict) else getattr(deps, name)


def run(state, deps, stats=None, **overrides):
    """-> {str mid: [smiles, ...]} (<= 25 each), exactly the notebook's final submission rows."""
    c = dict(CFG)
    c.update(state.get('consts') or {})
    c.update(overrides)
    stats = {} if stats is None else stats
    score_key, form = _dep(deps, 'score_key'), _dep(deps, 'form')
    ice_rerank, gl_multi = _dep(deps, 'ice_rerank'), _dep(deps, 'gl_rerank_multi')
    dreams = deps.get('dreams') if isinstance(deps, dict) else getattr(deps, 'dreams', None)

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

    # ---- A6: raw f.z per PubChem key, taken before the C8 PubChem rerank reorders pc/pc_keys
    if c['FZ_MERGE'] is not None:
        for p in PC.values():
            if isinstance(p, dict) and p.get('pc_rawfz') is not None and len(p['pc_rawfz']) == len(p.get('pc_keys', [])):
                p['_fzmap'] = dict(zip(p['pc_keys'], p['pc_rawfz']))

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
            if c['FZ_MERGE'] is not None:
                fm = None
                try:
                    fm = fz_merge_row(p, smis, keys, float(c['FZ_MERGE']))
                except Exception:
                    stats['fz_merge_err'] = stats.get('fz_merge_err', 0) + 1
                if fm is None:
                    stats['fz_merge_nofz'] = stats.get('fz_merge_nofz', 0) + 1
                else:
                    stats['fz_merge_mols'] = stats.get('fz_merge_mols', 0) + 1
                    stats['fz_merge_changed_vs_base'] = stats.get('fz_merge_changed_vs_base', 0) + int(fm[:25] != final[:25])
                    slot = next((q for q, s in enumerate(fm, 1) if s == p['pc'][0]), 0)
                    stats[f'fz_merge_pc1_slot_{slot}'] = stats.get(f'fz_merge_pc1_slot_{slot}', 0) + 1
                    final = fm
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
        out = _final_pass(sub, state, active, OVER, TAIL, ENG, PC, dreams)
    if c['METRIC_FUSION'] and ENG and c['USE_ENG']:
        out = _metric_overlay(out, PREFUSION, active, ENG, ICE, gl_of, c, window, score_key, form,
                              ice_rerank, gl_multi, stats)
    return out


def _metric_overlay(out, prefusion, active, ENG, ICE, gl_of, c, window, score_key, form, ice_rerank, gl_multi,
                    stats):
    """E006 cell 30 rerank_row (cell 28 metric_fusion + GL-present gate + optional ICE fallback), applied to the
    final rows: `original` is the reference E006 reads (C12 output, i.e. submission_reference.csv)."""
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
            elif c['ICE_FALLBACK'] and ice and not gl:
                order = ice_rerank(cands, keys, scores, [form(s) for s in cands], ice,
                                   lam=c['ICE_LAM'], top_n=len(cands))
                cands = [cands[i] for i in order]
                stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1
        except Exception:
            pass
        res[mid] = safe_extend(cands[:25], cands[25:] + original) if cands else original
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
