"""iso-rank: label-blind RRF of isomer children and their placement below the parent in a final top-25 row.

Pure functions, no I/O. Contract: research/c3/20261018-iso-rank/spec.md.

children  list of dicts {smiles, ik14, parent_ik14, parent_rank, op}, one per generated edge (ik14 = plain InChIKey[:14]).
scores    {list_name: {ik14: value}} for the lists ice, frag, nl, gl (higher better); mult and parent_rank are derived
          from the parent map when absent. A list missing for an ik14 is skipped for that ik14.
rec       {children, scores, parent_scores: {ik14: {ice, frag}}} for one molecule.
"""
from __future__ import annotations

import math
from functools import lru_cache

LISTS = ('ice', 'frag', 'mult', 'parent_rank', 'nl', 'gl')
LOWER_BETTER = frozenset({'parent_rank'})
DEFAULT_CFG = dict(ISO_M_ICE=0.0, ISO_RRF_K=10, ISO_LISTS=LISTS, ISO_TAIL=True)
ROW_LEN = 25
TAIL_START = 20


def parent_map(children):
    """Edges -> {ik14: {smiles, ik14, parent_ik14 (best-ranked parent), parent_rank, multiplicity}}.
    multiplicity = number of distinct (parent_ik14, op) pairs."""
    pairs, best, smi = {}, {}, {}
    for ch in children:
        k = ch.get('ik14')
        if not k:
            continue
        pairs.setdefault(k, set()).add((ch.get('parent_ik14'), ch.get('op')))
        smi.setdefault(k, ch.get('smiles'))
        pr = ch.get('parent_rank')
        pr = math.inf if pr is None else pr
        if k not in best or pr < best[k][0]:
            best[k] = (pr, ch.get('parent_ik14'))
    return {k: dict(smiles=smi[k], ik14=k, parent_ik14=best[k][1], parent_rank=best[k][0],
                    multiplicity=len(pairs[k])) for k in best}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _avg_ranks(items, lower_better):
    """items [(key, value)] -> {key: 1-based rank}, ties take the average rank."""
    srt = sorted(items, key=lambda kv: (kv[1] if lower_better else -kv[1]))
    out, i = {}, 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1][1] == srt[i][1]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            out[srt[t][0]] = r
        i = j + 1
    return out


def rrf(records, scores, k=10, lists=LISTS):
    """records {ik14: record} (parent_map output); -> ik14s sorted by RRF score desc, ties by ik14."""
    scores = scores or {}
    total = {ik: 0.0 for ik in records}
    for name in lists:
        if name == 'mult' and 'mult' not in scores:
            vals = {ik: r['multiplicity'] for ik, r in records.items()}
        elif name == 'parent_rank' and 'parent_rank' not in scores:
            vals = {ik: r['parent_rank'] for ik, r in records.items()}
        else:
            vals = scores.get(name) or {}
        items = [(ik, _num(vals.get(ik))) for ik in records]
        items = [(ik, v) for ik, v in items if v is not None]
        for ik, r in _avg_ranks(items, name in LOWER_BETTER).items():
            total[ik] += 1.0 / (k + r)
    return sorted(total, key=lambda ik: (-total[ik], ik))


@lru_cache(maxsize=200000)
def _keys(entry):
    """{plain InChIKey[:14], tautomer-canonical InChIKey[:14]} of a SMILES; a bare 14-letter key is its own key."""
    if not isinstance(entry, str):
        return frozenset()
    s = entry.strip()
    if len(s) == 14 and s.isalpha() and s.isupper():
        return frozenset({s})
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        m = Chem.MolFromSmiles(s)
        if m is None:
            return frozenset()
        Chem.RemoveStereochemistry(m)
        out = {Chem.MolToInchiKey(m)[:14]}
        out.add(Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(m))[:14])
        return frozenset(out)
    except Exception:
        return frozenset()


def _cfg(cfg):
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    return c


def apply(row, rec, cfg=None):
    """row: final top-25 entries (SMILES or IK14), best first. -> (new_row, {fired, tail}).

    Fire: the RRF-best child c not already in the row, with parent p in the row, fires iff ICE(c), ICE(p), frag(c),
    frag(p) all exist, ICE(c) - ICE(p) > ISO_M_ICE and frag(c) > frag(p). c goes directly below p (never rank 1; a
    parent at rank 25 leaves no room, no fire). IK14 dedupe, cut at 25. ISO_TAIL: ranks 21-25 (the fired child and its parent keep their slots,
    if any) are replaced by the next RRF children that are not in the row; originals only pad when the children
    run out."""
    c = _cfg(cfg)
    row = list(row)
    info = dict(fired=False, tail=0)
    children = (rec or {}).get('children') or []
    if not row or not children:
        return row, info
    recs = parent_map(children)
    if not recs:
        return row, info
    scores = rec.get('scores') or {}
    pscores = rec.get('parent_scores') or {}
    order = rrf(recs, scores, k=c['ISO_RRF_K'], lists=tuple(c['ISO_LISTS']))
    row_keys = [_keys(e) for e in row]

    def child_keys(ik):
        return set(_keys(recs[ik]['smiles'])) | {ik}

    # ---- fire
    cand = next((ik for ik in order if not any(child_keys(ik) & rk for rk in row_keys)), None)
    ins_idx = None
    if cand is not None:
        r = recs[cand]
        pos = next((i for i, rk in enumerate(row_keys) if r['parent_ik14'] in rk), None)
        ice_c, frag_c = _num((scores.get('ice') or {}).get(cand)), _num((scores.get('frag') or {}).get(cand))
        ps = pscores.get(r['parent_ik14']) or {}
        ice_p, frag_p = _num(ps.get('ice')), _num(ps.get('frag'))
        if (pos is not None and pos + 1 < ROW_LEN and None not in (ice_c, frag_c, ice_p, frag_p)
                and ice_c - ice_p > c['ISO_M_ICE'] and frag_c > frag_p):
            ins_idx = pos + 1
            row.insert(ins_idx, r['smiles'])
            row_keys.insert(ins_idx, frozenset(child_keys(cand)))
            info['fired'] = True
    fired_keys = set(row_keys[ins_idx]) if ins_idx is not None else set()
    row, row_keys = row[:ROW_LEN], row_keys[:ROW_LEN]
    if ins_idx is not None and ins_idx >= ROW_LEN:
        ins_idx = None

    # ---- tail ride-along
    if c['ISO_TAIL'] and len(row) > TAIL_START:
        used = set(fired_keys)
        for rk in row_keys[:TAIL_START]:
            used |= set(rk)
        pool = []
        for ik in order:
            ks = child_keys(ik)
            if ks & used:
                continue
            used |= ks
            pool.append(recs[ik]['smiles'])
        # the fired child and its parent keep their slots: c must stay directly below p
        keep_idx = {ins_idx, ins_idx - 1} if ins_idx is not None else set()
        slots = [i for i in range(TAIL_START, ROW_LEN) if i not in keep_idx]
        slots = [i for i in slots if i < len(row)]
        n = min(len(pool), len(slots))
        if n:
            taken = set()
            for s in pool[:n]:
                taken |= set(_keys(s))
            orig = [row[i] for i in slots if not (set(row_keys[i]) & taken)]
            for s, e in zip(slots, (pool[:n] + orig)[:len(slots)]):
                row[s] = e
            info['tail'] = n
    return row[:ROW_LEN], info
