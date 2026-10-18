"""A3 worker: one molecule -> a compact, truth-free record for iso-rank Stage A.

record = dict(mid, row, parents, kids, keymap, n_children, gen_s, frag_s, nl_s, stats, n_spec)
  row      the V22 final top-25 (SMILES)
  parents  [{smiles, ik14 (plain), rank (1-based fused rank), frag}] for row[:10]
  kids     [{smiles, ik14 (plain), op, pairs [(parent_index, op)], frag, nl}] the frag-cut top K (K = 150)
  keymap   {smiles: (plain ik14, tautomer-canonical ik14)} for the row and every kid
Truth is never an input of anything in this module."""
from __future__ import annotations

import time

from common import CAP, DEADLINE, N_PARENTS, TOP, gen_edges

_R = {}


def init():
    import casmi_post
    import frag_rescore
    import isomer_mut
    import iso_nl
    _R.update(cp=casmi_post, fr=frag_rescore, im=isomer_mut, nl=iso_nl)


def _keys_of(smiles):
    cp, im = _R["cp"], _R["im"]
    out = {}
    for s in smiles:
        if s not in out:
            try:
                taut = cp.canonicalize_smiles_to_ik14(s)
            except Exception:
                taut = None
            out[s] = (im.ik14_plain(s), taut)
    return out


def process(task):
    """task = dict(mid, row, spectra, exclude (list of SMILES whose plain IK14s are excluded), frozen (None | list of
    (smiles, ik14, op, frag) children to use instead of generating a new frag cut), k)"""
    fr, im, nl = _R["fr"], _R["im"], _R["nl"]
    mid, row, spectra = task["mid"], list(task["row"]), task.get("spectra") or []
    k_top = task.get("k", TOP)
    parents = row[:N_PARENTS]
    pks = [im.ik14_plain(p) for p in parents]
    excl = {k for k in (im.ik14_plain(s) for s in (task.get("exclude") or row)) if k}
    st = {}
    t0 = time.monotonic()
    sel, pairs = gen_edges(parents, cap=CAP, deadline_s=DEADLINE, exclude_keys=excl, stats=st)
    t1 = time.monotonic()
    cache = {}
    try:
        pfrag = fr.score_molecule(spectra, parents, cache=cache) if spectra else [None] * len(parents)
    except Exception as e:
        st["parent_frag_error"] = repr(e)[:120]
        pfrag = [None] * len(parents)
    frozen = task.get("frozen")
    if frozen is not None:     # pseudo_c3: the I1b.0 frozen top-150 (its frag score, its SMILES); edges from regeneration
        kids = [dict(smiles=s, ik14=k, op=op, frag=f, pairs=pairs.get(k, [])) for s, k, op, f in frozen]
        st["frozen_without_edge"] = sum(1 for kd in kids if not kd["pairs"])
        n_total = len(sel)
    else:
        try:
            sc = fr.score_molecule(spectra, [c[0] for c in sel], cache=cache) if (sel and spectra) else [None] * len(sel)
        except Exception as e:
            st["score_error"] = repr(e)[:120]
            sc = [None] * len(sel)
        order = sorted(range(len(sel)), key=lambda i: (-(sc[i] if sc[i] is not None else -1.0), i))[:k_top]
        kids = [dict(smiles=sel[i][0], ik14=sel[i][1], op=sel[i][2], frag=sc[i], pairs=pairs.get(sel[i][1], []))
                for i in order]
        n_total = len(sel)
    t2 = time.monotonic()
    try:
        nls = nl.nl_scores(spectra, [kd["smiles"] for kd in kids]) if (kids and spectra) else []
    except Exception as e:
        st["nl_error"] = repr(e)[:120]
        nls = []
    for i, kd in enumerate(kids):
        v = float(nls[i]) if len(nls) > i else float("nan")
        kd["nl"] = None if v != v else v
    t3 = time.monotonic()
    keymap = _keys_of(list(row) + [kd["smiles"] for kd in kids])
    return dict(mid=mid, row=row,
                parents=[dict(smiles=p, ik14=pks[i], rank=i + 1, frag=pfrag[i]) for i, p in enumerate(parents)],
                kids=kids, keymap=keymap, n_children=n_total, gen_s=t1 - t0, frag_s=t2 - t1, nl_s=t3 - t2,
                keymap_s=time.monotonic() - t3, stats=st, n_spec=len(spectra))
