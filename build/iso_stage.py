"""iso-rank fast-ship stage (research/c3/20261018-iso-rank): pure helpers the V22 notebook's last cell calls.

No truth, no FPNet. Phases (the notebook cell, build/patch_iso_rank.py, glues them to ICEBERG / GLACIER):
  generate()  CPU, spawn Pool: parents = final row top N_PARENTS -> isomer_mut.mutate_many (cap, per-molecule deadline,
              retrieved IK14s dropped) -> CPU frag on all children + parents (measured spectra) -> top K by frag -> NL.
  cands_for() what ICEBERG / GLACIER must score: parents + the K children.
  assemble()  per molecule: records -> iso_rank.apply (fast-ship cfg) -> new rows + counters.
"""
from __future__ import annotations

import time

N_PARENTS = 10
CAP = 600
DEADLINE = 120.0
FAST_CFG = dict(ISO_M_ICE=0.0, ISO_RRF_K=10, ISO_LISTS=('ice', 'frag', 'mult', 'parent_rank', 'nl', 'gl'), ISO_TAIL=True)
_R = {}


def _init():
    import frag_rescore
    import isomer_mut
    import iso_nl
    _R.update(fr=frag_rescore, im=isomer_mut, nl=iso_nl, cache={})
    try:
        frag_rescore.warmup()
    except Exception:
        pass


def _work(task):
    """task = (mid, row, spectra, t_limit, k). Returns a dict; skipped=True when the wall-clock guard fired first."""
    mid, row, spectra, t_limit, k = task
    if time.time() > t_limit:
        return dict(mid=mid, skipped=True)
    im, fr, nl = _R['im'], _R['fr'], _R['nl']
    t0 = time.monotonic()
    row = [s for s in row if isinstance(s, str) and s]
    parents = row[:N_PARENTS]
    st = {}
    out = dict(mid=mid, skipped=False, edges=[], frag={}, nl={}, parent_frag={}, n_children=0, stats=st)
    try:
        exclude = {kk for kk in (im.ik14_plain(s) for s in row) if kk}
        ch = im.mutate_many(parents, cap=CAP, deadline=t0 + DEADLINE, exclude_keys=exclude, stats=st)
        out['n_children'] = len(ch)
        out['gen_s'] = time.monotonic() - t0
        if not ch:
            return out
        pik = {p: im.ik14_plain(p) for p in parents}
        prank = {}
        for i, p in enumerate(row):
            prank.setdefault(im.ik14_plain(p), i + 1)
        _R['cache'].clear()
        fsc = [None] * len(ch)
        pfr = [None] * len(parents)
        if spectra:
            try:
                res = fr.score_molecule(spectra, [c[0] for c in ch] + parents, cache=_R['cache'])
                fsc, pfr = res[:len(ch)], res[len(ch):]
            except Exception as e:
                st['frag_error'] = repr(e)[:120]
        order = sorted(range(len(ch)), key=lambda i: (-(fsc[i] if fsc[i] is not None else -1.0), i))[:k]
        top = [ch[i] for i in order]
        for i in order:
            c = ch[i]
            out['edges'].append(dict(smiles=c[0], ik14=c[1], op=c[2], parent_ik14=pik.get(c[3]) or im.ik14_plain(c[3]),
                                     parent_rank=prank.get(pik.get(c[3]), N_PARENTS + 1)))
            if fsc[i] is not None:
                out['frag'][c[1]] = float(fsc[i])
        for p, v in zip(parents, pfr):
            if v is not None:
                out['parent_frag'][pik[p]] = float(v)
        if spectra and top:
            try:
                arr = nl.nl_scores(spectra, [c[0] for c in top], cache={})
                for c, v in zip(top, arr):
                    if v == v:
                        out['nl'][c[1]] = float(v)
            except Exception as e:
                st['nl_error'] = repr(e)[:120]
        out['total_s'] = time.monotonic() - t0
    except Exception as e:  # keep the molecule unchanged, never kill the pool
        out = dict(mid=mid, skipped=False, edges=[], frag={}, nl={}, parent_frag={}, n_children=0,
                   stats=dict(error=repr(e)[:200]))
    return out


def generate(tasks, procs=4, log=print):
    """tasks: list of (mid, row, spectra, t_limit, k) -> {mid: result}. spawn Pool (the notebook holds CUDA state)."""
    import multiprocessing as mp
    res, t0 = {}, time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(procs, _init) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, tasks), 1):
            res[r['mid']] = r
            if i % 25 == 0 or i == len(tasks):
                log(f'ISO_GEN {i}/{len(tasks)} {time.time() - t0:.0f}s')
    return res


def cands_for(rows, results, n_parents=N_PARENTS):
    """{mid: [parent smiles..., child smiles...]} for ICEBERG; molecules without children are left out."""
    out = {}
    for mid, r in results.items():
        if r.get('skipped') or not r.get('edges'):
            continue
        row = [s for s in rows[mid] if s][:n_parents]
        out[mid] = list(dict.fromkeys(row + [e['smiles'] for e in r['edges']]))
    return out


def _by_ik14(edges, sc):
    """{child smiles: v|None} -> {ik14: v}"""
    out = {}
    for e in edges:
        v = (sc or {}).get(e['smiles'])
        if v is not None:
            out[e['ik14']] = v
    return out


def assemble(rows, results, ice, gl, cfg=None, n_parents=N_PARENTS):
    """rows {mid: [smiles]}, results (generate), ice / gl {mid: {smiles: score|None}} -> (new_rows, counters)."""
    import iso_rank
    cfg = dict(FAST_CFG, **(cfg or {}))
    new = {m: list(v) for m, v in rows.items()}
    cnt = dict(iso_children=0, iso_fired=0, iso_tail=0, iso_skipped_time=0, n_mol=len(rows), iso_ice_mols=0, iso_gl_mols=0)
    for mid, r in results.items():
        if r.get('skipped'):
            cnt['iso_skipped_time'] += 1
            continue
        edges = r.get('edges') or []
        if not edges:
            continue
        cnt['iso_children'] += 1
        ic, g = (ice or {}).get(mid) or {}, (gl or {}).get(mid) or {}
        scores = dict(ice=_by_ik14(edges, ic), frag=dict(r['frag']), nl=dict(r['nl']), gl=_by_ik14(edges, g))
        cnt['iso_ice_mols'] += int(bool(scores['ice']))
        cnt['iso_gl_mols'] += int(bool(scores['gl']))
        row = [s for s in rows[mid] if s]
        pscores = {}
        for p in row[:n_parents]:
            pk = _plain(p)
            if pk:
                pscores[pk] = dict(ice=ic.get(p), frag=r['parent_frag'].get(pk))
        rec = dict(children=edges, scores=scores, parent_scores=pscores)
        out, info = iso_rank.apply(row, rec, cfg)
        cnt['iso_fired'] += int(info['fired'])
        cnt['iso_tail'] += int(info['tail'] > 0)
        new[mid] = out
    return new, cnt


def _plain(smi):
    import isomer_mut
    return isomer_mut.ik14_plain(smi)


def counters_line(cnt):
    return ('ISO_COUNTERS iso_children=%d iso_fired=%d iso_tail=%d iso_skipped_time=%d n_mol=%d'
            % (cnt['iso_children'], cnt['iso_fired'], cnt['iso_tail'], cnt['iso_skipped_time'], cnt['n_mol']))
