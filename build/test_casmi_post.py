"""Unit tests for build/casmi_post.py V22 baseline knobs (REL_PROMOTE / REL_PROMO, FRAG_GAP, E2R2) and base equivalence."""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402

SM = ['CCCCCCCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO', 'CC(C)O', 'CC(C)CO', 'OCCO', 'OCCCO', 'CC(O)CO']


def k(s):
    return cp.canonicalize_smiles_to_ik14(s)


def deps(gl_reverse=True):
    return dict(
        score_key=lambda s: cp.canonicalize_smiles_to_ik14(s),
        form=lambda s: 'F',
        ice_rerank=lambda smis, keys, scs, forms, ice, lam, top_n: list(range(len(smis)))[:top_n],
        gl_rerank_multi=(
            lambda smis, keys, scs, forms, cols, lams, top_n: list(range(len(smis)))[::-1][:top_n]
        ) if gl_reverse else (lambda *a, **kw: list(range(len(a[0])))),
        dreams=None,
    )


def state():
    base = {
        'm1': (SM[:3], [k(s) for s in SM[:3]], 0.95, [3.0, 2.0, 1.0], ['F'] * 3),
        'm2': (SM[:3], [k(s) for s in SM[:3]], 0.10, [3.0, 2.0, 1.0], ['F'] * 3),
        'm3': (SM[:2], [k(s) for s in SM[:2]], 0.95, [2.0, 1.0], ['F'] * 2),
    }
    pc = {
        'm1': dict(pc=[SM[5]], pc_keys=[k(SM[5])], pc_fz=[1.0], best_pool_fz=0.0, S=None, top_pop=0),
        'm2': dict(pc=[SM[5], SM[6]], pc_keys=[k(SM[5]), k(SM[6])], pc_fz=[900.0, 1.0], best_pool_fz=0.0, S=None, top_pop=0),
    }
    eng = {'m3': dict(smiles=[SM[7], SM[1]], keys=[k(SM[7]), k(SM[1])])}
    return dict(
        base=base, pc=pc, eng=eng, ice={'m3': {'x': 1.0}}, gl={'m3': {'x': 1.0}},
        te_mids=['m1', 'm2', 'm3'], sample_ids=['m1', 'm2', 'm3', 'mX'], targets={}, consts={}
    )


def test_backward_compatibility_defaults():
    assert cp.CFG['REL_PROMOTE'] == 250.0
    assert cp.CFG['FRAG_GAP'] is False
    assert cp.CFG['E2R2'] is False
    s = state()
    before = json.dumps(s, sort_keys=True, default=str)
    out = cp.run(s, deps())
    assert json.dumps(s, sort_keys=True, default=str) == before
    assert out['m1'][:3] == SM[:3]
    assert out['m2'][0] == SM[5]
    assert out['mX'] == ['CCO']


def test_rel_promote_150_and_rel_promo_alias():
    s = state()
    # Set rel = 200.0 (between 150.0 and 250.0)
    s['pc']['m2']['pc_fz'] = [200.0, 1.0]
    st_default, st_150, st_alias = {}, {}, {}
    out_def = cp.run(s, deps(), stats=st_default, USE_ENG=False, FINAL_PASS=False)
    out_150 = cp.run(s, deps(), stats=st_150, USE_ENG=False, FINAL_PASS=False, REL_PROMOTE=150.0)
    out_alias = cp.run(s, deps(), stats=st_alias, USE_ENG=False, FINAL_PASS=False, REL_PROMO=150)

    # Under default 250.0, rel=200.0 is NOT promoted (merged into gentle slot 4)
    assert out_def['m2'][0] == SM[0]
    assert st_default.get('promoted', 0) == 0

    # Under REL_PROMOTE=150.0 or REL_PROMO=150, rel=200.0 IS promoted to rank 1
    assert out_150['m2'][0] == SM[5]
    assert st_150.get('promoted', 0) == 1
    assert out_alias == out_150
    assert st_alias.get('promoted', 0) == 1


def test_frag_gap_rescoring_in_metric_overlay():
    s = state()
    # Make m3 completely blind in GLACIER and ICEBERG (all None or empty)
    s['gl'] = {'m3': {}}
    s['ice'] = {'m3': {'a': None, 'b': None}}
    s['frag_gap'] = {'m3': {SM[0]: 0.9, SM[1]: 0.1, SM[7]: 0.5}}

    # Capture gl_rerank_multi call arguments when FRAG_GAP=True
    calls = []

    def mock_gl_multi(cands, keys, scores, forms, cols, lams, top_n):
        calls.append((list(cands), list(cols), list(lams), top_n))
        # Sort candidates by the passed frag_gap score descending
        fg_col = cols[0]
        return sorted(range(len(cands)), key=lambda i: -fg_col.get(cands[i], 0.0))[:top_n]

    d = deps(gl_reverse=False)
    d['gl_rerank_multi'] = mock_gl_multi

    st_off, st_on = {}, {}
    out_off = cp.run(
        s, d, stats=st_off,
        METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, FRAG_GAP=False
    )
    assert len(calls) == 0
    assert st_off.get('frag_gap', 0) == 0

    out_on = cp.run(
        s, d, stats=st_on,
        METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, FRAG_GAP=True
    )
    assert len(calls) == 1
    assert calls[0][1] == [s['frag_gap']['m3']]
    assert calls[0][2] == [0.5]
    assert st_on.get('frag_gap') == 1
    assert out_on['m3'][0] == SM[0]
    assert out_on['m3'] != out_off['m3']


def test_frag_gap_skipped_when_ice_or_gl_present():
    s = state()
    s['gl'] = {}
    s['ice'] = {'m3': {SM[1]: 0.8}}  # finite ICEBERG score present
    s['frag_gap'] = {'m3': {SM[0]: 0.9, SM[1]: 0.1, SM[7]: 0.5}}
    st = {}
    cp.run(
        s, deps(gl_reverse=False), stats=st,
        METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, FRAG_GAP=True
    )
    assert st.get('frag_gap', 0) == 0
    assert st.get('ice_fallback', 0) == 1


def test_e2r2_row_pure_invariants():
    key = cp.e2r2_make_key(cp.metric_identity)

    # 1. Rank 1 is never displaced; x2 inserted at rank 2 when absent
    toks = [SM[0], SM[1], SM[2], SM[3]]
    new_toks, tag, moved = cp.e2r2_row(toks, [SM[7], SM[8]], key)
    assert tag == 'inserted' and moved == 'absent'
    assert new_toks == [SM[0], SM[7], SM[1], SM[2], SM[3]]
    assert new_toks[0] == toks[0]

    # 2. Metric twins of Rank 1 are NEVER inserted at Rank 2
    # 'CCCCO' and 'OCCCC' have identical metric_identity; stereo twins also share metric_identity
    toks_r1 = ['C[C@H](O)CC', SM[1], SM[2]]
    new_r1, tag_r1, moved_r1 = cp.e2r2_row(toks_r1, ['C[C@@H](O)CC'], key)
    assert tag_r1 == 'same_as_rank1' and moved_r1 is None
    assert new_r1 == toks_r1

    # 3. Already at Rank 2: stays at Rank 2, deeper metric twins are deduplicated
    toks_r2 = [SM[0], 'CCCCO', SM[1], 'OCCCC', SM[2]]
    new_r2, tag_r2, moved_r2 = cp.e2r2_row(toks_r2, ['OCCCC'], key)
    assert tag_r2 == 'already_rank2' and moved_r2 == '2'
    assert new_r2 == [SM[0], 'CCCCO', SM[1], SM[2]]

    # 4. Deeper rank promoted to Rank 2 and all deeper metric twins deduplicated
    toks_deep = [SM[0], SM[1], 'CCCCO', SM[2], 'OCCCC', SM[3]]
    new_deep, tag_deep, moved_deep = cp.e2r2_row(toks_deep, ['OCCCC'], key)
    assert tag_deep == 'inserted' and moved_deep == '3'
    assert new_deep == [SM[0], 'OCCCC', SM[1], SM[2], SM[3]]
    assert sum(key(t) == key('OCCCC') for t in new_deep) == 1

    # 5. List length stays <= 25 when inserting an absent candidate into a 25-element list
    long_toks = [f"{'C' * i}O" for i in range(3, 28)]  # 25 distinct alcohols C3..C27
    assert len(long_toks) == 25
    x2_new = 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCO'  # C28 alcohol
    new_long, tag_long, moved_long = cp.e2r2_row(long_toks, [x2_new], key, limit=25)
    assert tag_long == 'inserted' and moved_long == 'absent'
    assert len(new_long) == 25
    assert new_long[0] == long_toks[0]
    assert new_long[1] == x2_new
    assert new_long[2:] == long_toks[1:24]

    # 6. Placeholder / empty / invalid x2 handling
    assert cp.e2r2_row(['CCO'], [SM[0]], key) == (['CCO'], 'skipped', None)
    assert cp.e2r2_row([], [SM[0]], key) == ([], 'skipped', None)
    assert cp.e2r2_row([SM[0], SM[1]], [], key) == ([SM[0], SM[1]], 'no_eng', None)
    assert cp.e2r2_row([SM[0], SM[1]], ['not_a_smiles'], key) == ([SM[0], SM[1]], 'invalid_x2', None)
    assert cp.e2r2_row([SM[0], SM[1]], ['CCO'], key) == ([SM[0], SM[1]], 'invalid_x2', None)


def test_e2r2_integration_in_run():
    s = state()
    # For m1, lib_max=0.95 -> base list is [SM[0], SM[1], SM[2]].
    # Give m1 an Engine 2 list whose top candidate is SM[8] (not in m1's list).
    s['eng']['m1'] = dict(smiles=[SM[8], SM[0]], keys=[k(SM[8]), k(SM[0])])
    # Give m2 an Engine 2 list whose top candidate is SM[5] (which is already at Rank 2 of m2 after metric_fusion)
    s['base']['m2'] = (SM[:3], [k(x) for x in SM[:3]], 0.95, [3.0, 2.0, 1.0], ['F'] * 3)
    eng_m2 = [SM[1], SM[0]]
    s['eng']['m2'] = dict(smiles=eng_m2, keys=[k(x) for x in eng_m2])

    out_no_e2r2 = cp.run(s, deps(), METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, E2R2=False)
    st = {}
    out_e2r2 = cp.run(s, deps(), stats=st, METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, E2R2=True)
    # m1: Rank 1 unchanged, SM[8] inserted at Rank 2
    assert out_e2r2['m1'][0] == out_no_e2r2['m1'][0]
    assert out_e2r2['m1'][1] == SM[8]
    # m2: Rank 1 unchanged, top eng candidate at Rank 2
    assert out_e2r2['m2'][0] == out_no_e2r2['m2'][0]
    assert out_e2r2['m2'][1] == eng_m2[0]
    # mX (inactive placeholder): untouched
    assert out_e2r2['mX'] == ['CCO']
    assert st.get('e2r2_inserted', 0) >= 1
    assert st.get('e2r2_already_rank2', 0) >= 1
    for m in ('m1', 'm2', 'm3'):
        assert 1 <= len(out_e2r2[m]) <= 25
        assert len(set(map(cp.metric_identity, out_e2r2[m]))) == len(out_e2r2[m])
