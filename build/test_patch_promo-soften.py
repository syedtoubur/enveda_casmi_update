"""Tests for build/patch_promo-soften.py and the REL_PROMOTE knob of casmi_post (CPU, no Kaggle data).

G1: only the C9 merge/promotion cell differs from the E006 v1 base. G2: the C9 decision at REL_PROMOTE 250 / 500.

usage: python build/test_patch_promo-soften.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
from test_patch_harness_dump import SM, deps, k  # noqa: E402

pp = importlib.import_module('patch_promo-soften')
BASE_NB = json.load(open(pp.SRC, encoding='utf-8'))


def _code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def _fixture(rel_by_mid, spop=()):
    """Three molecules, lib_max 0.10 (C9 applies), pc_fz[0] - best_pool_fz = rel; 'inf' = non-finite pool fz."""
    base, pc = {}, {}
    for i, (mid, rel) in enumerate(rel_by_mid.items()):
        base[mid] = (SM[:3], [k(s) for s in SM[:3]], 0.10, [3.0, 2.0, 1.0], ['F'] * 3)
        top = SM[5 + i]
        pc[mid] = dict(pc=[top], pc_keys=[k(top)], pc_fz=[rel if rel != 'inf' else 1.0],
                       best_pool_fz=0.0 if rel != 'inf' else float('nan'),
                       S=7.0 if mid in spop else None, top_pop=6.0 if mid in spop else 0)
    mids = list(rel_by_mid)
    return dict(base=base, pc=pc, eng={}, ice={}, gl={}, te_mids=mids, sample_ids=mids, targets={}, consts={})


def _run(rel_by_mid, spop=(), **kw):
    st = {}
    out = cp.run(_fixture(rel_by_mid, spop), deps(), st, USE_ENG=False, FINAL_PASS=False, **kw)
    return out, st


REL = {'lo': 100.0, 'mid': 300.0, 'inf': 'inf'}


def test_default_reproduces_base():
    out, st = _run(REL)
    assert out['lo'][:3] == SM[:3] and out['lo'][3] == SM[5]      # rel 100: merged, gentle slot 4
    assert out['mid'][0] == SM[6] and out['inf'][0] == SM[7]      # rel 300 > 250 and rel 1e9: promoted
    assert st['promoted'] == 2 and st.get('promote_rel_skipped', 0) == 0
    out2, _ = _run(REL, REL_PROMOTE=250.0)
    assert out2 == out


def test_500_merges_only_the_band():
    out, st = _run(REL, REL_PROMOTE=500.0)
    assert out['mid'][:3] == SM[:3] and out['mid'][3] == SM[6]    # rel 300: merged, base rank 1 kept
    assert out['inf'][0] == SM[7]                                 # rel 1e9 still promoted
    assert st['promoted'] == 1 and st['promote_rel_skipped'] == 1


def test_spop_arm_still_promotes_in_band():
    out, st = _run({'mid': 300.0}, spop={'mid'}, REL_PROMOTE=500.0)
    assert out['mid'][0] == SM[5] and st['promoted'] == 1 and st.get('promote_rel_skipped', 0) == 0


def test_patcher_changes_only_c9_cell():
    nb = pp.build()
    base, new = _code(BASE_NB), _code(nb)
    assert len(base) == len(new)
    diff = [i for i, (a, b) in enumerate(zip(base, new)) if a != b]
    assert len(diff) == 1, diff
    cell = new[diff[0]]
    assert 'REL_PROMOTE = 500.0' in cell and 'rel > REL_PROMOTE' in cell and 'rel > 250.0' not in cell.replace(
        '# rel > 250 captures', '')
    assert "promote_rel_skipped=0" in cell and "stats['promote_rel_skipped'] += 1" in cell
    assert "print('merge stats', stats)" in cell
    assert [c for c in BASE_NB['cells'] if c['cell_type'] != 'code'] == [c for c in nb['cells'] if c['cell_type'] != 'code']
    compile(cell.replace('\n%', '\n#%'), 'c9', 'exec')


def test_patcher_default_250_is_base_decision():
    cell = _code(pp.build(rel_promote=250.0))
    assert any('REL_PROMOTE = 250.0' in c for c in cell)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
