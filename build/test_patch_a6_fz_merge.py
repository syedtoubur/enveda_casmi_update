"""Tests for casmi_post.fz_merge / FZ_MERGE and build/patch_a6_fz_merge.py (CPU).

usage: python -I build/test_patch_a6_fz_merge.py
The knob-off identity runs casmi_post.run on the harness dumps (E:/ENVEDA_casmi/research/harness/dumps/*) against the
exploit worktree's casmi_post; it is skipped when those files are absent.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import patch_a6_fz_merge as a6  # noqa: E402

INF = math.inf
DUMPS = "E:/ENVEDA_casmi/research/harness/dumps"
REF_POST = "E:/casmi/lines/exploit/build/casmi_post.py"


def test_internal_order_and_envelope():
    base, bk = ['A', 'B', 'C'], ['ka', 'kb', 'kc']
    pool = {'ka': -100.0, 'kb': -50.0, 'kc': -300.0}      # envelope: A -50, B -50, C -300
    pc, pck, fz = ['X', 'Y', 'Z'], ['kx', 'ky', 'kz'], [-60.0, -200.0, -10.0]
    out = cp.fz_merge(base, bk, pc, pck, fz, pool, 0.0)
    # X(-60) > env A(-50)? no -> A; X > env B(-50)? no -> B; X > env C(-300) yes -> X; Y(-200) > -300 -> Y; Z -> Z; C
    assert out == ['A', 'B', 'X', 'Y', 'Z', 'C'], out
    sub = [s for s in out if s in 'ABC']; assert sub == base
    assert [s for s in out if s in 'XYZ'] == pc                # PubChem keeps its own order even if fz is not monotone


def test_margin_inf_is_base_then_pc():
    base, bk = ['A', 'B'], ['ka', 'kb']
    out = cp.fz_merge(base, bk, ['X', 'Y'], ['kx', 'ky'], [1e9, 1e9], {'ka': 0.0, 'kb': 0.0}, INF)
    assert out == ['A', 'B', 'X', 'Y'], out
    out = cp.fz_merge(base, bk, ['X', 'Y'], ['kx', 'ky'], [1e9, 1e9], {}, INF)     # no envelope, still base first
    assert out == ['A', 'B', 'X', 'Y'], out


def test_huge_pc_fz_goes_first():
    base, bk = ['A', 'B'], ['ka', 'kb']
    out = cp.fz_merge(base, bk, ['X', 'Y'], ['kx', 'ky'], [1e9, 1e9], {'ka': 0.0, 'kb': 0.0}, 50.0)
    assert out == ['X', 'Y', 'A', 'B'], out


def test_dedup_truncate_and_unknowns():
    base = [f'b{i}' for i in range(20)]
    bk = [f'kb{i}' for i in range(20)]
    pc = [f'p{i}' for i in range(10)] + ['dupe']
    pck = [f'kp{i}' for i in range(10)] + ['kb3']             # a PC key already in BASE is dropped
    out = cp.fz_merge(base, bk, pc, pck, [0.0] * 11, {}, INF, n=25)
    assert len(out) == 25 and len(set(out)) == 25 and 'dupe' not in out
    out = cp.fz_merge(['A'], ['ka'], ['X', 'X2'], ['kx', 'kx'], [5.0, 5.0], {'ka': 0.0}, 0.0)   # PC-internal key dup
    assert out == ['X', 'A'], out
    out = cp.fz_merge(['A', 'B'], ['ka', 'kb'], ['X'], ['kx'], [None], {'ka': 0.0, 'kb': 0.0}, 0.0)   # unknown PC fz
    assert out == ['A', 'B', 'X'], out
    out = cp.fz_merge(['A'], ['ka'], ['X'], ['kx'], [float('nan')], {'ka': 0.0}, 0.0)
    assert out == ['A', 'X'], out
    out = cp.fz_merge([], [], ['X'], ['kx'], [1.0], {}, 0.0)
    assert out == ['X']


def test_fz_merge_row_missing_data():
    p = dict(pc=['X'], pc_keys=['kx'], pc_fz=[7.0], pool_fz={})
    assert cp.fz_merge_row(p, ['A'], ['ka'], 0.0) is None
    p = dict(pc=['X'], pc_keys=['kx'], pc_fz=[7.0], pool_fz={'ka': 0.0})
    assert cp.fz_merge_row(p, ['A'], ['ka'], 0.0) is None     # pc_fz is z-scored, never used as raw f.z
    p['pc_rawfz'] = [10.0]
    assert cp.fz_merge_row(p, ['A'], ['ka'], 0.0) == ['X', 'A']


def _fixture(**pc_extra):
    from test_patch_harness_dump import SM, k
    base = {'m1': (SM[:3], [k(s) for s in SM[:3]], 0.10, [3.0, 2.0, 1.0], ['F'] * 3),
            'm2': (SM[:3], [k(s) for s in SM[:3]], 0.95, [3.0, 2.0, 1.0], ['F'] * 3)}
    pc = {m: dict(pc=[SM[5], SM[6]], pc_keys=[k(SM[5]), k(SM[6])], pc_fz=[5.0, 1.0], best_pool_fz=0.0, S=None,
                  top_pop=0, fz_top=900.0, **pc_extra) for m in base}
    return dict(base=base, pc=pc, eng={}, ice={}, gl={}, te_mids=list(base), sample_ids=list(base), targets={}, consts={}), SM, k


def test_run_knob_paths():
    from test_patch_harness_dump import deps
    st, SM, k = _fixture()
    kw = dict(USE_ENG=False, FINAL_PASS=False)
    off = cp.run(st, deps(), {}, **kw)
    stats = {}
    nofz = cp.run(st, deps(), stats, FZ_MERGE=50.0, **kw)
    assert nofz == off and stats.get('fz_merge_nofz') == 1 and stats.get('fz_merge_mols', 0) == 0, stats
    st, SM, k = _fixture(pool_fz={k(s): -100.0 for s in SM[:3]}, pc_rawfz=[0.0, -500.0])
    stats = {}
    on = cp.run(st, deps(), stats, FZ_MERGE=50.0, **kw)
    assert on['m1'][0] == SM[5] and on['m1'][1:4] == SM[:3], on['m1']      # 0 > -100 + 50: PC leads; -500 trails
    assert on['m1'][4] == SM[6]
    assert on['m2'] == off['m2'] == SM[:3]                                   # lib_max >= LIB_TAU lock
    assert stats['fz_merge_mols'] == 1 and stats['fz_merge_changed_vs_base'] == 1 and stats['fz_merge_pc1_slot_1'] == 1
    assert cp.run(st, deps(), {}, **kw) == off                              # knob off ignores the new fields
    # pool_fz present but the PC rerank (ICE) permutes pc/pc_keys: the raw-fz map is taken before it
    st['ice'] = {'m1': {'x': 1}}
    cp.run(st, deps(), {}, FZ_MERGE=50.0, **kw)


def test_knob_off_identity_on_dumps():
    ref = None
    if not os.path.exists(REF_POST):
        print('  skip dump identity: reference post missing'); return
    spec = importlib.util.spec_from_file_location('casmi_post_ref', REF_POST)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    shards = [d for d in sorted(os.listdir(DUMPS)) if os.path.exists(f'{DUMPS}/{d}/post_dump/state.pkl')] \
        if os.path.isdir(DUMPS) else []
    if not shards:
        print('  skip dump identity: no dumps'); return
    n_ok = 0
    for sh in shards:
        d = f'{DUMPS}/{sh}/post_dump'
        state = pickle.load(open(f'{d}/state.pkl', 'rb'))
        sys.path.insert(0, f'{d}/src')
        for m in ('fuse', 'gl_fuse', 'casmi', 'casmi.chem'):
            sys.modules.pop(m, None)
        import fuse, gl_fuse  # noqa: E401
        from casmi import chem
        dp = dict(score_key=chem.score_key, form=lambda s: s, ice_rerank=fuse.rerank, gl_rerank_multi=gl_fuse.rerank_multi,
                  dreams=None)
        sub = dict(state, te_mids=state['te_mids'][:25])
        a = ref.run(sub, dp, {})
        sa, sb = {}, {}
        b = cp.run(sub, dp, sb)
        c = cp.run(sub, dp, sa, FZ_MERGE=50.0)                              # dumps lack pool_fz -> base path
        assert a == b, sh
        assert c == a and sa.get('fz_merge_mols', 0) == 0, (sh, sa)
        assert 'fz_merge_nofz' not in sb
        sys.path.remove(f'{d}/src'); n_ok += 1
    print(f'  knob-off identity ok on {n_ok} dumps: {shards}')


def _runner_cell(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8'))
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and "RUNNER = '" in ''.join(c['source'])
            and 'best = float((F @ z' in ''.join(c['source'])]
    assert len(hits) == 1
    return ''.join(hits[0]['source'])


def test_runner_patch_anchors_and_compiles():
    import ast
    text = _runner_cell(a6.SRC)
    line = [ln for ln in text.split('\n') if ln.startswith('RUNNER = ')]
    assert len(line) == 1
    runner = ast.literal_eval(line[0][len('RUNNER = '):])
    for old, _ in a6.RUNNER_EDITS:
        assert runner.count(old) == 1, old
    assert text.count(a6.LOAD_ANCHOR) == 1
    new = a6.patch_runner_cell(text)
    new_runner = ast.literal_eval([ln for ln in new.split('\n') if ln.startswith('RUNNER = ')][0][len('RUNNER = '):])
    compile(new_runner, 'pc_runner', 'exec')
    assert 'pool_fz' in new_runner and 'pc_rawfz' in new_runner and a6.LOAD_NEW in new
    compile(new.replace('%', '%'), 'cell', 'exec') if 'get_ipython' not in new else None


def test_c9_cell_anchor_and_compiles():
    nb = json.load(open(a6.SRC, encoding='utf-8'))
    hits = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and '(rel > 250.0)' in ''.join(c['source'])]
    assert len(hits) == 1
    for old, _ in (a6.A_KNOB, a6.A_STATS, a6.A_TAIL, a6.A_PRINT, (a6.A_DEF[0], '')):
        assert hits[0].count(old) == 1, old
    new = a6.patch_c9_text(hits[0], 50.0)
    assert 'FZ_MERGE = 50.0' in new and 'def fz_merge(' in new and 'def fz_merge_row(' in new
    assert 'FZ_MERGE = None' in a6.patch_c9_text(hits[0], None)


def test_generated_notebooks():
    import tempfile
    nb = a6.build(a6.SRC, 50.0)
    n_code = sum(c['cell_type'] == 'code' for c in nb['cells'])
    base = json.load(open(a6.SRC, encoding='utf-8'))
    assert n_code == sum(c['cell_type'] == 'code' for c in base['cells'])
    diff = [i for i, (x, y) in enumerate(zip(nb['cells'], base['cells'])) if x['source'] != y['source']]
    assert len(diff) == 2, diff                                           # runner cell + C9 cell only
    import patch_harness_dump as phd
    for sh in ('C2', 'C1_npx'):
        h = phd.build(sh, a6_runner=True)
        h0 = phd.build(sh)
        assert len(h['cells']) == len(h0['cells'])
        assert sum(x['source'] != y['source'] for x, y in zip(h['cells'], h0['cells'])) == 1


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('ok  ', name)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc(); print('FAIL', name, repr(e)); fails += 1
    raise SystemExit(1 if fails else 0)
