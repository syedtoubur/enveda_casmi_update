"""Tests for build/casmi_post.py and build/patch_harness_dump.py (CPU, no Kaggle data).

The real proof of equivalence runs inside the Kaggle notebook (`POST EQUIV`); these tests pin the pieces that can be
checked locally: the notebook's own merge/promote self-tests, the C9 gate, C10 fusion and the knobs, C12 dedupe, and
that the patch leaves every base cell untouched.

usage: python build/test_patch_harness_dump.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import patch_harness_dump as phd  # noqa: E402

# never 'CCO': the notebook's C10 drops that literal string as its empty-list placeholder (mirrored)
SM = ['CCCCCCCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO', 'CC(C)O', 'CC(C)CO', 'OCCO', 'OCCCO', 'CC(O)CO']


def deps(gl_reverse=True):
    return dict(score_key=lambda s: cp.canonicalize_smiles_to_ik14(s), form=lambda s: 'F',
                ice_rerank=lambda smis, keys, scs, forms, ice, lam, top_n: list(range(len(smis)))[:top_n],
                gl_rerank_multi=(lambda smis, keys, scs, forms, cols, lams, top_n: list(range(len(smis)))[::-1][:top_n])
                if gl_reverse else (lambda *a, **k: list(range(len(a[0])))), dreams=None)


def k(s):
    return cp.canonicalize_smiles_to_ik14(s)


def state():
    base = {'m1': (SM[:3], [k(s) for s in SM[:3]], 0.95, [3.0, 2.0, 1.0], ['F'] * 3),
            'm2': (SM[:3], [k(s) for s in SM[:3]], 0.10, [3.0, 2.0, 1.0], ['F'] * 3),
            'm3': (SM[:2], [k(s) for s in SM[:2]], 0.95, [2.0, 1.0], ['F'] * 2)}
    pc = {'m1': dict(pc=[SM[5]], pc_keys=[k(SM[5])], pc_fz=[1.0], best_pool_fz=0.0, S=None, top_pop=0),
          'm2': dict(pc=[SM[5], SM[6]], pc_keys=[k(SM[5]), k(SM[6])], pc_fz=[900.0, 1.0], best_pool_fz=0.0,
                     S=None, top_pop=0)}
    eng = {'m3': dict(smiles=[SM[7], SM[1]], keys=[k(SM[7]), k(SM[1])])}
    return dict(base=base, pc=pc, eng=eng, ice={'m3': {'x': 1.0}}, gl={'m3': {'x': 1.0}},
                te_mids=['m1', 'm2', 'm3'], sample_ids=['m1', 'm2', 'm3', 'mX'], targets={}, consts={})


def test_notebook_self_tests():
    _b, _bk, _p, _pk = ['A', 'B', 'C'], ['ka', 'kb', 'kc'], ['X', 'Y'], ['kx', 'ky']
    assert cp.merge(_b, _bk, _p, _pk, [2, 4]) == ['A', 'X', 'B', 'Y', 'C']
    assert cp.promote(_b, _bk, _p, _pk, [2, 4]) == ['X', 'A', 'B', 'Y', 'C']
    assert cp.merge(_b, _bk, [], [], [2, 4]) == ['A', 'B', 'C']


def test_gate_promotion_and_fallbacks():
    out = cp.run(state(), deps())
    assert out['m1'][:3] == SM[:3]                       # lib_max >= LIB_TAU: base list untouched
    assert out['m2'][0] == SM[5]                         # rel 900 > 250: PubChem top promoted
    assert out['mX'] == ['CCO']                          # sample id with no molecule
    off = cp.run(state(), deps(), PROMOTE=False)
    assert off['m2'][:3] == [SM[0], SM[5], SM[1]]        # aggressive slots [2, 4, ...] interleave instead


def test_fusion_and_gl_knobs():
    on = cp.run(state(), deps())['m3']
    no_gl = cp.run(state(), deps(), FUSED_GL=False)['m3']
    no_eng = cp.run(state(), deps(), USE_ENG=False)['m3']
    assert no_eng[:2] == SM[:2]                          # base order without Engine 2 (tail still appended)
    assert no_gl[0] == SM[1]                             # RRF: in both lists beats single-list entries
    assert on != no_gl                                   # GLACIER rerank (reversed in the fake) moves the list


def test_final_pass_dedupes_and_fills():
    s = state()
    s['base']['m1'] = (['CCCCO', 'OCCCC', 'CCCO'], [k('CCCCO'), k('OCCCC'), k('CCCO')], 0.95, [3, 2, 1], ['F'] * 3)
    out = cp.run(s, deps())['m1']
    assert out[:2] == ['CCCCO', 'CCCO'] and len(set(map(k, out))) == len(out)      # OCCCC == CCCCO dropped
    assert 'OCCCC' in cp.run(s, deps(), FINAL_PASS=False)['m1']


def test_state_not_mutated():
    s = state()
    before = json.dumps(s, sort_keys=True, default=str)
    cp.run(s, deps())
    assert json.dumps(s, sort_keys=True, default=str) == before


E006_OFF = dict(ICE_FALLBACK=False, METRIC_FUSION=False, FUSION_WINDOW=None)
E006_ON = dict(ICE_FALLBACK=True, METRIC_FUSION=True, FUSION_WINDOW=40)


def deps_ice_reverses():
    # fake ICEBERG that only moves the list when it is given a non-empty ICE column (the base passes {} in C10)
    d = deps()
    d['ice_rerank'] = lambda smis, keys, scs, forms, ice, lam, top_n: (
        list(range(len(smis)))[::-1] if ice else list(range(len(smis))))[:top_n]
    return d


def test_e006_defaults_are_base():
    for d in (deps(), deps_ice_reverses()):
        assert cp.run(state(), d) == cp.run(state(), d, **E006_OFF)
    assert cp.run(state(), deps(), FUSION_WINDOW=cp.CFG['FUSE_N']) == cp.run(state(), deps())


def test_ice_fallback_where_gl_blind():
    s = state()
    s['gl'] = {}                                         # GLACIER blind for m3, ICE present
    base = cp.run(s, deps_ice_reverses())['m3']
    fb = cp.run(s, deps_ice_reverses(), ICE_FALLBACK=True)['m3']
    assert base[0] == SM[1] and fb[0] == SM[7] and fb != base   # ICE (reversed in the fake) now reranks
    with_gl = cp.run(state(), deps_ice_reverses())['m3']
    assert cp.run(state(), deps_ice_reverses(), ICE_FALLBACK=True)['m3'] == with_gl   # GL present: no-op
    st = {}
    cp.run(s, deps_ice_reverses(), stats=st, ICE_FALLBACK=True, METRIC_FUSION=True)
    assert st.get('ice_fallback') == 1                   # fallback in the overlay only, not also in base C10


def state_dup():
    # m4: C9 list carries one identity twice (CCCCO == OCCCC); Engine 2 votes for CCCO
    s = state()
    s['base']['m4'] = (['CCCCO', 'OCCCC', 'CCCO'], [k('CCCCO'), k('OCCCC'), k('CCCO')], 0.95, [3, 2, 1], ['F'] * 3)
    s['eng']['m4'] = dict(smiles=['CCCO'], keys=[k('CCCO')])
    s['te_mids'].append('m4'); s['sample_ids'].append('m4')
    return s


def test_metric_fusion_one_vote_per_identity():
    # base RRF: CCCCO 1/4 + 1/5 = .45 > CCCO 1/6 + .6/4 = .317; metric: CCCCO 1/4 < CCCO 1/5 + .6/4 = .35
    base = cp.run(state_dup(), deps())['m4']
    mf = cp.run(state_dup(), deps(), METRIC_FUSION=True)['m4']
    assert base[0] == 'CCCCO' and mf[0] == 'CCCO'
    assert len(set(map(k, mf))) == len(mf) and 'OCCCC' not in mf
    assert cp.run(state_dup(), deps(), METRIC_FUSION=True)['mX'] == ['CCO']      # placeholder rows untouched
    c, sc, col = cp.metric_fusion(['CCCCO', 'OCCCC', 'CCCO'], ['CCCO'], cp.metric_identity)
    assert c == ['CCCO', 'CCCCO'] and col == 1 and abs(sc[0] - (1 / 5 + 0.6 / 4)) < 1e-12


def test_fusion_window_caps_shortlist():
    # m3 fused = [SM1, SM0, SM7], reversed by the fake GLACIER; a window of 1 leaves only SM1 to rerank
    assert cp.run(state(), deps())['m3'][0] == SM[7]
    assert cp.run(state(), deps(), FUSION_WINDOW=1)['m3'][0] == SM[1]
    assert cp.run(state(), deps(), METRIC_FUSION=True)['m3'][0] == SM[7]
    assert cp.run(state(), deps(), METRIC_FUSION=True, FUSION_WINDOW=1)['m3'][0] == SM[1]


def test_e006_config_runs_metric_unique():
    out = cp.run(state_dup(), deps_ice_reverses(), **E006_ON)
    for m in ('m1', 'm2', 'm3', 'm4'):
        assert 1 <= len(out[m]) <= 25 and len(set(map(k, out[m]))) == len(out[m])


def test_patch_only_adds():
    nb = phd.build('C2', ranker=RANKER)
    base = [''.join(c['source']) for c in json.load(open(phd.SRC, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    kept = [s.replace(phd.SNAP, '') for s in code if not s.startswith(('# === HARNESS', 'POST_SRC'))]
    assert kept == base and len(code) == len(base) + 6
    assert "VAL_SHARD = 'C2'" in code[2] and "VAL_SHARD = ''" in ''.join(''.join(c['source']) for c in phd.build('', ranker=RANKER)['cells'])
    assert sum(phd.SNAP in s for s in code) == 1 and 'POST EQUIV ok' in code[-1]


def test_inserted_cells_bind_their_names():
    # harness-c2 v1 died on `NameError: np`: the VAL cell sits above the base's numpy import, and the draft
    # (VAL_SHARD = '') never runs that branch. Rule R07 (research/rules/_nbnames.py).
    sys.path.insert(0, 'C:/Users/h/OneDrive/Desktop/ENVEDA_casmi/research/rules')
    from _nbnames import unbound
    for shard in phd.SHARDS:
        code = [''.join(c['source']) for c in phd.build(shard, ranker=RANKER)['cells'] if c['cell_type'] == 'code']
        assert unbound(code) == [], (shard, unbound(code))


# ------------------------------------------------------------------------------------------------
# Option A (2026-10-11): _pk fix, np shards, fpnet_0 bank swap, Engine 2 ranker-row filter, coverage guard
# ------------------------------------------------------------------------------------------------
RANKER = ('Q', 'np')          # test values; the real field/label come from the A0 kernel's NPZ printout


def _nbn():
    sys.path.insert(0, 'C:/Users/h/OneDrive/Desktop/ENVEDA_casmi/research/rules')
    import _nbnames
    return _nbnames


def _code(shard):
    return [''.join(c['source']) for c in phd.build(shard, ranker=RANKER)['cells'] if c['cell_type'] == 'code']


def test_harness_helpers_not_rebound():
    # harness-c1npx v1 died on `'list' object has no attribute 'dump'`: SNAP bound `_pk`, the base re-bound it
    nbn = _nbn()
    for shard in phd.SHARDS:
        code = _code(shard)
        assert nbn.rebound(code) == [], (shard, nbn.rebound(code))
    snap_old = "# >>> HARNESS\nimport pickle as _pk\ndef _d(x):\n    return _pk.dumps(x)\n# <<< HARNESS\n"
    assert nbn.rebound([snap_old + "_b, _pk = [1], [2]\n", "# === HARNESS\n_d(1)\n"]) == [(0, '_pk', 0)]
    assert nbn.rebound(["# === HARNESS\nimport pickle as _hx_pickle\n", "_pk = 1\n"]) == []


def test_np_shards_registered():
    assert 'C2_np' in phd.SHARDS and 'C1_np_fp0' in phd.SHARDS
    val = next(s for s in _code('C2_np') if 'VAL_SHARD = ' in s)
    assert "VAL_SHARD = 'C2_np'" in val
    # C2_np = the C1_npx records, every truth row banned by key, cls 2
    ns = {}
    exec(val[val.index('def _hx_val_records'):val.index('# end _hx_val_records')], ns)
    recs = [dict(molecule_id='V1', inchikey14='AAA', cohort='C1_npx', cls=1, ban_rows=[3], ban_keys=[]),
            dict(molecule_id='V2', inchikey14='BBB', cohort='C2', cls=2, ban_rows=[4], ban_keys=['BBB'])]
    c2np = ns['_hx_val_records'](recs, 'C2_np')
    assert [r['molecule_id'] for r in c2np] == ['V1'] and c2np[0]['ban_keys'] == ['AAA'] and c2np[0]['cls'] == 2
    assert recs[0]['ban_keys'] == []                                   # input not mutated
    fp0 = ns['_hx_val_records'](recs, 'C1_np_fp0')
    assert [r['molecule_id'] for r in fp0] == ['V1'] and fp0[0]['ban_keys'] == [] and fp0[0]['cls'] == 1
    assert [r['molecule_id'] for r in ns['_hx_val_records'](recs, 'C2')] == ['V2']
    out = phd.main(['--shard', 'C2_np', '--ranker-field', 'Q', '--ranker-drop', 'np', '--out', os.devnull])
    assert out == 0


def test_new_cells_placed():
    for shard in phd.SHARDS:
        code = _code(shard)
        i_bank = next(i for i, s in enumerate(code) if 'E = Engine(L, P, tfp' in s)
        assert code[i_bank + 1].startswith('# === HARNESS') and 'HARNESS fpnet_0 swaps 1' in code[i_bank + 1]
        i_eng = next(i for i, s in enumerate(code) if s.startswith('# Our engine (two-ranker'))
        assert 'HX_RANKER_FIELD' in code[i_eng - 1] and 'HARNESS ranker rows dropped' in code[i_eng + 1]
        i_c12 = next(i for i, s in enumerate(code) if 'Engine 2 did not cover all active molecules' in s
                     and not s.startswith('# === HARNESS'))
        assert 'HARNESS ENG fill' in code[i_c12 - 1]
    try:
        phd.build('C2_np')
    except SystemExit as e:
        assert 'ranker' in str(e)
    else:
        raise AssertionError('np shard built without a verified ranker field')


class _Bank:
    def __init__(self, paths, device='cpu'):
        self.paths, self.nbits = paths, 10226


def _cell(code, marker):
    return next(s for s in code if s.startswith('# === HARNESS') and marker in s)


def test_bank_swap_cell():
    import types
    cell = _cell(_code('C2_np'), 'HARNESS fpnet_0 swaps 1')
    fp = types.SimpleNamespace(ModelBank=_Bank)
    for shard, swapped in (('C2_np', True), ('C1_np_fp0', True), ('C2', False), ('', False)):
        E = types.SimpleNamespace(bank=_Bank(['full1.pt']))
        ns = dict(VAL_SHARD=shard, V4MOD='/m', fpnet=fp, dev='cpu', E=E, bank=E.bank, os=os)
        exec(cell, ns)
        assert (E.bank.paths == [os.path.join('/m', 'fpnet_0.pt')]) == swapped and ns['bank'] is E.bank, shard


def test_eng_guard_cell():
    import pandas as pd
    cell = _cell(_code('C2'), 'HARNESS ENG fill')
    te = pd.DataFrame(dict(molecule_id=[f'V{i}' for i in range(40)] * 2))
    eng = {f'V{i}': dict(smiles=['C'], keys=['K']) for i in range(39)}
    ns = dict(VAL_SHARD='C2', te=te, ENG=eng)
    exec(cell, ns)
    assert eng['V39'] == dict(smiles=[], keys=[]) and len(eng) == 40
    eng2 = {f'V{i}': {} for i in range(30)}
    try:
        exec(cell, dict(VAL_SHARD='C2', te=te, ENG=eng2))
    except AssertionError:
        pass
    else:
        raise AssertionError('coverage < 95% must fail')
    eng3 = {}
    exec(cell, dict(VAL_SHARD='', te=te, ENG=eng3))
    assert eng3 == {}                                                  # normal run: base assert untouched


def test_ranker_hook_filters_rows():
    import subprocess
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        site, eng = os.path.join(d, 'site'), os.path.join(d, 'eng')
        os.makedirs(site); os.makedirs(eng)
        open(os.path.join(site, 'hx_ranker_hook.py'), 'w').write(phd.HX_HOOK_SRC)
        open(os.path.join(eng, 'casmi_engine.py'), 'w').write(
            'import numpy as np\n'
            'def fit_rankers(p):\n'
            '    z = np.load(p, allow_pickle=True)\n'
            '    return z["X"].shape[0], [str(q) for q in z["Q"]], [str(b) for b in z["blocks"]]\n'
            'def main(p):\n'
            '    return fit_rankers(p)\n')
        np.savez(os.path.join(d, 'rows.npz'), X=np.zeros((5, 2)), Y=np.arange(5), Q=np.array(['np', 'a', 'np', 'b', 'a']),
                 blocks=np.array(['x', 'y']))
        env = dict(os.environ, HX_RANKER_FIELD='Q', HX_RANKER_DROP='np', HX_RANKER_OUT=os.path.join(d, 'out.npz'))
        prog = (f'import sys; sys.path.insert(0, {site!r}); import hx_ranker_hook; sys.path.insert(0, {eng!r}); '
                f'import casmi_engine as E; print(E.main({os.path.join(d, "rows.npz")!r}))')
        r = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        assert "(3, ['a', 'b', 'a'], ['x', 'y'])" in r.stdout and 'HARNESS ranker rows dropped 2 of 5' in r.stdout, r.stdout
        assert open(os.path.join(d, 'out.npz.dropped')).read().split() == ['2', '5']
        # int codes into a name table (sim_rank_rows_nofp.npz: Q int8 -> sets)
        np.savez(os.path.join(d, 'rows.npz'), X=np.zeros((5, 2)), Q=np.array([1, 0, 1, 2, 0], np.int8),
                 sets=np.array(['gnps', 'np', 'riken']), blocks=np.array(['x', 'y']))
        r = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True,
                           env=dict(env, HX_RANKER_NAMES='sets'))
        assert "(3, ['0', '2', '0']" in r.stdout and 'dropped 2 of 5' in r.stdout, (r.stdout, r.stderr[-500:])
        env.pop('HX_RANKER_FIELD')
        r = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, env=env)
        assert "(5," in r.stdout, r.stdout                               # no env -> hook inert


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    bad = 0
    for n, f in tests:
        try:
            f(); print('ok  ', n)
        except Exception as e:
            bad += 1; print('FAIL', n, repr(e))
    print(f'{len(tests) - bad}/{len(tests)} passed')
    sys.exit(1 if bad else 0)
