"""Tests for build/patch_b2_nl_sim.py and the NL_LAM knob of build/casmi_post.py (CPU, no Kaggle data).

Dump-based tests use E:/ENVEDA_casmi/research/harness/dumps/<shard>/post_dump (skipped if absent); runner tests use the
downloaded casmi26-glacier source in E:/casmi/data/sim_pkgs/glacier (skipped if absent). Nothing is executed from there:
the runner text is read, patched in memory, compiled.

usage: python -I build/test_patch_b2_nl_sim.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_b2_nl_sim as p  # noqa: E402

DUMPS = 'E:/ENVEDA_casmi/research/harness/dumps'
GL_RUNNER = 'E:/casmi/data/sim_pkgs/glacier/gl_runner.py'
E006_CFG = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40)


def _load_post():
    spec = importlib.util.spec_from_file_location('casmi_post_b2', os.path.join(HERE, 'casmi_post.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _runner_text():
    return open(GL_RUNNER, newline='').read()


def _runner_ns():
    """numpy-only namespace with the runner's real cleaning / entropy_sim code (njit -> identity) + the NL helpers."""
    t = _runner_text().replace('\r\n', '\n')
    a, b = t.index('LOG4 = np.log(4.0)'), t.index('# ---', t.index('def entropy_sim'))
    ns = {'np': np, 'TOL': 0.01, 'PPM': 20.0, 'njit': lambda *a, **k: (lambda f: f)}
    exec(t[a:b], ns)
    helpers = p._runner_ns()['B2_RUNNER_NL_HELPERS']
    exec(helpers, ns)
    return ns


# ------------------------------------------------------------------------------------------------ NL transform ----
def test_nl_transform_toy():
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    ns = _runner_ns()
    mz = np.array([50.0, 100.0, 150.0, 199.9, 200.0], np.float64)    # prec 200: NL 150, 100, 50, 0.1, 0.0
    w = np.array([0.1, 0.2, 0.3, 0.2, 0.2], np.float64)
    nl, nw = ns['nl_prep'](mz, w, 200.0)
    assert list(nl) == [50.0, 100.0, 150.0], nl                      # ascending, NL <= 0.5 dropped
    assert np.allclose(nw, np.array([0.3, 0.2, 0.1]) / 0.6) and abs(nw.sum() - 1) < 1e-12
    e, ew = ns['nl_prep'](np.array([199.9]), np.array([1.0]), 200.0)  # everything dropped
    assert len(e) == 0 and len(ew) == 0


def test_entropy_sim_identical_nl_is_one():
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    ns = _runner_ns()
    mz = np.array([60.1, 91.05, 120.2, 180.4, 250.0], np.float64)
    it = np.array([10.0, 100.0, 30.0, 5.0, 60.0], np.float64)
    qm, qp = ns['prep_query'](mz, it, 0.002, 256, 1.0, True)
    qn, qw = ns['nl_prep'](qm, qp, 256.0)
    s = ns['entropy_sim'](qn, qw, qn.copy(), qw.copy(), 0.01, 20.0)
    assert abs(s - 1.0) < 1e-9, s
    # a different loss pattern scores lower
    pm, pp = ns['prep_query'](mz + 3.0, it, 0.002, 256, 1.0, True)
    pn, pw = ns['nl_prep'](pm, pp, 256.0)
    assert ns['entropy_sim'](qn, qw, pn, pw, 0.01, 20.0) < 0.5


def test_same_fragments_different_precursor_losses():
    """same fragment ions, different precursor => fragment sim 1, NL sim 0 (the signal B2 adds)."""
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    ns = _runner_ns()
    mz = np.array([77.04, 105.03, 123.04], np.float64)
    w = np.array([0.3, 0.5, 0.2], np.float64)
    a = ns['nl_prep'](mz, w, 200.0)
    b = ns['nl_prep'](mz, w, 230.0)
    assert ns['entropy_sim'](a[0], a[1], b[0], b[1], 0.01, 20.0) == 0.0


# ------------------------------------------------------------------------------------------------ runner patch ----
def test_runner_patch_anchors_once_and_compiles():
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    t = _runner_text().replace('\r\n', '\n')
    for old, _ in p.NB_RUNNER_ANCHORS_FOR_TEST():
        assert t.count(old) == 1, old
    out = p.runner_patch_fn()(_runner_text())                        # CRLF input is accepted
    compile(out, 'gl_runner_b2', 'exec')
    assert out.count('nl_prep(') == 3 and "'.nl.json'" in out and 'nlsims.append' in out
    assert 'for qm, qp, ad, ins, ceb, prec in plan' in out and 'for _, _, ad, ins, ceb, _prec in plan' in out
    # idempotence guard: patching twice must raise (anchors gone), not double-patch
    try:
        p.runner_patch_fn()(out)
    except RuntimeError:
        pass
    else:
        raise AssertionError('second patch should fail')
    # an anchor edited away => raise (-> B2_NL_PATCH_FAILED fallback in the notebook)
    try:
        p.runner_patch_fn()(_runner_text().replace('entropy_sim(qm, qp, pm, pp, TOL, PPM)', 'entropy_sim(qm, qp, pm, pp)'))
    except RuntimeError:
        pass
    else:
        raise AssertionError('missing anchor should fail')


def test_patched_score_molecule_end_to_end():
    """run the patched score_molecule on a fake prediction cache with the runner's real numpy code (njit -> identity)."""
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    out = p.runner_patch_fn()(_runner_text())
    a = out.index('LOG4 = np.log(4.0)')
    b = out.index('# ---', out.index('def entropy_sim'))
    ns = {'np': np, 'TOL': 0.01, 'PPM': 20.0, 'FLOOR': 0.002, 'TOPK': 256, 'POWER': 1.0, 'ENTW': True,
          'njit': lambda *a, **k: (lambda f: f), 'SUM_CE': (10, 20, 40)}
    exec(out[a:b], ns)
    helpers = out[out.index('# ---- B2 neutral-loss'):out.index('def score_molecule')]
    s0 = out.index('def score_molecule')
    e0 = out.index('# ---', s0)
    exec(helpers, ns)
    exec(out[s0:e0], ns)
    prec = 250.0
    qm = np.array([60.0, 90.0, 120.0, 150.0]); qi = np.array([40.0, 100.0, 20.0, 60.0])
    q = ns['prep_query'](qm, qi, 0.002, 256, 1.0, True)
    plan = [(q[0], q[1], '[M+H]+', 'QTOF', 20, prec)]
    same = (np.array([60.0, 90.0, 120.0, 150.0], np.float32), np.array([0.4, 1.0, 0.2, 0.6], np.float32))
    shift = (np.array([62.0, 93.0, 121.0, 155.0], np.float32), np.array([0.4, 1.0, 0.2, 0.6], np.float32))
    cache = {('A', '[M+H]+', 'QTOF', 20): same, ('B', '[M+H]+', 'QTOF', 20): shift, ('C', '[M+H]+', 'QTOF', 20): None}
    res = ns['score_molecule'](plan, ['A', 'B', 'C'], cache)
    assert res['A'] > 0.99 and res['C'] is None and res['B'] < res['A']
    nl = ns['_NL_LAST']
    assert abs(nl['A'] - 1.0) < 1e-6 and nl['C'] is None and nl['B'] < 0.5, nl


# -------------------------------------------------------------------------------------------- notebook patcher ----
def _cells(nb):
    return [''.join(c['source']) for c in nb['cells']]


def test_patcher_anchors_once_and_cells_compile():
    nb = json.load(open(p.SRC, encoding='utf-8'))
    new = p.build(p.SRC, 0.5)
    old_c, new_c = _cells(nb), _cells(new)
    assert len(old_c) == len(new_c)
    changed = [i for i, (a, b) in enumerate(zip(old_c, new_c)) if a != b]
    assert len(changed) == 4, changed                                   # cell 7, cell 9, E004 overlay, E006 overlay
    for i in changed:
        src = new_c[i]
        compile(''.join(l for l in src.splitlines(keepends=True) if not l.lstrip().startswith(('!', '%'))), 'cell', 'exec')
        assert 'nl_of' in src
    c7 = new_c[changed[0]]
    assert 'NL_LAM, NL_SCORES, B2_MOLS = 0.5, {}, set()' in c7 and 'B2_NL_PATCH_FAILED' in c7
    assert 'nlm=m' in c7 and c7.count('nlm=m') == 2 and "b2_report('after_cell7')" in c7
    assert "b2_report('final')" in new_c[changed[3]]
    # unchanged cells are byte-identical
    assert all(a == b for i, (a, b) in enumerate(zip(old_c, new_c)) if i not in changed)
    # patching an already-patched notebook fails loudly instead of double-patching
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'x.ipynb')
        json.dump(new, open(f, 'w', encoding='utf-8'))
        try:
            p.build(f, 0.5)
        except SystemExit:
            pass
        else:
            raise AssertionError('double patch should fail')


def test_embedded_runner_patch_executes_in_notebook_cell():
    """the helper block embedded in cell 7 defines b2_patch_runner_text / b2_make_nl_pkg and works on the runner."""
    if not os.path.exists(GL_RUNNER):
        print('  skip (no runner)'); return
    c7 = _cells(p.build(p.SRC, 0.5))
    src = [s for s in c7 if 'B2_RUNNER_ANCHORS' in s][0]
    a = src.index('B2_RUNNER_NL_HELPERS')
    b = src.index('def nl_of(m)')
    ns = {'os': os}
    exec(src[a:b], ns)
    out = ns['b2_patch_runner_text'](_runner_text())
    assert out == p.runner_patch_fn()(_runner_text())
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as d:                           # b2_make_nl_pkg on a tiny fake package
        pkg = os.path.join(d, 'pkg'); os.makedirs(os.path.join(pkg, 'ckpt'))
        open(os.path.join(pkg, 'ckpt', 'glacier.pt'), 'wb').write(b'x')
        shutil.copy(GL_RUNNER, os.path.join(pkg, 'gl_runner.py'))
        dst = ns['b2_make_nl_pkg'](pkg, os.path.join(d, 'copy'))
        assert os.path.exists(os.path.join(dst, 'ckpt', 'glacier.pt'))
        assert 'nl_prep(' in open(os.path.join(dst, 'gl_runner.py')).read()
        assert 'nl_prep(' not in open(GL_RUNNER).read()               # original untouched


# ----------------------------------------------------------------------------------------- casmi_post knob -------
def _load_dump(shard):
    d = os.path.join(DUMPS, shard, 'post_dump')
    if not os.path.exists(os.path.join(d, 'state.pkl')):
        return None
    src = os.path.join(d, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    import fuse, gl_fuse  # noqa: E401
    from casmi import chem
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    cp = _load_post()

    def form(s):
        try:
            return CalcMolFormula(Chem.MolFromSmiles(s))
        except Exception:
            return s
    def memo(f):
        c = {}
        return lambda x: c[x] if x in c else c.setdefault(x, f(x))
    cp.canonicalize_smiles_to_ik14 = memo(cp.canonicalize_smiles_to_ik14)    # tautomer canonicalisation is slow
    form = memo(form)
    deps = dict(score_key=memo(chem.score_key), form=form, ice_rerank=fuse.rerank, gl_rerank_multi=gl_fuse.rerank_multi,
                dreams=None)
    return _subsample(pickle.load(open(os.path.join(d, 'state.pkl'), 'rb')), N_MOLS), deps, cp, form


N_MOLS = int(os.environ.get('B2_TEST_MOLS', '120'))


def _subsample(state, n):
    """first n molecules that GLACIER scored (keeps the test minutes, not hours; the harness caches RDKit, we don't)."""
    gl = state.get('gl') or {}
    keep = [str(m) for m in state['te_mids'] if str(m) in gl and any(v is not None for v in gl[str(m)].values())][:n]
    ks = set(keep)
    out = dict(state)
    out['te_mids'] = [m for m in state['te_mids'] if str(m) in ks]
    out['sample_ids'] = [m for m in state['sample_ids'] if str(m) in ks]
    for key in ('base', 'pc', 'eng', 'ice', 'gl', 'targets'):
        v = state.get(key)
        if isinstance(v, dict):
            out[key] = {m: x for m, x in v.items() if str(m) in ks}
    return out


def _synthetic_nl(state):
    """a deterministic gl_nl_mean for every candidate GLACIER scored (hash of the SMILES -> [0,1])."""
    import zlib
    nl = {}
    for m, g in (state.get('gl') or {}).items():
        if isinstance(g, dict):
            nl[m] = {s: (None if v is None else (zlib.crc32(s.encode()) % 1000) / 1000.0) for s, v in g.items()}
    return nl


def _identity_on(shard, cfgs):
    ld = _load_dump(shard)
    if ld is None:
        print(f'  skip {shard} (no dump)'); return
    state, deps, cp, form = ld
    nl = _synthetic_nl(state)
    assert any(any(v is not None for v in d.values()) for d in nl.values()), 'dump has no GL scores'
    st_nl = dict(state, nl=nl)
    for cfg in cfgs:
        ref = cp.run(state, deps, {}, **cfg)
        # (1) knob off (default), nl present in the state: byte-identical
        s_off = {}
        assert cp.run(st_nl, deps, s_off, **cfg) == ref
        assert 'nl_rerank_mols' not in s_off
        # (2) NL_LAM set but no nl dict: identical, counter absent
        s2 = {}
        assert cp.run(state, deps, s2, NL_LAM=0.5, **cfg) == ref and 'nl_rerank_mols' not in s2
        # (3) knob on with a synthetic nl: counted; every 3-column rerank call is slot-preserving per formula
        #     (checked at the gl_multi level: later dedup / tail steps legitimately compact the visible list)
        seen = {'n3': 0}
        real = deps['gl_rerank_multi']

        def spy(smis, keys, scs, forms, cols, lams, top_n):
            o = real(smis, keys, scs, forms, cols, lams, top_n=top_n)
            ref_o = real(smis, keys, scs, forms, cols[:2], lams[:2], top_n=top_n)
            if len(cols) == 3:
                seen['n3'] += 1
                m_ = min(int(top_n), len(smis))
                assert sorted(o) == list(range(len(smis)))
                assert all(forms[o[k_]] == forms[k_] for k_ in range(len(smis))), 'a non-same-formula slot moved'
                assert all(o[k_] == k_ for k_ in range(m_, len(smis)))
            return o
        s_on = {}
        out = cp.run(st_nl, dict(deps, gl_rerank_multi=spy), s_on, NL_LAM=0.5, **cfg)
        assert s_on.get('nl_rerank_mols', 0) > 0 and seen['n3'] > 0, (s_on, seen)
        moved = [m for m in ref if out[m] != ref[m]]
        assert moved, 'synthetic NL should move at least one list'
        print(f'  {shard} {cfg or "default"}: off-identical, on moves {len(moved)}/{len(ref)} lists, '
              f'nl_rerank_mols={s_on["nl_rerank_mols"]}, 3-col calls={seen["n3"]}')


def test_knob_dumps_c2():
    _identity_on('C2', [{}, E006_CFG])


def test_knob_dumps_c1_npx():
    _identity_on('C1_npx', [{}, E006_CFG])


def test_knob_synthetic_calls():
    """gl_multi is called with [ice, gl, nl] / [ICE_LAM, GL_LAM, NL_LAM] only when the knob is on and nl present."""
    cp = _load_post()
    calls = []

    def gl_multi(smis, keys, scs, forms, cols, lams, top_n):
        calls.append((len(cols), list(lams)))
        return list(range(len(smis)))[:top_n]
    SM = ['CCO', 'CCCO', 'CCCCO', 'OCCO']
    ik = cp.canonicalize_smiles_to_ik14
    base = {'m': (SM[:3], [ik(s) for s in SM[:3]], 0.1, [3.0, 2.0, 1.0], ['F'] * 3)}
    pc = {'m': dict(pc=[SM[3]], pc_keys=[ik(SM[3])], pc_fz=[1.0], pc_form=['F'], best_pool_fz=0.0, S=None, top_pop=0)}
    ice = {'m': {s: 0.1 for s in SM}}
    gl = {'m': {s: 0.2 for s in SM}}
    nl = {'m': {s: 0.3 for s in SM}}
    deps = dict(score_key=ik, form=lambda s: 'F', ice_rerank=lambda smis, keys, scs, forms, ice, lam, top_n: list(range(len(smis)))[:top_n],
                gl_rerank_multi=gl_multi, dreams=None)
    st = dict(base=base, pc=pc, eng={}, ice=ice, gl=gl, nl=nl, te_mids=['m'], sample_ids=['m'], targets={}, consts={})
    cp.run(st, deps, {}, USE_ENG=False, FINAL_PASS=False)
    assert calls and all(n == 2 for n, _ in calls), calls
    calls.clear()
    s = {}
    cp.run(st, deps, s, USE_ENG=False, FINAL_PASS=False, NL_LAM=0.5)
    assert calls and all(n == 3 and l[2] == 0.5 for n, l in calls), calls
    assert s['nl_rerank_mols'] == 1


TESTS = [v for k_, v in sorted(globals().items()) if k_.startswith('test_') and callable(v)]


if __name__ == '__main__':
    bad = 0
    for t in TESTS:
        try:
            t()
            print('PASS', t.__name__)
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print('FAIL', t.__name__, repr(e))
    print(f'{len(TESTS) - bad}/{len(TESTS)} passed')
    sys.exit(1 if bad else 0)
