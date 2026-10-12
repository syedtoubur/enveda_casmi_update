"""Tests for build/patch_engine2_fppair.py and the casmi_post knobs (CPU, no Kaggle data).

G1: only the Engine 2 cell differs from the E006 v1 base. G2: the patched engine's guards (A6(ii) fallback fixture).

usage: python build/test_patch_engine2_fppair.py
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import patch_engine2_fppair as pe  # noqa: E402
import numpy as np  # noqa: E402

BASE_NB = json.load(open(pe.SRC, encoding='utf-8'))


def _code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def _files(cell):
    line = next(l for l in cell.split('\n') if l.startswith(pe.ENG_LINE))
    return ast.literal_eval(ast.parse(line).body[0].value)


def _engine_cell(nb):
    return next(s for s in _code(nb) if s.startswith("'''Our engine (two-ranker"))


def test_knob_defaults_are_noop():
    assert cp.CFG['ENGINE2_ROWS'] == 'nofp' and cp.CFG['ENGINE2_W_PV'] == 0.88
    st = __import__('test_patch_harness_dump').state
    d = __import__('test_patch_harness_dump').deps
    e006 = dict(ICE_FALLBACK=True, METRIC_FUSION=True, FUSION_WINDOW=40)
    assert cp.run(st(), d(), **e006) == cp.run(st(), d(), ENGINE2_ROWS='fppair', ENGINE2_W_PV=0.5, **e006)  # replay reads stored ENG


def test_only_engine2_cell_changes():
    nb = pe.build()
    base, new = _code(BASE_NB), _code(nb)
    assert len(base) == len(new)
    diff = [i for i, (a, b) in enumerate(zip(base, new)) if a != b]
    assert len(diff) == 1 and base[diff[0]].startswith("'''Our engine (two-ranker"), diff
    other = [c for c in BASE_NB['cells'] if c['cell_type'] != 'code']
    assert other == [c for c in nb['cells'] if c['cell_type'] != 'code']


def test_patched_sources_compile_and_gate():
    cell = _engine_cell(pe.build())
    ast.parse(cell)
    files = _files(cell)
    for n, s in files.items():
        ast.parse(s)
    base_files = _files(_engine_cell(BASE_NB))
    assert files['pv.py'] == base_files['pv.py'] and files['pv_fp.py'] == base_files['pv_fp.py']
    assert "ENGINE2_ROWS, ENGINE2_W_PV = 'fppair', 0.5" in cell and 'E2_ROWS=ENGINE2_ROWS' in cell
    off = _engine_cell(pe.build(rows='nofp', w_pv=0.88))
    assert "ENGINE2_ROWS, ENGINE2_W_PV = 'nofp', 0.88" in off


def _engine(tmp):
    files = _files(_engine_cell(pe.build()))
    for n, s in files.items():
        open(os.path.join(tmp, n), 'w', encoding='utf-8').write(s)
    sys.path.insert(0, tmp)
    sys.modules.pop('pv', None); sys.modules.pop('casmi_engine', None)
    fake = types.ModuleType('pv_fp')
    sys.modules['pv_fp'] = fake
    try:
        import casmi_engine as E
    except Exception as e:                                   # pv needs a dep this machine lacks: skip, G4 covers it
        return None, fake, files, repr(e)
    return E, fake, files, None


def _recs(n=3, zero=False):
    return [dict(mid=f'm{i}', cand=np.arange(4), zlog_ours=np.zeros(6930, np.float32) if zero else np.ones(6930, np.float32))
            for i in range(n)]


def test_fallback_fixture_a6ii():
    with tempfile.TemporaryDirectory() as tmp:
        E, fake, files, why = _engine(tmp)
        if E is None:
            print('SKIP engine import:', why)
            return
        E.POOL = {'fp': np.packbits(np.zeros((4, 6930), np.uint8), axis=1)}
        buf = io.StringIO()
        # (1) FPNet missing -> load raises -> reason, no models
        def boom(paths, dev): raise FileNotFoundError('ours_fpnet_single_16k.pt')
        fake.load_fp_models = boom
        m, w = E._e2_load(['x.pt'], 'cpu')
        assert m is None and w.startswith('load')
        # (2) FPNet width != pool fp width
        fake.load_fp_models = lambda paths, dev: ([object()], [], 1000)
        m, w = E._e2_load(['x.pt'], 'cpu')
        assert m is None and 'nbits 1000' in w
        fake.load_fp_models = lambda paths, dev: ([object()], [object()], 6930)
        m, w = E._e2_load(['x.pt'], 'cpu')
        assert m is not None and w is None
        # (3) _e2_plan: every failed check returns the base call args and prints E2_ROWS FALLBACK
        calls = []
        E.BLOCKS = {'base': lambda r, cfp: np.zeros((len(cfp), 31), np.float32),
                    'libx': lambda r, cfp: np.zeros((len(cfp), 8), np.float32)}
        d = os.path.dirname(os.path.abspath(tmp))
        def npz(name, width, blocks):
            p = os.path.join(tmp, name)
            np.savez(p, X=np.zeros((5, width), np.float32), blocks=np.array(blocks))
            return p
        good, wrong = npz('good.npz', 39, ['base', 'libx']), npz('wrong.npz', 51, ['base', 'libx'])
        unk = npz('unk.npz', 39, ['base', 'nope'])
        base_args = ('nofp.npz', False, 'zlog', 0.88)
        ok = io.StringIO()
        with contextlib.redirect_stdout(ok):
            assert E._e2_plan(_recs(), (1, 2, 'cpu'), good, 'nofp.npz', 0.88, 0.5) == (good, True, 'zlog_ours', 0.5)
        out = ok.getvalue()
        assert 'E2_ROWS fppair rows=5 feats=39' in out and 'E2_FP ours models=2 nonzero_frac=1.000' in out, out
        for label, recs, path in (('width', _recs(), wrong), ('blocks', _recs(), unk), ('zero', _recs(zero=True), good),
                                  ('missing npz', _recs(), os.path.join(tmp, 'absent.npz'))):
            fb = io.StringIO()
            with contextlib.redirect_stdout(fb):
                try:
                    got = E._e2_plan(recs, (1, 2, 'cpu'), path, 'nofp.npz', 0.88, 0.5)
                except FileNotFoundError:
                    got = None                                # absent npz: the runner's find() catches it before main
            if got is not None:
                assert got == base_args and 'E2_ROWS FALLBACK' in fb.getvalue(), (label, got, fb.getvalue())
        # (4) no models2 (knob off or FPNet fallback) -> base call args, nothing printed
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            assert E._e2_plan(_recs(), None, good, 'nofp.npz', 0.88, 0.5) == base_args
        assert quiet.getvalue() == ''
        # (5) score_molecules default zkey is the base call
        import inspect
        assert 'zkey="zlog"' in inspect.getsource(E.score_molecules)


def test_runner_default_path_is_base():
    files = _files(_engine_cell(pe.build()))
    run = [l for l in _engine_cell(pe.build()).split('\n') if l.startswith(pe.RUNNER_LINE)][0]
    src = ast.literal_eval(ast.parse(run).body[0].value.args[0])
    assert "os.environ.get('E2_ROWS', 'nofp')" in src and "float(os.environ.get('E2_W_PV', '0.88'))" in src
    assert "if e2_rows == 'fppair':" in src and 'w_pv=0.88, **kw)' in src
    assert 'E2_TIME fp_ours=' in src and 'E2_BLEND w_pv=' in src and 'E2_ROWS FALLBACK missing' in src
    assert 'sim_rank_rows_fppair.npz' in src and 'ours_fpnet_single_16k.pt' in src and 'ours_fpnet_merged_26k.pt' in src
    assert files


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
