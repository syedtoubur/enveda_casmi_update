"""Tests for the A1 test-network vote (CPU, no Kaggle data).   usage: python build/test_patch_a1_test_network.py"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import patch_a1_test_network as pa  # noqa: E402

SMI = ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'c1ccccc1O', 'c1ccccc1N', 'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O', 'CCCCC(=O)O',
       'CCN', 'CCCN', 'CCCCN', 'c1ccccc1', 'CC(C)O', 'CC(C)N', 'CCOC', 'CCOCC', 'CCCl', 'CCBr', 'CCF', 'CCS', 'CCCS',
       'C1CCCCC1', 'C1CCCC1']


def _spec(seed, prec=300.0, n=20):
    r = np.random.default_rng(seed)
    return r.uniform(50, prec - 5, n), r.uniform(1, 100, n), prec, 'positive'


def test_identical_spectra_cosine_is_one():
    mz, it, p, _ = _spec(0)
    s, n = cp.modified_cosine(mz, it, p, mz, it, p)
    assert abs(s - 1.0) < 1e-9 and n == 20


def test_shifted_precursor_pairs_match():
    mz, it, p, _ = _spec(1)
    # second spectrum: first half of the peaks shifted by dprec (matches only via the shift), rest unchanged
    d = 14.0157
    mz2 = mz.copy(); mz2[:10] += d
    s_mod, n_mod = cp.modified_cosine(mz, it, p, mz2, it, p + d)
    assert n_mod == 20 and s_mod > 0.99
    s_plain, n_plain = cp.modified_cosine(mz, it, p, mz2, it, p)
    assert n_plain < 20 and s_plain < s_mod


def test_network_edges_and_threshold():
    a, b, c = _spec(2), _spec(2), _spec(3)
    net = cp.test_network({'a': [a], 'b': [b], 'c': [c]}, tau=0.8)
    assert [m for m, _ in net['a']] == ['b'] and [m for m, _ in net['b']] == ['a'] and 'c' not in net
    assert cp.test_network({'a': [a], 'b': [(a[0], a[1], a[2], 'negative')]}, tau=0.8) == {}   # polarity must agree


def _rows():
    return {'m1': list(SMI[:25]), 'm2': list(SMI[5:25]) + list(SMI[:5]), 'm3': list(SMI[:3])}


NET = {'m1': [('m2', 0.9)], 'm2': [('m1', 0.9)]}


def test_vote_keeps_slot0_and_tail_and_permutes():
    rows = _rows()
    new, st = cp.net_vote_rows(rows, NET, 5.0, cp.morgan_fp)
    for m in rows:
        assert sorted(new[m]) == sorted(rows[m])
        assert new[m][:1] == rows[m][:1] and new[m][10:] == rows[m][10:]
    assert st['net_mols'] == 2 and st['net_edges'] == 2 and st['net_vote_moved_mols'] >= 1
    assert new['m3'] == rows['m3']


def test_w0_identical():
    rows = _rows()
    new, st = cp.net_vote_rows(rows, NET, 0, cp.morgan_fp)
    assert new == rows and st['net_vote_moved_mols'] == 0


def test_patcher_anchor_once_and_cell_compiles():
    src = open(pa.SRC, encoding='utf-8').read()
    assert src.count(pa.FACTORIAL_ANCHOR) == 1
    compile(pa.ship_cell(0.8, 0.10), '<ship>', 'exec')
    nb = pa.build()
    assert 'NET_VOTE_FAILED' in ''.join(nb['cells'][-1]['source']) or any(
        'NET_VOTE_FAILED' in ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def _run_cell(te, sub_rows, cwd):
    import pandas as pd
    pd.DataFrame({'molecule_id': list(sub_rows), 'smiles': [';'.join(r) for r in sub_rows.values()]}).to_csv(
        os.path.join(cwd, 'submission.csv'), index=False)
    old = os.getcwd(); os.chdir(cwd)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(pa.ship_cell(0.8, 0.10), '<ship>', 'exec'), {'te': te})
    finally:
        os.chdir(old)
    return buf.getvalue(), pd.read_csv(os.path.join(cwd, 'submission.csv'), dtype=str, keep_default_na=False)


def test_ship_cell_on_fake_submission_and_failure_keeps_base():
    import pandas as pd
    mz, it, p, pol = _spec(5)
    te = pd.DataFrame([dict(molecule_id=m, ms2_mzs=mz, ms2_normalized_intensities=it, precursor_mz=p, ionization_mode=pol)
                       for m in ('m1', 'm2')])
    rows = {'m1': list(SMI[:25]), 'm2': list(SMI[5:25]) + list(SMI[:5])}
    with tempfile.TemporaryDirectory() as d:
        out, sub = _run_cell(te, rows, d)
        line = [x for x in out.splitlines() if x.startswith('NET_VOTE ')][0]
        st = json.loads(line[len('NET_VOTE '):])
        assert st['net_mols'] == 2 and st['tau'] == 0.8 and st['w'] == 0.1 and 'net_vote_moved_mols' in st
        for m, r in zip(sub.molecule_id, sub.smiles):
            assert sorted(r.split(';')) == sorted(rows[m])
    with tempfile.TemporaryDirectory() as d:        # broken te -> FAILED, file untouched
        out, sub = _run_cell(pd.DataFrame({'molecule_id': ['m1']}), rows, d)
        assert 'NET_VOTE_FAILED' in out and [r.split(';') for r in sub.smiles] == list(rows.values())


if __name__ == '__main__':
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f) and k != 'test_network':
            try:
                f(); print('ok  ', k)
            except Exception as e:
                fails += 1; print('FAIL', k, repr(e))
    raise SystemExit(1 if fails else 0)
