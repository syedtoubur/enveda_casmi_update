"""Tests for the lib-isomer-margin probe (CPU, no Kaggle data): the post rule, the knob seams in casmi_post.run, the
C7 graft in both notebooks, and the ship cell run on a fake submission.

usage: python build/test_patch_lib_isomer.py
"""
from __future__ import annotations

import contextlib
import csv
import inspect
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import casmi_post as cp  # noqa: E402
import patch_harness_dump as phd  # noqa: E402
import patch_lib_isomer as pl  # noqa: E402
import test_patch_harness_dump as T  # noqa: E402

SM = T.SM
A, TR, B = SM[0], SM[1], SM[2]          # A: library 0.70, TR: no reference (truth), B: library 0.90


def _ik(s):
    return s


def _run(state, **kw):
    return cp.run(state, T.deps(), **kw)


def _with_lib(lib):
    s = T.state()
    s['lib_sim'] = lib
    return s


def test_rule_swaps_library_slots_only():
    rows = [A, TR, B, SM[5], SM[6]]
    new, el, sw = cp.lib_isomer_rows(rows, {A: 0.70, B: 0.90}, 0.05, _ik)
    assert new == [B, TR, A, SM[5], SM[6]] and el and sw
    assert sorted(new) == sorted(rows)


def test_rule_margin_gap_and_single_candidate():
    rows = [A, TR, B]
    assert cp.lib_isomer_rows(rows, {A: 0.70, B: 0.90}, 0.5, _ik) == (rows, True, False)   # gap 0.2 <= 0.5
    assert cp.lib_isomer_rows(rows, {A: 0.70}, 0.05, _ik) == (rows, False, False)         # |S| = 1
    assert cp.lib_isomer_rows(rows, {}, 0.05, _ik) == (rows, False, False)


def test_knob_none_is_base_even_with_lib_sim():
    s = _with_lib({'m1': {A: 0.70, B: 0.90}})
    assert _run(s) == _run(s, LIB_ISOMER_MARGIN=None)
    assert _run(T.state()) == _run(s)                 # lib_sim present but knob off: base output


def test_no_lib_sim_key_is_identical_with_knob_on():
    base = _run(T.state())
    st = {}
    assert _run(T.state(), stats=st, LIB_ISOMER_MARGIN=0.05) == base
    assert st.get('lib_isomer_eligible', 0) == 0 and st.get('lib_sim_mols', 0) == 0


def test_top3_reorders_library_backed_slots():
    s = _with_lib({'m1': {A: 0.70, B: 0.90}})
    st = {}
    out = _run(s, stats=st, LIB_ISOMER_MARGIN=0.05)
    assert out['m1'][:3] == [B, TR, A]                # the no-reference truth stays in slot 2
    assert (st['lib_sim_mols'], st['lib_isomer_eligible'], st['lib_isomer_swapped']) == (1, 1, 1)
    base = _run(T.state())
    assert out['m1'][3:] == base['m1'][3:]            # ranks 4-25 untouched
    assert len({cp.canonicalize_smiles_to_ik14(x) for x in out['m1']}) == len(out['m1'])   # no duplicate ik14


def test_margin_larger_than_gap_is_unchanged():
    s = _with_lib({'m1': {A: 0.70, B: 0.90}})
    st = {}
    assert _run(s, stats=st, LIB_ISOMER_MARGIN=0.5)['m1'][:3] == [A, TR, B]
    assert st['lib_isomer_eligible'] == 1 and st['lib_isomer_swapped'] == 0


def test_single_library_candidate_in_top3_is_unchanged():
    s = _with_lib({'m1': {A: 0.70}})
    st = {}
    assert _run(s, stats=st, LIB_ISOMER_MARGIN=0.05)['m1'][:3] == [A, TR, B]
    assert st['lib_isomer_eligible'] == 0 and st['lib_sim_mols'] == 1


def test_lib_sim_resolves_through_ik14():
    # the reference is keyed by another SMILES spelling of the same molecule: the ik14 join still finds it
    assert B == 'CCCCO' and cp.canonicalize_smiles_to_ik14('OCCCC') == cp.canonicalize_smiles_to_ik14(B)
    s = _with_lib({'m1': {A: 0.70, 'OCCCC': 0.90}})
    assert _run(s, LIB_ISOMER_MARGIN=0.05)['m1'][:3] == [B, TR, A]


# ---------------------------------------------------------------- patch_lib_isomer.build (ship notebook)
def test_ship_notebook_patch():
    base = json.load(open(pl.SRC, encoding='utf-8'))['cells']
    nb = pl.build(pl.SRC, 0.05)
    fac = [i for i, c in enumerate(base) if pl.FACTORIAL_ANCHOR in ''.join(c['source'])]
    assert len(fac) == 1 and len(nb['cells']) == len(base) + 1
    src = ''.join(nb['cells'][fac[0] + 1]['source'])
    assert src.startswith('# === LIB_ISOMER') and 'LIB_ISOMER_MARGIN = 0.05' in src
    assert inspect.getsource(cp.lib_isomer_rows) in src
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert sum('LIB_SIM[mid] =' in s for s in code) == 1
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            lines = [('# ' + ln) if ln.lstrip().startswith(('!', '%')) else ln for ln in ''.join(c['source']).splitlines()]
            compile('\n'.join(lines), 'cell', 'exec')
            assert c['outputs'] == [] and c['execution_count'] is None
    assert sum('LIB_ISOMER_FAILED' in s for s in code) == 1


def _exec_ship(lib_sim, body_csv):
    cells = [''.join(c['source']) for c in pl.build(pl.SRC, 0.05)['cells'] if c['cell_type'] == 'code']
    src = next(c for c in cells if c.startswith('# === LIB_ISOMER'))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
                f.write('molecule_id,smiles\n' + body_csv)
            g = {} if lib_sim is None else {'LIB_SIM': lib_sim}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(src, 'ship', 'exec'), g)
            out = list(csv.DictReader(open('submission.csv', encoding='utf-8')))
            return buf.getvalue(), out
        finally:
            os.chdir(cwd)


def test_ship_cell_rewrites_submission_on_fake_rows():
    body = 'm1,CCCO;CCO;CCCCO;CCCCCO\nm2,CCO\n'
    out_txt, out = _exec_ship({'m1': {'CCCO': 0.70, 'CCCCO': 0.90}}, body)
    assert 'LIB_ISOMER {"lib_sim_mols": 1, "lib_isomer_eligible": 1, "lib_isomer_swapped": 1, "margin": 0.05}' in out_txt, out_txt
    assert out[0]['smiles'] == 'CCCCO;CCO;CCCO;CCCCCO' and out[1]['smiles'] == 'CCO'


def test_ship_cell_failure_keeps_submission():
    out_txt, out = _exec_ship(None, 'm1,CCCO;CCO;CCCCO\n')      # LIB_SIM undefined -> NameError inside the try
    assert 'LIB_ISOMER_FAILED' in out_txt and out[0]['smiles'] == 'CCCO;CCO;CCCCO'


# ---------------------------------------------------------------- graft in the dump notebook
def test_dump_notebook_has_graft_once():
    code = [''.join(c['source']) for c in phd.build('C1_npx', ranker=('Q', 'np'))['cells'] if c['cell_type'] == 'code']
    assert sum('LIB_SIM[mid] =' in s for s in code) == 1 and sum('LIB_SIM = {}' in s for s in code) == 1
    assert sum("lib_sim=dict(globals().get('LIB_SIM')" in s for s in code) == 1


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
