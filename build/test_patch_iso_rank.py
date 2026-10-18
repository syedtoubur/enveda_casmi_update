"""patch_iso_rank: only one cell is appended, and the final cell runs end to end on fakes (real spawn Pool, real
isomer_mut / frag / NL / iso_rank; ICEBERG / GLACIER faked) and obeys the counters, the 25-row and the dedupe contract."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

import patch_iso_rank as P

SAL = 'OC(=O)c1ccccc1O'
TYR = 'COc1ccc(CCN)cc1O'
FILL = ['C' * n + 'O' for n in range(1, 13)] + ['C' * n + '(=O)O' for n in range(1, 13)]


def _cells(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def test_only_one_cell_appended():
    assert os.path.exists(P.SRC), P.SRC
    base = json.load(open(P.SRC, encoding="utf-8"))
    a, b = _cells(base), _cells(P.build())
    assert len(b) == len(a) + 1 and a == b[:-1]
    assert b[-1].startswith("ISO_SRC = ") and "ISO_COUNTERS" in b[-1] and "ISO_K = 150" in b[-1]
    assert "ISO_K = 50" in _cells(P.build(k=50))[-1]


def test_helper_names_do_not_collide_with_base():
    base = "\n".join(_cells(json.load(open(P.SRC, encoding="utf-8"))))
    assert "_iso_" not in base and "ISO_SRC" not in base and "ISO_K" not in base


class _FakeFuse:
    @staticmethod
    def build_ice_input(te, cands, covered_only=True):
        return [dict(mid=str(m), spectra=[dict(adduct='[M+H]+')], cands=list(c)) for m, c in cands.items()]

    @staticmethod
    def run_ice(pkg, items, **kw):
        return {i['mid']: {c: (int(hashlib.md5(c.encode()).hexdigest()[:6], 16) % 1000) / 1000.0 for c in i['cands']} for i in items}

    run_gl = run_ice


def _run_final(rerun=False, k=6):
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            rows = {'m1': [SAL] + FILL[:24], 'm2': [TYR] + FILL[:24]}
            pd.DataFrame({'molecule_id': list(rows), 'smiles': [';'.join(v) for v in rows.values()]}).to_csv('submission.csv', index=False)
            te = pd.DataFrame([dict(molecule_id=m, ms2_mzs=[77.04, 95.05, 121.03, 139.04], ms2_normalized_intensities=[0.3, 0.2, 1.0, 0.5],
                                    precursor_mz=139.0390, adduct='[M+H]+', ionization_mode='positive') for m in rows])
            g = dict(ISO_SRC={m: open(os.path.join(P.HERE, m + '.py'), encoding='utf-8').read() for m in P.MODULES}, ISO_K=k,
                     T0=time.time(), te=te, IS_RERUN=rerun, ICE_PKG=d, ice_fuse=_FakeFuse, gl_fuse=_FakeFuse, GL_PKG=d)
            cell = P.FINAL_CELL.replace('/kaggle/working', d.replace(os.sep, '/'))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(cell, 'final', 'exec'), g)
            out = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
            return rows, out, buf.getvalue()
        finally:
            os.chdir(old)


def test_final_cell_end_to_end():
    rows, out, txt = _run_final()
    assert 'ISO_FAILED' not in txt, txt
    line = [x for x in txt.splitlines() if x.startswith('ISO_COUNTERS ')]
    assert len(line) == 1, txt
    kv = dict(x.split('=') for x in line[0].split()[1:])
    assert int(kv['n_mol']) == 2 and int(kv['iso_children']) == 2 and int(kv['iso_skipped_time']) == 0
    assert 0 <= int(kv['iso_fired']) <= 2
    for m, s in zip(out.molecule_id, out.smiles):
        r = s.split(';')
        assert 1 <= len(r) <= 25 and len(set(r)) == len(r)
        assert r[0] == rows[m][0]                       # never rank 1
        assert set(r[:20]) - set(rows[m][:20]) <= set(r)  # sanity
    assert 'ISO_RATES' in txt


def test_wall_clock_guard_skips():
    import iso_stage
    iso_stage._init()
    r = iso_stage._work(('m1', [SAL], [], time.time() - 1.0, 5))
    assert r['skipped'] is True
    new, cnt = iso_stage.assemble({'m1': [SAL, 'CCO']}, {'m1': r}, {}, {})
    assert cnt['iso_skipped_time'] == 1 and new['m1'] == [SAL, 'CCO'] and cnt['iso_children'] == 0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
