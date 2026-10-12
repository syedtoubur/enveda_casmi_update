"""Run the PATCHED E006 overlay cell (and the patched metric-audit defs) on a fake same-run scope.

Proves: control reconstruction still passes with the lock active (defect 9), the lock fires on the
treatment list only, and the report carries sg_events. Needs RDKit (metric identity).
Run: python .scratch/smart_gate/test_overlay_integration.py   (after build_nb.py)
"""
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smart_gate as SG  # noqa: E402

NB = HERE / 'casmi26-e006-smartgate.ipynb'


def cell_with(cells, anchor):
    hits = [c for c in cells if anchor in c]
    assert len(hits) == 1, anchor
    return hits[0]


def defs_only(src, names):
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


class _Chem:
    @staticmethod
    def score_key(s):
        return s


def main():
    cells = [''.join(c['source']) for c in json.loads(NB.read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
    ns = {'np': np, 'pd': pd}
    exec(cell_with(cells, 'def metric_identity'), ns)                       # canonicalize/metric_identity/safe_extend
    exec(defs_only(cell_with(cells, 'def metric_fusion'), {'metric_fusion'}), ns)
    overlay = cell_with(cells, 'def _factorial_entry')

    mids = ['m1', 'm2', 'm3', 'm4']
    pool = ['CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'OCCO', 'CC(=O)O', 'CCN', 'CCCN', 'c1ccccc1', 'c1ccccc1O',
            'CCOC', 'CCC(=O)O', 'NCCO', 'CC(N)C', 'OC1CCCCC1']
    rng = np.random.default_rng(0)
    pref = {m: list(rng.permutation(pool)[:10]) for m in mids}
    eng = {m: {'smiles': list(rng.permutation(pool)[:8])} for m in mids}
    ref_rows, ctrl_rows = [], []
    for m in mids:
        original = list(dict.fromkeys(pref[m] + eng[m]['smiles']))[:25]
        first = [x for x in pref[m] if x and x != 'CCO']          # rerank_row drops placeholders
        cands, _sc, _ = ns['metric_fusion'](first, eng[m]['smiles'], ns['metric_identity'], alpha=0.6, offset=3.0, limit=40)
        ctrl = ns['safe_extend'](cands[:25], cands[25:] + original)
        ref_rows.append((m, ';'.join(original))); ctrl_rows.append((m, ';'.join(ctrl)))
    ctrl_lists = {m: s.split(';') for m, s in ctrl_rows}
    gates = {
        'm1': dict(tier=1, smiles=ctrl_lists['m1'][4]),      # present at rank 5 -> pinned_moved
        'm2': dict(tier=2, smiles='CCCCCCO'),                  # absent -> inserted
        'm3': dict(tier=3, smiles=ctrl_lists['m3'][6]),      # -> top2_moved
        'm4': dict(tier=4, smiles=None),                       # no lock
    }

    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            pd.DataFrame(ref_rows, columns=['molecule_id', 'smiles']).to_csv('submission_reference.csv', index=False)
            pd.DataFrame(ctrl_rows, columns=['molecule_id', 'smiles']).to_csv('submission.csv', index=False)
            sha = hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()
            Path('e004_verification.json').write_text(json.dumps(dict(
                experiment='E004', hashes={'submission.csv': sha}, processed_ids=mids)))
            scope = dict(ns, te=pd.DataFrame({'molecule_id': mids}), ENG=eng, PREFUSION_LISTS=pref,
                         ALPHA=0.6, KRR=3.0, chem=_Chem, ICE_SCORES={}, gl_of=lambda m: {},
                         gl_fuse=None, ice_fuse=None, _form2=lambda s: s, ICE_LAM=1.0, GL_LAM=1.0,
                         SMART_GATES=({} if os.environ.get('SG_NO_GATES') else gates), SG=SG, SG_SUMMARY={'health': 'ok'}, sha256=lambda p: '')
            exec(overlay, scope)                                    # asserts control reconstruction inside
            rep = scope['FACTORIAL_REPORT']
            out = {r.molecule_id: r.smiles.split(';') for r in pd.read_csv('submission.csv').itertuples()}
        finally:
            os.chdir(cwd)

    assert rep['experiment'] == 'E006_SG' and rep['same_run_control_reconstructed']
    ev = rep['sg_events']
    assert ev['pinned_moved'] == 1 and ev['inserted'] == 1 and ev['top2_moved'] == 1, ev
    mi = ns['metric_identity']
    assert mi(out['m1'][0]) == mi(gates['m1']['smiles'])
    assert out['m2'][0] == 'CCCCCCO' and len(out['m2']) <= 25
    assert mi(out['m3'][1]) == mi(gates['m3']['smiles']) and out['m3'][0] == ctrl_lists['m3'][0]
    assert out['m4'] == ctrl_lists['m4']
    print('overlay integration OK', ev)


if __name__ == '__main__':
    main()
