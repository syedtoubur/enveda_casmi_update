"""Build casmi26-e006-smartgate.ipynb from the E006 notebook (plan: .scratch/smart_gate/PLAN.md).

Every patch is anchored on a string that must occur exactly once in exactly one cell; anything
else aborts the build. smart_gate.py is embedded as text, so the notebook needs no new dataset.
Run: python .scratch/smart_gate/build_nb.py
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = ROOT / 'casmi26-e006-window-40-ice-fallback-true.ipynb'
OUT = HERE / 'casmi26-e006-smartgate.ipynb'
SRC = (HERE / 'smart_gate.py').read_text(encoding='utf-8')

GATE_MD = '''## 6b. Smart library gate (Class-1 separation)

Masked spectral entropy against same-polarity, same-adduct, non-timsTOF library spectra of the same neutral mass
(Phase-3 operating curve; filters n_matched >= 4 and a matched fragment above half the precursor m/z).
T1 (S >= 0.85) and T2 (S >= 0.65) keep the main list free of PubChem and pin the library structure at rank 1;
T3 (S >= 0.40 on >= 2 spectra) gets PubChem only in slots 15/18/21 and keeps the library structure in the top 2.
A rival structure within 0.05 demotes one tier. `lib_max >= 0.9` stays untouched exactly as before.
'''

GATE_CELL = '''\'\'\'Smart library gate (.scratch/smart_gate): computed after the engine lists and outside the engine try.
Any failure leaves SMART_GATES empty, so every molecule keeps the legacy lib_max >= LIB_TAU behaviour.\'\'\'
import importlib.util as _sg_ilu
SMART_GATE_SRC = %s
open('/kaggle/working/smart_gate.py', 'w').write(SMART_GATE_SRC)
_sg_spec = _sg_ilu.spec_from_file_location('smart_gate', '/kaggle/working/smart_gate.py')
SG = _sg_ilu.module_from_spec(_sg_spec); _sg_spec.loader.exec_module(SG)
SMART_GATES, SG_SUMMARY = {}, dict(health='SG NOT RUN')
try:
    from casmi.build import LIBS as _SG_LIBS
    SMART_GATES, SG_SUMMARY = SG.run_all(L, mols, list(_SG_LIBS), chem.adduct_index,
                                         lambda sid: chem.score_key(str(L.struct_smiles[sid])), budget_s=900.0)
except Exception as e:
    SMART_GATES, SG_SUMMARY = {}, dict(health=f'SG FAILED: {e!r}')
print('SMART_GATE', json.dumps(SG_SUMMARY), f'{time.time()-T0:.0f}s', flush=True)
''' % repr(SRC)

OLD_LOOP_START = "rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)\n"
OLD_LOOP_END = "    rows.append((mid, ';'.join(final[:25])))\n"
NEW_LOOP = '''rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0, t0=0, t1=0, t2=0, t3=0, sg_margin_demoted=0)
for mid in te.molecule_id.unique():
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    p = PC.get(mid)
    # smart gate: T4/no decision reproduces the v3.6 loop exactly (tested in .scratch/smart_gate)
    _sg_dec = SMART_GATES.get(str(mid))
    final, _sg_labels = SG.merge_decision(smis, keys, lib_max, p, _sg_dec, LIB_TAU=LIB_TAU, REL_TH=REL_TH,
                                          SLOTS_AGG=SLOTS_AGG, SLOTS_GENTLE=SLOTS_GENTLE, S_TAU=S_TAU,
                                          POP_TAU=POP_TAU, merge=merge, promote=promote)
    for _l in _sg_labels:
        stats[_l] += 1
    stats['sg_margin_demoted'] += int(bool(_sg_dec and _sg_dec.get('margin_demoted')))
    if not final: final = ['CCO']
    rows.append((mid, ';'.join(final[:25])))
'''

LOCK = '''            audit["sg_event"] = None
            if active and scope.get("SMART_GATES"):
                # Smart-gate lock on the treatment list only; the control call above stays unlocked.
                result, audit["sg_event"] = scope["SG"].apply_lock(
                    result, scope["SMART_GATES"].get(mid), scope["metric_identity"], scope["safe_extend"]
                )
'''
REPORT = '''            "sg_events": {
                e: sum(a.get("sg_event") == e for a in audits) for e in scope["SG"].LOCK_EVENTS
            },
            "sg_summary": scope.get("SG_SUMMARY"),
'''


def code_cell(text):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=text.splitlines(keepends=True))


def md_cell(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))


def build():
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    cells = nb['cells']
    texts = [''.join(c['source']) for c in cells]

    def find(anchor):
        hits = [(i, t.count(anchor)) for i, t in enumerate(texts) if anchor in t]
        assert len(hits) == 1 and hits[0][1] == 1, f'anchor not unique: {anchor!r} -> {hits}'
        return hits[0][0]

    def patch(anchor, new, before=False, replace=False):
        i = find(anchor)
        rep = new if replace else (new + anchor if before else anchor + new)
        texts[i] = texts[i].replace(anchor, rep)
        return i

    # Cell 20: no PubChem simulator candidates for locked (T1/T2) molecules
    a = "if p and p.get('pc') and BASE[m][2] < LIB_TAU:"
    patch(a, "if p and p.get('pc') and BASE[m][2] < LIB_TAU and not SG.locked(SMART_GATES.get(str(m))):", replace=True)

    # Cell 22: merge loop -> SG.merge_decision; gate summary into run_manifest.json
    i = find(OLD_LOOP_START); j = find(OLD_LOOP_END)
    assert i == j
    s, e = texts[i].index(OLD_LOOP_START), texts[i].index(OLD_LOOP_END) + len(OLD_LOOP_END)
    texts[i] = texts[i][:s] + NEW_LOOP + texts[i][e:]
    patch('ice_n_scored=_ice_n, gl_n_scored=_gl_n,', '\n               smart_gate=SG_SUMMARY,')

    # Cell 31 (E006 overlay): treatment-only lock + counters; relabel the experiment
    patch('            assert 1 <= len(result) <= 25\n', LOCK, before=True)
    patch('            "same_run_control_reconstructed": True,\n', REPORT)
    patch("'experiment': 'E006',", "'experiment': 'E006_SG',", replace=True)

    out_cells = []
    gate_after = find('BASE_TAIL_CANDIDATES = {')
    for k, (c, t) in enumerate(zip(cells, texts)):
        c = copy.deepcopy(c)
        c['source'] = t.splitlines(keepends=True)
        c.get('metadata', {}).pop('papermill', None)
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
        out_cells.append(c)
        if k == gate_after:
            out_cells += [md_cell(GATE_MD), code_cell(GATE_CELL)]
    nb['cells'] = out_cells
    nb['metadata'].pop('papermill', None)

    for c in out_cells:              # every code cell must at least parse (magics are absent in this notebook)
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<cell>', 'exec')
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('wrote', OUT, len(out_cells), 'cells (base', len(cells), ')')


if __name__ == '__main__':
    sys.exit(build())
