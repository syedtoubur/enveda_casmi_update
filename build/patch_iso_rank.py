"""iso-rank fast-ship (research/c3/20261018-iso-rank): V22 + one appended iso stage, nothing else changes.

Base: scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb (read only). One appended cell, run after the E2R2 cell has
written the final submission.csv: parents = the final row's top 10 -> isomer children (isomer_mut) -> CPU frag + NL ->
ICEBERG + GLACIER on parents + top-K children -> iso_rank.apply (fast-ship cfg: m_ice 0, k 10, all six lists, tail on)
-> rewrite the rows. Rows only change where the stage fires or the tail rides along.
Draft (no KAGGLE_IS_COMPETITION_RERUN): the base already restricts `te` to 12 molecules, so the stage runs on those only.
Wall-clock guard: molecules not started before T0 + 8.3 h keep their V22 row (`iso_skipped_time`).
On the scoring rerun a failure keeps the V22 file (ISO_FAILED); on a commit run it raises.

usage: python build/patch_iso_rank.py [--k 150] [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "scratch", "baseline", "casmi26-v22-rel150-e2r2-clean.ipynb")
MODULES = ("isomer_mut", "frag_rescore", "iso_nl", "iso_rank", "iso_stage")
DEFAULT_K = 150

FINAL_CELL = r'''# === ISO_MUT iso-rank (patch_iso_rank.py): isomer children ranked by a truth-blind RRF, last step after E2R2 ===
import os as _iso_os, sys as _iso_sys, time as _iso_time, shutil as _iso_shutil, json as _iso_json
import numpy as _iso_np
import pandas as _iso_pd
_iso_dir = '/kaggle/working/iso_pkg'
_iso_os.makedirs(_iso_dir, exist_ok=True)
for _iso_n, _iso_s in ISO_SRC.items():
    open(_iso_os.path.join(_iso_dir, _iso_n + '.py'), 'w', encoding='utf-8').write(_iso_s)
if _iso_dir not in _iso_sys.path:
    _iso_sys.path.insert(0, _iso_dir)
import iso_stage as _iso_stage
_iso_strict = not bool(globals().get('IS_RERUN', True))      # commit run: fail loudly; scoring rerun: keep the V22 file
_iso_K = ISO_K
_iso_limit = T0 + 8.3 * 3600.0                                 # T0 = notebook start (base cell 1)
try:
    _iso_t = _iso_time.time()
    _iso_df = _iso_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _iso_shutil.copy('submission.csv', 'submission_base_iso.csv')
    _iso_active = set(te.molecule_id.astype(str))
    _iso_rows = {str(m): [x for x in str(s).split(';') if x] for m, s in zip(_iso_df.molecule_id, _iso_df.smiles)
                 if str(m) in _iso_active}
    _iso_spec = {}
    for _iso_r in te.itertuples(index=False):
        _iso_spec.setdefault(str(_iso_r.molecule_id), []).append(dict(
            mz=_iso_np.asarray(_iso_r.ms2_mzs, _iso_np.float64), it=_iso_np.asarray(_iso_r.ms2_normalized_intensities, _iso_np.float64),
            adduct=_iso_r.adduct, prec=float(_iso_r.precursor_mz), mode=1 if _iso_r.ionization_mode == 'positive' else -1))
    _iso_tasks = [(m, r, _iso_spec.get(m, [])[:16], _iso_limit, _iso_K) for m, r in _iso_rows.items()]
    print('ISO start: molecules', len(_iso_tasks), 'K', _iso_K, f'elapsed {_iso_time.time() - T0:.0f}s', flush=True)
    _iso_res = _iso_stage.generate(_iso_tasks, procs=4)
    _iso_gen_s = _iso_time.time() - _iso_t
    _iso_c = _iso_stage.cands_for(_iso_rows, _iso_res)
    print('ISO gen done', f'{_iso_gen_s:.0f}s', 'molecules with children', len(_iso_c), 'cands', sum(len(v) for v in _iso_c.values()), flush=True)
    _iso_items = ice_fuse.build_ice_input(te, _iso_c)
    _iso_items = [i for i in _iso_items if i['spectra']]
    _iso_rem = _iso_limit - _iso_time.time()
    _iso_ice_b = 900 if _iso_strict else max(600.0, _iso_rem - 3600.0)      # GLACIER keeps >= ~50 min, apply+write 10 min
    _iso_t1 = _iso_time.time()
    _iso_ice = ice_fuse.run_ice(ICE_PKG, _iso_items, workdir='/kaggle/working/iso_ice_work', device='cuda', budget_s=_iso_ice_b,
                                site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])
    _iso_ice_s = _iso_time.time() - _iso_t1
    _iso_gl, _iso_gl_s = {}, 0.0
    try:
        _iso_glp = globals().get('GL_PKG') or _iso_os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
        _iso_gl_b = 600 if _iso_strict else max(300.0, _iso_limit - _iso_time.time() - 600.0)
        _iso_t2 = _iso_time.time()
        _iso_gl = gl_fuse.run_gl(_iso_glp, _iso_items, workdir='/kaggle/working/iso_gl_work', device='cuda', budget_s=_iso_gl_b,
                                 site='/kaggle/working/ice_site', wheels=_iso_os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])
        _iso_gl_s = _iso_time.time() - _iso_t2
    except Exception as _iso_ge:
        print('ISO GL FAILED -> gl list absent:', repr(_iso_ge), flush=True)
    _iso_n_ice = sum(1 for m in _iso_ice.values() for v in m.values() if v is not None) if isinstance(_iso_ice, dict) else 0
    _iso_n_gl = sum(1 for m in _iso_gl.values() for v in m.values() if v is not None) if isinstance(_iso_gl, dict) else 0
    print('ISO_RATES ice_cand_scored', _iso_n_ice, 'ice_s', int(_iso_ice_s), 'ice_cand_per_s', round(_iso_n_ice / max(_iso_ice_s, 1.0), 2),
          '| gl_cand_scored', _iso_n_gl, 'gl_s', int(_iso_gl_s), 'gl_cand_per_s', round(_iso_n_gl / max(_iso_gl_s, 1.0), 2),
          '| gen_s', int(_iso_gen_s), 'items', len(_iso_items), 'elapsed_total_s', int(_iso_time.time() - T0), flush=True)
    if _iso_strict and _iso_c and _iso_n_ice == 0:
        raise RuntimeError('ISO: ICEBERG scored nothing on the draft')
    _iso_new, _iso_cnt = _iso_stage.assemble(_iso_rows, _iso_res, _iso_ice, _iso_gl)
    _iso_df['smiles'] = [';'.join(_iso_new[str(m)]) if str(m) in _iso_new else _iso_x
                         for m, _iso_x in zip(_iso_df.molecule_id, _iso_df.smiles)]
    _iso_df.to_csv('submission.csv', index=False)
    print(_iso_stage.counters_line(_iso_cnt), flush=True)
    print('ISO_COUNTERS_EXTRA iso_ice_mols=%d iso_gl_mols=%d' % (_iso_cnt['iso_ice_mols'], _iso_cnt['iso_gl_mols']), flush=True)
    _iso_after = _iso_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _iso_before = _iso_pd.read_csv('submission_base_iso.csv', dtype=str, keep_default_na=False)
    assert list(_iso_after.molecule_id) == list(_iso_before.molecule_id)
    assert all(1 <= len(s.split(';')) <= 25 for s in _iso_after.smiles)
    assert all(len(set(s.split(';'))) == len(s.split(';')) for s in _iso_after.smiles)
    print('ISO_CHANGED rows', int((_iso_after.smiles != _iso_before.smiles).sum()), 'of', len(_iso_after), flush=True)
except Exception as _iso_e:
    print('ISO_FAILED', repr(_iso_e), flush=True)
    if _iso_strict:
        raise
    if _iso_os.path.isfile('submission_base_iso.csv'):
        _iso_shutil.copy('submission_base_iso.csv', 'submission.csv')
'''


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def code_cell(text):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    set_src(c, text)
    return c


def build(src: str = SRC, k: int = DEFAULT_K) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    mods = {m: open(os.path.join(HERE, m + ".py"), encoding="utf-8").read() for m in MODULES}
    nb["cells"].append(code_cell(f"ISO_SRC = {mods!r}\nISO_K = {int(k)}\n" + FINAL_CELL))
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--out", default=os.path.join(ROOT, "casmi26-v22-iso-rank.ipynb"))
    a = ap.parse_args(argv)
    nb = build(a.src, a.k)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out, "K", a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
