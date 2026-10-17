"""gen-rank1-keep ship probe (campaign 20261017-c3-iso, R1): V22 + GEN1_KEEP, nothing else.

Base: scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb (read only). Two edits:
  1. the C8 rerank cell snapshots BASE top 5 (smiles, keys, forms) just before the loop rewrites it;
  2. a final cell applies casmi_post._gen1_keep to the rows of the finished V22 submission.csv. Only the rows where
     the knob fires change, so the draft differs from the V22 control only there (L38). Counters: `gen1_checked`
     (BASE #1 formulas examined, always > 0), `gen1_kept`, `gen1_moved` (may be 0 on the 12-molecule draft).
On the scoring rerun a failure keeps the V22 file (GEN1_FAILED); on a commit run it raises.

usage: python build/patch_gen1_keep.py [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "scratch", "baseline", "casmi26-v22-rel150-e2r2-clean.ipynb")

SNAP_ANCHOR = "for m in list(BASE):\n    smis, keys, lib_max, scs, forms = BASE[m]\n"
SNAP = (
    "# >>> GEN1_KEEP (patch_gen1_keep.py): BASE top 5 before the C8 rerank rewrites it\n"
    "_g1_base = {str(_g1_m): (list(_g1_v[0][:5]), list(_g1_v[1][:5]), _g1_v[2], list(_g1_v[3][:5]), list(_g1_v[4][:5]))\n"
    "            for _g1_m, _g1_v in BASE.items() if len(_g1_v) >= 5}\n"
    "# <<< GEN1_KEEP\n")

FINAL_CELL = r'''# === GEN1_KEEP (patch_gen1_keep.py): keep an ICE-agreeing generated BASE #1 at rank 1 (last step, after E2R2) ===
import os as _g1_os, sys as _g1_sys, shutil as _g1_shutil
import pandas as _g1_pd
open('/kaggle/working/casmi_post_gen1.py', 'w').write(POST_SRC)
_g1_sys.path.insert(0, '/kaggle/working')
import casmi_post_gen1 as _g1_post
_g1_strict = not bool(globals().get('IS_RERUN', True))      # commit run: fail loudly; scoring rerun: keep the V22 file
try:
    _g1_df = _g1_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _g1_shutil.copy('submission.csv', 'submission_base_gen1.csv')
    _g1_out = {str(m): str(s).split(';') for m, s in zip(_g1_df.molecule_id, _g1_df.smiles)}
    _g1_st = {'base': _g1_base}
    _g1_stats = {}
    _g1_checked = sum(1 for _g1_v in _g1_base.values() if _g1_v[0])
    _g1_new = _g1_post._gen1_keep(_g1_out, _g1_st, ICE_SCORES, set(_g1_base), _g1_stats)
    _g1_df['smiles'] = [';'.join(_g1_new[str(m)]) for m in _g1_df.molecule_id]
    _g1_df.to_csv('submission.csv', index=False)
    print('GEN1_KEEP gen1_checked', _g1_checked, flush=True)
    print('GEN1_KEEP gen1_kept', _g1_stats.get('gen1_kept', 0), flush=True)
    print('GEN1_KEEP gen1_moved', _g1_stats.get('gen1_moved', 0), flush=True)
    _g1_after = _g1_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    assert list(_g1_after.molecule_id) == list(_g1_pd.read_csv('submission_base_gen1.csv', dtype=str, keep_default_na=False).molecule_id)
    assert all(len(s.split(';')) <= 25 for s in _g1_after.smiles)
except Exception as _g1_e:
    print('GEN1_FAILED', repr(_g1_e), flush=True)
    if _g1_strict:
        raise
    if _g1_os.path.isfile('submission_base_gen1.csv'):
        _g1_shutil.copy('submission_base_gen1.csv', 'submission.csv')
'''


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def code_cell(text):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    set_src(c, text)
    return c


def build(src: str = SRC) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and SNAP_ANCHOR in src_of(c)]
    if len(hits) != 1:
        raise SystemExit(f"PATCH FAILED: C8 rerank loop anchor matched {len(hits)} cells")
    text = src_of(nb["cells"][hits[0]])
    if text.count(SNAP_ANCHOR) != 1:
        raise SystemExit("PATCH FAILED: C8 rerank loop anchor occurs more than once")
    set_src(nb["cells"][hits[0]], text.replace(SNAP_ANCHOR, SNAP + SNAP_ANCHOR))
    post_src = open(os.path.join(HERE, "casmi_post.py"), encoding="utf-8").read()
    nb["cells"].append(code_cell(f"POST_SRC = {post_src!r}\n" + FINAL_CELL))
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(ROOT, "casmi26-v22-gen1-keep.ipynb"))
    a = ap.parse_args(argv)
    nb = build(a.src)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
