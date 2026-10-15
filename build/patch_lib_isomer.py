"""lib-isomer-margin probe: E006 v1 + the C7 library-similarity graft + one final cell that reorders the library-backed
top-3 rows of submission.csv (rule: casmi_post.lib_isomer_rows, embedded from its own source at patch time).

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (E006 v1, read only). Changes:
  (a) the C7 graft (build/patch_harness_dump.py lib_sim_graft): LIB_SIM[mid] = {SMILES: best reference similarity};
  (b) one new final code cell after the FACTORIAL_REPORT cell: LIB_ISOMER_MARGIN, the rule, the rewrite of
      submission.csv, and a `LIB_ISOMER {...}` line with the three counters. Any exception -> `LIB_ISOMER_FAILED`
      and submission.csv stays as the base wrote it.
All outputs are cleared. Never hand-edit the generated .ipynb; change this script and regenerate.

usage: python build/patch_lib_isomer.py [--margin 0.05] [--src in.ipynb] [--out out.ipynb]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import casmi_post  # noqa: E402
import patch_harness_dump as phd  # noqa: E402

SRC = os.path.join(ROOT, "casmi26-e006-window-40-ice-fallback-true.ipynb")
OUT = os.path.join(ROOT, "casmi26-e006-lib-isomer.ipynb")
FACTORIAL_ANCHOR = "FACTORIAL_REPORT = _factorial_entry(globals())"

SHIP_CELL = r"""# === LIB_ISOMER (patch_lib_isomer.py): library-isomer margin on the final rows. A failure keeps the base submission ===
LIB_ISOMER_MARGIN = @MARGIN@
@FN@
_LI_IK = {}
def _li_ik14(s):
    if s not in _LI_IK:
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
            _m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            _LI_IK[s] = Chem.MolToInchiKey(_m)[:14] if _m is not None else None
        except Exception:
            _LI_IK[s] = None
    return _LI_IK[s]
try:
    import json as _li_json, os as _li_os
    import pandas as _li_pd
    _li_ls = {str(k): v for k, v in LIB_SIM.items()}
    _li_sub = _li_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _li_st = dict(lib_sim_mols=0, lib_isomer_eligible=0, lib_isomer_swapped=0)
    _li_out = []
    for _li_mid, _li_s in zip(_li_sub.molecule_id, _li_sub.smiles):
        _li_rows = str(_li_s).split(';')
        _li_d = _li_ls.get(str(_li_mid)) or {}
        _li_sims = {}
        for _li_smi, _li_v in _li_d.items():
            _li_k = _li_ik14(_li_smi)
            if _li_k is not None and float(_li_v) > _li_sims.get(_li_k, 0.0):
                _li_sims[_li_k] = float(_li_v)
        if _li_d:
            _li_st['lib_sim_mols'] += 1
        _li_new, _li_el, _li_sw = lib_isomer_rows(_li_rows, _li_sims, LIB_ISOMER_MARGIN, _li_ik14)
        assert sorted(_li_new) == sorted(_li_rows) and len(_li_new) == len(_li_rows), _li_mid
        _li_st['lib_isomer_eligible'] += int(_li_el)
        _li_st['lib_isomer_swapped'] += int(_li_sw)
        _li_out.append(';'.join(_li_new))
    _li_sub['smiles'] = _li_out
    _li_sub.to_csv('submission.csv.tmp', index=False)
    _li_os.replace('submission.csv.tmp', 'submission.csv')
    print('LIB_ISOMER', _li_json.dumps(dict(_li_st, margin=LIB_ISOMER_MARGIN)), flush=True)
except Exception as _li_e:
    print('LIB_ISOMER_FAILED', repr(_li_e), flush=True)
"""


def ship_cell(margin: float) -> str:
    fn = inspect.getsource(casmi_post.lib_isomer_rows)
    return (SHIP_CELL.replace("@MARGIN@", repr(float(margin))).replace("@FN@", fn.rstrip("\n")))


def build(src: str = SRC, margin: float = 0.05) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    nb = phd.lib_sim_graft(nb)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    f = phd.find(nb, lambda s: FACTORIAL_ANCHOR in s, "FACTORIAL_REPORT cell")
    nb["cells"].insert(f + 1, phd.code_cell(ship_cell(margin)))
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args(argv)
    nb = build(a.src, a.margin)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
