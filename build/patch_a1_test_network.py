"""A1 test-set network vote: E006 v1 + one final cell that reorders ranks 2-10 of submission.csv rows by the
similarity of each candidate to the top-3 picks of the molecule's spectral neighbours (modified-cosine network over
the test molecules). POST-HOC probe at tau=0.8 (user decision 2026-10-15; the K2 kill at tau=0.7 stands).

Base: casmi26-e006-window-40-ice-fallback-true.ipynb (read only). One new code cell after the FACTORIAL_REPORT cell;
the functions are embedded from casmi_post via inspect.getsource, so the harness replays the shipped code. A failure
prints NET_VOTE_FAILED and leaves submission.csv as the base wrote it. Never hand-edit the generated .ipynb.

usage: python build/patch_a1_test_network.py [--tau 0.8] [--w 0.10] [--src in.ipynb] [--out out.ipynb]
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
OUT = os.path.join(ROOT, "casmi26-e006-a1-test-network.ipynb")
FACTORIAL_ANCHOR = "FACTORIAL_REPORT = _factorial_entry(globals())"
FUNCS = ("_prep_spec", "_greedy", "_mcos_pair", "test_network", "morgan_fp", "net_vote_rows")

SHIP_CELL = r"""# === NET_VOTE (patch_a1_test_network.py): test-set network vote on ranks 2-10. A failure keeps the base submission ===
from functools import lru_cache
NET_TAU, NET_W, NET_MIN_MATCH, NET_MAX_NBRS = @TAU@, @W@, 4, 5
@FN@
try:
    import json as _nv_json, os as _nv_os
    import numpy as _nv_np, pandas as _nv_pd
    _nv_spec = {}
    for _r in te.itertuples():
        _nv_spec.setdefault(str(_r.molecule_id), []).append((_nv_np.asarray(_r.ms2_mzs), _nv_np.asarray(_r.ms2_normalized_intensities), float(_r.precursor_mz), _r.ionization_mode))
    _nv_net = test_network(_nv_spec, tau=NET_TAU, min_match=NET_MIN_MATCH, max_nbrs=NET_MAX_NBRS)
    _nv_sub = _nv_pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    _nv_rows = {str(m): str(s).split(';') for m, s in zip(_nv_sub.molecule_id, _nv_sub.smiles)}
    _nv_new, _nv_st = net_vote_rows(_nv_rows, _nv_net, NET_W, morgan_fp)
    for _m, _o in _nv_rows.items():
        assert sorted(_nv_new[_m]) == sorted(_o) and len(_nv_new[_m]) == len(_o), _m
        assert _nv_new[_m][:1] == _o[:1] and _nv_new[_m][10:] == _o[10:], _m
    _nv_sub['smiles'] = [';'.join(_nv_new[str(m)]) for m in _nv_sub.molecule_id]
    _nv_sub.to_csv('submission.csv.tmp', index=False)
    _nv_os.replace('submission.csv.tmp', 'submission.csv')
    print('NET_VOTE', _nv_json.dumps(dict(_nv_st, tau=NET_TAU, w=NET_W)), flush=True)
except Exception as _nv_e:
    print('NET_VOTE_FAILED', repr(_nv_e), flush=True)
"""


def ship_cell(tau: float = 0.8, w: float = 0.10) -> str:
    fn = "\n\n".join(inspect.getsource(getattr(casmi_post, n)).rstrip("\n") for n in FUNCS)
    return SHIP_CELL.replace("@TAU@", repr(float(tau))).replace("@W@", repr(float(w))).replace("@FN@", fn)


def build(src: str = SRC, tau: float = 0.8, w: float = 0.10) -> dict:
    nb = json.load(open(src, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    f = phd.find(nb, lambda s: FACTORIAL_ANCHOR in s, "FACTORIAL_REPORT cell")
    nb["cells"].insert(f + 1, phd.code_cell(ship_cell(tau, w)))
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--w", type=float, default=0.10)
    a = ap.parse_args(argv)
    nb = build(a.src, a.tau, a.w)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
