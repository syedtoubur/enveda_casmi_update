"""patch_gen1_keep: the anchor applies once, only the C8 cell and one appended cell differ, and the final cell is a
no-op when nothing fires and pins an ICE-agreeing generated BASE #1 when it does."""
import contextlib
import io
import json
import os
import tempfile

import pandas as pd

import patch_gen1_keep as P


def _cells(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def test_only_named_cells_differ():
    assert os.path.exists(P.SRC), P.SRC
    base = json.load(open(P.SRC, encoding="utf-8"))
    a, b = _cells(base), _cells(P.build())
    assert len(b) == len(a) + 1
    changed = [i for i in range(len(a)) if a[i] != b[i]]
    assert len(changed) == 1 and P.SNAP_ANCHOR in a[changed[0]]
    assert b[-1].startswith("POST_SRC = ") and "gen1_checked" in b[-1]


def _run_final(rows, base, ice):
    """-> (submission rows after the final cell, its stdout), run in a temp dir."""
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            pd.DataFrame({"molecule_id": list(rows), "smiles": [";".join(v) for v in rows.values()]}).to_csv(
                "submission.csv", index=False)
            g = {"POST_SRC": open(os.path.join(P.HERE, "casmi_post.py"), encoding="utf-8").read(),
                 "_g1_base": base, "ICE_SCORES": ice, "IS_RERUN": False}
            cell = P.FINAL_CELL.replace("/kaggle/working", d.replace("\\", "/"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(cell, "final", "exec"), g)
            out = pd.read_csv("submission.csv", dtype=str, keep_default_na=False)
            return out.smiles[0], buf.getvalue()
        finally:
            os.chdir(old)


def test_final_cell_noop_without_gen():
    rows = {"m1": ["CCO", "CCN", "CCC"]}
    base = {"m1": (["CCO", "CCN"], ["a", "b"], 0.0, [1.0, 0.5], ["C2H6O", "C2H7N"])}
    row, txt = _run_final(rows, base, {"m1": {"CCO": 0.9, "CCN": 0.1}})
    assert row == "CCO;CCN;CCC"
    assert "gen1_checked 1" in txt and "gen1_kept 0" in txt


def test_final_cell_pins_ice_agreeing_gen_only():
    rows = {"m1": ["CCN", "CCC", "CCO"]}
    base = {"m1": (["CCO", "CCN"], ["a", "b"], 0.0, [1.0, 0.5], ["C2H6O|gen", "C2H7N"])}
    row, txt = _run_final(rows, base, {"m1": {"CCO": 0.9, "CCN": 0.1}})
    assert row == "CCO;CCN;CCC"                                  # g first, deduped, nothing else reordered
    assert "gen1_kept 1" in txt and "gen1_moved 1" in txt
    row, txt = _run_final(rows, base, {"m1": {"CCO": 0.1, "CCN": 0.9}})   # ICE disagrees
    assert row == "CCN;CCC;CCO" and "gen1_kept 0" in txt


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
