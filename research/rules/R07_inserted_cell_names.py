ID = "R07"
TAGS = ["g4", "build"]
LESSON = "A cell a patch inserts must not read names the base binds later, and its helper names must not be re-bound by a base cell; a branch the draft never runs needs this static check."
SRC = "20261010-harness-c2: VAL cell inserted before the base imports numpy -> NameError np at 68 s; the draft (VAL_SHARD='') never ran it. 20261010-harness-c1npx v1: SNAP's `pickle as _pk` re-bound by the base's merge self-test -> 'list' object has no attribute 'dump'"
FLAGS = ["rules/fixtures/R07_cell_before_import", "rules/fixtures/R07_helper_rebound"]

from _nbnames import code_cells, rebound, unbound  # noqa: E402


def check(idea):
    out = []
    for nb in sorted((idea / "kaggle_push").glob("*.ipynb")):
        for i, name, ln in unbound(code_cells(nb)):
            out.append(f"kaggle_push/{nb.name}: code cell {i} line {ln} reads `{name}` before any cell binds it")
        for i, name, c in rebound(code_cells(nb)):
            out.append(f"kaggle_push/{nb.name}: harness name `{name}` (cell {i}) is re-bound by base cell {c}; "
                       f"use collision-proof `_hx_` names")
    return out
