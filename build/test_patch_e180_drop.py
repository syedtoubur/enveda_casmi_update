"""Unit and integration tests for build/patch_e180_drop.py (e180-drop-dreamsnp-pclite probe).

Validates:
1. Notebook is valid JSON and parses cleanly.
2. Cell count identical to baseline, outputs cleared.
3. Only Cell 34 and Cell 36 are modified.
4. All code cells compile cleanly as Python.
5. Cell 34 contains E180_DROP, DREAMS_FILL, e180 pruning and DreamsNP backfill logic.
6. Cell 36 contains E2R2 decoy guard in e2r2_row and E180_B8 counter emission.
7. e2r2_row logic functions correctly on synthetic test cases (guards against keyed decoys).

usage: python build/test_patch_e180_drop.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_e180_drop as P  # noqa: E402


def ok(cond: bool, msg: str):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")


def code_cells(nb):
    return [P.src_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def compiles(src: str):
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(body)


def test_baseline_and_patch_diff():
    base_nb = json.load(open(P.SRC, encoding="utf-8"))
    patched_nb = P.build(P.SRC, e180_drop=True, dreams_fill=True)

    ok(len(base_nb["cells"]) == len(patched_nb["cells"]), "cell count matches baseline")

    base_codes = code_cells(base_nb)
    patched_codes = code_cells(patched_nb)
    ok(len(base_codes) == len(patched_codes), "code cell count matches baseline")

    diff_indices = [i for i, (a, b) in enumerate(zip(base_codes, patched_codes)) if a != b]
    # Cells 34 and 36 in nb["cells"] correspond to code cell indices
    # Let's verify exactly 2 cells changed
    ok(len(diff_indices) == 2, f"exactly 2 code cells changed, got {diff_indices}")

    # Check output clearing
    for c in patched_nb["cells"]:
        if c["cell_type"] == "code":
            ok(c.get("outputs") == [], "cell outputs must be empty")
            ok(c.get("execution_count") is None, "execution_count must be None")
            compiles(P.src_of(c))

    # Verify Cell 34 code content
    c34_src = patched_codes[diff_indices[0]]
    ok("E180_DROP = True" in c34_src, "Cell 34 has E180_DROP = True")
    ok("DREAMS_FILL = True" in c34_src, "Cell 34 has DREAMS_FILL = True")
    ok("def _get_e180_keys():" in c34_src, "Cell 34 has _get_e180_keys")
    ok("def _get_dreams():" in c34_src, "Cell 34 has _get_dreams")
    ok("e180_drop=E180_DROP" in c34_src, "Cell 34 calls rerank_row with e180_drop=E180_DROP")
    ok("e180_drop=False" in c34_src, "Cell 34 calls rebuilt with e180_drop=False for control")
    ok('"e180_dropped": sum(dropped_mol.values())' in c34_src, "Cell 34 tracks e180_dropped")

    # Verify Cell 36 code content
    c36_src = patched_codes[diff_indices[1]]
    ok("def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO', e180_keys=None):" in c36_src,
       "Cell 36 e2r2_row accepts e180_keys")
    ok("e180_blocked" in c36_src, "Cell 36 tracks e180_blocked")
    ok("E180_B8" in c36_src, "Cell 36 prints E180_B8 counters")
    ok("'e2r2_e180_blocked': e180_blocked" in c36_src or '"e2r2_e180_blocked": e180_blocked' in c36_src,
       "Cell 36 includes e2r2_e180_blocked in E180_B8")


def test_e180_drop_false_variant():
    patched_nb = P.build(P.SRC, e180_drop=False, dreams_fill=False)
    joined = "\n".join(code_cells(patched_nb))
    ok("E180_DROP = False" in joined, "False variant has E180_DROP = False")
    ok("DREAMS_FILL = False" in joined, "False variant has DREAMS_FILL = False")


def test_generated_file_matches():
    # Verify the generated file at OUT
    ok(os.path.isfile(P.OUT), f"Generated file {P.OUT} must exist")
    with open(P.OUT, "r", encoding="utf-8") as f:
        nb = json.load(f)
    ok("cells" in nb and isinstance(nb["cells"], list), "valid notebook JSON structure")
    joined = "\n".join(code_cells(nb))
    ok("E180_DROP = True" in joined, "OUT file contains E180_DROP = True")
    ok("E180_B8" in joined, "OUT file contains E180_B8 counter")


def test_e2r2_row_behavior():
    # Extract and execute e2r2_row from patched cell 36
    patched_nb = P.build(P.SRC, e180_drop=True)
    c36_text = [P.src_of(c) for c in patched_nb["cells"] if "def _e2r2_apply(" in P.src_of(c)][0]

    ns = {}
    # Compile and exec e2r2_make_key and e2r2_row
    code_part = c36_text[:c36_text.index("def _e2r2_apply(")]
    exec(code_part, ns)
    e2r2_row = ns["e2r2_row"]
    e2r2_make_key = ns["e2r2_make_key"]

    # Dummy metric identity function: returns (smi, ())
    def dummy_identity(smi):
        return (smi, ()) if smi else None

    key = e2r2_make_key(dummy_identity)

    # Case 1: normal insertion (no e180 keys)
    toks = ["A", "B", "C"]
    eng = ["X", "Y"]
    res, tag, moved = e2r2_row(toks, eng, key)
    ok(res == ["A", "X", "B", "C"] and tag == "inserted", f"normal insertion: {res}, {tag}")

    # Case 2: top eng candidate "X" is an e180 decoy -> blocked, "Y" gets inserted!
    e180_keys = frozenset(["X"])
    res, tag, moved = e2r2_row(toks, eng, key, e180_keys=e180_keys)
    ok(res == ["A", "Y", "B", "C"] and tag == "inserted", f"e180 guard insertion: {res}, {tag}")

    # Case 3: all eng candidates are e180 decoys -> skipped / no_eng
    e180_keys_all = frozenset(["X", "Y"])
    res, tag, moved = e2r2_row(toks, eng, key, e180_keys=e180_keys_all)
    ok(res == ["A", "B", "C"] and tag == "no_eng", f"all blocked no_eng: {res}, {tag}")


def main():
    print("Running test_patch_e180_drop.py...")
    test_baseline_and_patch_diff()
    print("PASS: test_baseline_and_patch_diff")
    test_e180_drop_false_variant()
    print("PASS: test_e180_drop_false_variant")
    test_generated_file_matches()
    print("PASS: test_generated_file_matches")
    test_e2r2_row_behavior()
    print("PASS: test_e2r2_row_behavior")
    print("ALL TESTS PASSED in test_patch_e180_drop.py!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
