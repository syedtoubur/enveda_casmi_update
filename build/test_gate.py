"""Run Stage C v2's GATE cell, extracted from the shipped notebook, against real matrices.

The gate only earns its place if it rejects the matrix that actually lost 0.05 on the
leaderboard and accepts the one that scored 0.337. This runs the real cell code (not a
paraphrase) on both.
"""
import json
import sys
from pathlib import Path

import numpy as np

NB = Path(__file__).resolve().parent.parent / "casmi_rank_matrix.ipynb"
SRC = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]
       if c["cell_type"] == "code"]
GATE = next(s for s in SRC if "THE GATE" in s)
# the cell ends by writing the npz; stop before that so the test has no side effects
GATE = GATE[:GATE.index("np.savez_compressed('/kaggle/working/rank_train.npz'")]

CONST = [2, 7, 13, 14, 27, 28, 29, 30]


class CFG:
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03,
               min_samples_leaf=80, l2_regularization=1.0)


def run(name, path, expect):
    z = np.load(path)
    X, Y, M = z["X"], z["Y"].astype(np.int8), z["M"].astype(np.int8)
    if "G" in z:
        G = z["G"].astype(np.int64)
    else:  # v1 shipped no group column -- reconstruct from per-query-constant features
        _, G = np.unique(np.round(X[:, CONST], 6), axis=0, return_inverse=True)
        G = G.astype(np.int64)
    print("=" * 74)
    print(f"{name}   (expect the gate to {expect})")
    print("=" * 74)
    g = {"np": np, "X": X, "Y": Y, "M": M, "G": G, "CFG": CFG,
         "SystemExit": SystemExit, "print": print}
    try:
        exec(compile(GATE, "gate", "exec"), g)
        got = "PASS"
    except SystemExit as e:
        print(f"    SystemExit: {e}")
        got = "FAIL"
    ok = got == expect
    print(f"\n  -> gate said {got}; expected {expect}   {'[ok]' if ok else '[WRONG]'}\n")
    return ok


results = [
    run("v1 matrix (the 0.287 submission)", "rank_train.npz", "FAIL"),
    run("forked matrix (the 0.337 submission)", "rank_train_old.npz", "PASS"),
]
print("=" * 74)
if all(results):
    print("GATE VALIDATED: it rejects the matrix that regressed and accepts the one that worked.")
else:
    print("GATE IS WRONG - do not rely on it.")
    sys.exit(1)
