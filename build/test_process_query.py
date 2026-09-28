"""Exercise the Colab notebook's process_query() on synthetic data, with the REAL rank_features.

The invariant that matters: a query's M=0 and M=1 halves must be aligned candidate-for-
candidate, differing ONLY in the library and analog columns. In the forked matrix that shows up
as columns 14-24 being byte-identical between the two halves -- window size, the model's f.z
features and the MetFrag features are computed once and reused. If that breaks, the paired
design is gone and the ranker loses the controlled contrast it is supposed to learn from.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NB = Path(__file__).resolve().parent.parent / "casmi_rank_matrix_colab.ipynb"
SRC = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]
       if c["cell_type"] == "code"]
PQ = next(s for s in SRC if "def process_query" in s)
RF = next(s for s in SRC if "def rank_features" in s)

NBITS, NPOOL = 64, 3000
rng0 = np.random.default_rng(0)


class CFGs:
    PPM_CENTER, PPM_WIN, PPM_FALLBACK = 1.45, 8.5, 30.0
    N_ANALOG, SIM_POWER = 100, 4.0


class SCs:
    MAX_ROWS_PER_QUERY = 400


class Pool:
    def __init__(s):
        s.mass = np.sort(rng0.uniform(150, 1100, NPOOL))
        s.keys = np.array([f"K{i:013d}"[:14] for i in range(NPOOL)], dtype=object)
        s.smiles = np.array(["CCO"] * NPOOL, dtype=object)
        s._fp = (rng0.random((NPOOL, NBITS)) < 0.2).astype(np.uint8)
        s.k2i = {k: i for i, k in enumerate(s.keys)}

    def window(s, t, ppm):
        a = np.searchsorted(s.mass, t * (1 - ppm / 1e6), 'left')
        b = np.searchsorted(s.mass, t * (1 + ppm / 1e6), 'right')
        return np.arange(a, b)

    def fps(s, idx):
        return s._fp[np.asarray(idx)]


pool = Pool()
# put a query mass squarely inside a populated region and widen it so the window is non-trivial
TARGET = float(pool.mass[1500])
CFGs.PPM_WIN = 3000.0          # synthetic masses are uniform, so widen to get ~a real window
TRUTH = pool.keys[1500]

PK = {0: (np.sort(rng0.uniform(50, TARGET, 40)).astype(np.float32),
          rng0.random(40).astype(np.float32)),
      1: (np.sort(rng0.uniform(50, TARGET, 55)).astype(np.float32),
          rng0.random(55).astype(np.float32))}

g = {"np": np, "pd": pd, "time": __import__("time"), "print": print,
     "CFG": CFGs, "SC": SCs, "pool": pool, "NFEAT": 31, "rng": np.random.default_rng(1),
     "PK": PK,
     "_PREC": np.array([TARGET + 1.007, TARGET + 1.007]),
     "_AD": np.array(["[M+H]+", "[M+H]+"], dtype=object),
     "_ION": np.array(["positive", "positive"], dtype=object),
     "_INS": np.array(["timsTOF", "timsTOF"], dtype=object),
     "_CE": np.array([np.array([20.0]), np.array([35.0])], dtype=object),
     "neutral_mass": lambda mz, ad: np.asarray(mz, float) - 1.00728,
     "L": None, "rep": None, "rep_key": None, "rep_nm": None,
     "model_logits": lambda sub: rng0.standard_normal(NBITS).astype(np.float32),
     "frag_scores_fast": lambda smi, specs, mode: rng0.random(len(smi)).astype(np.float32),
     }
exec(compile(RF[RF.index("N_ANALOG = CFG.N_ANALOG"):], "rf", "exec"), g)

CALLS = {"lib": 0, "analog": 0}


def lib_sim_ban(L, specs, target, ban_rows=(), ban_keys=()):
    CALLS["lib"] += 1
    # library visible -> the truth is findable; library banned -> it is not
    out = {pool.keys[i]: float(rng0.random() * 0.3) for i in range(1480, 1520)}
    if not len(ban_keys):
        out[TRUTH] = 0.91
    else:
        out.pop(TRUTH, None)
    return out


def analog_sim_ban(L, specs, target, rep, rep_key, rep_nm, ban_rows=(), ban_keys=()):
    CALLS["analog"] += 1
    ks = [pool.keys[i] for i in range(1490, 1510) if pool.keys[i] not in set(ban_keys)]
    return [(k, float(v)) for k, v in zip(ks, np.sort(rng0.random(len(ks)))[::-1])]


g["lib_sim_ban"] = lib_sim_ban
g["analog_sim_ban"] = analog_sim_ban
exec(compile(PQ, "pq", "exec"), g)

rec = dict(ik=TRUTH, regime="unseen", rows=[0, 1], ban_vis=[0, 1], ban_all=[0, 1, 2, 3])
timing = {}
out = g["process_query"](rec, 42, timing)
assert out is not None, "process_query returned None on a well-formed query"
X, Y, M, G = out
n = len(X) // 2
print(f"[ok] process_query -> X{X.shape}  ({n} candidates x 2 ban regimes)")

assert X.shape[1] == 31, X.shape
assert set(np.unique(M)) == {0, 1}, f"both ban regimes must be emitted, got {np.unique(M)}"
assert (M[:n] == 0).all() and (M[n:] == 1).all(), "regimes must be contiguous halves"
assert (G == 42).all(), "group id must be the query index"
print(f"[ok] both regimes present, contiguous, G tagged")

assert Y[:n].sum() == 1 and Y[n:].sum() == 1, "each half must carry exactly one positive"
assert (Y[:n] == Y[n:]).all(), "the two halves must be aligned candidate-for-candidate"
print(f"[ok] one positive per half, label vectors aligned")

# THE INVARIANT: window size + model + MetFrag columns identical between halves
SHARED = list(range(14, 25))
same = [c for c in range(31) if np.allclose(X[:n, c], X[n:, c], atol=0, rtol=0)]
missing = [c for c in SHARED if c not in same]
assert not missing, (
    f"columns {missing} differ between the M=0 and M=1 halves - the expensive work is being "
    "recomputed per regime instead of shared, so the paired design is broken")
print(f"[ok] columns 14-24 byte-identical between halves -> {sorted(same)}")
print("     (matches the forked matrix, where cols 14-24 are identical across its two halves)")

lib_cols = [0, 1, 2, 3, 4]
assert not np.allclose(X[:n, 0], X[n:, 0]), "the library column must DIFFER between regimes"
print(f"[ok] library columns differ: M=0 truth lv={X[:n][Y[:n]==1, 0][0]:.3f}  "
      f"M=1 truth lv={X[n:][Y[n:]==1, 0][0]:.3f}")

assert CALLS == {"lib": 2, "analog": 2}, f"expected 2 library + 2 analog calls, got {CALLS}"
print(f"[ok] exactly 2 library + 2 analog calls; window/MetFrag/FPNet ran once")
assert set(timing) >= {"window", "metfrag", "fpnet", "library", "analog", "features"}
print(f"[ok] timing breakdown populated: {sorted(timing)}")

print("\nPROCESS_QUERY OK: paired two-regime design verified end to end.")
