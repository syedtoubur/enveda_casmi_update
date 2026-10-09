"""Checks for build/patch_supernext.py (Track 4 probe generator). Local, no Kaggle needed."""
from __future__ import annotations

import ast
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_supernext as P  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    N += 1


def cells(nb):
    return [P.src_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def compiles(src):
    # drop IPython magics / shell lines, which are not Python
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(body)


def gen(*flags):
    d = tempfile.mkdtemp()
    out = os.path.join(d, "v.ipynb")
    import json
    P.main(list(flags) + ["--out", out])
    return json.load(open(out, encoding="utf-8"))


src_nb = __import__("json").load(open(P.SRC, encoding="utf-8"))
SRC = cells(src_nb)

# -- every variant: all code cells compile, same cell count, outputs cleared, only 3 cells touched
VARIANTS = {(): None, ("--pop-precut",): None, ("--pop-fused",): None, ("--pop-fused", "0.3"): None,
            ("--c1-freeze",): None, ("--mu", "0.3"): None, ("--gen-only",): None,
            ("--pop-precut", "--pop-fused", "--c1-freeze"): None,
            ("--nplike", "0.5"): None, ("--classcons", "0.5"): None, ("--nplike", "0.5", "--classcons", "0.5"): None}
for fl in VARIANTS:
    nb = gen(*fl)
    VARIANTS[fl] = nb
    cs = cells(nb)
    ok(len(cs) == len(SRC), f"{fl}: cell count")
    for s in cs:
        compiles(s)
    ok(all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code"), f"{fl}: outputs cleared")
    changed = sum(a != b for a, b in zip(cs, SRC))
    ok(changed <= 3, f"{fl}: touched {changed} cells (max 3)")
    joined = "\n".join(cs)
    ok("KAGGLE_IS_COMPETITION_RERUN" in joined, f"{fl}: commit/rerun switch intact")
    ok("ENGINE FAILED" in joined, f"{fl}: engine fallback intact")
    ok("pop stats" in joined and "c1 frozen" in joined, f"{fl}: counters present")

J = lambda fl: "\n".join(cells(VARIANTS[fl]))

# -- base variant is behaviour-neutral
b = J(())
ok("TOPN_PRE = TOPN\n" in b, "base: TOPN_PRE == TOPN")
ok("MU_F, C1_FREEZE = 0.0, False" in b, "base: fused prior and freeze off")
ok("POOLPOP_MU = 0.25, 200, 0.15" in b, "base: POOLPOP_MU unchanged")
ok("POOLPOP_MU > 0 and len(scs) > 1" in b, "base: pop block condition unchanged")

# -- flags
ok("TOPN_PRE = max(TOPN, int(POP_UNION))" in J(("--pop-precut",)), "precut: TOPN_PRE widened")
ok("if len(smis) >= TOPN_PRE: break" in J(("--pop-precut",)), "precut: loop cuts at TOPN_PRE")
ok("smis[:TOPN], keys[:TOPN]" in J(("--pop-precut",)), "precut: cut back to TOPN after re-sort")
ok("MU_F, C1_FREEZE = 0.15, False" in J(("--pop-fused",)), "pop-fused default 0.15")
ok("MU_F, C1_FREEZE = 0.3, False" in J(("--pop-fused", "0.3")), "pop-fused explicit value")
ok("MU_F, C1_FREEZE = 0.0, True" in J(("--c1-freeze",)), "c1-freeze on")
ok("POOLPOP_MU = 0.25, 200, 0.3" in J(("--mu", "0.3")), "mu sweep")
g = J(("--gen-only",))
ok("POOLPOP_MU = 0.25, 200, 0.0" in g and "POOLPOP_MU >= 0 and" in g, "gen-only keeps the block at mu 0")
ok("'POOL_POP' not in locals()" not in b, "stale locals() guard replaced")

# -- behaviour of the fusion helpers, executed against stubs
fuse_src = [s for s in cells(VARIANTS[("--pop-fused",)]) if "def fuse2" in s][0]
helpers = fuse_src[fuse_src.index("# === TRACK 4"):fuse_src.index("\nif ENG:")]
ns = dict(np=np, BASE={"m1": ([], [], 0.95), 2: ([], [], 0.1)}, RAW_TOP1={"m1": "CCO"},
          POPK={"A": 0.0, "B": 10.0, "C": 1.0})
exec(helpers, ns)
f = ns["_pop_fuse"]
sm, k, s, moved = f(["a", "b", "c"], ["A", "B", "C"], [0.30, 0.29, 0.28])
ok(moved == 1 and k[0] == "B", "pop-fused promotes a much more popular near-tie")
sm, k, s, moved = f(["a", "b"], ["A", "Z"], [0.30, 0.10])
ok(k == ["A", "Z"], "missing key gets the median, does not crash")
ns["MU_F"] = 0.0
ok(f(["a", "b"], ["A", "B"], [0.3, 0.1])[3] == 0, "MU_F 0 is a no-op")
ok(ns["_LIBMAX"] == {"m1": 0.95, "2": 0.1} and ns["_RAWTOP"] == {"m1": "CCO"}, "str-keyed lookups")


# -- Track 4b NP block: present only with its flag, executed against stubs
ok("TRACK 4b" not in "\n".join(cells(VARIANTS[()])), "no NP block without a flag")
ok("NP_MU_LIKE, NP_MU_CLS = 0.5, 0.0" in "\n".join(cells(VARIANTS[("--nplike", "0.5")])), "nplike mu")
ok("NP_MU_LIKE, NP_MU_CLS = 0.0, 0.5" in "\n".join(cells(VARIANTS[("--classcons", "0.5")])), "classcons mu")
ice_src = [s for s in cells(VARIANTS[("--nplike", "0.5", "--classcons", "0.5")]) if "# === TRACK 4b" in s][0]
np_block = ice_src[ice_src.index("# === TRACK 4b"):ice_src.index("print('ICE rerank stats', ice_stats,")]
import pandas as pd  # noqa: E402
d = tempfile.mkdtemp()
flav = "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"            # apigenin (flavonoid)
syn = "O=C(Nc1ccccc1)c1ccc(S(=O)(=O)N2CCCC2)cc1"           # synthetic sulfonamide
syn2 = "Cc1ccc(S(=O)(=O)Nc2ccc(C(=O)N3CCOCC3)cc2)cc1"
# KBASE is labelled by the table; the flavone candidate (key KX) only via kNN to the table's apigenin row
pd.DataFrame({"scorer_key": ["KAPI", "KBASE"], "smiles": [flav, "CCO"], "np_superclass": ["Flavones", "Flavones"]}).to_parquet(
    os.path.join(d, "natural_products.parquet"))
PCs = {"m1": dict(pc=[syn, syn2, flav], pc_keys=["KSYN", "KSYN2", "KX"], pc_fz=[900.0, 800.0, 700.0], best_pool_fz=850.0),
       "m2": dict(pc=[syn], pc_keys=["KSYN"], pc_fz=[5.0], best_pool_fz=1.0)}
ns = dict(np=np, pd=pd, os=os, sys=sys, PC=PCs, BASE={"m1": (["CCO"], ["KBASE"], 0.1)},
          find=lambda pat: os.path.join(d, pat))
exec(np_block, ns)
st = ns["NP_STATS"]
ok(st["molecules"] == 1, f"single-candidate list skipped: {st}")
# mu 0.5 is a tie-breaker: the flavone climbs from rank 3 to 2 but does not overturn the spectral top-1
ok(PCs["m1"]["pc"][1] == flav and st["reordered"] == 1 and st["top1_changed"] == 0,
   f"NP flavone moves up one place, top-1 kept: {PCs['m1']['pc']}")
ok(PCs["m1"]["pc_fz"][0] == 900.0, "pc_fz[0] (rel gate) unchanged")
ok(PCs["m1"]["pc_keys"][1] == "KX" and len(PCs["m1"]["pc"]) == 3, "keys follow smiles")
ok(st["base_labelled"] == 1 and st["labelled_cands"] == 1 and st["cands"] == 3, f"labels: {st}")
ns2 = dict(ns, PC={"m1": dict(pc=[syn, flav], pc_keys=["A", "B"], pc_fz=[1.0, 0.5], best_pool_fz=0.0)},
           find=lambda pat: (_ for _ in ()).throw(FileNotFoundError(pat)))
ns2["NP_MU_LIKE"] = 0.0
src2 = np_block.replace("NP_MU_LIKE, NP_MU_CLS = 0.5, 0.5", "NP_MU_LIKE, NP_MU_CLS = 0.0, 0.5")
exec(src2, ns2)
ok(ns2["PC"]["m1"]["pc"] == [syn, flav], "missing dataset -> printed no-op, list untouched")

print(f"test_patch_supernext: {N} checks passed")
