"""Checks for build/patch_adduct_ice.py (adduct-partitioned ICEBERG rerank probe). Local, no Kaggle needed."""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import types

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_adduct_ice as P  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    N += 1


def code_cells(nb):
    return [P.src_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def compiles(src):
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(body)


# ---------------------------------------------------------------- G1: notebook delta = ICE cell + fusion cell
out = os.path.join(tempfile.mkdtemp(), "probe.ipynb")
P.main(["--out", out])
nb = json.load(open(out, encoding="utf-8"))
base = code_cells(json.load(open(P.SRC, encoding="utf-8")))
var = code_cells(nb)
ok(len(base) == len(var), "same number of code cells")
diff = [i for i, (a, b) in enumerate(zip(base, var)) if a != b]
ok(len(diff) == 2 and "ice_fuse.run_ice" in base[diff[0]] and "def fuse2" in base[diff[1]],
   f"only the ICE and fusion cells change: {diff}")
ice_cell, fuse_cell = var[diff[0]], var[diff[1]]
compiles(ice_cell)
compiles(fuse_cell)
ok("BASE_ICE = False" in ice_cell and "CELL19_ICE = False" in fuse_cell, "base switches untouched")
ok("TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True" in "".join(var), "ICE settings untouched")
ok(all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code"), "outputs cleared")
ok(ice_cell.index("ADDUCT_ICE_MIDS = adduct_ice_mids") < ice_cell.index("cands = {m:"), "eligibility before ICE input")
ok(fuse_cell.count("(ice if (CELL19_ICE or _aice) else {})") == 1, "only the single-ICE fused call gets the column")
ok("[(ice if CELL19_ICE else {}), _g]" in fuse_cell, "GLACIER rerank_multi call untouched")
ok(fuse_cell.index("[ADDUCT_ICE] eligible") < fuse_cell.index("V1 inference complete"), "counter report printed")

# ---------------------------------------------------------------- G2b: pure eligibility
ns = {}
exec(P.FUNCS, ns)
te = pd.DataFrame(dict(
    molecule_id=[1, 2, 2, 3, 4, 4, 5, 6],
    adduct=["[M+H]+", "[M-H]-", "[M-H]-", "[M+Na]+", "[M+H]+", "[M-H]-", "[M+NH4]+", "[M+Na]+"]))
mids = ns["adduct_ice_mids"](te, {"[M-H]-", "[M+Na]+"})
ok(mids == {"2", "3", "6"}, f"only non-[M+H]+ covered ids, as str: {mids}")
ok(ns["adduct_ice_mids"](te, set()) == set(), "empty adduct set -> rollback")
ns["ADDUCT_ICE_MIDS"] = mids
on = ns["adduct_ice_on"]
ok(on(2, {"a": 0.5}, {}) and on("3", {"a": 0.5}, {}), "eligible id with live ICE -> on (int or str)")
ok(not on(1, {"a": 0.5}, {}) and not on(4, {"a": 0.5}, {}), "[M+H]+ molecule never on, even with GLACIER empty")
ok(not on(2, {}, {}) and not on(2, {"a": None}, {}), "no live ICE -> off")
ok(not on(2, {"a": 0.5}, {"a": 0.1}), "GLACIER present -> off")

# ---------------------------------------------------------------- base-list rerank loop (exec'd from the cell)
calls = []


def fake_rerank(smis, keys, scs, forms, ice, lam=1.0, top_n=25):
    calls.append(dict(n=len(smis), ice=ice))
    return list(range(len(smis)))[::-1]


ice_fuse = types.SimpleNamespace(rerank=fake_rerank)
elig = ice_cell[ice_cell.index("# === ADDUCT_ICE"):ice_cell.index("try:\n    ICE_PKG = ")]
loop = ice_cell[ice_cell.index("def gl_of(m):"):ice_cell.index("print('ICE rerank stats'")]
te2 = pd.DataFrame(dict(molecule_id=["m1", "m2", "m3"], adduct=["[M+H]+", "[M-H]-", "[M+NH4]+"]))
lst = (["A", "B", "C"], ["ka", "kb", "kc"], 0.1, [0.3, 0.2, 0.1], ["F", "F", "F"])
ICE = {m: {"ka": 0.1, "kb": 0.2, "kc": 0.9} for m in ("m1", "m2", "m3")}
g = dict(te=te2, BASE_ICE=False, BASE={m: lst for m in ("m1", "m2", "m3")}, ICE_SCORES=ICE, GL_SCORES={},
         ICE_PC=False, PC={}, TOPN=60, ICE_LAM=1.0, GL_LAM=1.0, ice_fuse=ice_fuse,
         ice_stats=dict(molecules=0, changed_top25=0, changed_top1=0), gl_stats={})
exec(elig + loop, g)
st = g["ADDUCT_ICE_STATS"]
ok(st["eligible"] == 1 and g["ADDUCT_ICE_MIDS"] == {"m2"}, f"one eligible molecule: {g['ADDUCT_ICE_MIDS']}")
ok(len(calls) == 1 and calls[0]["ice"] is ICE["m2"], f"GLACIER failed: only the [M-H]- molecule is reranked: {calls}")
ok(g["BASE"]["m2"][0] == ["C", "B", "A"] and g["BASE"]["m1"][0] == ["A", "B", "C"], "rerank applied to m2 only")
ok(st["base_reranked"] == 1 and st["base_changed_top25"] == 1, f"base counters: {st}")

calls.clear()
g["ADDUCT_ICE_MIDS"] = set()
g["BASE"] = {m: lst for m in ("m1", "m2", "m3")}
exec(loop, g)
ok(calls == [] and g["BASE"]["m2"][0] == ["A", "B", "C"], "rollback (empty set) -> base behaviour")

# ---------------------------------------------------------------- fusion cell end to end
calls.clear()
d = tempfile.mkdtemp()
pd.DataFrame(dict(molecule_id=["m1", "m2", "m3"], smiles=["CCO;CCN;CCC"] * 3)).to_csv(os.path.join(d, "submission.csv"), index=False)
multi = []
gl_fuse = types.SimpleNamespace(rerank_multi=lambda smis, keys, scs, forms, ices, lams, top_n: multi.append(ices) or list(range(len(smis))))
ENG = {m: dict(smiles=["CCCC", "CCN"], keys=["k4", "kN"]) for m in ("m1", "m2", "m3")}
chem = types.SimpleNamespace(score_key=lambda s: "k" + s)
GLS = {"m1": {"kCCN": 0.3}}
g2 = dict(pd=pd, ENG=ENG, chem=chem, ICE_SCORES=ICE, ice_fuse=ice_fuse, gl_fuse=gl_fuse, ICE_LAM=1.0, GL_LAM=1.0,
          gl_of=lambda m: GLS.get(str(m), {}), ADDUCT_ICE_MIDS={"m2"}, ADDUCT_ICE_STATS=dict(st),
          adduct_ice_on=g["adduct_ice_on"])
g["ADDUCT_ICE_MIDS"] = {"m2"}
cwd = os.getcwd()
os.chdir(d)
try:
    exec(fuse_cell, g2)
finally:
    os.chdir(cwd)
ok(len(calls) == 3, f"fused rerank called for every ENG molecule: {len(calls)}")
ok(calls[0]["ice"] == {} and calls[2]["ice"] == {}, "[M+H]+ / uncovered molecules keep the {} ICE column")
ok(calls[1]["ice"] is ICE["m2"], "eligible molecule gets the real ICE column")
ok(len(multi) == 1 and multi[0][0] == {}, "GLACIER rerank_multi keeps CELL19_ICE behaviour")
ok(g2["ADDUCT_ICE_STATS"]["fused_reranked"] == 1, f"fused counter: {g2['ADDUCT_ICE_STATS']}")
sub = pd.read_csv(os.path.join(d, "submission.csv"))
ok(len(sub) == 3 and sub.molecule_id.is_unique, "submission written")

# guard: ICE cell never defined the helpers -> off, no NameError
calls.clear()
g3 = {k: v for k, v in g2.items() if k not in ("ADDUCT_ICE_MIDS", "ADDUCT_ICE_STATS", "adduct_ice_on")}
pd.DataFrame(dict(molecule_id=["m2"], smiles=["CCO;CCN"])).to_csv(os.path.join(d, "submission.csv"), index=False)
os.chdir(d)
try:
    exec(fuse_cell, g3)
finally:
    os.chdir(cwd)
ok(calls and calls[0]["ice"] == {} and g3["ADDUCT_ICE_STATS"]["fused_reranked"] == 0, "missing ICE-cell helpers -> off")

print(f"test_patch_adduct_ice: {N} checks passed")
