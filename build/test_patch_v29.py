"""Checks for build/patch_v29.py (Track 4 v2 probe generator on the 0.411 notebook). Local, no Kaggle needed."""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_v29 as P  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    N += 1


def cells(nb):
    return [P.src_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def compiles(src):
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(body)


def gen(*flags):
    out = os.path.join(tempfile.mkdtemp(), "v.ipynb")
    P.main(list(flags) + ["--out", out])
    return json.load(open(out, encoding="utf-8"))


SRC = cells(json.load(open(P.SRC, encoding="utf-8")))

VARIANTS = {(): None, ("--mu", "0.25"): None, ("--pop-eng", "0.25"): None, ("--np-bonus", "1"): None,
            ("--pop-precut",): None, ("--lib-gate",): None,
            ("--mu", "0.25", "--pop-eng", "0.25", "--np-bonus", "1", "--pop-precut", "--lib-gate"): None}
for fl in VARIANTS:
    nb = VARIANTS[fl] = gen(*fl)
    cs = cells(nb)
    ok(len(cs) == len(SRC), f"{fl}: cell count")
    for s in cs:
        compiles(s)
    ok(all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code"), f"{fl}: outputs cleared")
    ok(sum(a != b for a, b in zip(cs, SRC)) <= 5, f"{fl}: touched > 5 cells")
    j = "\n".join(cs)
    ok("IS_RERUN = True" not in j, f"{fl}: hard-coded IS_RERUN removed")
    ok("IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'" in j,
       f"{fl}: commit/rerun switch restored")
    ok("SMOKE_N = 12" in j and "ICE_BUDGET = 300" in j, f"{fl}: smoke branch intact")
    ok("PUBCHEM CHANNEL FAILED" in j and "poolpop failed" in j, f"{fl}: fallbacks intact")
    ok("TRACK4v2 pop stats" in j and "TRACK4v2 MU_ENG" in j, f"{fl}: counters present")

J = lambda fl: "\n".join(cells(VARIANTS[fl]))

# -- base variant: behaviour-neutral
b = J(())
ok("POOLPOP_MU = 0.15\n" in b, "base: mu unchanged")
ok("MU_ENG, LIB_GATE_F = 0.0, False" in b, "base: engine pop and fused lib gate off")
ok("LIB_GATE = True" not in b and "TOPN_PRE" not in b and "CASMI_NP_BONUS" not in b, "base: no flag code")
ok("if len(smis) >= TOPN: break" in b, "base: loop cut unchanged")
changed_base = [i for i, (x, y) in enumerate(zip(cells(VARIANTS[()]), SRC)) if x != y]
ok(len(changed_base) == 3, f"base touches only the run-mode, BASE and fusion cells: {changed_base}")

# -- flags
ok("POOLPOP_MU = 0.25\n" in J(("--mu", "0.25")), "mu")
ok("MU_ENG, LIB_GATE_F = 0.25, False" in J(("--pop-eng", "0.25")), "pop-eng value")
ok("fuse2(vs, _es, _ek)" in J(("--pop-eng", "0.25")), "pop-eng wired into fuse2")
n = J(("--np-bonus", "1"))
ok("NP_BONUS = 1.0" in n and "find('pool_np.npy')" in n and "(_pnp & 6) > 0" in n, "np bonus in pool prior")
ok("os.environ['CASMI_NP_BONUS'] = '1.0'" in n, "np bonus passed to the PubChem subprocess")
pc = J(("--pop-precut",))
ok("if len(smis) >= TOPN_PRE: break" in pc and "smis[:TOPN], keys[:TOPN]" in pc, "precut")
lg = J(("--lib-gate",))
ok("not (LIB_GATE and lib_max >= LIB_TAU)" in lg and "MU_ENG, LIB_GATE_F = 0.0, True" in lg, "lib gate in C8 and C10")

# -- the V17_PATCH replacements apply exactly once to the real 0.411 string, and the result compiles
c4 = [s for s in SRC if s.startswith("V17_PATCH = ")][0]
v17 = ast.literal_eval(c4.splitlines()[0][len("V17_PATCH = "):])
ok(v17.count(P._V17_OLD1) == 1 and v17.count(P._V17_OLD2) == 1, "V17 anchors present once")
v17b = v17.replace(P._V17_OLD1, P._V17_NEW1).replace(P._V17_OLD2, P._V17_NEW2)
ast.parse(v17b)
ok("_W['npf']" in v17b and "& 6) > 0" in v17b, "V17 np bonus code")
# the generated C4 block itself performs the replacement at run time
blk = [s for s in cells(VARIANTS[("--np-bonus", "1")]) if s.startswith("V17_PATCH = ")][0]
ns = {"os": os, "V17_PATCH": v17}
code = blk[blk.index("# === TRACK 4 v2"):blk.index("# PubChem-only channel in its own process")]
exec(code, ns)
ok(ns["V17_PATCH"] == v17b, "C4 block rewrites V17_PATCH as intended")

# -- engine re-sort helper, executed against stubs
fuse_src = [s for s in cells(VARIANTS[("--pop-eng", "0.25")]) if "def fuse2" in s][0]
helpers = fuse_src[fuse_src.index("# === TRACK 4 v2"):fuse_src.index("\ndef _form2")]


class _Pool:
    key = np.array(["A", "B", "C"], dtype=object)


ns = dict(np=np, BASE={"m1": ([], [], 0.95), 2: ([], [], 0.1)}, P=_Pool(), POOL_POP=np.array([0.0, 12.0, 1.0]))
exec(helpers, ns)
f = ns["_pop_eng"]
es, ek = f(["a", "b", "c"], ["A", "B", "C"])
ok(ek[0] == "B", "a far more popular candidate at rank 2 moves to rank 1")
es, ek = f(["a", "z"], ["A", "Z"])
ok(ek == ["A", "Z"], "missing key gets the median, does not crash or reorder ties")
ok(f(["a"], ["A"]) == (["a"], ["A"]), "single-item list untouched")
ok(ns["_LIBMAX"] == {"m1": 0.95, "2": 0.1}, "str-keyed lib_max lookup")
ok(ns["N_ENGPOP_HIT"] == 4 and ns["N_ENGPOP_ALL"] == 5, "key-hit counters")
ns["_KPOP"] = None
ok(f(["a", "b"], ["A", "B"]) == (["a", "b"], ["A", "B"]), "MU_ENG 0 / no map is a no-op")

# ================================================================ 0.415 source + P3e (--np-expand)
SRC415 = os.path.join(P.ROOT, "casmi26-v29-fusion-pop-glmh-0-415.ipynb")
if os.path.exists(SRC415):
    import pandas as pd
    S4 = cells(json.load(open(SRC415, encoding="utf-8")))
    V4 = {fl: gen("--src", SRC415, *fl) for fl in [(), ("--np-expand",), ("--np-bonus", "1")]}
    for fl, nb in V4.items():
        cs = cells(nb)
        ok(len(cs) == len(S4), f"0.415 {fl}: cell count")
        for s in cs:
            compiles(s)
        ok(all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code"), f"0.415 {fl}: outputs cleared")
        rm = [s for s in cs if "IS_RERUN =" in s][0]
        ok(rm == [s for s in S4 if "IS_RERUN =" in s][0], f"0.415 {fl}: run-mode cell untouched")
    d = lambda fl: [i for i, (x, y) in enumerate(zip(cells(V4[fl]), S4)) if x != y]
    eng_i = [i for i, s in enumerate(S4) if "ENG_FILES = {" in s][0]
    fus_i = [i for i, s in enumerate(S4) if "def fuse2" in s][0]
    base_d = d(())
    ok(eng_i not in base_d, "0.415 base: engine cell untouched")
    ok(set(d(("--np-expand",))) == set(base_d) | {eng_i}, f"0.415 npx touches only the engine cell beyond base: {d(('--np-expand',))}")
    ok(fus_i in base_d, "fusion cell carries the counters")
    nx = cells(V4[("--np-expand",)])
    ok("_npx_count(e['keys']" in nx[fus_i] and "TRACK4 NPX keys" in nx[fus_i], "npx counters wired")
    ok("_npx_count" not in "\n".join(cells(V4[()])) and "natural_products" not in "\n".join(cells(V4[()])), "base: no npx code")

    # the engine-cell block rewrites the real casmi_engine.py string exactly once, and the result compiles
    line = [l for l in S4[eng_i].splitlines() if l.startswith("ENG_FILES = {")][0]
    ef = ast.literal_eval(line.split("=", 1)[1].strip())
    blk = nx[eng_i][nx[eng_i].index("# === TRACK 4 P3e"):nx[eng_i].index("for _n, _src in ENG_FILES.items():")]
    ns = {"ENG_FILES": dict(ef)}
    exec(blk, ns)
    ce = ns["ENG_FILES"]["casmi_engine.py"]
    ast.parse(ce)
    ok(ce == ef["casmi_engine.py"].replace(P._NPX_OLD, P._NPX_NEW) and "TRACK4 NP expansion" in ce, "engine source patched once")

    # the inserted build_pool lines, executed against a stub table
    tmp = tempfile.mkdtemp()
    pd.DataFrame({"ik14": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"],
                  "scorer_key": ["KA", "KB", "KC", "KD", "KE", "KE", "KG"],
                  "smiles": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
                  "mono_mass": [300.0, 300.0, 100.0, 300.0, 300.0, 300.0, 300.0],
                  "in_casmi_pool": [False, True, False, False, False, False, False]}).to_parquet(os.path.join(tmp, "natural_products.parquet"))
    logs = []
    snippet = "\n".join(l[4:] for l in P._NPX_NEW.splitlines()[1:])
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        ns = dict(pd=pd, os=os, np=np, log=logs.append, find=lambda n: os.path.join(tmp, n),
                  co_keys=np.array(["KD"], dtype=object), tr=pd.DataFrame({"ik": ["GGGG"], "smi": ["X"]}))
        exec(snippet, ns)
        new = list(ns["tr"].ik[1:])
        keys_file = open(os.path.join(tmp, "npx_keys.txt")).read().split("\n")
    finally:
        os.chdir(cwd)
    ok(new == ["KA", "KE"], f"npx: pool rows, <150 Da, coco keys, source-ik14 hits and duplicates dropped: {new}")
    ok(keys_file == ["KA", "KE"] and logs == ["TRACK4 NP expansion +2 structures"], f"npx: keys file + log {keys_file} {logs}")
    ns = dict(pd=pd, os=os, log=logs.append, find=lambda n: (_ for _ in ()).throw(FileNotFoundError(n)),
              co_keys=np.array([], dtype=object), tr=pd.DataFrame({"ik": ["A"], "smi": ["X"]}))
    exec(snippet, ns)
    ok(len(ns["tr"]) == 1 and "failed" in logs[-1], "npx: missing table falls back to the base pool")

    # the fusion counters, executed against stubs
    h = nx[fus_i][nx[fus_i].index("# === TRACK 4 P3e"):nx[fus_i].index("def _npx_count")]
    f_src = nx[fus_i][nx[fus_i].index("def _npx_count"):]
    f_src = f_src[:f_src.index("# === TRACK 4 v2")]
    open(os.path.join(tmp, "npx_keys.txt"), "w").write("KA\nKE")
    ns = dict(os=os, ENG_DIR=tmp)
    exec(h + f_src, ns)
    ns["_npx_count"](["KA", "Q", "Q"], ["Z", "KA", "KE"])
    ns["_npx_count"](["Q"], ["Q"])
    ok((ns["N_NPX_ENG"], ns["N_NPX_TOP"], ns["N_NPX_R1"], ns["N_ENG_DUP"]) == (1, 1, 0, 1) and ns["NPX_BOUND"] == 0.5,
       "npx counters and bound")

print(f"test_patch_v29: {N} checks passed")
