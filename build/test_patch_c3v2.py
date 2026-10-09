"""Tests for build/patch_c3v2.py.  Run: python build/test_patch_c3v2.py

compare.py aligns cells by position, so an inserted cell makes every later cell look modified.
These checks match cells by content instead, then execute the splice block against stubs: the
splice is the only code that can move a retrieved answer, so its behaviour is pinned here.
"""
import ast
import collections
import json
import os
import sys
import tempfile
import textwrap
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import patch_c3v2 as P
from patch_v4n_mutator import src_of

FAIL = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAIL.append(msg)


TMP = tempfile.mkdtemp()
OUT = os.path.join(TMP, "c3v2_test.ipynb")
OUT_N1 = os.path.join(TMP, "c3v2_test_norank1.ipynb")
OUT_NS = os.path.join(TMP, "c3v2_test_nosplice.ipynb")
P.main(["--out", OUT])
P.main(["--out", OUT_N1, "--no-rank1"])
P.main(["--out", OUT_NS, "--no-splice"])
SRC = json.load(open(P.SRC, encoding="utf-8"))
NB = json.load(open(OUT, encoding="utf-8"))


def codes(nb):
    return [src_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def test_structure():
    print("test_structure")
    a, b = codes(SRC), codes(NB)
    check(len(b) == len(a) + 1, "exactly one code cell added (%d -> %d)" % (len(a), len(b)))
    check(len(NB["cells"]) == len(SRC["cells"]) + 2, "plus one markdown cell")
    changed = [s for s in a if s not in b]
    check(len(changed) == 5, "exactly 5 source code cells modified (%d)" % len(changed))
    want = ["TOPN, ICE_LAM, ICE_BUDGET, ICE_PC", "eng_runner.py", "pc_runner.py", "ICE_UNION", "def fuse2"]
    check(all(any(w in s for s in changed) for w in want), "and they are config, engine, PubChem, ICE, fusion")
    check(all(not c.get("outputs") for c in NB["cells"] if c["cell_type"] == "code"), "outputs cleared")
    check("papermill" not in NB.get("metadata", {}), "papermill metadata cleared")
    try:
        P.main(["--src", OUT, "--out", os.path.join(TMP, "again.ipynb")])
        refused = False
    except SystemExit:
        refused = True
    check(refused and not os.path.exists(os.path.join(TMP, "again.ipynb")),
          "re-patching a patched notebook is refused, nothing written")


def test_compiles():
    print("test_compiles")
    bad = []
    for i, s in enumerate(codes(NB)):
        try:
            ast.parse(s)
        except SyntaxError:
            try:        # only then strip IPython magics; a blind strip can cut a line out of a string
                ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith(("!", "%"))))
            except SyntaxError as e:
                bad.append((i, str(e)))
    check(not bad, "every code cell parses %s" % bad[:2])


def test_contents():
    print("test_contents")
    mut = [s for s in codes(NB) if P.MARK in s]
    check(len(mut) == 1, "one mutator cell")
    m = mut[0]
    for f in ("delta_m_mutator.py", "mut_runner.py", "c3_gate.py"):
        check(repr(open(os.path.join(HERE, f), encoding="utf-8").read()) in m,
              "%s embedded byte-identical to build/" % f)
    check("MUT_V2='1'" in m, "runner launched in v2 mode")
    check("'MUT_TAU': '8'" in m, "tau=8 passed (oracle: 0 truths lost at depth 2)")
    eng = [s for s in codes(NB) if "eng_runner.py" in s][0]
    check("c3 diagnostics" in eng and ".c3.json" in eng, "engine exports the anchors")
    pc = [s for s in codes(NB) if "pc_runner.py" in s and "RUNNER" in s][0]
    check("pc_z.npz" in pc, "PubChem exports z")
    ice = [s for s in codes(NB) if "ICE_UNION" in s and "import gc" in s][0]
    check("C3V2 ICE_UNION_MUT" in ice and "MUT = globals().get('MUT') or {}" in ice,
          "generated candidates join ICE input, behind a NameError guard")
    check("1.0, 5200" in ice, "GL budget raised")
    fu = [s for s in codes(NB) if "def fuse2" in s][0]
    check(fu.index(P.SPLICE_MARK) > fu.index("vs = fsm[:25]"), "splice runs after the fused rerank")
    check("if ENG or MUT:" in fu, "fusion runs when only the mutator list exists")
    n1 = [s for s in codes(json.load(open(OUT_N1, encoding="utf-8"))) if P.MARK in s][0]
    check("'C3_ALLOW_RANK1': '0'" in n1, "--no-rank1 sets C3_ALLOW_RANK1=0")
    check("C3_SPLICE_ON = True" in m, "default build splices")
    ns = codes(json.load(open(OUT_NS, encoding="utf-8")))
    check(any("C3_SPLICE_ON = False" in x for x in ns), "--no-splice sets C3_SPLICE_ON = False")
    a = [x for x in codes(NB) if P.MARK not in x]
    b = [x for x in ns if P.MARK not in x]
    check(a == b, "--no-splice: every other cell identical to C1 (same plumbing, ICE union, budgets)")


# ------------------------------------------------------------------ the splice, executed
class _Chem:
    @staticmethod
    def score_key(s):
        return "K_" + s


def _merge_src():
    s = [x for x in codes(NB) if "def merge(base, base_keys" in x][0]
    return s[s.index("def merge"):s.index("rows, stats")]


def run_splice(mut, out_rows, ice, gl, base, eng, splice_on=True):
    g = dict(np=np, pd=pd, sys=sys, collections=collections, MUT_DIR=HERE, chem=_Chem,
             C3_SPLICE_ON=splice_on,
             MUT=mut, out=list(out_rows), ICE_SCORES=ice, gl_of=lambda m: gl.get(str(m), {}),
             BASE=base, ENG=eng,
             v4=pd.DataFrame(out_rows, columns=["molecule_id", "smiles"]))
    exec(_merge_src(), g)
    body = textwrap.dedent(P.SPLICE.replace(P.SPLICE_MARK, ""))
    exec(body, g)
    return dict(g["out"])


RET = ";".join("R%d" % i for i in range(25))


def _mut(tier, n=4):
    return dict(smiles=["G%d" % i for i in range(n)], keys=["K_G%d" % i for i in range(n)],
                formula=["C1|gen"] * n, fz=[10.0 - i for i in range(n)], tier=tier,
                sig=dict(ana_max=0.85, lib_exact=0.2))


def test_splice():
    print("test_splice")
    os.environ["C3_SURE_FRAC"] = "0.5"
    os.environ["C3_FIRE_FRAC"] = "0.9"
    os.environ["C3_DISABLE_FRAC"] = "0.95"
    import importlib
    import c3_gate
    importlib.reload(c3_gate)
    try:
        rows = [("m1", RET), ("m2", RET), ("m3", RET), ("m4", RET)]
        ice = {"m1": {"K_G0": 0.5, "K_R0": 0.3}, "m2": {"K_G0": 0.5, "K_R0": 0.3},
               "m3": {"K_G0": 0.5, "K_R0": 0.3}, "m4": {"K_G0": 0.1, "K_R0": 0.4}}
        base = {"m1": ([], [], 0.3), "m2": ([], [], 0.95), "m3": ([], [], 0.3), "m4": ([], [], 0.3)}
        eng = {"m1": {"keys": ["K_X"]}, "m2": {"keys": ["K_X"]}, "m3": {"keys": ["K_X"]},
               "m4": {"keys": ["K_X"]}}
        mut = {"m1": _mut("sure"), "m2": _mut("sure"), "m3": _mut("likely"), "m4": _mut("likely")}
        res = run_splice(mut, rows, ice, {}, base, eng)
        l1, l2, l3, l4 = (res[m].split(";") for m in ("m1", "m2", "m3", "m4"))
        check(l1[:3] == ["G0", "G1", "G2"] and l1[3] == "R0", "sure + ICE agrees -> ranks 1-3, retrieved top-1 at 4")
        check(l2 == RET.split(";"), "real lib_max >= 0.9 -> untouched")
        check(l3[0] == "R0" and l3[1:4] == ["G0", "G1", "G2"], "likely -> ranks 2-4, rank 1 kept")
        check(l4 == RET.split(";"), "ICE clearly prefers the retrieved top-1 -> untouched")
        check(all(len(x.split(";")) <= 25 and len(set(x.split(";"))) == len(x.split(";"))
                  for x in res.values()), "<= 25 entries, no duplicates")
        eng_agree = dict(eng, m1={"keys": ["K_R0"]})
        res2 = run_splice(mut, rows, ice, {}, base, eng_agree)
        check(res2["m1"].split(";")[0] == "R0", "engines agree on top-1 -> sure demoted, rank 1 kept")
        res3 = run_splice({}, rows, ice, {}, base, eng)
        check(res3 == dict(rows), "MUT empty -> lists byte-identical to 0.415")
        storm = {("s%d" % i): _mut("likely") for i in range(30)}
        srows = [("s%d" % i, RET) for i in range(30)]
        os.environ["C3_DISABLE_FRAC"] = "0.25"
        importlib.reload(c3_gate)
        res4 = run_splice(storm, srows, {}, {}, {k: ([], [], 0.3) for k in storm},
                          {k: {"keys": ["K_X"]} for k in storm})
        check(all(v == RET for v in res4.values()), "fires on 100% of the run -> channel disabled")
        import contextlib
        import io
        os.environ["C3_DISABLE_FRAC"] = "0.95"
        importlib.reload(c3_gate)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res6 = run_splice(mut, rows, ice, {}, base, eng, splice_on=False)
        log = buf.getvalue()
        check(res6 == dict(rows), "--no-splice: fires that would splice leave the lists byte-identical")
        check("C3V2 STAGE 2:" in log and "C3V2 CAP:" in log, "--no-splice: stage-2 and cap counters still print")
        check("SPLICE DISABLED" in log and "would splice 2 molecules" in log,
              "--no-splice: logs what it would have spliced")
        bad = dict(mut)
        res5 = run_splice(bad, rows, "not-a-dict", {}, base, eng)
        check(res5["m2"] == RET, "garbage ICE_SCORES cannot break the run")
    finally:
        for k in ("C3_SURE_FRAC", "C3_FIRE_FRAC", "C3_DISABLE_FRAC"):
            os.environ.pop(k, None)
        importlib.reload(c3_gate)


if __name__ == "__main__":
    for fn in [test_structure, test_compiles, test_contents, test_splice]:
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
