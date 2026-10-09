"""Checks for build/patch_zenith.py (Track 4c class-vote probe generator). Local, no Kaggle needed."""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import tempfile
import time
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_zenith as P  # noqa: E402

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


def gen(mu):
    out = os.path.join(tempfile.mkdtemp(), "v.ipynb")
    P.main(["--classvote", str(mu), "--out", out])
    return json.load(open(out, encoding="utf-8"))


# ---------------------------------------------------------------- notebook delta = the fusion cell only
base = code_cells(json.load(open(P.SRC, encoding="utf-8")))
nb = gen(0.5)
var = code_cells(nb)
ok(len(base) == len(var), "same number of code cells")
diff = [i for i, (a, b) in enumerate(zip(base, var)) if a != b]
ok(len(diff) == 1 and "def fuse2" in base[diff[0]], f"only the fusion cell changes: {diff}")
fuse = var[diff[0]]
compiles(fuse)
ok(fuse.count("fsm, fk, fsc = cv_hook(mid, fsm, fk, fsc)") == 1, "hook inserted once")
ok(fuse.index("fsm, fk, fsc = cv_hook") > fuse.index("fsm, fk, fsc = fuse2(vs"), "hook right after fuse2")
ok(fuse.index("fsm, fk, fsc = cv_hook") < fuse.index("gl_fuse.rerank_multi"), "hook before the GLACIER rerank")
ok("CV_MU, CV_K" in fuse and "= 0.5, 10," in fuse, "mu written")
ok(fuse.count("fs.to_csv('submission.csv', index=False)") == 1 and "TRACK4c dry-reach" in fuse, "report added")
ok(all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code"), "outputs cleared")
ok("CELL19_ICE = False" in fuse and "BASE_ICE = False" in "".join(var), "zenith switches untouched")

# drift: the embedded deltas are specs_scratch's NP_DELTAS
emb = ast.literal_eval(re.search(r"CV_DELTAS = (\{.*?\})\n", fuse).group(1))
ok(emb == P.np_deltas() and len(emb) == 13 and abs(emb["C6H10O5 hexose"] - 162.052824) < 1e-9,
   "embedded NP_DELTAS == scratch_pkg analog.NP_DELTAS")

# ---------------------------------------------------------------- pure vote / term / apply
ns = dict(np=np)
exec(P.VOTE_FUNCS, ns)
lab = {"a": "Flavonoids", "b": "Flavonoids", "c": "Coumarins", "d": None, "e": "Flavonoids|Coumarins"}
share, win, n = ns["cv_vote"]({"a": 0.9, "b": 0.8, "c": 0.6, "d": 0.95, "x": 0.4}, lab.get)
ok(n == 3 and abs(win - 1.7 / 2.3) < 1e-9 and abs(share["Coumarins"] - 0.6 / 2.3) < 1e-9,
   f"weighted vote, unlabelled abstains, below min_sim ignored: {share} {win} {n}")
ok(ns["cv_vote"]({"d": 0.9}, lab.get) == ({}, 0.0, 0), "no labelled voter -> no vote")
s2, _, _ = ns["cv_vote"]({"e": 0.8}, lab.get)
ok(abs(s2["Flavonoids"] - 0.5) < 1e-9 and abs(s2["Coumarins"] - 0.5) < 1e-9, "multi-valued voter label splits its weight")
t = ns["cv_term"](["Flavonoids", "Coumarins", "Alkaloids", None, "Alkaloids|Flavonoids"], share, win)
ok(abs(t[0] - win) < 1e-9 and abs(t[1] - share["Coumarins"]) < 1e-9, "agree -> +share")
ok(abs(t[2] + win) < 1e-9, "labelled with an unvoted class -> -winner share")
ok(t[3] == 0.0, "unlabelled -> 0")
ok(abs(t[4] - win) < 1e-9, "multi-valued candidate takes its best voted part")
sm, km, fm, o = ns["cv_apply"](["A", "B", "C"], ["ka", "kb", "kc"], [0.30, 0.29, 0.10], [-0.8, 0.8, 0.0], 1.0)
ok(sm[0] == "B" and km[0] == "kb" and o == [1, 0, 2], f"mu 1.0 overturns a close top-1: {sm} {o}")
sm, _, _, o = ns["cv_apply"](["A", "B", "C"], ["ka", "kb", "kc"], [0.30, 0.29, 0.10], [-0.8, 0.8, 0.0], 0.0)
ok(o == [0, 1, 2] and sm == ["A", "B", "C"], "mu 0 keeps the order")

# ---------------------------------------------------------------- the block end to end on a toy library
d = tempfile.mkdtemp()
FLAV, COUM, ALK = "O=c1cc(-c2ccccc2)oc2ccccc12", "O=c1ccc2ccccc2o1", "CN1CCC[C@H]1c1cccnc1"
npq = os.path.join(d, "natural_products.parquet")
pd.DataFrame(dict(scorer_key=["KFLAV", "KCOUM", "KALK"], smiles=[FLAV, COUM, ALK],
                  np_superclass=["Flavonoids", "Coumarins", "Pyridine alkaloids"])).to_parquet(npq)

# library: 4 spectra; sid 0,1 flavonoid analogs (one at +hexose), sid 2 coumarin, spectrum 3 has no structure
M = 222.0681
lib = types.SimpleNamespace(
    sid=np.array([0, 1, 2, -1]), mode=np.array([1, 1, 1, 1]),
    smass=np.array([M, M + 162.052824, M, M]),
    struct_smiles=np.array(["Oc1ccc(-c2cc(=O)c3ccccc3o2)cc1", FLAV, COUM], dtype=object),
    struct_key=np.array(["KX", "KFLAV", "KCOUM"], dtype=object),
    off=np.zeros(5, np.int64), mz=np.zeros(0), it=np.zeros(0))
lib.window = lambda target, ppm: np.where(np.abs(lib.smass - target) <= target * ppm * 1e-6)[0]
lib.prep = lambda mz, it, cfg: (np.asarray(mz), np.asarray(it))
SIMS = {0: 0.9, 1: 0.8, 2: 0.55, 3: 0.99}
fake_sp = types.ModuleType("casmi.spectra")
fake_sp.search_shift = lambda qm, qp, cand, shift, *a: np.array([SIMS[int(c)] for c in cand], np.float32)
fake_lib = types.ModuleType("casmi.library")
fake_lib.SearchCfg = lambda: types.SimpleNamespace(tol=0.01, ppm_tol=20.0, floor=0.002, topk=256, power=1.0, ent_weight=True)
fake_casmi = types.ModuleType("casmi")
fake_casmi.spectra, fake_casmi.library = fake_sp, fake_lib
sys.modules.update({"casmi": fake_casmi, "casmi.spectra": fake_sp, "casmi.library": fake_lib})
te = pd.DataFrame(dict(molecule_id=["m1", "m2"], nm=[M * (1 + 1.45e-6)] * 2, ionization_mode=["positive"] * 2,
                       ms2_mzs=[[100.0, 150.0]] * 2, ms2_normalized_intensities=[[1.0, 0.5]] * 2))


def run_block(mu, rerun=True, find=lambda pat: npq, base=None):
    src = P.CV_BLOCK.replace("@MU@", repr(mu)).replace("@DELTAS@", repr(P.np_deltas())).replace("@VOTE_FUNCS@", P.VOTE_FUNCS)
    g = dict(np=np, pd=pd, time=time, json=json, L=lib, te=te, find=find, IS_RERUN=rerun, LIB_TAU=0.9,
             BASE=base or {"m1": ([], [], 0.3), "m2": ([], [], 0.95)})
    exec(src, g)
    return g


fsm, fk, fsc = [ALK, FLAV, COUM], ["KALK", "KFLAV", "KCOUM"], [0.30, 0.29, 0.10]
g = run_block(1.0)
s2, k2, f2 = g["cv_hook"]("m1", fsm, fk, fsc)
st = g["CV_STATS"]
ok(s2[0] == FLAV and k2[0] == "KFLAV" and len(s2) == 3, f"confident flavonoid vote lifts the flavone to rank 1: {k2}")
ok(st["fired"] == 1 and st["top1_changed"] == 1 and st["searched"] == 1, f"counters: {st}")
ok(st["labelled_voters"] == 3 and st["voters"] == 3, f"structure-less spectrum excluded, 3 labelled voters: {st}")
s3, _, _ = g["cv_hook"]("m2", fsm, fk, fsc)
ok(s3 == fsm and g["CV_STATS"]["gated_libmax"] == 1, "library match (lib_max >= 0.9) untouched")
ok(fsm == [ALK, FLAV, COUM], "input lists not mutated")

g0 = run_block(0.0)
ok(g0["cv_hook"]("m1", fsm, fk, fsc)[0] == fsm and g0["CV_STATS"]["fired"] == 0, "mu 0 -> zenith's order")

gd = run_block(1.0, rerun=False)
gd["cv_hook"]("m2", fsm, fk, fsc)
ok(gd["CV_DRY"]["fired"] == 1 and gd["CV_DRY"]["top1_changed"] == 1 and gd["CV_STATS"]["fired"] == 0,
   f"commit mode: dry reach counts the gated molecule without applying: {gd['CV_DRY']}")

SIMS.update({1: 0.4, 2: 0.4})                       # one labelled voter left (sid 0) -> below CV_MIN_VOTERS
gs = run_block(1.0)
ok(gs["cv_hook"]("m1", fsm, fk, fsc)[0] == fsm and gs["CV_STATS"]["fired"] == 0 and gs["CV_STATS"]["voted"] == 1,
   f"a single voter does not fire: {gs['CV_STATS']}")
SIMS.update({0: 0.6, 1: 0.4, 2: 0.9})               # flavonoid 0.6 vs coumarin 0.9: 2 voters, winner share 0.6
gs = run_block(1.0)
ok(gs["cv_hook"]("m1", fsm, fk, fsc)[0] == [FLAV, ALK, COUM],
   "coumarin-led vote: off-class alkaloid demoted, coumarin too far down the RRF to climb")
SIMS.update({0: 0.9, 1: 0.8, 2: 0.55})

gf = run_block(1.0, find=lambda pat: (_ for _ in ()).throw(FileNotFoundError(pat)))
ok(gf["cv_hook"]("m1", fsm, fk, fsc) == (fsm, fk, fsc), "missing NP table -> printed no-op")

gb = run_block(1.0)
gb["L"] = None                                      # search crashes inside the hook
ok(gb["cv_hook"]("m1", fsm, fk, fsc) == (fsm, fk, fsc) and gb["CV_STATS"]["failed"] == 1, "hook failure -> order kept")

print(f"test_patch_zenith: {N} checks passed")
