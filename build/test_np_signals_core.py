"""Checks for build/np_signals_core.py (Track 4b NP signals). Local, no Colab needed."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import np_signals_core as C  # noqa: E402

N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    N += 1


# ---- expected_rank_metrics: ties are broken uniformly at random
rr, hit = C.expected_rank_metrics([3.0, 2.0, 1.0], 0)
ok(rr == 1.0 and hit[1] == 1.0, "unique top-1")
rr, hit = C.expected_rank_metrics([1.0, 1.0, 1.0, 1.0], 2)
ok(abs(rr - np.mean([1, 1 / 2, 1 / 3, 1 / 4])) < 1e-12 and hit[1] == 0.25, "4-way tie averages ranks 1..4")
rr, hit = C.expected_rank_metrics([5.0] * 30 + [1.0], 30)
ok(rr == 0.0 and hit[25] == 0.0, "truth below 30 others is outside @25")
rr, hit = C.expected_rank_metrics([5.0] * 24 + [1.0, 1.0], 24)
ok(abs(rr - 0.5 * (1 / 25)) < 1e-12 and hit[25] == 0.5, "tie straddling the @25 cut")

# ---- vendor sources: category column required, ids mapped from display names
st = pd.DataFrame({"Source Name": ["Enamine", "ChEBI", "MolPort"],
                   "Source Category": ["Chemical Vendors", "Curation Efforts", "Chemical Vendors; Legacy"]})
sn = pd.DataFrame({0: ["12345", "ChEBI", "MolPort"], 1: ["Enamine", "ChEBI", "MolPort"]})
v = C.vendor_source_ids(st, sn)
ok({"Enamine", "MolPort", "12345"} <= v and "ChEBI" not in v, f"vendor set {v}")
try:
    C.vendor_source_ids(pd.DataFrame({"Source Name": ["x"]}))
    ok(False, "missing category column must raise")
except KeyError:
    ok(True, "")

# ---- accumulate + vendor_only
nonv = np.zeros(10, bool); anys = np.zeros(10, bool)
C.accumulate_sid_map(np.array(["12345", "12345", "ChEBI", "12345", "ChEBI"], object),
                     np.array([1, 2, 2, np.nan, 99]), v, nonv, anys)
ok(anys[1] and anys[2] and not anys[3], "anysid set from CIDs present")
ok(not nonv[1] and nonv[2], "CID 2 has a non-vendor SID")
vo = C.vendor_only_flags([1, 2, 3, 1, -1], [0, 0, 0, 1.2, 0], nonv, anys)
ok(list(vo) == [1, 0, 0, 0, 0], f"vendor_only {list(vo)}: vendor-only CID with no PMID only")

# ---- NP-likeness orders NP-like above a synthetic sulfonamide
s = C.np_scores(["CC1=CC(=O)C2=C(O1)C=C(O)C=C2O", "O=C(Nc1ccccc1)c1ccc(S(=O)(=O)N2CCCC2)cc1", "not_a_smiles"])
ok(s[0] > 0 > s[1] and np.isnan(s[2]), f"np scores {s}")
z = C.zscore_window(s)
ok(z[2] == 0 and z[0] > 0 > z[1], "zscore: NaN neutral")
ok((C.zscore_window([1.0, 1.0]) == 0).all(), "constant window -> 0")

# ---- class consistency
cc = C.class_consistency(["Flavonoids", None, "Coumarins"], ["Flavonoids", "Coumarins", None], [0.75, 0.25, 0.9])
ok(abs(cc[0] - 0.75) < 1e-12 and cc[1] == 0 and abs(cc[2] - 0.25) < 1e-12, f"class cons {cc}")
ok((C.class_consistency(["Flavonoids"], [None], [0.9]) == 0).all(), "no labelled analog -> abstain")
bank = [C.morgan_fp(x) for x in ["c1ccccc1O", "CCCCCCCC", "c1ccccc1OC"]]
lab, sim = C.knn_label(C.morgan_fp("c1ccccc1O"), bank, ["phenols", "alkanes", "anisoles"])
ok(lab == "phenols" and sim == 1.0, "knn exact")
lab, _ = C.knn_label(C.morgan_fp("CC(=O)Nc1ccc(O)cc1"), bank, ["phenols", "alkanes", "anisoles"], thr=0.95)
ok(lab is None, "knn below threshold -> None")
idx, sims = C.nearest_analogs(C.morgan_fp("c1ccccc1O"), bank, k=2, exclude={0})
ok(0 not in idx and len(idx) >= 1, "nearest analogs exclude self")

# ---- formula helpers and tier slicing on a toy memmap layout
ok(C.canon_formula("C6H12O6") == C.canon_formula("H12C6O6") == "C6H12O6", "canon formula")
smis = ["CCO", "COC", "CCCO"]
mass = np.array([46.0419, 46.0419, 60.0575])
blob = "".join(smis).encode()
off = np.cumsum([0] + [len(x) for x in smis])
buf = np.frombuffer(blob, np.uint8)
a, b, got = C.tier_slice(mass, off, buf, 46.0419, 5)
ok((a, b, got) == (0, 2, ["CCO", "COC"]), f"tier slice {a, b, got}")
ok(list(C.same_formula(got + ["CCCO"], "C2H6O")) == [True, True, False], "same formula")
ok(list(C.dedupe_first(["k1", "k2", "k1", "k3"])) == [0, 1, 3], "dedupe first")

# ---- eval_query on a toy tier: an NP truth among synthetic same-formula decoys
# C9H8O4: caffeic acid (truth, NP) + three same-formula synthetic-looking isomers + one other-formula row
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Descriptors import ExactMolWt  # noqa: E402

tsmi = ["OC(=O)C=Cc1ccc(O)c(O)c1", "COC(=O)c1ccc(O)cc1C=O", "CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccc(OCC=O)cc1",
        "CCCCCCCCCC"]
tik = [Chem.MolToInchiKey(Chem.MolFromSmiles(x))[:14] for x in tsmi]
tm = np.array([ExactMolWt(Chem.MolFromSmiles(x)) for x in tsmi])
o_ = np.argsort(tm, kind="stable")
tsmi = [tsmi[i] for i in o_]; tik = [tik[i] for i in o_]; tm = tm[o_]
blob = "".join(tsmi).encode()
C.G.update(mass=tm, off=np.cumsum([0] + [len(x) for x in tsmi]), buf=np.frombuffer(blob, np.uint8),
           ik14=np.array([k.encode() for k in tik], "S14"), ppm=5.0,
           lsid=np.ones(len(tsmi), np.float16), lpmid=np.zeros(len(tsmi), np.float16),
           vendor=np.array([0 if "C=Cc1ccc(O)c(O)c1" in x else 1 for x in tsmi], np.uint8),
           label_of={}, lotus_fps=[C.morgan_fp("OC(=O)C=Cc1ccc(O)c(O)c1")], lotus_labels=["Phenylpropanoids"],
           train_fps=[C.morgan_fp("COc1cc(C=CC(=O)O)ccc1O")], train_labels=["Phenylpropanoids"], train_rows_of={})
truth_ik = Chem.MolToInchiKey(Chem.MolFromSmiles("OC(=O)C=Cc1ccc(O)c(O)c1"))[:14]
r = C.eval_query(("q1", truth_ik, "C9H8O4", ExactMolWt(Chem.MolFromSmiles("OC(=O)C=Cc1ccc(O)c(O)c1")),
                  "OC(=O)C=Cc1ccc(O)c(O)c1", "L"))
ok(r["found"] and r["n"] == 4, f"window: found={r['found']} n={r['n']} (4 same-formula, other formula excluded)")
ok(abs(r["rr|pop"] - np.mean([1, 1 / 2, 1 / 3, 1 / 4])) < 1e-9, "pop-only: 4-way tie")
ok(r["rr|vendor@0.25"] == 1.0 and r["truth_vendor"] == 0, "vendor penalty lifts the non-vendor truth to rank 1")
ok(r["rr|nplike@1.0"] >= r["rr|pop"], "NP-likeness does not hurt the NP truth")
ok(r["has_lab_analog"] and r["truth_lab"] and r["cc_truth"] > 0, "kNN label + labelled analog -> class signal")
r2 = C.eval_query(("q2", "NOTPRESENTXXXX", "C9H8O4", ExactMolWt(Chem.MolFromSmiles("OC(=O)C=Cc1ccc(O)c(O)c1")), "CCO", "T"))
ok(not r2["found"], "absent truth -> found False")


ok("rr|member@0.5" in r, "membership ablation variant present")
C.G.update(exclude_keys={tik[i] for i, x in enumerate(tsmi) if x.startswith("COC(=O)")}, exclude_cohorts=("P",))
rP = C.eval_query(("q3", truth_ik, "C9H8O4", ExactMolWt(Chem.MolFromSmiles("OC(=O)C=Cc1ccc(O)c(O)c1")),
                   "OC(=O)C=Cc1ccc(O)c(O)c1", "P"))
ok(rP["found"] and rP["n"] == 3, f"cohort P drops pool members from the window (n={rP['n']})")
rL = C.eval_query(("q4", truth_ik, "C9H8O4", ExactMolWt(Chem.MolFromSmiles("OC(=O)C=Cc1ccc(O)c(O)c1")),
                   "OC(=O)C=Cc1ccc(O)c(O)c1", "L"))
ok(rL["n"] == 4, "other cohorts keep the full window")
C.G.pop("exclude_keys"); C.G.pop("exclude_cohorts")

# ---- the Colab notebook builds, every code cell compiles, and it embeds the current core verbatim
import ast  # noqa: E402
import mk_np_signals as MK  # noqa: E402

nb = MK.build()
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i, src in enumerate(code):
    try:
        ast.parse(src)
    except SyntaxError as e:
        ok(False, f"cell {i} does not compile: {e}")
ok(any(repr(open(os.path.join(HERE, "np_signals_core.py"), encoding="utf-8").read()) in s for s in code),
   "notebook embeds the current np_signals_core.py")
joined = "\n".join(code)
for k in ("GATE_L, GATE_T = 0.010, -0.005", "GATE_V0 = 0.01", "GATE_C0_ANALOG, GATE_C0_MULTI = 0.50, 0.20",
          "N_COHORT, SEED, PPM_WIN = 2000, 0, 2.0"):
    ok(k in joined, f"pre-registered constant present: {k}")


nb2 = MK.build(*MK.BUILDS[1])
code2 = ["".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code"]
for i, src in enumerate(code2):
    try:
        ast.parse(src)
    except SyntaxError as e:
        ok(False, f"realism cell {i} does not compile: {e}")
j2 = "\n".join(code2)
ok("N_COHORT, SEED, PPM_WIN = 2000, 0, 2.0" in j2 and "R1_MIN = 0.005" in j2, "realism: pre-registered constants")
ok("exclude_cohorts=(\"P\",)" in j2, "realism: pool members excluded for cohort P only")
print(f"test_np_signals_core: {N} checks passed")
