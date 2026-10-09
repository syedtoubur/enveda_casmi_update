"""Tests for build/patch_c3cal.py.  Run: python build/test_patch_c3cal.py

Builds the calibration notebook, checks the embedded runners compile with their edits, and
executes the data-builder cell against a synthetic train.parquet with the real column schema:
the library ban, the pseudo-test, the truth table and the ban file must all come out right.
"""
import ast
import json
import os
import re
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import patch_c3cal as C
from patch_v4n_mutator import src_of

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


TMP = tempfile.mkdtemp()
OUT = os.path.join(TMP, "cal.ipynb")
C.main(["--out", OUT, "--n-c3", "3", "--n-c2", "2", "--n-c1", "2"])
NB = json.load(open(OUT, encoding="utf-8"))
CODES = [src_of(c) for c in NB["cells"] if c["cell_type"] == "code"]


def test_notebook():
    print("test_notebook")
    check(all(_parses(s) for s in CODES), "every code cell parses")
    eng = [s for s in CODES if "eng_runner.py" in s][0]
    i = eng.find("open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write(")
    er = ast.literal_eval(eng[i + len("open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write("):eng.find("\n", i) - 1])
    check(_parses(er) and "C3CAL engine pool" in er, "engine runner compiles with the pool ban")
    pc = [s for s in CODES if "RUNNER = " in s][0]
    rr = ast.literal_eval(re.search(r"^RUNNER = (.*)$", pc, re.M).group(1))
    check(_parses(rr) and "_BANL" in rr and "probe skipped" in rr and "pc_z.npz" in rr,
          "PubChem runner compiles: pool-window mask, probe skipped, z exported")
    check("imap_unordered(pc.probe_one" not in rr, "  and the PubChem probe really is gone")
    mut = [s for s in CODES if C.P.MARK in s][0]
    check("MUT_KEEP_DECLINED='1'" in mut, "mutator keeps declined lists for recall measurement")
    check(not any("def fuse2" in s or "ICE_UNION" in s for s in CODES), "v4n/ICE/fusion cells dropped")
    check("PRE-REGISTERED" in CODES[-1], "evaluation cell last")


def _parses(s):
    try:
        ast.parse(s)
        return True
    except SyntaxError:
        return False


def test_builder():
    print("test_builder")
    comp, inp = os.path.join(TMP, "comp"), os.path.join(TMP, "input")
    os.makedirs(comp); os.makedirs(inp)
    iks = ["C3KEYAAAAAAAAA", "C3KEYBBBBBBBBB", "C2KEYAAAAAAAAA"] + ["OTHER%09d" % i for i in range(3000)]
    rows = []
    for k in iks:
        for j in range(20 if k.startswith("C") else 1):
            rows.append(dict(inchikey14=k, normalized_smiles="CCO" if k != "C2KEYAAAAAAAAA" else "CCN",
                             adduct="[M+H]+", precursor_mz=47.05, ionization_mode="positive",
                             ms2_mzs=[29.0, 31.0], ms2_normalized_intensities=[0.5, 1.0],
                             molecular_formula="C2H6O", instrument_type="timsTOF",
                             collision_energy=[20.0]))
    t = pa.Table.from_pylist(rows)
    pq.write_table(t, os.path.join(comp, "train.parquet"), row_group_size=500)
    base = dict(smiles="CCO", in_scope=True)
    pd.DataFrame([dict(base, molecule_id="P0", inchikey14="C3KEYAAAAAAAAA", lib_ban_keys=["C3KEYAAAAAAAAA"],
                       pool_ban_keys=["C3KEYAAAAAAAAA"]),
                  dict(base, molecule_id="P1", inchikey14="C3KEYBBBBBBBBB", lib_ban_keys=["C3KEYBBBBBBBBB"],
                       pool_ban_keys=["C3KEYBBBBBBBBB"])]).to_parquet(os.path.join(inp, "pseudo_c3.parquet"))
    pd.DataFrame([dict(base, smiles="CCN", molecule_id="D0", inchikey14="C2KEYAAAAAAAAA",
                       lib_ban_keys=["C2KEYAAAAAAAAA"], pool_ban_keys=[])]).to_parquet(
        os.path.join(inp, "c2_displace.parquet"))
    cell = [s for s in CODES if "C3 CALIBRATION: pseudo-test" in s][0]
    cal = os.path.join(TMP, "cal_out")
    cell = cell.replace("/kaggle/working/cal", cal.replace("\\", "/"))
    g = dict(os=os, json=json, COMP=comp, find=lambda n: os.path.join(inp, n))
    exec(cell, g)
    tr = pd.read_csv(os.path.join(cal, "truth.csv"))
    te = pd.read_parquet(os.path.join(cal, "test.parquet"))
    lib = pq.read_table(os.path.join(cal, "train.parquet")).to_pandas()
    ban = json.load(open(os.path.join(cal, "ban.json")))
    check(set(tr[tr.cohort == "c3"].molecule_id) == {"P0", "P1"} and
          set(tr[tr.cohort == "c2"].molecule_id) == {"D0"}, "positives and Class-2 negatives in the truth table")
    check(0 < (tr.cohort == "c1").sum() <= 2, "Class-1 controls sampled and capped (%d)" % (tr.cohort == "c1").sum())
    check(not lib.inchikey14.isin(["C3KEYAAAAAAAAA", "C3KEYBBBBBBBBB", "C2KEYAAAAAAAAA"]).any(),
          "every banned library row removed")
    c1k = set(tr[tr.cohort == "c1"].inchikey14)
    check(lib.inchikey14.isin(c1k).sum() == len(c1k), "Class-1 control spectra stay in the library")
    check(len(lib) == len(rows) - 60, "and nothing else was removed (%d rows)" % len(lib))
    check(te.groupby("molecule_id").size().max() <= 16, "<= 16 spectra per query molecule")
    check(set(["molecule_id", "spectrum_id", "precursor_mz", "adduct", "ionization_mode", "ms2_mzs",
               "ms2_normalized_intensities", "collision_energy_ev"]) <= set(te.columns),
          "test.parquet has the competition columns")
    check("inchikey14" not in te.columns and "cohort" not in te.columns, "and no answer leaks into it")
    check(set(ban["pool_ban"]) == {"C3KEYAAAAAAAAA", "C3KEYBBBBBBBBB"}, "pool ban = pseudo_c3 truths only")
    check([a[0] for a in ban["pool_add"]] == ["C2KEYAAAAAAAAA"], "Class-2 structure re-added to the pool")
    check(g["COMP"] == cal.replace("\\", "/") and os.environ.get("CASMI_BAN_KEYS", "").endswith("ban.json"),
          "COMP and CASMI_BAN_KEYS redirected")
    os.environ.pop("CASMI_BAN_KEYS", None)


if __name__ == "__main__":
    test_notebook()
    test_builder()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")
