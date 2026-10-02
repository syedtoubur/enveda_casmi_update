"""Execute a mutator notebook's MUTATOR CHANNEL cell against stubbed notebook globals.

A syntax check cannot see a NameError path, an f-string that only evaluates on the happy branch, a
sanity check that reads a field the runner does not emit, or a counter that divides by zero when
the channel produced nothing. This runs the real cell text with `subprocess.run` stubbed, over the
scenarios that matter: both sources, engine failure, each way the sanity block can trip, an empty
result, a missing `pc_z.npz`, and an empty PubChem list.

Every failure path must leave `MUT == {}` -- that is what makes the notebook fall back to the 0.399
pipeline instead of dying mid-run and producing no submission.

usage: python build/test_mut_cell.py [notebook.ipynb]      (default: the bio notebook)
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "casmi26-v4n-mutator-bio-submit.ipynb")
_BASE = os.path.basename(NB)
C3 = "c3" in _BASE                       # Class-3 routed: a superset of the bio variant
BIO = "bio" in _BASE or C3               # engine-A scaffolds, MUT_K 40, MUT_DEADLINE

nb = json.load(io.open(NB, encoding="utf-8"))
cell = [c for c in ("".join(x["source"]) for x in nb["cells"]) if "=== MUTATOR CHANNEL" in c][0]

TMP = tempfile.mkdtemp(prefix="mut_cell_test_")
WORK = os.path.join(TMP, "kaggle_working")
os.makedirs(WORK)
cell = cell.replace("/kaggle/working", WORK.replace("\\", "/"))

MUT_OUT: dict = {}
FAIL: list = []


def ck(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


class FakeRun:
    returncode = 0

    def __init__(self, out, err=""):
        self.stdout, self.stderr = out, err


def scenario(name, mut_payload, eng, pc, make_z=True, expect_disabled=False, expect_eng=None,
             c3_diag=None):
    """Run the cell once. `expect_eng=None` skips the MUT_ENG assertion."""
    global MUT_OUT
    print(name)
    zp = os.path.join(WORK, "pc_z.npz")
    if make_z:
        io.open(zp, "wb").write(b"x")
    elif os.path.exists(zp):
        os.remove(zp)
    io.open(os.path.join(WORK, "eng_lists.json"), "w").write(json.dumps(eng))
    c3p = os.path.join(WORK, "eng_lists.json.c3.json")
    if c3_diag is None:
        if os.path.exists(c3p):
            os.remove(c3p)
    else:
        io.open(c3p, "w").write(json.dumps(c3_diag))
    seen_env = {}

    def fake_run(cmd, **kw):
        seen_env.update({k: v for k, v in kw.get("env", {}).items()
                         if k.startswith("MUT_") or k.startswith("C3_")})
        json.dump(mut_payload, io.open(os.path.join(WORK, "mut_lists.json"), "w"))
        return FakeRun("mut_runner done: stub\nMUT SOURCES: scaffolds pc=1 eng=2 eng-only=1\n")

    sp = types.ModuleType("subprocess")
    sp.run = fake_run
    g = dict(os=os, sys=sys, json=json, time=time, subprocess=sp, T0=time.time(),
             PC=pc, ENG=eng, V3CODE="/v3", POOL_DIR="/pool", IS_RERUN=False, MUT_MAXM=3)
    exec(compile(cell, "<mutator cell>", "exec"), g)
    MUT_OUT = g.get("MUT")
    ck(isinstance(MUT_OUT, dict), "MUT is a dict after the cell runs")
    if expect_disabled:
        ck(MUT_OUT == {}, "the channel disabled itself and left 0.399 behaviour")
    if expect_eng is not None:
        ck(("MUT_ENG" in seen_env) == bool(expect_eng),
           f'MUT_ENG {"passed" if expect_eng else "withheld"}')
    return seen_env


PC = {"m1": {"pc_keys": ["PCKEY00000001"], "pc": ["c1ccccc1"]},
      "m2": {"pc_keys": ["PCKEY00000002"], "pc": ["c1ccccc1"]}}
ENG = {"m1": {"smiles": ["c1ccc2ccccc2c1"], "keys": ["ENGKEY00000001"]},
       "m2": {"smiles": ["c1ccc2ccccc2c1"], "keys": ["ENGKEY00000002"]}}
GOOD = {"m1": dict(smiles=["CCO", "CCC"], keys=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
                   formula=["C2H6O", "C3H8"], fz=[9.0, 8.0], src=["pc", "eng"], tier="sure"),
        "m2": dict(smiles=["CCN"], keys=["CCCCCCCCCCCCCC"], formula=["C2H7N"], fz=[7.0],
                   src=["ana"], tier="likely")}
# what the engine exports for the Class-3 gate: exact-mass library similarity plus the
# mass-shifted analogs as (smiles, mass, similarity)
C3DIAG = {"m1": dict(target=400.0, lib_exact=0.1, ana=[["c1ccc2ccccc2c1", 300.0, 0.9]]),
          "m2": dict(target=410.0, lib_exact=0.1, ana=[["c1ccc2ccccc2c1", 310.0, 0.9]])}


def main():
    print(f'notebook: {os.path.basename(NB)}  ({"bio" if BIO else "PubChem-only"} variant)')

    env = scenario("happy path", GOOD, ENG, PC, expect_eng=True if BIO else None,
                   c3_diag=C3DIAG if C3 else None)
    ck(env.get("MUT_K") == ("40" if BIO else "25"), f'MUT_K reaches the runner, got {env.get("MUT_K")}')
    ck(len(MUT_OUT) == 2, "both molecules survive the sanity checks")

    if C3:
        ck(env.get("MUT_C3", "").endswith(".c3.json"),
           f'MUT_C3 reaches the runner, got {env.get("MUT_C3")}')
        ck(env.get("C3_MAX_STEPS") == "1",
           f'C3_MAX_STEPS reaches the runner at the measured band, got {env.get("C3_MAX_STEPS")}')
        # the whole point of the gate is that it can decline; the cell must survive the diagnostics
        # file being absent rather than silently switching to a gate it cannot calibrate unannounced
        env2 = scenario("no c3 diagnostics -> rank gate", GOOD, ENG, PC, c3_diag=None)
        ck("MUT_C3" not in env2, "MUT_C3 is withheld when the diagnostics file is absent")
        ck(len(MUT_OUT) == 2, "and the channel still runs on the old rank gate")

    if BIO:
        ck(env.get("MUT_DEADLINE") == "45", f'MUT_DEADLINE reaches the runner, got {env.get("MUT_DEADLINE")}')

        scenario("engine failed -> ENG empty", GOOD, {}, PC, expect_eng=False)
        ck(len(MUT_OUT) == 2, "the mutator still runs PubChem-only")

        bad = {"m1": dict(GOOD["m1"], keys=["ENGKEY00000001", "BBBBBBBBBBBBBB"])}
        scenario("candidate overlaps the engine list", bad, ENG, PC, expect_disabled=True)

        scenario("ragged src", {"m1": dict(GOOD["m1"], src=["pc"])}, ENG, PC, expect_disabled=True)

    bad3 = {"m1": dict(GOOD["m1"], keys=["PCKEY00000001", "BBBBBBBBBBBBBB"])}
    scenario("candidate overlaps its own PubChem list", bad3, ENG, PC, expect_disabled=True)

    scenario("duplicate keys", {"m1": dict(GOOD["m1"], keys=["AAAAAAAAAAAAAA"] * 2)}, ENG, PC,
             expect_disabled=True)

    n = 41 if BIO else 26
    bad4 = {"m1": dict(smiles=["CCO"] * n, keys=[f"K{i:013d}" for i in range(n)],
                       formula=["C2H6O"] * n, fz=[1.0] * n, src=["pc"] * n)}
    scenario("more than MUT_K candidates", bad4, ENG, PC, expect_disabled=True)

    scenario("empty mutator output", {}, ENG, PC)
    ck(MUT_OUT == {}, "an empty result is handled without dividing by zero")

    scenario("no pc_z.npz -> channel skipped", GOOD, ENG, PC, make_z=False)
    ck(MUT_OUT == {}, "the channel skips cleanly when z was never exported")

    scenario("no PubChem lists -> channel skipped", GOOD, ENG, {})
    ck(MUT_OUT == {}, "the channel skips cleanly when PC is empty")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
