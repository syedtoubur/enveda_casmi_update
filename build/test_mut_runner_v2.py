"""End-to-end test of mut_runner.py in MUT_V2 mode against the mocked `casmi` of test_mut_runner.

Four synthetic molecules, each built to exercise one branch of the anchored v2 path:

  m_glc   query = quercetin 3-O-glucoside; the only analog is quercetin at exactly -162.0528 Da,
          and z is the glucoside's own fingerprint (an oracle).  Must fire, and the truth must
          come back first, tagged '|gen', with the stage-1 signals attached.
  m_lib   same, but the exact-window library match is 0.95 -> Class 1, declined before generation.
  m_mass  the analog sits at a mass no <= 2-step move combination explains -> declined.
  m_pool  same as m_glc but best_pool_fz above anything generated -> "a database structure fits
          the predicted fingerprint better", declined after generation.

Run: python build/test_mut_runner_v2.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_mut_runner as T  # noqa: E402  (mock casmi + build_env)
import delta_m_mutator as M  # noqa: E402

FAIL = []
Q = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
QG = "OCC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def mass(s):
    return M.vec_mass(M.vec_of_smiles(s))


def main():
    tmp = tempfile.mkdtemp(prefix="mut_v2_test_")
    try:
        mids = ["m_glc", "m_lib", "m_mass", "m_pool"]
        tgt = mass(QG) * (1 + 1.45e-6)              # as measured: timsTOF reads +1.45 ppm high
        pc = {m: dict(pc=["CCO"], pc_fz=[0.0], pc_keys=[], fz_top=None, best_pool_fz=-1e9,
                      target=tgt) for m in mids}
        pc["m_pool"]["best_pool_fz"] = 1e9
        v3, pool = T.build_env(tmp, mids, pc)
        # oracle z: +1 on the truth's selected bits, -1 elsewhere
        sys.path.insert(0, v3)
        from casmi import chem
        bits = np.load(os.path.join(pool, "fp_bits.npy"))
        f = chem.raw_fingerprint(QG)[bits].astype(np.float32)
        z = np.stack([2 * f - 1] * len(mids)).astype(np.float32)
        np.savez_compressed(os.path.join(tmp, "pc_z.npz"), mids=np.array(mids, dtype=object), z=z)
        ana = [[Q, mass(Q), 0.85]]
        c3 = {"m_glc": dict(target=tgt, lib_exact=0.2, ana=ana),
              "m_lib": dict(target=tgt, lib_exact=0.95, ana=ana),
              "m_mass": dict(target=tgt, lib_exact=0.2, ana=[[Q, mass(Q) + 7.77, 0.85]]),
              "m_pool": dict(target=tgt, lib_exact=0.2, ana=ana)}
        c3f = os.path.join(tmp, "c3.json")
        json.dump(c3, open(c3f, "w"))
        out = os.path.join(tmp, "mut.json")
        env = dict(os.environ, MUT_V2="1", MUT_C3=c3f, MUT_K="10", MUT_TAU="8", MUT_DEADLINE="600",
                   PYTHONIOENCODING="utf-8")
        env.pop("MUT_ENG", None)
        r = subprocess.run([sys.executable, os.path.join(HERE, "mut_runner.py"), v3, pool,
                            os.path.join(tmp, "pc_lists.json"), os.path.join(tmp, "pc_z.npz"), out, "1"],
                           capture_output=True, text=True, env=env, cwd=tmp, timeout=1800)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit("mut_runner exited %d" % r.returncode)
        res = json.load(open(out))
        diag = {d["molecule_id"]: d for d in json.load(open(out + ".diag.json"))}

        print("test_v2_fires_and_recovers")
        g = res.get("m_glc")
        check(g is not None, "the glucoside molecule fires")
        if g:
            check(chem.score_key(g["smiles"][0]) == chem.score_key(QG),
                  "the truth is regenerated from quercetin and ranked first")
            check(g["tier"] in ("sure", "likely"), "with a stage-1 tier (%s)" % g["tier"])
            check(all(x.endswith("|gen") for x in g["formula"]), "every formula carries |gen")
            check(set(g["sig"]) >= {"ana_max", "fz_best", "fz_anchor", "best_pool_fz"},
                  "stage-1 signals attached for stage 2")
            check(g["sig"]["fz_best"] > g["sig"]["fz_anchor"], "fz(best) > fz(anchor)  (I0)")
            check(g["src"][0].startswith("a0:") and "hexose" in g["src"][0],
                  "provenance names the anchor and the move (%s)" % g["src"][0])

        print("test_v2_declines")
        check("m_lib" not in res and diag["m_lib"].get("gate") == "class1_library_match",
              "library match declined before generation")
        check(diag["m_lib"].get("n_gen", 0) == 0, "  and nothing was generated for it")
        check("m_mass" not in res and diag["m_mass"].get("gate") == "delta_unmatched",
              "unexplained mass delta declined")
        check("m_pool" not in res and diag["m_pool"].get("gate") == "pool_explains_fp",
              "pool fits the fingerprint better -> declined after generation")
        check("MUT V2 STATS" in r.stdout and "MUT V2: anchored" in r.stdout, "v2 counters printed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for x in FAIL:
            print("  -", x)
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
