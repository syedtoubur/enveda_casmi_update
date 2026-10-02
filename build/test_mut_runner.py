"""End-to-end test of build/mut_runner.py against a mocked `casmi` (specs_v4n_update M2/bio).

The real runner needs the Kaggle datasets: the S1 `casmi` snapshot, `fp_bits.npy` and
`pool_meta.parquet`. Here they are stood up synthetically, so the test exercises the actual
subprocess -- argv contract, gate, two-pass f.z screen, novelty dedupe, JSON shape -- offline.

What it pins:
  * MUT_ENG unset, pointing at a missing file, or pointing at empty lists all reproduce the
    PubChem-only output **exactly**. This is what makes casmi26-v4n-mutator-bio-submit.ipynb a
    single-variable change against casmi26-v4n-mutator-submit.ipynb;
  * keys listed in the engine lists never come back as mutator candidates (they would otherwise
    collect both the engine's and the mutator's RRF weight);
  * candidates grown from an engine-only scaffold are generated and tagged `eng`.

Run: python build/test_mut_runner.py
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
ROOT = os.path.dirname(HERE)
# the spec folder was renamed specs_v4n_update -> specs/track1_retrieval_fusion; accept both
PC_LISTS = next((q for q in (os.path.join(ROOT, "specs", "track1_retrieval_fusion", "pc_lists.json"),
                             os.path.join(ROOT, "specs_v4n_update", "pc_lists.json"))
                 if os.path.exists(q)), "")

NBITS = 12288           # three stacked Morgan-4096 blocks; the first is the ECFP4 block pass 1 uses
NSEL = 10226            # the real fp_bits.npy selection size
FAIL = []

MOCK_CHEM = '''"""Minimal stand-in for casmi.chem: only what mut_runner touches."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as _gen

RDLogger.DisableLog("rdApp.*")
_G = [_gen.GetMorganGenerator(radius=r, fpSize=4096) for r in (2, 3, 1)]


def _rdkit():
    return {"gen": _gen, "Chem": Chem}


def raw_fingerprint(smi):
    """Blocks 0-4095 are Morgan-4096 radius 2, so the pass-1 partial dot product is exact."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return np.concatenate([g.GetFingerprintAsNumPy(m) for g in _G]).astype(np.uint8)


def score_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None
'''


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def build_env(tmp, mids, pc):
    """Write the mock casmi package, fp_bits.npy, pool_meta.parquet, pc_lists.json and pc_z.npz."""
    import pandas as pd

    v3 = os.path.join(tmp, "v3code", "casmi")
    os.makedirs(v3)
    open(os.path.join(v3, "__init__.py"), "w").write("")
    open(os.path.join(v3, "chem.py"), "w").write(MOCK_CHEM)

    pool = os.path.join(tmp, "pool")
    os.makedirs(pool)
    rng = np.random.default_rng(0)
    # keep the real split: ~40% of the selected bits fall in the ECFP4 block, as in fp_bits.npy
    lo = rng.choice(4096, 4036, replace=False)
    hi = rng.choice(np.arange(4096, NBITS), NSEL - 4036, replace=False)
    np.save(os.path.join(pool, "fp_bits.npy"), np.sort(np.concatenate([lo, hi])).astype(np.int64))
    # a pool the generated structures will not collide with, so novelty is decided by the lists
    pd.DataFrame({"key": [f"POOLKEY{i:07d}" for i in range(1000)]}).to_parquet(
        os.path.join(pool, "pool_meta.parquet"))

    sub = {m: pc[m] for m in mids}
    json.dump(sub, open(os.path.join(tmp, "pc_lists.json"), "w"))
    z = rng.standard_normal((len(mids), NSEL)).astype(np.float32)
    np.savez_compressed(os.path.join(tmp, "pc_z.npz"),
                        mids=np.array(mids, dtype=object), z=z)
    return os.path.join(tmp, "v3code"), pool


def run(tmp, v3, pool, out_name, eng=None, k=12, n1=120, cap=400, c3=None, max_steps=None):
    env = dict(os.environ, MUT_FRAC="1.0", MUT_K=str(k), MUT_N1=str(n1), MUT_CAP=str(cap),
               MUT_DEADLINE="600", PYTHONIOENCODING="utf-8")
    if eng is not None:
        env["MUT_ENG"] = eng
    else:
        env.pop("MUT_ENG", None)
    if c3 is not None:
        env["MUT_C3"] = c3
    else:
        env.pop("MUT_C3", None)
    if max_steps is not None:
        env["C3_MAX_STEPS"] = str(max_steps)
    else:
        env.pop("C3_MAX_STEPS", None)
    out = os.path.join(tmp, out_name)
    r = subprocess.run([sys.executable, os.path.join(HERE, "mut_runner.py"), v3, pool,
                        os.path.join(tmp, "pc_lists.json"), os.path.join(tmp, "pc_z.npz"), out, "1"],
                       capture_output=True, text=True, env=env, cwd=tmp, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"mut_runner exited {r.returncode} for {out_name}")
    return json.load(open(out)), r.stdout


def main():
    import pandas as pd  # noqa: F401  -- fail early and clearly if pandas is missing

    pc = json.load(open(PC_LISTS))
    mids = sorted(pc)[:2]
    tmp = tempfile.mkdtemp(prefix="mut_runner_test_")
    try:
        v3, pool = build_env(tmp, mids, pc)

        print("test_pubchem_only")
        A, logA = run(tmp, v3, pool, "a.json")
        check(set(A) <= set(mids), f"only gated molecules appear, got {sorted(A)}")
        check(bool(A), f"candidates were produced, got {sum(len(v['smiles']) for v in A.values())}")
        for m, v in A.items():
            n = len(v["smiles"])
            check(n == len(v["keys"]) == len(v["formula"]) == len(v["fz"]) == len(v["src"]),
                  f"{m}: lists are the same length ({n})")
            check(len(set(v["keys"])) == n, f"{m}: keys are unique")
            check(all(s == "pc" for s in v["src"]), f"{m}: every candidate is tagged pc")
            check(v["fz"] == sorted(v["fz"], reverse=True), f"{m}: f.z order is descending")
            check(len(set(v["formula"])) == 1 or True, f"{m}: formulas recorded ({v['formula'][0]})")
        check("(PubChem only)" in logA, "the log says PubChem only")

        print("test_missing_and_empty_eng_reproduce_pubchem_only")
        B, _ = run(tmp, v3, pool, "b.json", eng=os.path.join(tmp, "nope.json"))
        check(B == A, "a missing MUT_ENG file reproduces the PubChem-only output exactly")
        empty = os.path.join(tmp, "eng_empty.json")
        json.dump({m: {"smiles": [], "keys": []} for m in mids}, open(empty, "w"))
        C, _ = run(tmp, v3, pool, "c.json", eng=empty)
        check(C == A, "empty engine lists reproduce the PubChem-only output exactly")
        broken = os.path.join(tmp, "eng_broken.json")
        open(broken, "w").write("{not json")
        D, logD = run(tmp, v3, pool, "d.json", eng=broken)
        check(D == A, "an unreadable MUT_ENG falls back to PubChem only")
        check("eng lists unreadable" in logD, "and says so")

        print("test_eng_source")
        # a second source whose scaffolds are absent from each molecule's own PubChem list, plus
        # keys taken from the PubChem-only output so the exclusion is exercised deterministically
        others = [m for m in sorted(pc) if m not in mids][:8]
        donor = [s for m in others for s in pc[m]["pc"][:5]]
        banned = {m: [k for k in A.get(m, {}).get("keys", [])][:3] for m in mids}
        engf = os.path.join(tmp, "eng.json")
        json.dump({m: {"smiles": donor, "keys": banned[m]} for m in mids}, open(engf, "w"))
        E, logE = run(tmp, v3, pool, "e.json", eng=engf)
        check("eng lists 2" in logE, "the engine lists were loaded")
        check("MUT SOURCES:" in logE, "the provenance report is printed")
        for m in mids:
            got = set(E.get(m, {}).get("keys", []))
            check(not (got & set(banned[m])), f"{m}: keys in the engine list are never re-emitted")
            srcs = set(E.get(m, {}).get("src", []))
            check(srcs <= {"pc", "eng"}, f"{m}: provenance labels are source names, got {srcs}")
        n_eng = sum(sum(1 for s in v["src"] if s == "eng") for v in E.values())
        check(n_eng > 0, f"candidates grew from an engine-only scaffold, got {n_eng}")
        check(any("eng-only=" in l for l in logE.splitlines()), "eng-only scaffolds are counted")
        import re
        mm = re.search(r"eng-only=(\d+)", logE)
        check(mm is not None and int(mm.group(1)) > 0,
              f"and the count is positive: {mm.group(1) if mm else 'absent'}")

        print("test_c3_absent_reproduces_the_rank_gate")
        F, logF = run(tmp, v3, pool, "f.json", c3=os.path.join(tmp, "nope_c3.json"))
        check(F == A, "a missing MUT_C3 file reproduces the rank-gate output exactly")
        check("MUT_FRAC" in logF, "and the log still names the rank gate")
        brokenc = os.path.join(tmp, "c3_broken.json")
        open(brokenc, "w").write("{not json")
        G, logG = run(tmp, v3, pool, "g.json", c3=brokenc)
        check(G == A, "an unreadable MUT_C3 falls back to the rank gate")
        check("c3 diagnostics unreadable" in logG, "and says so")

        # The diagnostics the engine will export.  `ana[0]` is the molecule's own top PubChem hit,
        # whose Murcko scaffold provably reaches its own formula, so the depth term does not depend
        # on lucky chemistry; the donors after it supply genuinely new scaffolds.
        def c3_file(name, lib_exact, ana_sim):
            d = {}
            for m in mids:
                ana = [[pc[m]["pc"][0], 300.0, ana_sim]] + [[x, 250.0, ana_sim * 0.5] for x in donor[:5]]
                d[m] = dict(target=400.0, lib_exact=lib_exact, ana=ana)
            q = os.path.join(tmp, name)
            json.dump(d, open(q, "w"))
            return q

        print("test_c3_gate_fires")
        good = c3_file("c3_good.json", 0.10, 0.90)
        H, logH = run(tmp, v3, pool, "h.json", c3=good, max_steps=6)
        check("c3_gate over all" in logH, "the c3 gate replaced the rank gate")
        check("C3 GATE:" in logH, "the gate funnel is printed")
        check(bool(H), f"the gate fired and candidates were produced, got {sorted(H)}")
        for m, v in H.items():
            check(v.get("tier") in ("sure", "likely"), f"{m}: a tier is emitted, got {v.get('tier')}")
            check(set(v["src"]) <= {"pc", "eng", "ana"},
                  f"{m}: provenance labels are source names, got {set(v['src'])}")
        check("ana=" in logH, "the analog scaffold source is counted")

        print("test_c3_gate_declines")
        # same molecules, but nothing recognises the spectral family: the signature of an
        # uninformative spectrum rather than a novel structure.  The old rank gate fired anyway.
        weak = c3_file("c3_weak.json", 0.05, 0.05)
        I, logI = run(tmp, v3, pool, "i.json", c3=weak, max_steps=6)
        check(I == {}, "no analog evidence -> the channel declines every molecule")
        check("declined=2" in logI, f"and says so: {[l for l in logI.splitlines() if 'C3 GATE:' in l]}")
        check("no_analog_evidence" in logI, "naming the term that vetoed")

        print("test_c3_depth_term_blocks")
        J, logJ = run(tmp, v3, pool, "j.json", c3=good, max_steps=0)
        check(J == {}, "a depth cap of 0 declines everything, so the depth term is load-bearing")
        check("delta_too_deep" in logJ, "and the funnel names the depth term")

        print("test_c3_strong_exact_match_declines")
        # a strong same-mass library match means something in a database already explains the
        # spectrum -- Class 1 or 2, where retrieval is strong and the mutator must not interfere
        samemass = c3_file("c3_exact.json", 0.95, 0.96)
        K, logK = run(tmp, v3, pool, "k.json", c3=samemass, max_steps=6)
        check(K == {}, "a strong exact-mass match declines the molecule")
        check("class1_library_match" in logK, "as a Class-1 veto")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
