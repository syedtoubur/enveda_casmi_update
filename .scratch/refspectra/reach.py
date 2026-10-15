"""Read-only reach diagnostic for `refspectra-lib` (campaign 20261013, C).

C's claim: a third engine built on external reference spectra (MassBank/MoNA) gives real spectral matches
for molecules our own library cannot see, so it acts where retrieval currently has no channel at all.

Class 2 is *defined* as "in databases, no reference spectra", so the claim is testable and could be false:
if MassBank carries no structure for the C2 truths, C cannot act on the cohort it is meant to fix.

Measured here: for each val molecule, (a) does the MassBank harmonized library contain the truth structure
at all (exact metric identity), (b) how many spectra does it carry, (c) does the current pipeline already
rank that truth at 1. No writes, no model, no submission.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(r"E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H  # noqa: E401

LIB = REPO / ".scratch" / "refspectra" / "lib"


def ik14_of(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    k = Chem.MolToInchiKey(m)
    return k[:14] if isinstance(k, str) and len(k) >= 14 else None


def main():
    structs = pd.read_parquet(LIB / "structures_with_fingerprints.parquet",
                              columns=["inchikey", "smiles", "formula"])
    spec = pd.read_parquet(LIB / "spectra.parquet", columns=["inchikey"])
    n_spectra = spec["inchikey"].value_counts()

    # library side: canonical metric identities (RDKit layer-1 skeleton == InChIKey14)
    lib_by_key = {}
    for ik, smi, form in zip(structs["inchikey"], structs["smiles"], structs["formula"]):
        if not isinstance(ik, str):
            continue
        k = ik[:14]
        n = int(n_spectra.get(ik, 0))
        if k in lib_by_key:
            lib_by_key[k] = (lib_by_key[k][0], lib_by_key[k][1] + n)
        else:
            lib_by_key[k] = (smi, n)
    print(f"MassBank library: {len(structs)} structures, {len(lib_by_key)} distinct InChIKey14, "
          f"{len(spec)} spectra records")

    T = H.truths()
    for shard in ("C2", "C1_npx"):
        mids = [m for m in T if T[m].get("cohort") == shard]
        hit = hit_spec = miss_now_covered = covered_rank1 = 0
        spec_hist = []
        ex = []
        for mid in mids:
            truth = cp_canon(T[mid]["smiles"]) if CP else T[mid].get("inchikey14")
            if not truth:
                continue
            k = truth[:14]
            rec = lib_by_key.get(k)
            if rec is None:
                continue
            hit += 1
            n = rec[1]
            if n > 0:
                hit_spec += 1
            spec_hist.append(n)
            r = current_rank(mid, shard, k)
            if r == 1:
                covered_rank1 += 1
            elif r != 1:
                miss_now_covered += 1
                if len(ex) < 8:
                    ex.append((mid, k, n, r))
        tot = len(mids)
        print(f"\n=== {shard}: {tot} molecules")
        print(f"  truth structure present in MassBank:        {hit:4d} ({100*hit/max(tot,1):.1f}%)")
        print(f"  ... and it carries >=1 measured spectrum:   {hit_spec:4d} ({100*hit_spec/max(tot,1):.1f}%)")
        print(f"  covered AND currently NOT rank 1 (C's target): {miss_now_covered:4d}")
        print(f"  covered AND already rank 1 (no gain available): {covered_rank1:4d}")
        if spec_hist:
            h = np.array(spec_hist)
            print(f"  spectra per covered truth: median {np.median(h):.0f}, p90 {np.quantile(h,0.9):.0f}, max {h.max():.0f}")
        for mid, k, n, r in ex:
            print(f"    {mid}  {k}  spectra={n}  current rank={r}")


CP = None


def cp_canon(s):
    return CP.canonicalize_smiles_to_ik14(s)


RANK_CACHE = {}


def current_rank(mid, shard, truth14):
    """Is the truth at rank 1 of the primary BASE list in the dump? Returns 1, r>1, or 0."""
    if shard not in RANK_CACHE:
        state, deps, cp = H.load_dump(shard)
        RANK_CACHE[shard] = (state, cp)
    state, cp = RANK_CACHE[shard]
    prim = (state["base"].get(mid) or ([], [], None))[0] or []
    for i, s in enumerate(prim[:25], 1):
        if (cp.canonicalize_smiles_to_ik14(s) or "")[:14] == truth14:
            return i
    return 99


if __name__ == "__main__":
    state, deps, CP = H.load_dump("C2")
    main()
