"""Side-chat probe (NOT part of the loop): does a mass-preserving isomer edit create
DIFFERENTIAL fragment ions that MetFrag-lite could use to separate parent from child?

Reuses the real operators (isomer_mut_base.orig) and the real scorer's fragment_masses
(casmi_scratch.metfrag). Loads a handful of NP structures from pseudo_c3.parquet only.
Light, single-threaded, no spectrum IO.
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, "scratch_pkg")
from casmi_scratch.metfrag import fragment_masses  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO = Path(".")
MUT_ORIG = REPO / "research/c3/20261017-isomer-mut-ice/isomer_mut_base.py.orig"

# load the mutator from its .orig file
_spec = importlib.util.spec_from_loader(
    "isomut", importlib.machinery.SourceFileLoader("isomut", str(MUT_ORIG)))
isomut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(isomut)

H = 1.00782503207
TOL = 0.01  # MetFrag-lite match tolerance (metfrag.py explain_score default)


def ion_neutral_set(smi):
    """Neutral fragment masses (0/1/2-bond cuts) rounded to tolerance buckets."""
    fr = fragment_masses(smi)
    return np.array(sorted(set(np.round(fr / TOL).astype(int) * TOL)))


def diff_report(parent_smi, child_smi):
    """Classify the child's fragment set vs the parent's, in Da."""
    fp, fc = ion_neutral_set(parent_smi), ion_neutral_set(child_smi)
    # adjacency identical => same cut partitions => no separable ion
    def adj(s):
        m = Chem.MolFromSmiles(s)
        return frozenset(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in m.GetBonds())
    shared = len([x for x in fc if np.min(np.abs(fp - x)) <= TOL + 1e-9]) if len(fp) and len(fc) else 0
    uniq_child = len(fc) - shared
    uniq_parent = len(fp) - shared
    return {
        "n_frag_parent": len(fp), "n_frag_child": len(fc),
        "shared": shared, "child_unique": uniq_child, "parent_unique": uniq_parent,
        "adjacency_same": adj(parent_smi) == adj(child_smi),
    }


def main():
    import pandas as pd
    df = pd.read_parquet(REPO / "p1_novel/pseudo_c3.parquet")
    smis = df["smiles"].dropna().tolist()
    # keep molecules that actually mutate (ring + side chain) and are not huge
    chosen = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() > 40:
            continue
        kids = isomut.mutate(s, max_moves=1, cap=60)
        if kids:
            chosen.append((s, kids))
        if len(chosen) >= 25:
            break

    print(f"parents sampled: {len(chosen)}\n")
    by_op = defaultdict(lambda: {"pairs": 0, "adj_same": 0, "sep_total": 0, "sep_nonzero": 0,
                                 "nfrag": 0})
    example = None
    for s, kids in chosen:
        for child, op, nm in kids:
            base = op.split("+")[0]
            rep = diff_report(s, child)
            d = by_op[base]
            d["pairs"] += 1
            d["nfrag"] += rep["n_frag_child"]
            if rep["adjacency_same"]:
                d["adj_same"] += 1
            sep = rep["child_unique"] + rep["parent_unique"]
            d["sep_total"] += sep
            if sep > 0:
                d["sep_nonzero"] += 1
            # remember one rich example (group_move / subst_shift)
            if base in ("subst_shift", "group_move", "methyl_swap") and sep >= 3 and example is None:
                example = (s, child, op, rep)

    print(f"{'operator':<13} {'pairs':>5} {'adj_same':>9} {'%sep>0':>7} {'mean_nfrag':>10} {'mean_sep':>9}")
    for op, d in sorted(by_op.items(), key=lambda kv: -kv[1]["pairs"]):
        pct = 100.0 * d["sep_nonzero"] / max(d["pairs"], 1)
        mnf = d["nfrag"] / max(d["pairs"], 1)
        msep = d["sep_total"] / max(d["pairs"], 1)
        print(f"{op:<13} {d['pairs']:>5} {d['adj_same']:>9} {pct:>6.0f}% {mnf:>10.1f} {msep:>9.2f}")

    print("\n'adj_same' = child graph has the IDENTICAL bond set (cut partitions unchanged) -> MetFrag-lite sees ZERO difference")
    print("'mean_sep' = mean number of *separable* fragment masses (child-unique + parent-unique) out of ~mean_nfrag total")

    if example:
        s, child, op, rep = example
        fp, fc = ion_neutral_set(s), ion_neutral_set(child)
        cu = [round(float(x), 3) for x in fc if len(fp) == 0 or np.min(np.abs(fp - x)) > TOL + 1e-9]
        pu = [round(float(x), 3) for x in fp if len(fc) == 0 or np.min(np.abs(fc - x)) > TOL + 1e-9]
        print(f"\nExample ({op}):")
        print(f"  parent {s}")
        print(f"  child  {child}")
        print(f"  formula parent={formula(s)} child={formula(child)}")
        print(f"  child-unique ion masses (Da): {cu[:15]}")
        print(f"  parent-unique ion masses (Da): {pu[:15]}")


def formula(s):
    m = Chem.MolFromSmiles(s)
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return CalcMolFormula(m) if m else "?"


if __name__ == "__main__":
    main()
