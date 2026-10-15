"""Read-only reach diagnostic for `formula-window` (campaign 20261013, E).

E's claim: a truth sitting in the WRONG formula group can never be rescued by any reranker, so a better
formula assignment raises recall upstream of everything tried so far.

Testable form on the val dumps: for each molecule, does ANY candidate the pipeline already retrieved carry
the truth's molecular formula? If not, the loss is a formula-window / pool loss and E could act on it.
If yes, the truth's formula group WAS present and E does nothing for that molecule (it is a ranking loss).
"""
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

REPO = Path(r"E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H  # noqa: E401

MASS_TOL = 0.008   # nominal: same-mass formula groups collide inside the ppm window anyway


def formula_of(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return CalcMolFormula(m) if m else None


def run(shard):
    state, deps, cp = H.load_dump(shard)
    T = H.truths()
    mids = [m for m in state["te_mids"] if m in T and T[m].get("cohort") == shard]
    tf = deps["form"]           # the dump's own formula fn, same one the pipeline groups by
    tot = win_lost = pool_lost = rank_loss = 0
    examples = []
    for mid in mids:
        truth_smi = T[mid]["smiles"]
        tk = cp.canonicalize_smiles_to_ik14(truth_smi)
        tform = formula_of(truth_smi)
        if tform is None:
            continue
        tot += 1
        prim, keys = state["base"][mid][0], None
        e = (state.get("eng") or {}).get(mid) or {}
        pc = (state.get("pc") or {}).get(mid) or {}
        cands = list(prim) + list(e.get("smiles") or []) + list(pc.get("pc") or [])
        # did the truth make it into the retrieved lists at all?
        got = any(cp.canonicalize_smiles_to_ik14(s) == tk for s in prim)
        if got:
            rank_loss += 1
            continue
        # otherwise: was its FORMULA represented among the retrieved candidates?
        has_form = any(tf(s) == tform for s in cands if s)
        if has_form:
            win_lost += 1        # right formula retrieved, truth's own structure not -> grouping/ranking inside group
            examples.append((mid, tform, "formula present, truth absent"))
        else:
            pool_lost += 1       # no candidate with that formula anywhere -> window or pool miss: E's real target
            examples.append((mid, tform, "no candidate with the truth formula"))
    print(f"\n=== {shard}: {tot} molecules scored")
    print(f"  truth already in the primary list (pure ranking loss, E cannot help): {rank_loss} "
          f"({100*rank_loss/max(tot,1):.1f}%)")
    print(f"  truth absent but its FORMULA was retrieved (in-group loss): {win_lost} ({100*win_lost/max(tot,1):.1f}%)")
    print(f"  NO candidate carries the truth formula (window/pool miss = E's target): {pool_lost} "
          f"({100*pool_lost/max(tot,1):.1f}%)")
    for mid, f, why in examples[:8]:
        print(f"    {mid}  {f:<16} {why}")


if __name__ == "__main__":
    for shard in ("C2", "C1_npx"):
        try:
            run(shard)
        except Exception as exc:
            print(f"\n=== {shard}: FAILED {type(exc).__name__}: {exc}")
