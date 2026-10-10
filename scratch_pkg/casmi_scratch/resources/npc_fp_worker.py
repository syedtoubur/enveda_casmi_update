"""NPClassifier fingerprints under the RDKit the network was trained with.

RDKit changed the Morgan bit assignment between 2021.03.5 and 2021.09.5; NP-Classifier's V1 weights expect the old
bits (R2.0, gate_report). So this runs in its own env (never in the notebook's interpreter, which keeps RDKit
2026.03.3): Python 3.9 + rdkit-pypi 2021.3.5 + numpy, started by `casmi_scratch.npc.fp_subprocess`.
Standalone: numpy + rdkit only (plus the repo's fingerprint_handler when --ref is given).

    in.txt  : one SMILES per line
    out.npz : F (n, 2048) int16 radius-0 counts, B (n, 4096) int16 radius-1/2 counts, ok (n,) bool, rdkit str,
              parity int (rows where the repo's calculate_fingerprint gives the same arrays; -1 without --ref)

    python npc_fp_worker.py --inp in.txt --out out.npz --workers 8 [--ref DIR_WITH_fingerprint_handler] [--any-rdkit]
"""
import argparse
import sys
from multiprocessing import Pool

import numpy as np

OLD_MAX = (2021, 3)            # last RDKit release with the bits NP-Classifier V1 was trained on


def fingerprint(smiles, radi=2):
    """NP-Classifier's calculate_fingerprint arithmetic -> (formula (2048,), binary (4096,)) or None."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    formula, binary = np.zeros(2048, np.int16), np.zeros(2048 * radi, np.int16)
    for r in range(radi + 1):
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=bi, nBits=2048)
        for i in fp.GetOnBits():
            n = sum(1 for e in bi[i] if e[1] == r)
            if n == 0:
                continue
            if r == 0:
                formula[i] = n
            else:
                binary[2048 * (r - 1) + i] = n
    return formula, binary


def block(smiles):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    n = len(smiles)
    F, B, ok = np.zeros((n, 2048), np.int16), np.zeros((n, 4096), np.int16), np.zeros(n, bool)
    for j, s in enumerate(smiles):
        fp = fingerprint(s)
        if fp is not None:
            F[j], B[j], ok[j] = fp[0], fp[1], True
    return F, B, ok


def parity(smiles, F, B, ok, ref_dir):
    sys.path.insert(0, ref_dir)
    import fingerprint_handler as REF          # the repo's own code
    same = 0
    for j, s in enumerate(smiles):
        if ok[j]:
            a, b = REF.calculate_fingerprint(s, 2)
            same += bool(np.array_equal(a[0], F[j]) and np.array_equal(b[0], B[j]))
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--any-rdkit", action="store_true", help="tests only: skip the RDKit version check")
    a = ap.parse_args()

    import rdkit
    ver = tuple(int(x) for x in rdkit.__version__.split(".")[:2])
    if ver > OLD_MAX and not a.any_rdkit:
        sys.exit(f"npc_fp_worker: RDKit {rdkit.__version__} gives the post-2021.09 Morgan bits; need <= 2021.03")
    with open(a.inp, encoding="utf-8") as fh:
        smiles = fh.read().split("\n")
    if smiles and smiles[-1] == "":
        smiles = smiles[:-1]
    parts = [smiles[i::max(1, a.workers)] for i in range(max(1, a.workers))]
    if a.workers > 1 and len(smiles) > 1000:
        with Pool(a.workers) as p:
            res = p.map(block, parts)
    else:
        res = [block(x) for x in parts]
    n = len(smiles)
    F, B, ok = np.zeros((n, 2048), np.int16), np.zeros((n, 4096), np.int16), np.zeros(n, bool)
    for i, (f, b, o) in enumerate(res):                    # undo the strided split
        F[i::len(parts)], B[i::len(parts)], ok[i::len(parts)] = f, b, o
    par = parity(smiles, F, B, ok, a.ref) if a.ref else -1
    np.savez(a.out, F=F, B=B, ok=ok, rdkit=rdkit.__version__, parity=par)


if __name__ == "__main__":
    main()
