"""Second probe: isomer separability in the ACTUAL ion space MetFrag matches in
(each fragment expanded +proton and +/-2H), vs neutral-mass space."""
import importlib.util
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "scratch_pkg")
from casmi_scratch.metfrag import fragment_masses  # noqa: E402

RDLogger.DisableLog("rdApp.*")
spec = importlib.util.spec_from_loader(
    "isomut", importlib.machinery.SourceFileLoader(
        "isomut", "research/c3/20261017-isomer-mut-ice/isomer_mut_base.py.orig"))
im = importlib.util.module_from_spec(spec)
spec.loader.exec_module(im)

H, PROTON, TOL = 1.00782503207, 1.00782503207 - 0.00054858, 0.01


def bucket(vals):
    return np.array(sorted(set(np.round(np.asarray(vals) / TOL).astype(int))), dtype=object)


def neutral(smi):
    return bucket(fragment_masses(smi))


def ionize(smi):
    fr = fragment_masses(smi)
    ions = [m + PROTON + dh * H for m in fr for dh in (-2, -1, 0, 1, 2)]
    return bucket(ions)


def sym_diff(a, b):
    """#items in a not within 1 bucket of b + vice versa (bucket units ~0.01 Da)."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    ai, bi = np.array(a), np.array(b)
    ub = sum(1 for x in bi if np.min(np.abs(ai - x)) > 1)
    ab = sum(1 for x in ai if np.min(np.abs(bi - x)) > 1)
    return ub + ab


df = pd.read_parquet("p1_novel/pseudo_c3.parquet")
chosen = []
for s in df["smiles"].dropna().tolist():
    m = Chem.MolFromSmiles(s)
    if m is None or m.GetNumAtoms() > 40:
        continue
    k = im.mutate(s, max_moves=1, cap=60)
    if k:
        chosen.append((s, k))
    if len(chosen) >= 25:
        break

by = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # pairs, nfrag_ion, sum_neutsep, sum_ionsep, ionsep>0, ionsep>=5
for s, ks in chosen:
    fp_neu, fp_ion = neutral(s), ionize(s)
    for child, op, nm in ks:
        fc_neu, fc_ion = neutral(child), ionize(child)
        base = op.split("+")[0]
        nsep = sym_diff(fp_neu, fc_neu)
        isep = sym_diff(fp_ion, fc_ion)
        d = by[base]
        d[0] += 1
        d[1] += len(fc_ion)
        d[2] += nsep
        d[3] += isep
        d[4] += 1 if isep > 0 else 0
        d[5] += 1 if isep >= 5 else 0

hdr = "operator       pairs  nIonFrags  neutral_sep  ION_sep  %IONsep>0  %IONsep>=5"
print(hdr)
for op, d in sorted(by.items(), key=lambda kv: -kv[1][0]):
    p = d[0]
    print("{:<14}{:>5}{:>11}{:>13}{:>9}{:>10}{:>11}".format(
        op, p, round(d[1] / p, 1), round(d[2] / p, 1), round(d[3] / p, 1),
        round(100 * d[4] / p), round(100 * d[5] / p)))
