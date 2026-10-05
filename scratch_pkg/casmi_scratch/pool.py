"""Candidate pool: structures sorted by neutral exact mass, one row per ik14."""
import os
import pickle
import time
from multiprocessing import Pool as MPool

import numpy as np
import pyarrow.parquet as pq


class CandidatePool:
    def __init__(self, mass, keys, smiles, flags=None):
        o = np.argsort(np.asarray(mass, np.float64), kind="stable")
        self.mass = np.asarray(mass, np.float64)[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.asarray(smiles, dtype=object)[o]
        self.flags = None if flags is None else np.asarray(flags)[o]   # provenance bits (R1)

    def __len__(self):
        return len(self.mass)

    def bounds(self, target, ppm):
        """(lo, hi) index range within +-ppm of target."""
        return (int(np.searchsorted(self.mass, target * (1 - ppm / 1e6), "left")),
                int(np.searchsorted(self.mass, target * (1 + ppm / 1e6), "right")))

    def window(self, target, ppm):
        return np.arange(*self.bounds(target, ppm))

    def without(self, keys):
        """A copy without the given ik14 (validation's Class-3 arm)."""
        keep = ~np.isin(self.keys, np.asarray(list(keys), dtype=object))
        return CandidatePool(self.mass[keep], self.keys[keep], self.smiles[keep],
                             None if self.flags is None else self.flags[keep])


def _exact_mass(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    return float(ExactMolWt(m)) if m is not None else np.nan


def build_r0_pool(coco_dir, train_path, workers=4):
    """COCONUT + train structures (the Track 3 base pool, without the unused fingerprints).

    COCONUT rows are deduplicated on ik14 (first occurrence) with their precomputed mass; train structures not in
    COCONUT get RDKit's exact mass from their SMILES; rows without a finite mass are dropped.
    """
    t0 = time.time()
    cm = pickle.load(open(os.path.join(coco_dir, "coco_meta.pkl"), "rb"))
    co_mass = np.load(os.path.join(coco_dir, "coco_mass.npy")).astype(np.float64)
    co_k14 = np.array([str(k)[:14] for k in cm["keys"]], dtype=object)
    _, first = np.unique(co_k14, return_index=True)
    first = np.sort(first)
    co_k14, co_smi, co_mass = co_k14[first], np.asarray(cm["smiles"], dtype=object)[first], co_mass[first]

    tr = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    tr = tr.dropna().drop_duplicates("inchikey14")
    tr = tr[~tr.inchikey14.isin(set(co_k14))]
    smis = list(tr.normalized_smiles)
    if workers > 1:
        with MPool(workers) as mp:
            tr_mass = np.array(mp.map(_exact_mass, smis, chunksize=500), np.float64)
    else:
        tr_mass = np.array([_exact_mass(s) for s in smis], np.float64)

    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_k14, tr.inchikey14.to_numpy(dtype=object)])
    smiles = np.concatenate([co_smi, tr.normalized_smiles.to_numpy(dtype=object)])
    good = np.isfinite(mass)
    pool = CandidatePool(mass[good], keys[good], smiles[good])
    print(f"[pool] COCONUT {len(co_k14):,} + train-only {len(tr):,} -> {len(pool):,} structures "
          f"({time.time() - t0:.1f}s)", flush=True)
    return pool
