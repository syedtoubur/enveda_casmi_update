"""Candidate pool: structures sorted by neutral exact mass, one row per ik14."""
import json
import os
import pickle
import time
from multiprocessing import Pool as MPool

import numpy as np
import pyarrow.parquet as pq


class CandidatePool:
    def __init__(self, mass, keys, smiles, flags=None, npc=None):
        o = np.argsort(np.asarray(mass, np.float64), kind="stable")
        self.mass = np.asarray(mass, np.float64)[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.asarray(smiles, dtype=object)[o]
        self.flags = None if flags is None else np.asarray(flags)[o]   # provenance bits (R1)
        # NPClassifier superclasses (R2): (codes (n, 3) int16, -1 = none; probs (n, 3); names)
        self.npc = None if npc is None else (np.asarray(npc[0])[o], np.asarray(npc[1])[o], list(npc[2]))

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
                             None if self.flags is None else self.flags[keep],
                             None if self.npc is None else (self.npc[0][keep], self.npc[1][keep], self.npc[2]))


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


# ---------------------------------------------------------------- R1: the published unified pool
# provenance bits of pool_flags.npy (casmi26-scratch-pool); pool_build writes them
F_TRAIN, F_E180, F_TRAIN_OTHER, F_COCO, F_V2, F_BIO, F_NPT, F_LOTUS, F_NPATLAS = (1 << i for i in range(9))
FLAG_NAMES = dict(train=F_TRAIN, e180=F_E180, train_other=F_TRAIN_OTHER, coco=F_COCO, v2=F_V2, bio=F_BIO,
                  np_table=F_NPT, lotus=F_LOTUS, npatlas=F_NPATLAS)
NP_SOURCES = F_TRAIN_OTHER | F_COCO | F_BIO | F_NPT


def e180_only(flags):
    """Structures whose only provenance is enveda-180 (synthetic): no other train library, COCONUT, BIO or NP table.

    v2-pool membership does not count: v2 = train + COCONUT, so every enveda-180 structure is in it.
    """
    flags = np.asarray(flags, np.int64)
    return ((flags & F_E180) != 0) & ((flags & NP_SOURCES) == 0)


def load_pool(pool_dir, drop_e180=False, with_npc=False):
    """casmi26-scratch-pool -> (CandidatePool keyed on scorer_key, info dict). with_npc: attach pool_npc.parquet."""
    t0 = time.time()
    mass = np.load(os.path.join(pool_dir, "pool_mass.npy"))
    keys = np.load(os.path.join(pool_dir, "pool_key.npy")).astype(str).astype(object)
    flags = np.load(os.path.join(pool_dir, "pool_flags.npy"))
    smiles = pq.read_table(os.path.join(pool_dir, "pool_smiles.parquet")).column("smiles").to_numpy(
        zero_copy_only=False).astype(object)
    assert len(mass) == len(keys) == len(flags) == len(smiles) and np.all(np.diff(mass) >= 0), "pool arrays misaligned"
    drop = e180_only(flags) if drop_e180 else np.zeros(len(mass), bool)
    npc = None
    if with_npc:
        from .npc import load_npc
        sc, sp, names = load_npc(pool_dir)
        npc = (sc[~drop], sp[~drop], names)
    pool = CandidatePool(mass[~drop], keys[~drop], smiles[~drop], flags[~drop], npc)
    man = os.path.join(pool_dir, "MANIFEST.json")
    info = dict(pool_size=len(pool), dropped_e180=int(drop.sum()),
                pool_version=json.load(open(man)).get("version") if os.path.exists(man) else None)
    print(f"[pool] R1 {len(pool):,} structures (dropped enveda-180-only {info['dropped_e180']:,}) "
          f"({time.time() - t0:.1f}s)", flush=True)
    return pool, info


def load_keymap(pool_dir):
    """train inchikey14 -> scorer_key (the R1 pool's key), for remapping the library."""
    t = pq.read_table(os.path.join(pool_dir, "train_keymap.parquet")).to_pandas()
    return dict(zip(t.inchikey14, t.scorer_key))


def load_np_labels(pool_dir):
    """scorer_key-indexed NP side table (labels, label_source, knn_sim, metadata, np_score)."""
    return pq.read_table(os.path.join(pool_dir, "pool_np.parquet")).to_pandas().set_index("key")


def npc_top1(pool_dir, keys):
    """scorer_key -> top-1 NPClassifier superclass, for the given keys (the library's: the vote side). Uses the whole
    pool file, so library structures dropped from R1b keep their label."""
    from .npc import load_npc
    sc, _, names = load_npc(pool_dir)
    allk = np.load(os.path.join(pool_dir, "pool_key.npy")).astype(str)
    want = set(map(str, keys))
    return {k: names[c] for k, c in zip(allk, sc[:, 0]) if c >= 0 and k in want}
