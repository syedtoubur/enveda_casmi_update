"""NP labels for every pool row: exact NP-table match, else the nearest labelled NP-table structure (kNN), plus Ertl
NP-likeness. Ported from build/np_signals_core.py (Track 4b: morgan_fp, knn_label, np_scores).

label_source: 0 none (neutral), 1 table (exact scorer_key match), 2 knn (Morgan r2 2048, Tanimoto >= thr).
"""
import os
import sys

import numpy as np
import pandas as pd

from .pool_build import run_chunked

LABELS = ["np_pathway", "np_superclass", "np_class"]
KNN_THR = 0.7
_G = {}


def morgan_fp(smiles, nbits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = _G.setdefault("gen", rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits))
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return None if m is None else gen.GetFingerprint(m)


def _init_bank(bank_smiles):
    fps = [morgan_fp(s) for s in bank_smiles]
    _G["bank_idx"] = np.array([i for i, f in enumerate(fps) if f is not None], np.int64)
    _G["bank"] = [f for f in fps if f is not None]


def _knn_chunk(smis):
    """-> [(bank row index or -1, best Tanimoto)] against the bank set up by _init_bank."""
    from rdkit import DataStructs
    out = []
    for s in smis:
        fp = morgan_fp(s)
        if fp is None or not _G["bank"]:
            out.append((-1, 0.0))
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, _G["bank"]))
        j = int(sims.argmax())
        out.append((int(_G["bank_idx"][j]), float(sims[j])))
    return out


def knn_nearest(query_smiles, bank_smiles, ckpt_dir, chunk=20_000, workers=8, log=print):
    """Nearest bank structure per query (checkpointed) -> DataFrame(bank_j, sim) in query order."""
    return run_chunked(list(query_smiles), _knn_chunk, ckpt_dir, "knn", ["bank_j", "sim"], chunk, workers,
                       initializer=_init_bank, initargs=(list(bank_smiles),), log=log)


def assign_labels(keys, smiles, table, nearest=None, thr=KNN_THR):
    """Pool keys -> label frame (index key): LABELS, label_source, knn_sim.

    table: NP-table metadata indexed by recomputed key (LABELS + extra columns). nearest: optional function
    (query smiles list, bank smiles list) -> DataFrame(bank_j, sim); without it, keys outside the table stay unlabelled.
    """
    keys = pd.Index(np.asarray(keys, dtype=object), name="key")
    out = pd.DataFrame(index=keys, columns=LABELS, dtype=object)
    out["label_source"] = np.int8(0)
    out["knn_sim"] = np.float32(np.nan)
    hit = keys.isin(table.index)
    t = table.reindex(keys[hit])
    has = t.np_superclass.notna().to_numpy() | t.np_pathway.notna().to_numpy()
    out.loc[keys[hit][has], LABELS] = t.loc[has, LABELS].to_numpy()
    out.loc[keys[hit][has], "label_source"] = np.int8(1)

    bank = table[table.np_superclass.notna() | table.np_pathway.notna()]
    need = out.label_source.to_numpy() == 0
    if nearest is not None and need.any() and len(bank):
        q = np.asarray(smiles, dtype=object)[need]
        nn = nearest(list(q), list(bank["smiles"]))
        j, s = nn.bank_j.to_numpy(np.int64), nn.sim.to_numpy(np.float64)
        ok = (j >= 0) & (s >= thr)
        idx = keys[need]
        out.loc[idx[ok], LABELS] = bank.iloc[j[ok]][LABELS].to_numpy()
        out.loc[idx[ok], "label_source"] = np.int8(2)
        out.loc[idx, "knn_sim"] = np.where(j >= 0, s, np.nan).astype(np.float32)
    return out


# ---------------------------------------------------------------- NP-likeness (Ertl, RDKit Contrib)

def _np_model():
    if "np_m" not in _G:
        from rdkit.Chem import RDConfig
        p = os.path.join(RDConfig.RDContribDir, "NP_Score")
        if p not in sys.path:
            sys.path.append(p)
        import npscorer
        _G["np_f"], _G["np_m"] = npscorer, npscorer.readNPModel()
    return _G["np_f"], _G["np_m"]


def _np_chunk(smis):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    f, m = _np_model()
    out = []
    for s in smis:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        v = np.nan
        if mol is not None:
            try:
                v = float(f.scoreMol(mol, m))
            except Exception:
                pass
        out.append((v,))
    return out


def np_scores(smiles, ckpt_dir, chunk=50_000, workers=8, log=print):
    """Ertl NP-likeness per SMILES (NaN where RDKit cannot parse), checkpointed -> float32 array."""
    return run_chunked(list(smiles), _np_chunk, ckpt_dir, "npscore", ["np_score"], chunk, workers,
                       log=log).np_score.to_numpy(np.float32)
