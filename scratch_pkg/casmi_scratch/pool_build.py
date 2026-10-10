"""R1 unified candidate pool: v2-pool ∪ R0 (COCONUT + train) ∪ BIO (ChEBI / LIPID MAPS) ∪ NP knowledge table.

Every row is keyed on the scorer_key (InChIKey14 of the RDKit 2026.03.3 canonical tautomer, as the competition
scorer does it). Duplicates collapse to one row: the SMILES of the highest-priority source wins (train > v2-pool >
COCONUT > BIO > NP table), its mass is recomputed with ExactMolWt, and the provenance bits are OR'd.
Rows RDKit cannot parse or key are dropped; everything else is kept verbatim (no salt stripping).

Runs on Colab (`drivers/r1_pool_colab.py`); the RDKit work is chunked and checkpointed so a restart resumes.
Spec: specs/specs_scratch/r1_pool_plan.md.
"""
import hashlib
import json
import os
import pickle
import time
from multiprocessing import Pool as MPool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .pool import (F_BIO, F_COCO, F_E180, F_LOTUS, F_NPATLAS, F_NPT, F_TRAIN, F_TRAIN_OTHER, F_V2, FLAG_NAMES,
                   e180_only)

E180_LIB = "enveda-180"
RANK = dict(train=0, v2=1, coco=2, bio=3, np_table=4)        # lower wins the SMILES of a duplicated key
NP_META = ["np_pathway", "np_superclass", "np_class", "kingdoms", "n_organisms"]
ROW_COLS = ["smiles", "src_ik", "flags", "rank", "source"]


def _rows(smiles, src_ik, flags, source):
    n = len(smiles)
    return pd.DataFrame({"smiles": np.asarray(smiles, dtype=object),
                         "src_ik": np.asarray(src_ik, dtype=object) if src_ik is not None else np.full(n, None, object),
                         "flags": np.broadcast_to(np.asarray(flags, np.int64), (n,)).copy(),
                         "rank": RANK[source], "source": source})


# ---------------------------------------------------------------- sources -> rows (smiles, src_ik, flags, rank)

def train_rows(train_path, e180=E180_LIB):
    """One row per train inchikey14 (first normalized_smiles, as R0), flagged by the libraries it has spectra in."""
    t = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles", "ingest_lib"]).to_pandas()
    t = t.dropna(subset=["inchikey14", "normalized_smiles"])
    lib = t.ingest_lib.astype(str).to_numpy()
    t = t.assign(e=lib == e180, o=lib != e180)
    g = t.groupby("inchikey14", sort=False)
    first = g.normalized_smiles.first()
    anyf = g[["e", "o"]].any().loc[first.index]
    flags = F_TRAIN | np.where(anyf.e, F_E180, 0) | np.where(anyf.o, F_TRAIN_OTHER, 0)
    return _rows(first.to_numpy(dtype=object), first.index.to_numpy(dtype=object), flags, "train")


def v2_rows(pool_meta_path):
    """casmi26-v2-pool pool_meta.parquet (train + COCONUT, 710,701)."""
    cols = pq.read_schema(pool_meta_path).names
    smi = next((c for c in ("smiles", "SMILES", "canonical_smiles", "smi") if c in cols), None)
    assert smi is not None, f"no SMILES column in {pool_meta_path}: {cols}"
    t = pq.read_table(pool_meta_path, columns=[smi] + (["key"] if "key" in cols else [])).to_pandas()
    return _rows(t[smi].to_numpy(dtype=object), t["key"].astype(str).str[:14].to_numpy(dtype=object)
                 if "key" in cols else None, F_V2, "v2")


def coco_rows(coco_dir):
    """The phase1 COCONUT arrays R0 uses (coco_meta.pkl keys / smiles)."""
    cm = pickle.load(open(os.path.join(coco_dir, "coco_meta.pkl"), "rb"))
    return _rows(np.asarray(cm["smiles"], dtype=object), np.array([str(k)[:14] for k in cm["keys"]], dtype=object),
                 F_COCO, "coco")


def bio_rows(bio_dir):
    """chebi-lipidmaps-casmi26 bio_meta.pkl (keys / smiles)."""
    bm = pickle.load(open(os.path.join(bio_dir, "bio_meta.pkl"), "rb"))
    return _rows(np.asarray(bm["smiles"], dtype=object), np.array([str(k)[:14] for k in bm["keys"]], dtype=object),
                 F_BIO, "bio")


def np_table_rows(np_path):
    """NP knowledge table rows -> (rows, metadata frame aligned to rows by position)."""
    t = pq.read_table(np_path).to_pandas()
    flags = F_NPT | np.where(t.in_lotus.astype(bool), F_LOTUS, 0) | np.where(t.in_npatlas.astype(bool), F_NPATLAS, 0)
    rows = _rows(t.smiles.to_numpy(dtype=object), t.scorer_key.to_numpy(dtype=object), flags, "np_table")
    return rows, t[NP_META].reset_index(drop=True)


# ---------------------------------------------------------------- RDKit: smiles -> key, mass, hygiene

_G = {}


def smiles_props(smi):
    """-> (scorer_key or None, exact mass or NaN, n_fragments, charged)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _G:
        _G["te"] = rdMolStandardize.TautomerEnumerator()
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None, np.nan, 0, False
    try:
        key = Chem.MolToInchiKey(_G["te"].Canonicalize(m))[:14] or None
    except Exception:
        key = None
    return (key, float(ExactMolWt(m)), len(Chem.GetMolFrags(m)),
            any(a.GetFormalCharge() != 0 for a in m.GetAtoms()))


def _props_chunk(smis):
    return [smiles_props(s) for s in smis]


def run_chunked(items, fn, ckpt_dir, name, columns, chunk=50_000, workers=8, initializer=None, initargs=(),
                log=print):
    """fn(list of items) -> list of tuples, run over fixed-size chunks with one checkpoint parquet per chunk.

    Chunk files carry a hash of the items and the chunk size, so a rerun with the same items resumes where it
    stopped and a changed input never mixes with stale chunks. Returns the concatenated frame, in item order.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    items = list(items)
    n_chunks = (len(items) + chunk - 1) // chunk
    tag = hashlib.sha1((chr(10).join(map(str, items)) + f"|{chunk}").encode()).hexdigest()[:10]
    parts, t0, mp = [], time.time(), None
    try:
        for i in range(n_chunks):
            path = os.path.join(ckpt_dir, f"{name}_{tag}_{i:05d}_{n_chunks:05d}.parquet")
            if os.path.exists(path):
                parts.append(pd.read_parquet(path))
                continue
            s = items[i * chunk:(i + 1) * chunk]
            if workers > 1:
                if mp is None:
                    mp = MPool(workers, initializer=initializer, initargs=initargs)
                res = [r for part in mp.map(fn, [list(x) for x in np.array_split(np.asarray(s, dtype=object),
                                                                                  workers * 8) if len(x)])
                       for r in part]
            else:
                if initializer is not None and not _G.get("_init_" + name):
                    initializer(*initargs)
                    _G["_init_" + name] = True
                res = fn(s)
            df = pd.DataFrame(res, columns=columns)
            df.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            parts.append(df)
            log(f"  {name} chunk {i + 1}/{n_chunks} | {time.time() - t0:.0f}s")
    finally:
        if mp is not None:
            mp.close()
            mp.join()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def compute_props(smiles, ckpt_dir, chunk=50_000, workers=8, props_fn=_props_chunk, log=print):
    """Unique SMILES -> DataFrame(index smiles; key, mass, nfrag, charged), checkpointed per chunk."""
    u = sorted({s for s in smiles if isinstance(s, str)})
    df = run_chunked(u, props_fn, ckpt_dir, "props", ["key", "mass", "nfrag", "charged"], chunk, workers, log=log)
    df.index = pd.Index(u, name="smiles")
    return df


# ---------------------------------------------------------------- merge

def merge_rows(rows, props):
    """Rows of every source + their props -> (pool frame, stats).

    Pool frame: key, smiles, mass, flags, nfrag, charged, source (winning source), one row per key, sorted by mass.
    """
    r = rows.join(props, on="smiles")
    good = r.key.notna() & np.isfinite(r.mass.astype(np.float64)) & (r.key.astype(str).str.len() == 14)
    stats = dict(rows_in=r.groupby("source").size().to_dict(),
                 dropped_unkeyable=r[~good].groupby("source").size().to_dict())
    r = r[good].copy()
    has_src = r.src_ik.notna()
    stats["key_disagree"] = (r[has_src].src_ik.astype(str).str[:14] != r[has_src].key).groupby(
        r[has_src].source).sum().astype(int).to_dict()
    r["_pos"] = np.arange(len(r))
    r = r.sort_values(["key", "rank", "_pos"], kind="stable")
    k = r.key.to_numpy(dtype=object)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    flags = np.bitwise_or.reduceat(r["flags"].to_numpy(np.int64), starts)
    win = r.iloc[starts]
    out = pd.DataFrame({"key": win.key.to_numpy(dtype=object), "smiles": win.smiles.to_numpy(dtype=object),
                        "mass": win.mass.to_numpy(np.float64), "flags": flags.astype(np.uint16),
                        "nfrag": win.nfrag.to_numpy(np.int16), "charged": win.charged.to_numpy(bool),
                        "source": win.source.to_numpy(dtype=object)})
    out = out.sort_values(["mass", "key"], kind="stable").reset_index(drop=True)
    stats.update(pool_size=len(out), multi_fragment=int((out.nfrag > 1).sum()), charged=int(out.charged.sum()),
                 per_flag={n: int(((out["flags"].to_numpy(np.int64) & b) != 0).sum()) for n, b in FLAG_NAMES.items()},
                 winner_source=out.source.value_counts().to_dict(),
                 e180_only=int(e180_only(out["flags"]).sum()))
    return out, stats


def keymap_from_rows(rows, props):
    """train inchikey14 -> scorer_key (from the train rows' SMILES)."""
    t = rows[rows.source == "train"].join(props, on="smiles")
    t = t[t.key.notna()]
    return pd.DataFrame({"inchikey14": t.src_ik.astype(str).to_numpy(), "scorer_key": t.key.astype(str).to_numpy()})


def np_meta_by_key(np_rows, np_meta, props):
    """NP-table metadata per recomputed key (first table row per key) + parity of the table's own scorer_key."""
    k = np_rows.smiles.map(props.key)
    ok = k.notna().to_numpy()
    meta = np_meta[ok].assign(key=k[ok].to_numpy(), smiles=np_rows.smiles[ok].to_numpy())
    parity = dict(n=int(ok.sum()), table_key_differs=int((np_rows.src_ik[ok].to_numpy() != k[ok].to_numpy()).sum()))
    return meta.drop_duplicates("key").set_index("key"), parity


def superset_check(r0_smiles, props, r1_keys):
    """R1.1: R0 pool rows whose structure is not in R1 (expected: none).

    An R0 row is lost if RDKit parses it (R0 had a mass for it) but gives no scorer key, or if its key is not in R1.
    """
    p = props.reindex(pd.Index(np.asarray(r0_smiles, dtype=object)))
    parsed = np.isfinite(p.mass.to_numpy(np.float64))
    keyed = p.key.notna().to_numpy()
    lost_unkeyable = parsed & ~keyed
    miss = sorted(set(p.key[keyed]) - set(r1_keys))
    return dict(r0_rows=int(len(p)), r0_parsed=int(parsed.sum()), r0_keys=int(p.key[keyed].nunique()),
                lost_unkeyable=int(lost_unkeyable.sum()), missing_keys=len(miss),
                missing=int(lost_unkeyable.sum()) + len(miss),
                examples=list(p.index[lost_unkeyable][:5]) + miss[:5])


# ---------------------------------------------------------------- write

def _sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_pool(pool, np_side, keymap, out_dir, manifest):
    """pool frame (mass-sorted) + NP side table (key-indexed) + keymap -> casmi26-scratch-pool files + MANIFEST."""
    os.makedirs(out_dir, exist_ok=True)
    assert np.all(np.diff(pool.mass.to_numpy()) >= 0) and not pool.key.duplicated().any()
    np.save(os.path.join(out_dir, "pool_mass.npy"), pool.mass.to_numpy(np.float64))
    np.save(os.path.join(out_dir, "pool_key.npy"), pool.key.to_numpy().astype("S14"))
    np.save(os.path.join(out_dir, "pool_flags.npy"), pool["flags"].to_numpy(np.uint16))
    pq.write_table(pa.table({"smiles": pool.smiles.astype(str).to_numpy()}), os.path.join(out_dir, "pool_smiles.parquet"))
    side = np_side.reindex(pool.key.to_numpy())
    side.index.name = "key"
    pq.write_table(pa.Table.from_pandas(side.reset_index(), preserve_index=False),
                   os.path.join(out_dir, "pool_np.parquet"))
    pq.write_table(pa.Table.from_pandas(keymap.drop_duplicates("inchikey14"), preserve_index=False),
                   os.path.join(out_dir, "train_keymap.parquet"))
    files = sorted(f for f in os.listdir(out_dir) if f != "MANIFEST.json")
    manifest = dict(manifest, files={f: dict(bytes=os.path.getsize(os.path.join(out_dir, f)),
                                             sha1=_sha1(os.path.join(out_dir, f))) for f in files},
                    flag_bits={n: int(b) for n, b in FLAG_NAMES.items()})
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


# ---------------------------------------------------------------- the whole build (drivers/r1_pool_colab.py)

def build(inputs, out_dir, ckpt_dir, version, workers=8, log=print):
    """inputs: dict(train=train.parquet, v2=pool_meta.parquet, coco=dir, bio=dir, np_table=natural_products.parquet).

    Every RDKit stage is checkpointed under ckpt_dir; re-running resumes. -> MANIFEST dict.
    """
    from rdkit import __version__ as rdkit_version

    from . import np_labels
    t0 = time.time()
    os.makedirs(ckpt_dir, exist_ok=True)
    rows_path, meta_path = os.path.join(ckpt_dir, "rows.parquet"), os.path.join(ckpt_dir, "np_meta_rows.parquet")
    if os.path.exists(rows_path) and os.path.exists(meta_path):
        rows, np_meta = pd.read_parquet(rows_path), pd.read_parquet(meta_path)
    else:
        npr, np_meta = np_table_rows(inputs["np_table"])
        rows = pd.concat([train_rows(inputs["train"]), v2_rows(inputs["v2"]), coco_rows(inputs["coco"]),
                          bio_rows(inputs["bio"]), npr], ignore_index=True)
        rows.to_parquet(rows_path, index=False)
        np_meta.to_parquet(meta_path, index=False)
    log(f"[1 rows] {rows.groupby('source').size().to_dict()} | {rows.smiles.nunique():,} unique SMILES "
        f"| {time.time() - t0:.0f}s")

    props = compute_props(rows.smiles, os.path.join(ckpt_dir, "props"), workers=workers, log=log)
    log(f"[2 keys] {len(props):,} SMILES, unkeyable {int(props.key.isna().sum()):,} | {time.time() - t0:.0f}s")

    pool, stats = merge_rows(rows, props)
    keymap = keymap_from_rows(rows, props)
    np_tab, parity = np_meta_by_key(rows[rows.source == "np_table"].reset_index(drop=True), np_meta, props)
    r1_1 = superset_check(rows.smiles[rows.source.isin(["coco", "train"])], props, pool.key)
    log(f"[3 merge] {json.dumps(dict(stats, np_parity=parity, R1_1=r1_1), default=str)}")
    log(f"R1.1 no loss: {'PASS' if r1_1['missing'] == 0 else 'FAIL'} ({r1_1['missing']} R0 rows lost)")

    labels = np_labels.assign_labels(
        pool.key, pool.smiles, np_tab,
        nearest=lambda q, b: np_labels.knn_nearest(q, b, os.path.join(ckpt_dir, "knn"), workers=workers, log=log))
    side = labels.join(np_tab[["kingdoms", "n_organisms"]], how="left")
    side["np_score"] = np_labels.np_scores(pool.smiles, os.path.join(ckpt_dir, "npscore"), workers=workers,
                                           log=log).astype(np.float16)
    side["knn_sim"] = side.knn_sim.astype(np.float16)
    lstats = dict(label_source={int(k): int(v) for k, v in side.label_source.value_counts().sort_index().items()},
                  superclass_top=side.np_superclass.value_counts().head(10).to_dict(),
                  np_score_nan=int(side.np_score.isna().sum()))
    log(f"[4 labels] {json.dumps(lstats, default=str)} | {time.time() - t0:.0f}s")

    man = write_pool(pool, side, keymap, out_dir, dict(
        version=version, built=time.strftime("%Y-%m-%d %H:%M"), rdkit=rdkit_version,
        inputs={k: str(v) for k, v in inputs.items()}, stats=stats, np_parity=parity, R1_1=r1_1, labels=lstats,
        build_s=round(time.time() - t0, 1)))
    log(f"[5 write] {out_dir}: {len(pool):,} structures | {time.time() - t0:.0f}s")
    return man
