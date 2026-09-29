"""specs_novel P3 training data: the structure pass, the training index and the epoch sampler.

Pre-registered in `specs_novel/gate_report.md` -> G3. Numpy / pandas only, no torch: the notebook
turns a batch into tensors. That keeps this module testable on its own.

    build_structures      every unique ik14 -> (status, can, selfies, formula); multiprocessing
    build_index           dec_split + corpus_meta + corpus_ce + structures -> a compact index
    EmbeddingStore        memory-maps `_novel/emb_dbec3a0_lx/emb_*.npy` and gathers rows
    make_batch            one batch: spectra sampled per molecule, with CE dropout

Three things here are easy to get wrong and are therefore spelled out:

1. **The oracle formula comes from the structure, not the label.** `prep_structure(can)` returns the
   RDKit formula with hydrogen included; 165 corpus structures disagree with `molecular_formula`
   (G1 finding 4). Only the structure formula is used, for the memory token, the search masks and
   the leaf check alike.
2. **Embedding row == corpus_meta row.** chunk = row // CHUNK, offset = row % CHUNK. Only
   `emb_dbec3a0_lx` is a valid source; `emb_dbec3a0` and `_dreams_out/emb` were built with the old
   `dreams_peaks` and must never be used for specs_novel (G0.1-R2).
3. **The training set is `dec_split == "train"`.** `dec_val` and every `heldout_*` label are the
   model's exclusion set, not merely an eval holdout.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

import novel_cohort as nc
import novel_decoder as nd

CHUNK = 200_000                      # rows per embedding chunk file
EMB_DIM = 1024
MAX_LEN = 171                        # P0.3, p99.9 of corpus SELFIES token lengths
MAX_SPEC_PER_MOL = 4                 # G3
CE_DROPOUT = 0.3                     # G3
ADDUCT_IX = {a: i for i, a in enumerate(nd.TEST_ADDUCTS)}
HILL = ("C", "H")                    # Hill order: C, H, then the rest alphabetically


def formula_str(counts):
    """{element: count} -> a canonical Hill-order string, so a formula round-trips through parquet."""
    rest = sorted(e for e in counts if e not in HILL and counts[e])
    out = []
    for e in list(HILL) + rest:
        c = int(counts.get(e, 0))
        if c:
            out.append(e if c == 1 else f"{e}{c}")
    return "".join(out)


# --------------------------------------------------------------------------------------------- #
# The structure pass
# --------------------------------------------------------------------------------------------- #

_VOCAB = None


def _init_worker(vocab_path):
    global _VOCAB
    _VOCAB = nd.SelfiesVocab.from_json(vocab_path)


def _structure(args):
    """(ik14, smiles) -> dict. `in_scope` means prep ok, every token in the vocabulary, and
    a length within MAX_LEN, i.e. the molecule is reachable by the decoder at all."""
    ik, smi = args
    status, can, selfies, formula = nd.prep_structure(smi)
    rec = dict(inchikey14=ik, status=status, can=can, selfies=selfies,
               formula=None, n_tokens=0, n_unk=0, in_scope=False)
    if status != "ok":
        return rec
    ids = _VOCAB.encode(selfies)
    rec["formula"] = formula_str(formula)
    rec["n_tokens"] = len(ids)
    rec["n_unk"] = sum(1 for i in ids if i == _VOCAB.unk_id)
    rec["in_scope"] = rec["n_unk"] == 0 and 0 < len(ids) <= MAX_LEN
    return rec


def build_structures(ik14, smiles, vocab_path, workers=None, chunksize=256):
    """One row per unique ik14. Uses a fork pool, as the P1 notebook does."""
    args = list(zip(ik14, smiles))
    workers = workers or os.cpu_count() or 1
    if workers > 1:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(str(vocab_path),)) as pool:
            recs = pool.map(_structure, args, chunksize=chunksize)
    else:
        _init_worker(str(vocab_path))
        recs = [_structure(a) for a in args]
    return pd.DataFrame.from_records(recs)


# --------------------------------------------------------------------------------------------- #
# The training index
# --------------------------------------------------------------------------------------------- #

class Index:
    """A compact, fork-friendly training index. Ragged lists are flat arrays plus offsets, so a
    forked worker shares them instead of copying a Python object graph.

    tok_flat / tok_off   SELFIES token ids per molecule
    row_flat / row_off   corpus rows (== embedding rows) per molecule
    row_adduct           adduct index into nd.TEST_ADDUCTS, per row of row_flat
    row_ce               CE bin per row of row_flat (nd.CE_UNKNOWN where the CE is unknown)
    fvec                 (N, len(ALLOWED_ELEMENTS)) formula counts, hydrogen included
    """

    __slots__ = ("ik14", "formula", "tok_flat", "tok_off", "row_flat", "row_off", "row_adduct",
                 "row_ce", "fvec", "meta")

    def __init__(self, ik14, formula, tok_flat, tok_off, row_flat, row_off, row_adduct, row_ce,
                 fvec, meta=None):
        self.ik14, self.formula = ik14, formula
        self.tok_flat, self.tok_off = tok_flat, tok_off
        self.row_flat, self.row_off = row_flat, row_off
        self.row_adduct, self.row_ce = row_adduct, row_ce
        self.fvec = fvec
        self.meta = meta or {}

    def __len__(self):
        return len(self.ik14)

    def tokens(self, i):
        return self.tok_flat[self.tok_off[i]:self.tok_off[i + 1]]

    def rows(self, i):
        return self.row_flat[self.row_off[i]:self.row_off[i + 1]]

    def n_spectra(self, i):
        return int(self.row_off[i + 1] - self.row_off[i])

    def save(self, path):
        path = Path(path)
        np.savez(path, ik14=np.asarray(self.ik14, object), formula=np.asarray(self.formula, object),
                 tok_flat=self.tok_flat, tok_off=self.tok_off, row_flat=self.row_flat,
                 row_off=self.row_off, row_adduct=self.row_adduct, row_ce=self.row_ce,
                 fvec=self.fvec, meta=np.asarray(json.dumps(self.meta), object))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(list(z["ik14"]), list(z["formula"]), z["tok_flat"], z["tok_off"], z["row_flat"],
                   z["row_off"], z["row_adduct"], z["row_ce"], z["fvec"],
                   json.loads(str(z["meta"])))


def row_quality(meta):
    """Per corpus row: is it usable for training? -> bool array.

    G1's label-quality filter, plus the requirement that the adduct is one of the ten test adducts
    (the decoder has an embedding row per adduct and nothing else)."""
    ok, _neutral, _truth = nc.label_ok(meta["precursor_mz"].to_numpy(),
                                       meta["adduct"].to_numpy(object),
                                       meta["molecular_formula"].to_numpy(object))
    known = meta["adduct"].map(ADDUCT_IX).notna().to_numpy()
    return ok & known


def build_index(meta, ce, split, structures, vocab, keep_splits=("train",)):
    """-> Index over the molecules of `keep_splits` that are in scope and have >= 1 quality-OK row.

    meta        corpus_meta.parquet, row i == embedding row i
    ce          corpus_ce.parquet with `row` and `ce_mean`, or None
    split       dec_split.parquet with `inchikey14` and `split`
    structures  build_structures() output
    vocab       SelfiesVocab -- passed explicitly, because parquet does not keep DataFrame.attrs
    """
    keep_ik = set(split.loc[split["split"].isin(keep_splits), "inchikey14"])
    st = structures[structures["in_scope"] & structures["inchikey14"].isin(keep_ik)]
    st = st.set_index("inchikey14")

    ok = row_quality(meta)
    ce_mean = pd.Series(np.nan, index=meta.index, dtype="float64")
    if ce is not None and len(ce):
        ce_mean.iloc[ce["row"].to_numpy()] = ce["ce_mean"].to_numpy(np.float64)

    use = pd.DataFrame({"row": np.arange(len(meta), dtype=np.int64),
                        "ik14": meta["inchikey14"].to_numpy(object),
                        "adduct": meta["adduct"].to_numpy(object),
                        "ce_mean": ce_mean.to_numpy()})[ok]
    use = use[use["ik14"].isin(st.index)]
    use = use.sort_values("row", kind="stable")

    groups = use.groupby("ik14", sort=True)
    ik14 = list(groups.groups.keys())
    counts = groups.size().reindex(ik14).to_numpy(np.int64)

    row_flat = use["row"].to_numpy(np.int64)
    order = np.argsort(use["ik14"].to_numpy(object), kind="stable")
    row_flat = row_flat[order]
    adducts = use["adduct"].to_numpy(object)[order]
    ces = use["ce_mean"].to_numpy()[order]
    row_adduct = np.array([ADDUCT_IX[a] for a in adducts], np.int8)
    row_ce = np.array([nd.ce_bin(c) for c in ces], np.int8)
    row_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    toks, tok_off, formula, fvec = [], [0], [], []
    for ik in ik14:
        rec = st.loc[ik]
        ids = vocab.encode(rec["selfies"])
        toks.extend(ids)
        tok_off.append(len(toks))
        formula.append(rec["formula"])
        fvec.append(nd.formula_vector(nd.parse_formula(rec["formula"])))
    meta_info = dict(n_molecules=len(ik14), n_rows=int(row_off[-1]), keep_splits=list(keep_splits),
                     max_len=MAX_LEN, vocab_size=len(vocab))
    return Index(ik14, formula, np.asarray(toks, np.int32), np.asarray(tok_off, np.int64),
                 row_flat, row_off, row_adduct, row_ce,
                 np.asarray(fvec, np.float32), meta_info)


# --------------------------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------------------------- #

class EmbeddingStore:
    """Memory-maps the chunked DreaMS cache. Row r lives at chunk r // CHUNK, offset r % CHUNK."""

    def __init__(self, directory, n_rows=None, chunk=CHUNK, dim=EMB_DIM, pattern="emb_{:04d}.npy"):
        self.dir, self.chunk, self.dim = Path(directory), chunk, dim
        assert "emb_dbec3a0_lx" in str(self.dir) or os.environ.get("NOVEL_ALLOW_ANY_EMB"), (
            f"{self.dir} is not the G0.1-R2 cache; only emb_dbec3a0_lx may be used for specs_novel")
        self.parts, i = [], 0
        while True:
            p = self.dir / pattern.format(i)
            if not p.exists():
                break
            self.parts.append(np.load(p, mmap_mode="r"))
            i += 1
        assert self.parts, f"no embedding chunks under {self.dir}"
        for j, a in enumerate(self.parts):
            assert a.ndim == 2 and a.shape[1] == dim, (j, a.shape)
            # row -> (chunk, offset) is only correct if every chunk but the last is exactly full.
            # A short middle chunk would silently return the WRONG embedding for every later row,
            # which no downstream check could catch, so it is refused here.
            assert a.shape[0] == chunk or j == len(self.parts) - 1, (
                f"chunk {j} has {a.shape[0]} rows, expected {chunk}: row -> chunk mapping "
                f"would be wrong for every later row")
        assert self.parts[-1].shape[0] <= chunk, (len(self.parts) - 1, self.parts[-1].shape)
        self.n_rows = sum(a.shape[0] for a in self.parts)
        if n_rows is not None:
            assert self.n_rows == n_rows, f"cache has {self.n_rows} rows, corpus has {n_rows}"

    def gather(self, rows):
        """-> (len(rows), dim) float32. Reads each touched chunk once, in row order."""
        rows = np.asarray(rows, np.int64)
        out = np.empty((len(rows), self.dim), np.float32)
        if not len(rows):
            return out
        ci = rows // self.chunk
        for c in np.unique(ci):
            sel = ci == c
            out[sel] = self.parts[int(c)][rows[sel] % self.chunk]
        return out


# --------------------------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------------------------- #

def sample_spectra(index, i, rng, max_spec=MAX_SPEC_PER_MOL):
    """Up to `max_spec` of molecule i's rows, drawn without replacement each epoch. This both
    augments the data and matches the test's 1-16 spectra per molecule."""
    rows = index.rows(i)
    if len(rows) <= max_spec:
        return np.arange(index.row_off[i], index.row_off[i + 1])
    off = index.row_off[i]
    return off + rng.choice(len(rows), max_spec, replace=False)


def make_batch(index, mol_ids, store, rng, *, max_spec=MAX_SPEC_PER_MOL, ce_dropout=CE_DROPOUT,
               pad_id, sos_id, eos_id):
    """One training batch as numpy arrays.

    -> dict with
       emb        (B, K, EMB_DIM) float32     K = the batch's largest spectrum count
       adduct     (B, K) int64
       ce         (B, K) int64                CE_UNKNOWN where unknown or dropped
       spec_mask  (B, K) bool                 True = a real spectrum
       fvec       (B, E) float32
       tgt        (B, L+2) int64              <s> tokens </s>, pad-filled
       ce_known   (B, K) bool                 before dropout, for the CE-split reporting
    """
    sel = [sample_spectra(index, i, rng, max_spec) for i in mol_ids]
    k = max(len(s) for s in sel)
    b = len(mol_ids)
    flat = np.concatenate(sel) if b else np.zeros(0, np.int64)
    gathered = store.gather(index.row_flat[flat]) if len(flat) else np.zeros((0, store.dim), np.float32)

    emb = np.zeros((b, k, store.dim), np.float32)
    adduct = np.zeros((b, k), np.int64)
    ce = np.full((b, k), nd.CE_UNKNOWN, np.int64)
    ce_known = np.zeros((b, k), bool)
    spec_mask = np.zeros((b, k), bool)
    at = 0
    for j, s in enumerate(sel):
        n = len(s)
        emb[j, :n] = gathered[at:at + n]
        adduct[j, :n] = index.row_adduct[s]
        raw = index.row_ce[s].astype(np.int64)
        ce_known[j, :n] = raw != nd.CE_UNKNOWN
        if ce_dropout > 0:
            drop = rng.random(n) < ce_dropout
            raw = np.where(drop, nd.CE_UNKNOWN, raw)
        ce[j, :n] = raw
        spec_mask[j, :n] = True
        at += n

    lens = [index.tok_off[i + 1] - index.tok_off[i] for i in mol_ids]
    lmax = int(max(lens)) + 2
    tgt = np.full((b, lmax), pad_id, np.int64)
    for j, i in enumerate(mol_ids):
        t = index.tokens(i)
        tgt[j, 0] = sos_id
        tgt[j, 1:1 + len(t)] = t
        tgt[j, 1 + len(t)] = eos_id
    return dict(emb=emb, adduct=adduct, ce=ce, spec_mask=spec_mask,
                fvec=index.fvec[np.asarray(mol_ids, np.int64)], tgt=tgt, ce_known=ce_known)


def epoch_order(n, rng, batch_size, length=None, bucket=8):
    """Molecule ids for one epoch, shuffled, then length-bucketed so a batch pads little.

    Shuffle first, then sort within a window of `bucket` batches: batches stay random across epochs
    but hold similar-length sequences, which is where the padding saving comes from."""
    ids = rng.permutation(n)
    if length is None:
        return [ids[i:i + batch_size] for i in range(0, n, batch_size)]
    out, window = [], batch_size * bucket
    for s in range(0, n, window):
        w = ids[s:s + window]
        w = w[np.argsort(length[w], kind="stable")]
        out.extend(w[i:i + batch_size] for i in range(0, len(w), batch_size))
    rng.shuffle(out)
    return out
