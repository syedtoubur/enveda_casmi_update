"""specs_novel P3 evaluation helpers: cohorts, resumable generation, and the G3 metrics.

Pre-registered in `specs_novel/gate_report.md` -> G3. Shared by the GPU eval notebook (generation and
the bake-off) and the CPU null notebook (RandomDeNovo, MCES and the verdict).

    build_queries      a cohort parquet or a dec_val Index -> one query per molecule
    ChunkStore         append candidates to Drive every N molecules, with a skip list, so a
                       disconnected generation run resumes instead of restarting
    score_molecule     top-25 max Tanimoto and exact@k for one molecule, per selection mode
    summarise          per-cohort medians and the delta against the null arm
    g3_verdict         the pre-registered G3.1 / G3.2 decision

Two invariants this module enforces rather than documents:

* **The oracle formula comes from the structure.** `formula` in a cohort parquet is the *label*
  formula and 165 corpus structures disagree with it (G1 finding 4). Queries carry the RDKit formula
  of `can`, hydrogen included, and that one formula feeds the memory token, the search masks and the
  leaf check alike.
* **Both arms see the same molecules and the same formula.** `score_molecule` takes one query and one
  candidate list, so the decoder arm and the RandomDeNovo arm are scored by identical code. G3 is a
  comparison; an asymmetry anywhere in it would make the number an artefact.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import novel_decoder as nd
import novel_p3_data as p3d

TOP_K = 25
EXACT_KS = (1, 10, 25)
SELECT_MODES = ("maxmin", "logp")
DEC_VAL_MAX_SPEC = 16          # the test's cap on spectra per molecule


def oracle_formula(can):
    """The formula the gates use: RDKit's, from the prepared structure, hydrogen included."""
    Chem = nd._chem()
    m = Chem.MolFromSmiles(can)
    if m is None:
        return None
    return p3d.formula_str(nd.mol_formula(m))


def build_queries(cohort, limit=None, seed=None):
    """A cohort parquet (pseudo_c3 / ts_dec / c2_displace) -> a list of query dicts.

    Each query: molecule_id, ik14, can, formula (ORACLE), rows, adducts, ces, src_lib, timstof,
    n_ce_known. Molecules out of scope or whose structure will not parse are dropped and reported by
    the caller -- they are unreachable by the decoder, so scoring them as misses would blame the
    search for a vocabulary limit."""
    df = cohort
    if limit is not None and len(df) > limit:
        rng = np.random.default_rng(seed)
        df = df.iloc[np.sort(rng.choice(len(df), limit, replace=False))]
    out, skipped = [], []
    for _, r in df.iterrows():
        f = oracle_formula(r["can"])
        if f is None or not bool(r.get("in_scope", True)):
            skipped.append(str(r["molecule_id"]))
            continue
        ces = list(r["ces"]) if r["ces"] is not None else []
        out.append(dict(molecule_id=str(r["molecule_id"]), ik14=str(r["inchikey14"]),
                        can=str(r["can"]), formula=f,
                        rows=np.asarray(r["rows"], np.int64),
                        adducts=[str(a) for a in r["adducts"]],
                        ces=[None if c is None or not np.isfinite(c) else float(c) for c in ces],
                        src_lib=str(r.get("src_lib", "")), timstof=bool(r.get("timstof", False)),
                        n_ce_known=int(sum(1 for c in ces if c is not None and np.isfinite(c)))))
    return out, skipped


def queries_from_index(index, structures, limit=None, seed=None, max_spec=DEC_VAL_MAX_SPEC):
    """The same query shape from a `novel_p3_data.Index` (used for the dec_val bake-off subset)."""
    can_of = structures.set_index("inchikey14")["can"].to_dict()
    ids = np.arange(len(index))
    if limit is not None and len(ids) > limit:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(len(ids), limit, replace=False))
    out, skipped = [], []
    for i in ids:
        ik = index.ik14[i]
        can = can_of.get(ik)
        f = oracle_formula(can) if can else None
        if f is None:
            skipped.append(ik)
            continue
        sl = slice(index.row_off[i], index.row_off[i] + min(max_spec, index.n_spectra(i)))
        ce = index.row_ce[sl]
        out.append(dict(molecule_id=ik, ik14=ik, can=can, formula=f,
                        rows=index.row_flat[sl].astype(np.int64),
                        adducts=[nd.TEST_ADDUCTS[a] for a in index.row_adduct[sl]],
                        ce_bins=ce.astype(np.int64).tolist(), ces=None,
                        src_lib="dec_val", timstof=False,
                        n_ce_known=int((ce != nd.CE_UNKNOWN).sum())))
    return out, skipped


def query_memory(model, query, store, device, torch):
    """-> (memory, mem_pad) for one query: its spectra plus the oracle formula token."""
    emb = store.gather(query["rows"])
    if "ce_bins" in query and query["ce_bins"] is not None:
        ce = list(query["ce_bins"])
    else:
        ce = [nd.ce_bin(c) for c in (query["ces"] or [None] * len(query["rows"]))]
    ce = (ce + [nd.CE_UNKNOWN] * len(emb))[:len(emb)]
    ad = [nd.TEST_ADDUCTS.index(a) if a in nd.TEST_ADDUCTS else 0 for a in query["adducts"]]
    ad = (ad + [0] * len(emb))[:len(emb)]
    fv = nd.formula_vector(nd.parse_formula(query["formula"]))
    t = lambda a, dt: torch.as_tensor(np.asarray(a)[None], dtype=dt, device=device)
    return model.build_memory(t(emb, torch.float32), t(ad, torch.long), t(ce, torch.long),
                              t(fv, torch.float32))


# --------------------------------------------------------------------------------------------- #
# Resumable generation
# --------------------------------------------------------------------------------------------- #

def randomdenovo_smiles(rd, formula):
    """One RandomDeNovo draw -> [smiles].

    `generate_random_molecule_graphs_via_traversal` is typed `-> list[Mol]`
    (MSGYM_novel/massspecgym/models/de_novo/random.py:746-750) and returns up to `max_top_k`
    molecules per call, never a single Mol. Handing that list straight to `Chem.MolToSmiles` raises
    `ArgumentError`, which a caller's `except Exception` silently turns into "this molecule produced
    no candidates" -- and a null arm of exactly zero is not a baseline, it hands G3.1 a free pass at
    delta = the treatment's own score. So the unpacking lives here, in the module both notebooks
    import, and is covered by a test.

    Note also that `RandomDeNovo.__init__` sets `cache_results=True` by default, so repeat calls for
    the same formula return the identical list object: a bigger pool needs `cache_results=False`,
    not more calls."""
    from rdkit import Chem
    got = rd.generate_random_molecule_graphs_via_traversal(formula)
    if got is None:
        return []
    if not isinstance(got, (list, tuple)):
        got = [got]
    out = []
    for g in got:
        if g is None:
            continue
        try:
            smi = g if isinstance(g, str) else Chem.MolToSmiles(g)
        except Exception:
            continue
        if smi:
            out.append(smi)
    return out


class ChunkStore:
    """Candidates flushed to Drive every `chunk` molecules, with a done-list beside them.

    Generation over hundreds of molecules is the longest leg of P3, so it must survive a Colab
    disconnect the way training does. Reloading yields the set of molecule_ids already finished; the
    caller skips those. The parquet is only ever appended to, one file per flush, so a crash during a
    write cannot corrupt earlier work."""

    def __init__(self, directory, tag, chunk=50):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag, self.chunk = tag, chunk
        self.done_path = self.dir / f"done_{tag}.json"
        self.buf, self.n_flush = [], 0

    def done(self):
        if not self.done_path.exists():
            return set()
        try:
            return set(json.loads(self.done_path.read_text()))
        except Exception:
            return set()

    def parts(self):
        # The chunk index is anchored to digits because tags nest: a plain `_*` glob on tag
        # "randomdenovo_pseudo_c3" also matches "cands_randomdenovo_pseudo_c3_pool_0000.parquet",
        # which silently merged the pool null's 300 candidates into the gated null's 25.
        return sorted(self.dir.glob(f"cands_{self.tag}_[0-9]*.parquet"))

    def add(self, molecule_id, cands, funnel, seconds):
        for rank, (smi, logp) in enumerate(cands):
            self.buf.append(dict(molecule_id=molecule_id, rank=rank, smiles=smi, logp=float(logp)))
        if not cands:                              # keep the molecule on the record even with 0 hits
            self.buf.append(dict(molecule_id=molecule_id, rank=-1, smiles=None, logp=np.nan))
        self._meta(molecule_id, funnel, seconds)
        if len({r["molecule_id"] for r in self.buf}) >= self.chunk:
            self.flush()

    def _meta(self, molecule_id, funnel, seconds):
        rec = dict(molecule_id=molecule_id, seconds=float(seconds), **{k: int(v) for k, v in
                                                                      (funnel or {}).items()})
        with open(self.dir / f"funnel_{self.tag}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def flush(self, extra_done=()):
        if self.buf:
            df = pd.DataFrame(self.buf)
            tmp = self.dir / f"cands_{self.tag}_{self.n_flush:04d}.parquet.tmp"
            dest = self.dir / f"cands_{self.tag}_{self.n_flush:04d}.parquet"
            while dest.exists():
                self.n_flush += 1
                tmp = self.dir / f"cands_{self.tag}_{self.n_flush:04d}.parquet.tmp"
                dest = self.dir / f"cands_{self.tag}_{self.n_flush:04d}.parquet"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, dest)
            self.n_flush += 1
            ids = {r["molecule_id"] for r in self.buf}
            self.buf = []
        else:
            ids = set()
        ids |= set(extra_done)
        allids = self.done() | ids
        tmp = Path(str(self.done_path) + ".tmp")
        tmp.write_text(json.dumps(sorted(allids)))
        os.replace(tmp, self.done_path)
        return len(allids)

    def load(self):
        """-> {molecule_id: [(smiles, logp), ...]} over every flushed part, rank-ordered."""
        parts = self.parts()
        if not parts:
            return {}
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df[df["rank"] >= 0].sort_values(["molecule_id", "rank"], kind="stable")
        return {m: list(zip(g["smiles"], g["logp"])) for m, g in df.groupby("molecule_id")}

    def funnels(self):
        p = self.dir / f"funnel_{self.tag}.jsonl"
        if not p.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])


# --------------------------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------------------------- #

def score_molecule(query, cands, modes=SELECT_MODES, k=TOP_K, exact_ks=EXACT_KS):
    """One molecule, one candidate list -> the G3 numbers, per selection mode.

    Used for BOTH arms, so the decoder and RandomDeNovo cannot be scored differently."""
    truth_ik = query["ik14"]
    out = dict(molecule_id=query["molecule_id"], ik14=truth_ik, n_cands=len(cands),
               src_lib=query.get("src_lib", ""), timstof=bool(query.get("timstof", False)),
               n_ce_known=int(query.get("n_ce_known", 0)))
    for mode in modes:
        sel = nd.select_top25(cands, mode, k=k) if cands else []
        out[f"maxtan_{mode}"] = nd.max_tanimoto(sel, query["can"])
        keys = [nd.ik14(s) for s, _ in sel]
        for kk in exact_ks:
            out[f"exact{kk}_{mode}"] = int(truth_ik in keys[:kk])
        out[f"nsel_{mode}"] = len(sel)
    return out


def summarise(scored, label):
    """-> dict of medians / means / rates over a list of score_molecule records."""
    df = pd.DataFrame(scored)
    s = dict(label=label, n=len(df))
    if not len(df):
        return s
    for mode in SELECT_MODES:
        s[f"median_maxtan_{mode}"] = float(df[f"maxtan_{mode}"].median())
        s[f"mean_maxtan_{mode}"] = float(df[f"maxtan_{mode}"].mean())
        for kk in EXACT_KS:
            s[f"exact{kk}_{mode}"] = float(df[f"exact{kk}_{mode}"].mean())
    s["median_n_cands"] = float(df["n_cands"].median())
    s["mean_n_cands"] = float(df["n_cands"].mean())
    s["zero_cand_share"] = float((df["n_cands"] == 0).mean())
    return s


def split_summaries(scored, label):
    """The G3 recorded-not-gating splits: per src_lib, per CE known/unknown, per timsTOF."""
    df = pd.DataFrame(scored)
    out = []
    if not len(df):
        return out
    for lib, g in df.groupby("src_lib"):
        out.append(summarise(g.to_dict("records"), f"{label}|src_lib={lib}"))
    for known, g in df.groupby(df["n_ce_known"] > 0):
        out.append(summarise(g.to_dict("records"), f"{label}|ce_known={bool(known)}"))
    for ts, g in df.groupby("timstof"):
        out.append(summarise(g.to_dict("records"), f"{label}|timstof={bool(ts)}"))
    return out


def pick_winner(bake, order=("prefix_seeded", "diverse_beam", "masked_beam")):
    """The pre-registered selection rule: highest median top-25 max Tanimoto on the dec_val subset
    under `maxmin`; ties go to A, then C, then B."""
    key = "median_maxtan_maxmin"
    best, best_name = None, None
    for name in order:
        if name not in bake:
            continue
        v = bake[name].get(key)
        if v is None:
            continue
        if best is None or v > best + 1e-12:       # strictly greater, so `order` breaks ties
            best, best_name = v, name
    return best_name, best


G3_EPS = 1e-9


def g3_verdict(treatment, null, threshold=0.10, converged=None, eps=G3_EPS):
    """G3.1: median top-25 max Tanimoto beats the null by >= `threshold` on pseudo_c3, under
    `maxmin`. G3.2: the loss converged (passed in from train_meta.json). Both must hold.

    The comparison carries `eps` because the registered rule is ">= 0.10" on a difference of two
    binary floats: 0.30 - 0.20 evaluates to 0.09999999999999998, so a delta that is mathematically
    exactly at the threshold would otherwise FAIL. eps only admits values within 1e-9 of the
    threshold, far below any difference that could matter to the gate."""
    t = treatment.get("median_maxtan_maxmin")
    n = null.get("median_maxtan_maxmin")
    delta = None if (t is None or n is None) else t - n
    g31 = bool(delta is not None and delta >= threshold - eps)
    items = {"G3.1": "PASS" if g31 else "FAIL"}
    if converged is not None:
        items["G3.2"] = "PASS" if converged else "FAIL"
    verdict = "PASS" if all(v == "PASS" for v in items.values()) else "FAIL"
    return dict(items=items, verdict=verdict, threshold=threshold, eps=eps,
                treatment_median=t, null_median=n, delta=delta)
