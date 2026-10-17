"""c3-iso I1: can mass-preserving isomers of the V22 fused top 10 reach the truth? (CPU only)

    python -u build/isomer_reach.py --cohort pseudo_c3 [--limit 600|--sample N --seed 0] [--procs 4]

Phase 1  `generate()`  - NO truth is read. V22 rows (casmi_post.run, V22 cfg) -> fused top 10 -> isomer_mut.mutate_many
                         (cap 300, 60 s deadline, drop IK14s of every retrieved candidate and of the parents) -> CPU
                         fragment pre-screen (frag_rescore.score_molecule) against the query's MEASURED MS/MS (R10) ->
                         top 20 -> frozen to disk (FROZEN file) together with every child.
Phase 2  `measure()`   - only now are truths loaded: reach_all (truth in ALL children, plain-IK14 compare), reach_20
                         (truth tautomer-IK14 in the frozen 20), rank of the truth among the 20, splits, operators.

Query spectra: pseudo_c3 -> train.parquet rows `train_rows` of p1_novel/pseudo_c3.parquet (the very rows the dump's query
spectra came from, patch_harness_dump.C3_VAL_CELL); other cohorts -> val_scripts/val.parquet by molecule_id.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

V22 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True)
TRAIN = "E:/casmi/data/comp_raw/train.parquet"
VAL = "E:/ENVEDA_casmi/val_scripts/val.parquet"
WORK = "E:/casmi/c3_iso"
IDEA = "E:/ENVEDA_casmi/research/c3/20261017-isomer-mut-sim"
N_PARENTS, CAP, DEADLINE, TOP = 10, 300, 60.0, 20
_R = {}


# ------------------------------------------------------------------ spectra
def spectra_pseudo_c3(mids):
    """{mid: [spectrum dict]} from train.parquet rows (peaks only; no structure column is read)."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    c3 = pd.read_parquet(Path("E:/ENVEDA_casmi/p1_novel/pseudo_c3.parquet"),
                         columns=["molecule_id", "train_rows", "timstof"])
    c3 = c3[c3.molecule_id.isin(set(mids))]
    want = {int(r): m for m, v in zip(c3.molecule_id, c3.train_rows) for r in v}
    pf = pq.ParquetFile(TRAIN)
    offs = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    sel = np.array(sorted(want), dtype=np.int64)
    out = {m: [] for m in c3.molecule_id}
    cols = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", "ionization_mode"]
    for rg in range(pf.metadata.num_row_groups):
        lo, hi = offs[rg], offs[rg + 1]
        loc = sel[(sel >= lo) & (sel < hi)] - lo
        if len(loc) == 0:
            continue
        t = pf.read_row_group(rg, columns=cols).take(loc).to_pandas()
        for g, r in zip(sel[(sel >= lo) & (sel < hi)], t.itertuples()):
            out[want[int(g)]].append(dict(
                mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1))
    tims = dict(zip(c3.molecule_id, c3.timstof.astype(bool)))
    return out, tims


def spectra_val(mids):
    import numpy as np
    import pandas as pd
    vq = pd.read_parquet(VAL)
    vq = vq[vq.molecule_id.isin(set(mids))]
    out, tims = {}, {}
    for mid, g in vq.groupby("molecule_id", sort=False):
        out[str(mid)] = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                              adduct=r.adduct, prec=float(r.precursor_mz),
                              mode=1 if r.ionization_mode == "positive" else -1) for r in g.itertuples()][:16]
        tims[str(mid)] = bool((g.instrument_type == "timsTOF").any())
    return out, tims


# ------------------------------------------------------------------ phase 1
def _init():
    import frag_rescore
    import isomer_mut
    _R["fr"], _R["im"], _R["cache"] = frag_rescore, isomer_mut, {}


def _work(task):
    mid, fused, extra, spectra = task
    im, fr = _R["im"], _R["fr"]
    t0 = time.monotonic()
    keys = {k for k in (im.ik14_plain(s) for s in list(fused) + list(extra)) if k}
    st = {}
    ch = im.mutate_many(list(fused[:N_PARENTS]), cap=CAP, deadline=t0 + DEADLINE, exclude_keys=keys, stats=st)
    t1 = time.monotonic()
    sc = [None] * len(ch)
    if ch and spectra:
        _R["cache"].clear()
        try:
            sc = fr.score_molecule(spectra, [c[0] for c in ch], cache=_R["cache"])
        except Exception as e:  # keep the molecule; unscored children rank last, in generation order
            st["score_error"] = repr(e)[:120]
    t2 = time.monotonic()
    order = sorted(range(len(ch)), key=lambda i: (-(sc[i] if sc[i] is not None else -1.0), i))[:TOP]
    return dict(mid=mid, children=ch, scores=sc, top=order, gen_s=t1 - t0, score_s=t2 - t1, stats=st,
                timed_out=(t1 - t0) >= DEADLINE, n_spec=len(spectra or []))


def generate(cohort, mids_sel, procs, out_path, log=print):
    """No truth anywhere in this function. Writes the FROZEN file and returns its path."""
    import harness
    state, deps, cp = harness.load_dump(cohort, Path(HERE) / "casmi_post.py")
    mids = [m for m in state["te_mids"] if (mids_sel is None or m in mids_sel)]
    t = time.time()
    fused = cp.run(state, deps, **V22)
    log(f"V22 rows for {len(state['te_mids'])} molecules in {time.time() - t:.0f}s; selected {len(mids)}")
    if cohort == "pseudo_c3":
        spec, tims = spectra_pseudo_c3(mids)
        src = f"{TRAIN} rows=pseudo_c3.parquet:train_rows"
    else:
        spec, tims = spectra_val(mids)
        src = VAL
    log(f"spectra from {src}: {sum(len(v) > 0 for v in spec.values())}/{len(mids)} molecules with spectra")
    tasks = []
    for m in mids:
        extra = list((state.get("eng") or {}).get(m, {}).get("smiles") or []) + \
            list(((state.get("pc") or {}).get(m) or {}).get("pc") or []) + list((state["base"][m][0]) or [])
        tasks.append((m, list(fused.get(m, [])), extra, spec.get(m, [])))
    res = {}
    t0 = time.time()
    with Pool(procs, _init) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, tasks), 1):
            res[r["mid"]] = r
            if i % 25 == 0 or i == len(tasks):
                log(f"  generated {i}/{len(tasks)} {time.time() - t0:.0f}s")
    frozen = dict(cohort=cohort, cfg=dict(V22=V22, n_parents=N_PARENTS, cap=CAP, deadline=DEADLINE, top=TOP),
                  spectra_source=src, fused={m: fused.get(m, []) for m in mids}, timstof=tims, res=res,
                  wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(frozen, open(out_path, "wb"), protocol=4)
    log(f"FROZEN {out_path}")
    return frozen


# ------------------------------------------------------------------ phase 2
def wilson(k, n, z=1.6449):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _rate(flags):
    n, k = len(flags), int(sum(flags))
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, rate=(k / n if n else None), ci90=[lo, hi])


def measure(frozen, log=print):
    """Truth is loaded HERE, after the frozen lists exist on disk."""
    import harness
    from rdkit import Chem
    cohort = frozen["cohort"]
    state, deps, cp = harness.load_dump(cohort, Path(HERE) / "casmi_post.py")
    import isomer_mut as im
    T = harness.truths()
    rows, ops_all, ops_20 = [], {}, {}
    for m, r in frozen["res"].items():
        if m not in T:
            continue
        t = T[m]
        tk_taut = cp.canonicalize_smiles_to_ik14(t["smiles"])          # tautomer-canonical, as the metric
        tk_plain = im.ik14_plain(t["smiles"])
        tm = Chem.MolFromSmiles(t["smiles"])
        ch = r["children"]
        fused = frozen["fused"][m]
        in_all = [c for c in ch if c[1] == tk_plain]
        top = [ch[i] for i in r["top"]]
        top_taut = [cp.canonicalize_smiles_to_ik14(c[0]) for c in top]
        in20 = [j for j, k in enumerate(top_taut) if k == tk_taut]
        fused_taut = [cp.canonicalize_smiles_to_ik14(s) for s in fused[:N_PARENTS]]
        fused25 = [cp.canonicalize_smiles_to_ik14(s) for s in fused[:25]]
        rows.append(dict(
            mid=m, n_children=len(ch), gen_s=r["gen_s"], score_s=r["score_s"], timed_out=r["timed_out"],
            reach_20=bool(in20), reach_all=bool(in_all) or bool(in20), rank20=(in20[0] + 1 if in20 else 0),
            op_all=(in_all[0][2] if in_all else (top[in20[0]][2] if in20 else None)),
            op_20=(top[in20[0]][2] if in20 else None),
            truth_in_top10=tk_taut in fused_taut, truth_in_top25=tk_taut in fused25,
            acyclic=bool(tm is not None and tm.GetRingInfo().NumRings() == 0),
            timstof=bool(frozen["timstof"].get(m, False)), n_spec=r["n_spec"], stats=r["stats"]))
    return rows


def summarize(rows):
    def sub(f):
        s = [r for r in rows if f(r)]
        return dict(n=len(s), reach_all=_rate([r["reach_all"] for r in s]), reach_20=_rate([r["reach_20"] for r in s]))
    ops = {}
    for r in rows:
        if r["reach_20"]:
            ops[r["op_20"]] = ops.get(r["op_20"], 0) + 1
    ops_all = {}
    for r in rows:
        if r["reach_all"] and r["op_all"]:
            ops_all[r["op_all"]] = ops_all.get(r["op_all"], 0) + 1
    n = max(len(rows), 1)
    rk = [r["rank20"] for r in rows if r["rank20"]]
    return dict(
        n=len(rows), reach_all=_rate([r["reach_all"] for r in rows]), reach_20=_rate([r["reach_20"] for r in rows]),
        truth_in_fused_top10=sum(r["truth_in_top10"] for r in rows), truth_in_fused_top25=sum(r["truth_in_top25"] for r in rows),
        not_in_top25=sub(lambda r: not r["truth_in_top25"]),
        acyclic=sub(lambda r: r["acyclic"]), ring=sub(lambda r: not r["acyclic"]),
        timstof=sub(lambda r: r["timstof"]), not_timstof=sub(lambda r: not r["timstof"]),
        children_per_mol=dict(mean=sum(r["n_children"] for r in rows) / n,
                              zero=sum(r["n_children"] == 0 for r in rows),
                              capped=sum(r["n_children"] >= CAP for r in rows)),
        hit_ops_20=ops, hit_ops_all=ops_all, truth_rank_in_20=rk,
        mean_truth_rank_in_20=(sum(rk) / len(rk) if rk else None),
        cpu_s_per_mol=dict(gen=sum(r["gen_s"] for r in rows) / n, score=sum(r["score_s"] for r in rows) / n,
                           total=sum(r["gen_s"] + r["score_s"] for r in rows) / n,
                           max=max((r["gen_s"] + r["score_s"] for r in rows), default=0)),
        timed_out=sum(r["timed_out"] for r in rows),
        formula_drops=sum(r["stats"].get("formula_drop", 0) for r in rows),
        score_errors=sum("score_error" in r["stats"] for r in rows),
        no_spectra=sum(r["n_spec"] == 0 for r in rows))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="pseudo_c3")
    ap.add_argument("--limit", type=int, default=0, help="first N after a seeded shuffle (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    assert a.procs <= 4
    import random
    st = pickle.load(open(f"E:/ENVEDA_casmi/research/harness/dumps/{a.cohort}/post_dump/state.pkl", "rb"))
    mids = list(st["te_mids"])
    if a.limit:
        random.Random(a.seed).shuffle(mids)
        mids = mids[:a.limit]
    sel = set(mids)
    tag = a.tag or (f"{a.cohort}_n{len(sel)}")
    frozen_path = f"{WORK}/frozen_{tag}.pkl"
    frozen = generate(a.cohort, sel, a.procs, frozen_path)
    rows = measure(frozen)
    summ = summarize(rows)
    summ.update(cohort=a.cohort, tag=tag, spectra_source=frozen["spectra_source"], wall_s=frozen["wall_s"], seed=a.seed)
    json.dump(dict(summary=summ, rows=rows), open(f"{WORK}/measure_{tag}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "truth_rank_in_20"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
