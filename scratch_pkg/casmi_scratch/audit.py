"""R1 coverage audit (gates R1.2, R1.3), the class-vote diagnostic (gate CV) and the R2 discrimination gate (R2.3)
on slice I.

Slice I = the enveda-np-examples molecules of train (timsTOF natural products), one molecule per inchikey14.
Each pool is audited in its own key space: R0 keys are plain inchikey14, R1 keys are scorer keys (the truth is
mapped through the train keymap). Every library ban drops the query's own structure from the library.
"""
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .analog import analog_hits, class_vote, vote_report
from .metfrag import frag_scores
from .fpnet import molecule_spectra
from .pipeline import analog_scores, class_bonus, features, molecule_target, plan_molecule, scored_range
from .rank import rank_two_tier
from .sim import rows_that_fit

SLICE_I_LIB = "enveda-np-examples"
SLICE_COLS = ["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ionization_mode", "collision_energy_ev",
              "ms2_mzs", "ms2_normalized_intensities", "ingest_lib", "instrument_type"]


def load_slice(train_path, lib_name=SLICE_I_LIB):
    """-> [(inchikey14, spectra frame)] for one ingest library."""
    t = pq.read_table(train_path, columns=SLICE_COLS)
    t = t.filter(pc.equal(t.column("ingest_lib").cast("string"), lib_name)).to_pandas()
    return [(k, g.reset_index(drop=True)) for k, g in t.groupby("inchikey14")]


def window_keys(pool, target, cfg, scored=False):
    lo, hi = scored_range(pool, target, cfg) if scored else pool.bounds(target, cfg.ppm_win)
    return pool.keys[lo:hi]


def coverage(mols, pool, truth_of, cfg):
    """Truth in the +-ppm_win window / in the scored range / in the pool at all, and window sizes."""
    allkeys = set(pool.keys)
    rows = []
    for ik, sub in mols:
        t = molecule_target(sub, cfg)
        truth = truth_of(ik)
        w = window_keys(pool, t, cfg) if np.isfinite(t) else np.zeros(0, object)
        s = window_keys(pool, t, cfg, scored=True) if np.isfinite(t) else np.zeros(0, object)
        rows.append(dict(ik=ik, in_pool=truth in allkeys, in_window=bool(np.isin(truth, w)),
                         in_scored=bool(np.isin(truth, s)), n_window=len(w), n_scored=len(s)))
    return rows


def summarize_coverage(rows):
    a = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    return dict(n=len(rows), recall_window=float(a("in_window").mean()), recall_scored=float(a("in_scored").mean()),
                in_pool=float(a("in_pool").mean()), window_median=float(np.median(a("n_window"))),
                window_p95=float(np.percentile(a("n_window"), 95)), scored_median=float(np.median(a("n_scored"))))


def growth(rows0, rows1):
    """Per-molecule window growth R1 / R0 (+-ppm_win window)."""
    n0 = np.array([r["n_window"] for r in rows0], float)
    n1 = np.array([r["n_window"] for r in rows1], float)
    ratio = n1 / np.maximum(n0, 1.0)
    return dict(ratio_median=float(np.median(ratio)), ratio_p95=float(np.percentile(ratio, 95)),
                added_median=float(np.median(n1 - n0)), added_p95=float(np.percentile(n1 - n0, 95)),
                recovered=int(sum(b["in_window"] and not a["in_window"] for a, b in zip(rows0, rows1))),
                lost=int(sum(a["in_window"] and not b["in_window"] for a, b in zip(rows0, rows1))))


def own_ban(lib, key):
    return lib.ik == key


def sim_budget_per_mol(cfg, n_test_mol=400):
    """Simulation requests per test molecule that fit the time budget (R0's rows_that_fit, rate from Config)."""
    rows = rows_that_fit(cfg.hard_budget_h * 3600.0 - cfg.pass2_s_per_mol * n_test_mol - 300.0, cfg.rate,
                         cfg.startup_s)
    return rows / n_test_mol


def truth_in_sim(mols, lib, pool, truth_of, cfg, req_per_mol, frag_fn=frag_scores, log=print):
    """R1.3: share of molecules whose truth would be simulated, own spectra banned.

    A molecule's request budget is spread over its (adduct, CE) conditions, so the truth is simulated iff its
    priority rank < req_per_mol / n_conditions (an even split of the global budget; the real run shares it).
    """
    hit, n, t0 = [], 0, time.time()
    for i, (ik, sub) in enumerate(mols):
        t = molecule_target(sub, cfg)
        if not np.isfinite(t):
            continue
        truth = truth_of(ik)
        p = plan_molecule(sub, t, lib, pool, cfg.with_(use_sim=True), frag_fn=frag_fn,
                          lib_exclude=own_ban(lib, truth))
        keys = pool.keys[p["sim_idx"]]
        j = np.flatnonzero(keys == truth)
        n += 1
        hit.append(bool(len(j)) and int(p["prio"][j[0]]) < req_per_mol / max(1, len(p["conds"])))
        if i % 50 == 0:
            log(f"  truth-in-sim {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    return dict(n=n, req_per_mol=float(req_per_mol), truth_in_sim=float(np.mean(hit)) if hit else float("nan"))


def class_vote_diag(mols, lib, truth_of, label_of, cfg, modes=("delta", "open"), k=10, min_sim=0.5, log=print):
    """Gate CV: per mode, vote purity vs the majority baseline and coverage, at superclass and pathway level.

    label_of(key, level) -> label or None. The truth's own structure is banned from the library.
    """
    out = {}
    for mode in modes:
        preds = {"np_superclass": [], "np_pathway": []}
        truths = {"np_superclass": [], "np_pathway": []}
        n_hits, t0 = [], time.time()
        for i, (ik, sub) in enumerate(mols):
            t = molecule_target(sub, cfg)
            truth = truth_of(ik)
            pol = 1 if (sub.ionization_mode == "positive").mean() >= 0.5 else -1
            specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
            hits = analog_hits(lib, specs, t, cfg, mode, polarity=pol, exclude=own_ban(lib, truth)) \
                if np.isfinite(t) else {}
            n_hits.append(sum(1 for s in hits.values() if s >= min_sim))
            for lvl in preds:
                preds[lvl].append(class_vote(hits, lambda kk: label_of(kk, lvl), k, min_sim)[0])
                truths[lvl].append(label_of(truth, lvl))
            if i % 25 == 0:
                log(f"  CV {mode} {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
        rep = {lvl: vote_report(preds[lvl], truths[lvl]) for lvl in preds}
        sc = rep["np_superclass"]
        rep["hits_ge_min_median"] = float(np.median(n_hits)) if n_hits else float("nan")
        rep["pass"] = bool(sc["n_voted"] and sc["purity"] >= sc["baseline"] + 0.10 and sc["coverage"] >= 0.50)
        out[mode] = rep
    return out


def discrimination(mols, lib, pool, truth_of, vote_label, cfg, frag_fn=frag_scores, log=print):
    """R2.3 per molecule, own structure banned from the library, no simulator (sv = 0), scored range.

    -> rows: voted label, conf, the truth's P(voted) vs the window median, the truth's rank without / with the bonus,
    whether the top 25 changed, and the tier-2 top-1 / top-2 fr gap.
    """
    c = cfg.with_(class_vote=True, use_sim=False)
    rows, t0 = [], time.time()
    for i, (ik, sub) in enumerate(mols):
        t = molecule_target(sub, c)
        if not np.isfinite(t):
            continue
        truth = truth_of(ik)
        plan = plan_molecule(sub, t, lib, pool, c, frag_fn=frag_fn, lib_exclude=own_ban(lib, truth),
                             vote_label=vote_label)
        lo, hi = scored_range(pool, t, c)
        cand = np.arange(lo, hi)
        lv, sv, fr, ppm = features(plan, cand, pool, {}, c, frag_fn)
        bonus = class_bonus(plan, cand, pool, c)
        o0, o1 = rank_two_tier(lv, sv, fr, ppm, c.tau), rank_two_tier(lv, sv, fr, ppm, c.tau, bonus)
        voted, conf = plan["vote"]
        pv = bonus / (c.vote_beta * conf) if voted is not None and conf > 0 else np.zeros(len(cand))
        j = np.flatnonzero(pool.keys[cand] == truth)
        t2 = np.sort(fr[lv < c.tau])[::-1]
        rows.append(dict(ik=ik, voted=voted, conf=conf, n_scored=len(cand), truth_in_scored=bool(len(j)),
                         truth_p=float(pv[j[0]]) if len(j) else None,
                         window_median_p=float(np.median(pv)) if len(pv) else None,
                         rank0=int(np.flatnonzero(o0 == j[0])[0]) + 1 if len(j) else None,
                         rank1=int(np.flatnonzero(o1 == j[0])[0]) + 1 if len(j) else None,
                         changed=not np.array_equal(o0[:c.topn], o1[:c.topn]),
                         fr_gap=float(t2[0] - t2[1]) if len(t2) > 1 else None))
        if i % 25 == 0:
            log(f"  R2.3 {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    return rows


def summarize_discrimination(rows, topn=25):
    """R2.3 verdict: (1) truth P(voted) > window median in >= 60% of voted molecules with the truth in range, and
    (2) the bonus moves the truth up in more molecules than down."""
    v = [r for r in rows if r["voted"] is not None and r["truth_in_scored"]]
    above = [r["truth_p"] > r["window_median_p"] for r in v]
    up = sum(r["rank1"] < r["rank0"] for r in v)
    down = sum(r["rank1"] > r["rank0"] for r in v)
    rr = lambda k: float(np.mean([1.0 / r[k] if r[k] <= topn else 0.0 for r in rows if r["truth_in_scored"]]))  # noqa
    gaps = [r["fr_gap"] for r in rows if r["fr_gap"] is not None]
    out = dict(n=len(rows), n_voted=sum(r["voted"] is not None for r in rows), n_voted_in_range=len(v),
               truth_above_median=float(np.mean(above)) if above else float("nan"), up=up, down=down,
               reach=float(np.mean([r["changed"] for r in rows])) if rows else float("nan"),
               mrr_before=rr("rank0"), mrr_after=rr("rank1"),
               fr_gap_median=float(np.median(gaps)) if gaps else float("nan"))
    out["pass"] = bool(above and out["truth_above_median"] >= 0.60 and up > down)
    return out


# ---------------------------------------------------------------------------------------------- R3 FPNet channel
def load_cohorts(val_dir, cohorts=("C2", "C2_hard")):
    """val_scripts/{val.parquet, val_key.pkl} -> [(molecule_id, inchikey14, query frame, ban_rows, ban_keys)].

    C2 / C2_hard structures (C2_hard: their whole scaffold family) were banned from own-FPNet training; ban_rows
    are train.parquet row indices (= library spectrum indices) to drop from the library for that query."""
    import os
    import pickle
    key = [r for r in pickle.load(open(os.path.join(val_dir, "val_key.pkl"), "rb")) if r["cohort"] in cohorts]
    q = pq.read_table(os.path.join(val_dir, "val.parquet")).to_pandas()
    g = dict(list(q.groupby("molecule_id")))
    return [(r["molecule_id"], r["inchikey14"], g[r["molecule_id"]].reset_index(drop=True),
             list(r.get("ban_rows") or []), list(r.get("ban_keys") or [])) for r in key if r["molecule_id"] in g]


def ban_mask(lib, ban_rows, ban_keys, keymap=None):
    m = np.zeros(len(lib), bool)
    rows = np.asarray(ban_rows, np.int64)
    m[rows[(rows >= 0) & (rows < len(lib))]] = True
    keys = {keymap.get(k, k) if keymap else k for k in ban_keys}
    if keys:
        m |= np.isin(lib.ik, list(keys))
    return m


def fp_discrimination(mols, lib, pool, truth_of, bank, cfg, bans=None, frag_fn=frag_scores, log=print):
    """R3.audit per molecule: scored range, no simulator (sv = 0), class vote off; the truth's rank without the FP
    channel (rank0) and with the RRF fusion (rank1). mols: [(mid, truth ik14, frame)]; bans: {mid: library mask}
    (default: the truth's own structure)."""
    c = cfg.with_(class_vote=False, use_sim=False, fp_channel=bank.kind)
    rows, t0 = [], time.time()
    for i, (mid, ik, sub) in enumerate(mols):
        t = molecule_target(sub, c)
        if not np.isfinite(t):
            continue
        truth = truth_of(ik)
        excl = bans.get(mid) if bans is not None else own_ban(lib, truth)
        plan = plan_molecule(sub, t, lib, pool, c, frag_fn=frag_fn, lib_exclude=excl)
        z = bank.molecule_z(molecule_spectra(sub))
        lo, hi = scored_range(pool, t, c)
        cand = np.arange(lo, hi)
        lv, sv, fr, ppm = features(plan, cand, pool, {}, c, frag_fn)
        fz = bank.fz(z, [pool.keys[k] for k in cand], [pool.smiles[k] for k in cand]) if z is not None else None
        o0 = rank_two_tier(lv, sv, fr, ppm, c.tau)
        o1 = rank_two_tier(lv, sv, fr, ppm, c.tau, None, fz, c.fp_rrf_k) if fz is not None else o0
        j = np.flatnonzero(pool.keys[cand] == truth)
        fz_rank = None
        if len(j) and fz is not None and np.isfinite(fz[j[0]]):
            fz_rank = int(np.sum(np.nan_to_num(fz, nan=-np.inf) > fz[j[0]])) + 1
        rows.append(dict(mid=mid, ik=ik, n_scored=len(cand), truth_in_scored=bool(len(j)), fp_ok=z is not None,
                         tier1_top=bool(lv[o0[0]] >= c.tau) if len(o0) else False, fz_rank=fz_rank,
                         rank0=int(np.flatnonzero(o0 == j[0])[0]) + 1 if len(j) else None,
                         rank1=int(np.flatnonzero(o1 == j[0])[0]) + 1 if len(j) else None,
                         changed=not np.array_equal(o0[:c.topn], o1[:c.topn])))
        if i % 25 == 0:
            log(f"  R3 {bank.kind} {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    return rows


def summarize_fp(rows, topn=25, bar=0.02):
    """R3.audit verdict: MRR@25 with FP - without >= bar and the truth moves up in more molecules than down."""
    v = [r for r in rows if r["truth_in_scored"]]
    rr = lambda k: float(np.mean([1.0 / r[k] if r[k] <= topn else 0.0 for r in v])) if v else float("nan")  # noqa
    up = sum(r["rank1"] < r["rank0"] for r in v)
    down = sum(r["rank1"] > r["rank0"] for r in v)
    fr = [r["fz_rank"] for r in v if r["fz_rank"] is not None]
    out = dict(n=len(rows), n_in_range=len(v), n_fp_ok=sum(r["fp_ok"] for r in rows), up=up, down=down,
               reach=float(np.mean([r["changed"] for r in rows])) if rows else float("nan"),
               mrr_before=rr("rank0"), mrr_after=rr("rank1"),
               fz_only_mrr=float(np.mean([1.0 / x if x <= topn else 0.0 for x in fr])) if fr else float("nan"),
               fz_top1=float(np.mean([x == 1 for x in fr])) if fr else float("nan"))
    out["delta"] = out["mrr_after"] - out["mrr_before"]
    out["pass"] = bool(v and out["delta"] >= bar and up > down)
    return out


def fp_self_rank(bank, mols, pool, truth_of, cfg, n=5, k=5):
    """R3.run smoke (b): for n molecules the bank trained on, is the truth's f.z in the top k of its +-ppm_win
    window? -> (n_hit, rows). Catches a wrong bit order, adduct vocabulary or peak prep."""
    rows = []
    for ik, sub in mols:
        if len(rows) >= n:
            break
        t = molecule_target(sub, cfg)
        if not np.isfinite(t):
            continue
        truth = truth_of(ik)
        cand = pool.window(t, cfg.ppm_win)
        j = np.flatnonzero(pool.keys[cand] == truth)
        if len(cand) < 10 or not len(j):
            continue
        z = bank.molecule_z(molecule_spectra(sub))
        fz = bank.fz(z, [pool.keys[c] for c in cand], [pool.smiles[c] for c in cand])
        if not np.isfinite(fz[j[0]]):
            continue
        rank = int(np.sum(np.nan_to_num(fz, nan=-np.inf) > fz[j[0]])) + 1
        rows.append(dict(ik=ik, n_window=len(cand), rank=rank))
    return sum(r["rank"] <= k for r in rows), rows


# ---------------------------------------------------------------------------------------------- R5 analogue channel
AN_MODES = ("none", "pssm", "tani")


def analog_discrimination(mols, lib, pool, truth_of, fp_bank, an_bank, cfg, vote_label=None, frag_fn=frag_scores,
                          log=print):
    """R5.audit per molecule: own structure banned from the library, no simulator (sv = 0), and the truth counted
    as non-train (the analogue list scores it; every other train structure gets the neutral term), as for an NP
    truth in the hidden test. The truth's rank under every arm: FP list off/on x analogue list none/pssm/tani x
    class vote off/on (vote arms only with vote_label). mols: [(inchikey14, frame)]."""
    c = cfg.with_(use_sim=False, class_vote=vote_label is not None, analog_channel="pssm")
    rows, t0 = [], time.time()
    for i, (ik, sub) in enumerate(mols):
        t = molecule_target(sub, c)
        if not np.isfinite(t):
            continue
        truth = truth_of(ik)
        plan = plan_molecule(sub, t, lib, pool, c, frag_fn=frag_fn, lib_exclude=own_ban(lib, truth),
                             vote_label=vote_label)
        hits = plan.pop("an_hits") or {}
        profs = {m: an_bank.profile(hits, m) for m in AN_MODES[1:]}
        lo, hi = scored_range(pool, t, c)
        cand = np.arange(lo, hi)
        lv, sv, fr, ppm = features(plan, cand, pool, {}, c, frag_fn)
        bonus = class_bonus(plan, cand, pool, c) if vote_label is not None else None
        z = fp_bank.molecule_z(molecule_spectra(sub)) if fp_bank is not None else None
        fz = fp_bank.fz(z, [pool.keys[k] for k in cand], [pool.smiles[k] for k in cand]) if z is not None else None
        ans = {m: analog_scores(dict(an_prof=profs[m], an_free={truth}), cand, pool, c.with_(analog_channel=m),
                                an_bank) for m in AN_MODES[1:]}
        j = np.flatnonzero(pool.keys[cand] == truth)
        row = dict(ik=ik, n_scored=len(cand), truth_in_scored=bool(len(j)), fp_ok=fz is not None,
                   n_hits=len(hits), an_s=round(plan["an_s"], 2),
                   tier1_top=bool(len(cand)) and bool(lv[rank_two_tier(lv, sv, fr, ppm, c.tau)[0]] >= c.tau))
        for m in AN_MODES[1:]:
            row[f"n_{m}"] = profs[m]["n"] if profs[m] is not None else 0
            a = ans[m]
            row[f"{m}_rank"] = (int(np.sum(np.nan_to_num(a, nan=-np.inf) > a[j[0]])) + 1
                                if len(j) and a is not None and np.isfinite(a[j[0]]) else None)
            row[f"{m}_scored"] = int(np.isfinite(a).sum()) if a is not None else 0
        for fp_on in (False, True):
            for m in AN_MODES:
                for vote in (False, True):
                    if vote and bonus is None:
                        continue
                    o = rank_two_tier(lv, sv, fr, ppm, c.tau, bonus if vote else None, fz if fp_on else None,
                                      c.fp_rrf_k, None, ans.get(m), c.analog_w)
                    row[f"fp{int(fp_on)}_{m}_v{int(vote)}"] = int(np.flatnonzero(o == j[0])[0]) + 1 if len(j) \
                        else None
        rows.append(row)
        if i % 25 == 0:
            log(f"  R5 {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    return rows


def summarize_analog(rows, topn=25, base="fp1_none_v1", bar=0.010):
    """R5.audit: MRR@25 per arm (molecules with the truth in range), up / down vs base for each arm, coverage, the
    analogue-only MRR, and the verdict per scorer: arm fp1_<mode>_v1 - base >= bar and up > down."""
    v = [r for r in rows if r["truth_in_scored"]]
    if not v:
        return dict(n=len(rows), n_in_range=0)
    if base not in v[0]:
        base = base.replace("_v1", "_v0")
    rr = lambda k: float(np.mean([1.0 / r[k] if r[k] <= topn else 0.0 for r in v]))  # noqa: E731
    arms = sorted(k for k in v[0] if k.startswith("fp") and k[2] in "01")
    out = dict(n=len(rows), n_in_range=len(v), base=base, mrr={k: rr(k) for k in arms}, up={}, down={},
               an_s_median=float(np.median([r["an_s"] for r in rows])),
               an_s_total=float(np.sum([r["an_s"] for r in rows])))
    for k in arms:
        out["up"][k] = sum(r[k] < r[base] for r in v)
        out["down"][k] = sum(r[k] > r[base] for r in v)
    for m in AN_MODES[1:]:
        ranks = [r[f"{m}_rank"] for r in v if r[f"{m}_rank"] is not None]
        out[f"{m}_coverage"] = float(np.mean([r[f"n_{m}"] > 0 for r in rows]))
        out[f"{m}_n_median"] = float(np.median([r[f"n_{m}"] for r in rows]))
        out[f"{m}_only_mrr"] = float(np.mean([1.0 / x if x <= topn else 0.0 for x in ranks])) if ranks else None
        arm = base.replace("none", m)
        out[f"{m}_delta"] = out["mrr"][arm] - out["mrr"][base]
        out[f"{m}_pass"] = bool(out[f"{m}_delta"] >= bar and out["up"][arm] > out["down"][arm])
    return out
