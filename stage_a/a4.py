"""A4/A5 analysis for iso-rank Stage A. Pure functions over the A3 tables (dict mid -> record) + truth dicts.

freeze(tables, truth_taut_A, mids_A)   uses ONLY half-A molecules; returns the frozen choices.
summarize(...)                         scores replays (half B) with the frozen choices.
Truth only scores / groups; it never selects children."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "build")):
    if p not in sys.path:
        sys.path.insert(0, p)
import iso_rank as ir  # noqa: E402

CAND_LISTS = ("ice", "frag", "mult", "parent_rank", "nl")
M_GRID = (0.0, 0.01, 0.02, 0.05, 0.1)


# ---------------------------------------------------------------- record -> apply input
def patch_keys(keymap):
    """Make iso_rank._keys a table lookup (no tautomer canonicalisation in the analysis)."""
    orig = ir._keys.__wrapped__ if hasattr(ir._keys, "__wrapped__") else ir._keys

    def look(entry):
        v = keymap.get(entry)
        if v is not None:
            return frozenset(x for x in v if x)
        if isinstance(entry, str) and len(entry) == 14 and entry.isalpha() and entry.isupper():
            return frozenset({entry})
        return orig(entry)
    ir._keys = look


def build_rec(r, frag_only=False):
    """A3 record -> (row, rec) for iso_rank.apply. frag_only: ICE conjunct made trivially true (child 1.0 > parent 0.0)."""
    pk = [p["ik14"] for p in r["parents"]]
    edges, sc = [], dict(ice={}, frag={}, nl={})
    for kd in r["kids"]:
        prs = kd.get("pairs") or []
        if prs:
            for pi, op in prs:
                if pi < len(pk) and pk[pi]:
                    edges.append(dict(smiles=kd["smiles"], ik14=kd["ik14"], parent_ik14=pk[pi], parent_rank=pi + 1, op=op))
        else:
            edges.append(dict(smiles=kd["smiles"], ik14=kd["ik14"], parent_ik14=None, parent_rank=None, op=kd.get("op")))
        if kd.get("frag") is not None:
            sc["frag"][kd["ik14"]] = kd["frag"]
        if kd.get("nl") is not None:
            sc["nl"][kd["ik14"]] = kd["nl"]
        if frag_only:
            sc["ice"][kd["ik14"]] = 1.0
        elif kd.get("ice") is not None:
            sc["ice"][kd["ik14"]] = kd["ice"]
    ps = {}
    for p in r["parents"]:
        if p["ik14"]:
            ps[p["ik14"]] = dict(ice=(0.0 if frag_only else p.get("ice")), frag=p.get("frag"))
    return list(r["row"]), dict(children=edges, scores=sc, parent_scores=ps)


def _taut(km, s):
    return (km.get(s) or (None, None))[1]


def truth_rank(row, truth_taut, km):
    if not truth_taut:
        return 0
    for i, s in enumerate(row[:25], 1):
        if _taut(km, s) == truth_taut:
            return i
    return 0


def rr(rank):
    return 1.0 / rank if rank > 0 else 0.0


def replay_one(r, truth_taut, cfg, frag_only=False):
    """-> dict(rr0, rr1, rr_wc, fired, tail). rr_wc: every entry not in the V22 row is treated as sure-wrong."""
    row, rec = build_rec(r, frag_only)
    km = r["keymap"]
    new, info = ir.apply(row, rec, cfg)
    orig = {_taut(km, s) for s in row} - {None}
    r0 = truth_rank(row, truth_taut, km)
    r1 = truth_rank(new, truth_taut, km)
    rwc = 0
    for i, s in enumerate(new[:25], 1):
        t = _taut(km, s)
        if truth_taut and t == truth_taut and t in orig:
            rwc = i
            break
    return dict(rr0=rr(r0), rr1=rr(r1), rr_wc=rr(rwc), fired=bool(info["fired"]), tail=int(info["tail"]),
                has_kids=bool(r["kids"]))


# ---------------------------------------------------------------- freeze (half A only)
def pool_mrr20(tables, truth_taut, mids, lists, k=10):
    """Within-pool MRR@20: molecules whose truth is among the children; RRF order over the children."""
    vals = []
    for m in mids:
        r = tables[m]
        _, rec = build_rec(r)
        recs = ir.parent_map(rec["children"])
        if not recs:
            continue
        tt = truth_taut[m]
        hit = {kd["ik14"] for kd in r["kids"] if tt and _taut(r["keymap"], kd["smiles"]) == tt}
        if not hit:
            continue
        order = ir.rrf(recs, rec["scores"], k=k, lists=tuple(lists))
        pos = min(i for i, ik in enumerate(order, 1) if ik in hit)
        vals.append(rr(pos) if pos <= 20 else 0.0)
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def forward_select(tables, truth_taut, mids_a, cands=CAND_LISTS):
    chosen, best, trace = [], -1.0, []
    while True:
        scored = []
        for l in cands:
            if l in chosen:
                continue
            v, n = pool_mrr20(tables, truth_taut, mids_a, chosen + [l])
            scored.append((v, l, n))
        if not scored:
            break
        v, l, n = max(scored, key=lambda t: (t[0], -cands.index(t[1])))
        trace.append(dict(step=len(chosen) + 1, tried={s[1]: s[0] for s in scored}, n_pool=n))
        if v > best + 1e-12:
            chosen, best = chosen + [l], v
        else:
            break
    return chosen, best, trace


def delta_mean(tables, truth_taut, mids, cfg, frag_only=False):
    d = [replay_one(tables[m], truth_taut[m], cfg, frag_only) for m in mids]
    return (float(np.mean([x["rr1"] - x["rr0"] for x in d])) if d else float("nan")), d


def freeze(tables, truth_taut_a, mids_a, cands=CAND_LISTS, grid=M_GRID):
    """Only half-A inputs. Returns the frozen dict (lists, m_ice, traces)."""
    lists, best, trace = forward_select(tables, truth_taut_a, mids_a, cands)
    grid_res = {str(m): delta_mean(tables, truth_taut_a, mids_a, dict(ISO_M_ICE=m, ISO_LISTS=tuple(lists)))[0]
                for m in grid}
    m_best = max(grid, key=lambda m: (grid_res[str(m)], -m))
    return dict(lists=lists, m_ice=m_best, pool_mrr20_A=best, forward_trace=trace, grid_delta_A=grid_res,
                n_A=len(mids_a), rule=("forward: add the list with the highest within-pool MRR@20 while it rises; "
                                       "m_ice = argmax half-A Delta MRR@25 over the grid, ties -> smaller"))


# ---------------------------------------------------------------- score (half B)
def boot(delta, groups, n=2000, seed=0):
    d = np.asarray(delta, float)
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    G = gid.max() + 1
    swd, sw = np.bincount(gid, d, G), np.bincount(gid, minlength=G).astype(float)
    idx = np.random.default_rng(seed).integers(0, G, (n, G))
    m = swd[idx].sum(1) / np.maximum(sw[idx].sum(1), 1e-300)
    return float(d.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def summarize(res, groups=None, strata=None):
    d = np.array([x["rr1"] - x["rr0"] for x in res])
    out = dict(n=len(res), delta=float(d.mean()) if len(d) else None,
               delta_wc=float(np.mean([x["rr_wc"] - x["rr0"] for x in res])) if res else None,
               fire_rate=float(np.mean([x["fired"] for x in res])) if res else None,
               tail_rate=float(np.mean([x["tail"] > 0 for x in res])) if res else None,
               children_rate=float(np.mean([x["has_kids"] for x in res])) if res else None,
               up=int((d > 1e-12).sum()), down=int((d < -1e-12).sum()))
    if groups is not None and len(d):
        _, lo, hi = boot(d, groups)
        out["ci90"] = [lo, hi]
    for name, flags in (strata or {}).items():
        idx = [i for i, f in enumerate(flags) if f]
        out[name] = dict(n=len(idx), delta=float(d[idx].mean()) if idx else None)
    return out


def composite(c1, c2, c3, pi):
    """Equal split of the non-C3 share between C1 and C2."""
    return pi * c3 + (1 - pi) / 2.0 * (c1 + c2)
