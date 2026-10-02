"""Measure the mutator against the Phase-1 novelty cohorts (GC.0 / GC.1).

`val_scripts/val_key.pkl` carries `cls` in {1, 2} only -- **zero Class-3 molecules** -- and
`build/mk_stage_d.py` explains why it never can: "every val truth comes from train.parquet, which
is inside the pool, so pool presence is ~1.0 by construction".  `p1_novel/pseudo_c3.parquet` is the
first honest Class-3 set available offline: 600 natural products with `pool_ban_keys =
[inchikey14]` and every MassSpecGym / ICEBERG-seen structure excluded.

This harness does three things.

**1. It reproduces the oracle-conditions number** so the rest can be trusted: seed from each truth's
own Murcko scaffold (the `scaffold` column) toward its own formula, which is what
`scratch/eval_mutator_pseudo_c3.py` did.  Expect 418 reachable of 471 cyclic and 72 exact hits.

**2. It cuts that number down to what production can reach.**  In production the scaffold must be
*harvested from a retrieved hit*, so a truth that is the only molecule carrying its Murcko scaffold
can never be seeded.  `group_size` (cross-checked against `dec_split.parquet`) decides this, and
the cut is a *valid* ceiling rather than a loose bound: Murcko scaffolds are identical across a
scaffold group by construction, so whenever a relative exists the oracle scaffold is exactly what a
relative would have supplied.  Nothing is given away for `group_size > 1`.

**3. It measures the dilution the report left out.**  "Exact hit" there meant the truth appeared
anywhere among up to 300 generated candidates.  Promotion into slots [1,2,3] needs the truth in the
top 3 of the mutator list, a far stronger requirement, and the `f.z` screen that has to find it
recovered only 21.6% under oracle conditions (GM.0b).  `f.z` needs FPNet logits and cannot be
computed here, so this reports the candidate-count distribution and `3/n_candidates` -- the
probability a *random* screen lands a hit in the top 3.  That is an honest floor for the real rate,
and it is what makes depth <= 2 valuable: shallow deltas generate far fewer candidates to choose
between.

usage: python build/eval_mutator_cohort.py [cohort ...] [--limit N] [--beam N] [--cap N]
       cohorts default to: pseudo_c3 c2_displace ts_dec   (the latter two are controls)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

import delta_m_mutator as M

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "p1_novel")
DEFAULT_COHORTS = ("pseudo_c3", "c2_displace", "ts_dec")


def load(cohort):
    df = pd.read_parquet(os.path.join(P1, cohort + ".parquet"))
    split = os.path.join(P1, "dec_split.parquet")
    if os.path.exists(split):
        s = pd.read_parquet(split)
        cnt = s.groupby("scaffold_key").size()
        m = df[["inchikey14"]].merge(s[["inchikey14", "scaffold_key"]], on="inchikey14", how="left")
        univ = m.scaffold_key.map(cnt).values.astype(float)
        agree = int(np.nansum(univ == df.group_size.values))
        print("  group_size cross-checked against dec_split: %d/%d agree (%d unmatched)"
              % (agree, len(df), int(np.isnan(univ).sum())))
    return df


def one(row, reach, beam, cap):
    """Oracle-conditions run for one molecule -> per-molecule outcomes."""
    out = dict(molecule_id=row.molecule_id, group_size=int(row.group_size),
               harvestable=bool(row.group_size > 1), cyclic=False, reachable=False,
               depth=-1, exact_hit=False, n_candidates=0, secs=0.0)
    t0 = time.time()
    truth = Chem.MolFromSmiles(row.can)
    if truth is None:
        return out
    if str(row.scaffold).startswith("acyclic:"):
        out["secs"] = time.time() - t0          # Murcko strips everything; 21.5% of pseudo_c3
        return out
    out["cyclic"] = True
    scaf = Chem.MolFromSmiles(str(row.scaffold))
    if scaf is None:
        return out
    sv, tv = M.elem_vec(scaf), M.elem_vec(truth)
    if sv is None or tv is None:
        return out
    delta = tuple(t - s for t, s in zip(tv, sv))
    if any(d < 0 for d, e in zip(delta, M.EL) if e != "H"):
        return out
    decs = reach.get(delta)
    if decs is None:
        out["secs"] = time.time() - t0
        return out
    out["reachable"] = True
    out["depth"] = min(len(d) for d in decs)

    cands, seen = [], set()
    for dec in decs:
        for s in M.grow(scaf, dec, tv, beam, cap - len(cands)):
            if s not in seen:
                seen.add(s)
                cands.append(s)
        if len(cands) >= cap:
            break
    out["n_candidates"] = len(cands)
    for s in cands:
        m = Chem.MolFromSmiles(s)
        if m is not None and inchi.MolToInchiKey(m)[:14] == row.inchikey14:
            out["exact_hit"] = True
            break
    out["secs"] = time.time() - t0
    return out


def report(cohort, df, beam, cap):
    print("\n" + "=" * 78)
    print("%s: %d molecules" % (cohort, len(df)))
    print("=" * 78)
    reach = M.build_reach()
    d = pd.DataFrame([one(r, reach, beam, cap) for r in df.itertuples()])
    n = len(d)
    cyc = d[d.cyclic]
    rc = d[d.reachable]

    print("  cyclic %d (%.1f%%)   acyclic %d (%.1f%%; Murcko returns nothing, cannot be seeded)"
          % (len(cyc), 100 * len(cyc) / n, n - len(cyc), 100 * (n - len(cyc)) / n))
    print("  formula-reachable (depth <= %d): %d/%d of cyclic (%.1f%%), %.1f%% overall"
          % (M.MAXOPS, len(rc), len(cyc), 100 * len(rc) / max(1, len(cyc)), 100 * len(rc) / n))

    print("\n  exact 2D InChIKey14 recovery, cut successively to what production can reach:")
    print("  %-34s%5s%6s%8s%13s" % ("cut", "n", "hits", "rate", "% of cohort"))
    for label, sub in [("all reachable cyclic (oracle)", rc),
                       ("+ harvestable scaffold (gs > 1)", rc[rc.harvestable]),
                       ("+ delta within 2 steps", rc[rc.harvestable & (rc.depth <= 2)])]:
        h = int(sub.exact_hit.sum())
        print("  %-34s%5d%6d%7.1f%%%12.1f%%"
              % (label, len(sub), h, 100 * h / max(1, len(sub)), 100 * h / n))

    print("\n  by depth (oracle conditions):")
    print("  %6s%6s%6s%8s%11s%9s" % ("depth", "n", "hits", "rate", "cands med", "3/cands"))
    for k in sorted(rc.depth.unique()):
        s = rc[rc.depth == k]
        med = float(s.n_candidates.median())
        # 3/cands is meaningless where nothing was generated or nothing was recovered, and a bare
        # "100%" in a report is the kind of number that gets quoted out of context later
        frac = "-" if (med < 1 or not s.exact_hit.any()) else "%.1f%%" % (100 * min(1.0, 3.0 / med))
        print("  %6d%6d%6d%7.1f%%%11.0f%9s"
              % (k, len(s), int(s.exact_hit.sum()), 100 * s.exact_hit.mean(), med, frac))

    hp = rc[rc.harvestable & (rc.depth <= 2) & rc.exact_hit]
    if len(hp):
        floor = float(np.mean(np.minimum(1.0, 3.0 / np.maximum(1.0, hp.n_candidates.values))))
        print("\n  production-viable hits (harvestable, depth <= 2): %d" % len(hp))
        print("  candidates generated for them: median %.0f, max %.0f"
              % (hp.n_candidates.median(), hp.n_candidates.max()))
        print("  random-screen floor for landing one in the mutator top 3: %.1f%%" % (100 * floor))
        print("  => q_3 floor over the cohort: %.2f%%  (the f.z screen must beat random for the "
              "real rate to exceed this)" % (100 * len(hp) * floor / n))

    print("\n  runtime: %.0fs total, %.0f ms/molecule (beam=%d, cap=%d)"
          % (d.secs.sum(), 1000 * d.secs.mean(), beam, cap))
    return d


def main(argv):
    cohorts, limit, beam, cap = [], 0, 100, 300
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--limit":
            limit = int(argv[i + 1])
            i += 2
        elif a == "--beam":
            beam = int(argv[i + 1])
            i += 2
        elif a == "--cap":
            cap = int(argv[i + 1])
            i += 2
        else:
            cohorts.append(a)
            i += 1
    cohorts = cohorts or list(DEFAULT_COHORTS)

    out = {}
    for c in cohorts:
        p = os.path.join(P1, c + ".parquet")
        if not os.path.exists(p):
            print("%s: missing %s -- skipped" % (c, p))
            continue
        df = load(c)
        if limit:
            df = df.head(limit)
        out[c] = report(c, df, beam, cap)
        os.makedirs(os.path.join(ROOT, "scratch"), exist_ok=True)
        out[c].to_csv(os.path.join(ROOT, "scratch", "mutator_%s_cuts.csv" % c), index=False)

    if len(out) > 1:
        print("\n" + "=" * 78)
        print("Controls: recovery must be far lower on the non-novel cohorts")
        print("=" * 78)
        print("  %-16s%5s%13s%13s" % ("cohort", "n", "viable hits", "% of cohort"))
        for c, d in out.items():
            v = d[d.reachable & d.harvestable & (d.depth <= 2) & d.exact_hit]
            print("  %-16s%5d%13d%12.1f%%" % (c, len(d), len(v), 100 * len(v) / len(d)))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
