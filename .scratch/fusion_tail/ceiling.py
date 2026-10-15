"""Read-only reach/ceiling diagnostic for `fusion-tail` (campaign 20261013, cycle 1, stage-5 evidence).

Question: the primary engine computes TOPN=60 candidates and reranks them, then `casmi_post.py:206`
keeps only 25 before fusion, while Engine 2 contributes up to 40 with weight ALPHA=0.6.
How many molecules does the primary's discarded tail 26-40 change, when it is allowed to vote?

This replays RRF ONLY (no ICE/GL simulator rerank), so it is a REACH estimate, not the harness number.
It imports the harness read-only and edits nothing.
"""
import sys, collections
import numpy as np
from pathlib import Path

REPO = Path(r"E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H  # noqa: E401

ALPHA, KRR, WINDOW, LIMIT = 0.6, 3.0, 40, 25


def run(shard):
    state, deps, cp = H.load_dump(shard)
    tk = H.truths()
    identity = getattr(cp, "metric_identity", None) or deps["score_key"]  # E006 overlay identity
    rows = []
    for mid, prim in state["base"].items():
        mid = str(mid)
        rec = tk.get(mid) or {}
        if rec.get("cohort") != shard:      # truths() spans all cohorts
            continue
        # harness.compare:201 canonicalises the truth SMILES, it does not trust the stored key
        truth = cp.canonicalize_smiles_to_ik14(rec["smiles"])
        if not truth:
            continue
        p_full = [s for s in (prim[0] or []) if s]
        e = (state.get("eng") or {}).get(mid) or {}
        e_smis = [s for s in (e.get("smiles") or []) if s]
        e_keys = list(e.get("keys") or [])
        if not p_full:
            continue
        out = {}
        for cap in (LIMIT, 40, 60):
            first = p_full[:cap]
            try:
                cands, scores, _ = cp.metric_fusion(first, e_smis, identity, alpha=ALPHA,
                                                    offset=KRR, limit=WINDOW)
            except TypeError:                      # older signature without limit=
                cands, scores, _ = cp.metric_fusion(first, e_smis, identity, alpha=ALPHA, offset=KRR)
            out[cap] = H.rank_of(cands, truth, cp)   # rank_of canonicalises the SMILES itself
        rows.append((mid, len(p_full), out[25], out[40], out[60]))
    n = len(rows)
    r25 = np.array([r[2] for r in rows], float)
    r40 = np.array([r[3] for r in rows], float)
    r60 = np.array([r[4] for r in rows], float)
    rr = H.rr    # rank_of returns 0 when the truth is outside the top 25, so this IS MRR@25
    deep = sum(1 for r in rows if r[1] > LIMIT)
    print(f"\n=== {shard}: {n} scored molecules, {deep} carry a primary list deeper than 25 "
          f"(median depth {int(np.median([r[1] for r in rows]))})")
    for tag, rb in (("primary cap 25 -> 40", r40), ("primary cap 25 -> 60", r60)):
        d = rr(rb) - rr(r25)
        lo, hi = H.bootstrap(d)
        print(f"  {tag}: MRR@25 {rr(r25).mean():.4f} -> {rr(rb).mean():.4f}  delta {d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  up {int((d>0).sum())} down {int((d<0).sum())} flat {int((d==0).sum())}")
    # where do the truths that could benefit actually sit?
    hist = collections.Counter()
    for mid, depth, a, b, c in rows:
        if depth <= LIMIT:
            hist["list already <=25 (no change possible)"] += 1
        elif a <= LIMIT and b <= a:
            hist["truth already in top 25, stays or improves"] += 1
        elif a > LIMIT and b <= LIMIT:
            hist["RESCUE: truth enters top 25 only with the tail"] += 1
        elif a > LIMIT and b > LIMIT:
            hist["truth outside top 25 either way"] += 1
        else:
            hist["other (truth in 25, tail can only reorder)"] += 1
    for k, v in hist.most_common():
        print(f"  {v:4d}  {k}")
    # ceiling: every truth that the tail can put at rank 1
    best = np.where((r40 > 0) & ((r25 == 0) | (r40 < r25)), r40, r25)
    ceil = rr(best) - rr(r25)
    print(f"  CEILING (tail truths forced to rank 1): +{ceil.mean():.4f} MRR@25")


if __name__ == "__main__":
    for shard in ("C2", "C1_npx"):
        try:
            run(shard)
        except Exception as exc:
            print(f"\n=== {shard}: FAILED {type(exc).__name__}: {exc}")
