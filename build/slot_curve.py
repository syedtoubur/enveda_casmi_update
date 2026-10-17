"""S1 c3-slot-curve (instrument): insert the generator top-1 at a fixed slot s for EVERY molecule, no detector.

Gate: research/harness/gate_report.md `2026-10-17: c3-iso campaign, PRE-REGISTERED`, bullet S1.
  python build/slot_curve.py rows --shard pseudo_c3     # V22 replay once per cohort -> E:/casmi/c3_gen/slot_rows_<shard>.pkl
  python build/slot_curve.py curve --idea <idea dir>    # all slot insertions in memory -> result.json / result.md
The generator top-1 comes from p4_raw.pkl (decoder log-p beam, mut-first) only; no FPNet score picks x (attack A3).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

POST = Path(HERE) / "casmi_post.py"
V22 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True)
POS = "pseudo_c3"
OTHERS = ("C1_npx", "C1_xlib", "C2", "C2_hard")
ADVISORY = ("C2_np",)
SHARDS = (POS,) + OTHERS + ADVISORY
CACHE = "E:/casmi/c3_gen/slot_rows_{}.pkl"
RAW = "E:/casmi/c3_gen/p4_raw.pkl"
SLOTS = list(range(2, 26))
PIS = (0.15, 0.25, 0.35)
PLACEHOLDER = "__SURE_WRONG__"
SANITY = {"pseudo_c3": 0.0562, "C2": 0.5907, "C1_npx": 0.9285, "C2_hard": 0.5707, "C1_xlib": 0.9153, "C2_np": 0.5432}


# ---------------------------------------------------------------- pure functions (unit tested)
def insert_at(row: list, x, s: int, k: int = 25, keep_higher: bool = False) -> list:
    """Row of IK14s with x at 1-based slot s: x removed from the rest first (no duplicates), cut at k.
    s > k is the no-op (row returned unchanged); x None means no insert. keep_higher (supplementary variant, not
    the pre-registered one): if x already sits above slot s the row is left untouched instead of demoting x to s."""
    if x is None or s > k:
        return list(row)
    if keep_higher and x in row[:s - 1]:
        return list(row)
    rest = [r for r in row if r != x]
    return (rest[:s - 1] + [x] + rest[s - 1:])[:k]


def rank_in(row: list, truth: str) -> int:
    for i, r in enumerate(row[:25], 1):
        if r == truth:
            return i
    return 0


def rr(ranks) -> np.ndarray:
    ranks = np.asarray(ranks)
    return np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)


def mutfirst_ik(raw_rec: dict, key):
    """Generator top-1 IK14 (mut-first), reusing build/c3_detector.gen_top1; None when the molecule has none."""
    from c3_detector import gen_top1
    g = gen_top1(raw_rec or {}, key)["mutfirst"]
    return g[1] if g else None


def delta_for_slot(rows: list, truths: list, xs: list, s: int, keep_higher: bool = False) -> np.ndarray:
    base = rr([rank_in(r, t) for r, t in zip(rows, truths)])
    new = rr([rank_in(insert_at(r, x, s, keep_higher=keep_higher), t) for r, t, x in zip(rows, truths, xs)])
    return new - base


def boot_means(d: np.ndarray, groups, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Distribution of the mean over whole-scaffold resamples; same draw as metrics.paired_bootstrap."""
    _, gid = np.unique(np.asarray(groups), return_inverse=True)
    G = gid.max() + 1
    swd, sw = np.bincount(gid, d, G), np.bincount(gid, None, G).astype(float)
    idx = np.random.default_rng(seed).integers(0, G, (n, G))
    return swd[idx].sum(1) / np.maximum(sw[idx].sum(1), 1e-300)


def ci90(m: np.ndarray):
    return [float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))]


# ---------------------------------------------------------------- replay
def replay_rows(shard: str) -> dict:
    import harness as H
    state, deps, cp = H.load_dump(shard, POST)
    key = cp.canonicalize_smiles_to_ik14
    T = H.truths()
    mids = [m for m in state["te_mids"] if m in T]
    tk = [key(T[m]["smiles"]) for m in mids]
    out = cp.run(state, deps, **V22)
    rows = [[key(s) for s in out.get(m, [])[:25]] for m in mids]
    base_rank = np.array([rank_in(r, t) for r, t in zip(rows, tk)])
    leak = [bool(state["base"][m][0]) and key(state["base"][m][0][0]) == t for m, t in zip(mids, tk)]
    res = dict(shard=shard, mids=mids, truth=tk, rows=rows, leak=leak, smiles=[T[m]["smiles"] for m in mids],
               mrr=float(rr(base_rank).mean()))
    if shard == POS:
        raw = pickle.load(open(RAW, "rb"))["raw"]
        res["x"] = [mutfirst_ik(raw.get(m), key) for m in mids]
        import pandas as pd
        tf = pd.read_parquet(H.PSEUDO_C3, columns=["molecule_id", "timstof"]).set_index("molecule_id")["timstof"]
        res["timstof"] = [bool(tf.get(m, False)) for m in mids]
    H.save_caches()
    return res


def cmd_rows(shard: str) -> int:
    r = replay_rows(shard)
    exp = SANITY.get(shard)
    ok = exp is None or abs(r["mrr"] - exp) < 5e-5
    print(f"{shard}: n={len(r['mids'])} V22 MRR {r['mrr']:.4f} expected {exp} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        return 2
    pickle.dump(r, open(CACHE.format(shard), "wb"), protocol=4)
    return 0


# ---------------------------------------------------------------- curve
def summarize(shards: list) -> dict:
    import harness as H
    data = {sh: pickle.load(open(CACHE.format(sh), "rb")) for sh in shards}
    res = {"cohorts": {}, "slots": SLOTS, "pis": PIS}
    dist = {}   # (shard, s, 'all'|'other') -> bootstrap means
    for sh, r in data.items():
        groups = np.array(H.scaffold_groups(r["smiles"]))
        leak = np.array(r["leak"])
        xs = r.get("x") or [PLACEHOLDER] * len(r["rows"])
        n_x = int(sum(x is not None for x in r["x"])) if "x" in r else None
        per = {}
        for s in SLOTS:
            d = delta_for_slot(r["rows"], r["truth"], xs, s)
            ma = boot_means(d, groups)
            dist[(sh, s, "all")] = ma
            e = dict(delta=float(d.mean()), ci90=ci90(ma), delta_leak_hit=float(d[leak].mean()) if leak.any() else None,
                     delta_other=float(d[~leak].mean()) if (~leak).any() else None)
            if (~leak).any():
                mo = boot_means(d[~leak], groups[~leak])
                dist[(sh, s, "other")] = mo
                e["ci90_other"] = ci90(mo)
            if sh == POS:
                tf = np.array(r["timstof"])
                e["timstof15"] = dict(n=int(tf.sum()), delta=float(d[tf].mean()),
                                      ci90=ci90(boot_means(d[tf], groups[tf])))
            dk = delta_for_slot(r["rows"], r["truth"], xs, s, keep_higher=True)
            e["keep_higher"] = dict(delta=float(dk.mean()), delta_other=float(dk[~leak].mean()) if (~leak).any() else None,
                                    delta_leak_hit=float(dk[leak].mean()) if leak.any() else None)
            per[s] = e
        res["cohorts"][sh] = dict(n=len(r["rows"]), n_leak_hit=int(leak.sum()), mrr_v22=r["mrr"], n_with_x=n_x, slots=per)
    res["expected"] = {}
    for variant in ("other", "all"):
        key = "delta_other" if variant == "other" else "delta"
        for pi in PIS:
            rows = {}
            for s in SLOTS:
                dp = dist[(POS, s, "all")]
                mix = np.mean([dist[(c, s, variant)] for c in OTHERS], axis=0)
                m = pi * dp + (1 - pi) * mix
                point = pi * res["cohorts"][POS]["slots"][s]["delta"] + (1 - pi) * float(
                    np.mean([res["cohorts"][c]["slots"][s][key] for c in OTHERS]))
                rows[s] = dict(delta=float(point), ci90=ci90(m))
            res["expected"][f"{variant}/pi={pi}"] = rows
    res["expected_keep_higher"] = {}
    for variant in ("other", "all"):
        key = "delta_other" if variant == "other" else "delta"
        for pi in PIS:
            res["expected_keep_higher"][f"{variant}/pi={pi}"] = {s: float(
                pi * res["cohorts"][POS]["slots"][s]["keep_higher"]["delta"] + (1 - pi) * np.mean(
                    [res["cohorts"][c]["slots"][s]["keep_higher"][key] for c in OTHERS])) for s in SLOTS}
    kh = res["expected_keep_higher"]["other/pi=0.25"]
    res["s_star_keep_higher"] = dict(s=max(SLOTS, key=lambda s: kh[s]), delta=max(kh.values()))
    best = max(SLOTS, key=lambda s: res["expected"]["other/pi=0.25"][s]["delta"])
    res["s_star"] = dict(s=best, **res["expected"]["other/pi=0.25"][best])
    return res


def report_md(res: dict) -> str:
    f = lambda x: "-" if x is None else f"{x:+.4f}"   # noqa: E731
    S = res["s_star"]
    L = ["# S1 c3-slot-curve result (instrument)", "",
         f"**s\\* = {S['s']}** (argmax of expected Delta at pi 0.25, other-variant): {f(S['delta'])}, 90% CI "
         f"[{f(S['ci90'][0])}, {f(S['ci90'][1])}].", "",
         "V22 MRR@25 per cohort: " + ", ".join(f"{c} {v['mrr_v22']:.4f} (n={v['n']}, leak-hit {v['n_leak_hit']})"
                                               for c, v in res["cohorts"].items()), "",
         f"pseudo_c3 molecules with a generator top-1: {res['cohorts'][POS]['n_with_x']}.", "",
         "## Delta(s) per cohort (all molecules; 90% scaffold-bootstrap CI), pseudo_c3 leak-hit/other, expected Delta", "",
         "| s | " + " | ".join(res["cohorts"]) + " | c3 leak-hit | c3 other | E pi.15 oth | E pi.25 oth | E pi.35 oth | E pi.15 all | E pi.25 all | E pi.35 all |",
         "|" + "---|" * (len(res["cohorts"]) + 9)]
    for s in SLOTS:
        cells = []
        for c, v in res["cohorts"].items():
            e = v["slots"][s]
            cells.append(f"{f(e['delta'])} [{f(e['ci90'][0])},{f(e['ci90'][1])}]")
        p = res["cohorts"][POS]["slots"][s]
        ex = [f(res["expected"][f"{v}/pi={pi}"][s]["delta"]) for v in ("other", "all") for pi in PIS]
        L.append(f"| {s} | " + " | ".join(cells) + f" | {f(p['delta_leak_hit'])} | {f(p['delta_other'])} | " + " | ".join(ex) + " |")
    L += ["", "## Other-only (leak-free) Delta(s) on C1/C2", "", "| s | " + " | ".join(OTHERS + ADVISORY) + " |",
          "|" + "---|" * (len(OTHERS) + len(ADVISORY) + 1)]
    for s in SLOTS:
        L.append(f"| {s} | " + " | ".join(
            f(res["cohorts"][c]["slots"][s]["delta_other"]) for c in OTHERS + ADVISORY if c in res["cohorts"]) + " |")
    t = res["cohorts"][POS]["slots"]
    kh = res["expected_keep_higher"]
    L += ["", "## Supplementary: keep-higher variant (x already above slot s is left in place, not demoted to s)", "",
          f"s* (keep-higher, pi 0.25 other) = {res['s_star_keep_higher']['s']}: {f(res['s_star_keep_higher']['delta'])}. "
          "Not the pre-registered variant; the literal spec removes x from the rest and so demotes a generated "
          "candidate the pipeline already ranked above s.", "",
          "| s | c3 all | c3 leak-hit | c3 other | E pi.15 oth | E pi.25 oth | E pi.35 oth | E pi.25 all |", "|---|---|---|---|---|---|---|---|"]
    for s in SLOTS:
        k = t[s]["keep_higher"]
        L.append(f"| {s} | {f(k['delta'])} | {f(k['delta_leak_hit'])} | {f(k['delta_other'])} | {f(kh['other/pi=0.15'][s])} | "
                 f"{f(kh['other/pi=0.25'][s])} | {f(kh['other/pi=0.35'][s])} | {f(kh['all/pi=0.25'][s])} |")
    L += ["", f"## 15-timsTOF sub-curve (pseudo_c3, n={t[2]['timstof15']['n']}; flag only, n tiny)", "",
          "| s | Delta | 90% CI |", "|---|---|---|"]
    for s in SLOTS:
        e = t[s]["timstof15"]
        L.append(f"| {s} | {f(e['delta'])} | [{f(e['ci90'][0])},{f(e['ci90'][1])}] |")
    return "\n".join(L) + "\n"


def cmd_curve(idea: str) -> int:
    shards = [s for s in SHARDS if os.path.exists(CACHE.format(s))]
    res = summarize(shards)
    os.makedirs(idea, exist_ok=True)
    json.dump(res, open(os.path.join(idea, "result.json"), "w"), indent=1)
    open(os.path.join(idea, "result.md"), "w", encoding="utf-8").write(report_md(res))
    print(json.dumps(res["s_star"]), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rows", "curve"])
    ap.add_argument("--shard")
    ap.add_argument("--idea", default="E:/ENVEDA_casmi/research/c3/20261017-c3-slot-curve")
    a = ap.parse_args(argv)
    return cmd_rows(a.shard) if a.cmd == "rows" else cmd_curve(a.idea)


if __name__ == "__main__":
    raise SystemExit(main())
