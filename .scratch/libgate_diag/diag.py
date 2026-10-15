"""Class-1 library-gate diagnosis (pre-registered: research/harness/gate_report.md, 2026-10-14 libgate block).

Per battery dump: engine lib_max (state['base'][m][2]) vs truth, a stage-by-stage loss decomposition of Class 1
(BASE -> 0.433 control -> E006, plus E006 with ICE removed), the instrument split of each C1 truth's remaining
library references, and the "protect library hits from the overlay" counterfactual at the pre-registered taus.
Usage: python .scratch/libgate_diag/diag.py            (writes .scratch/libgate_diag/diag.json)
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("E:/ENVEDA_casmi")
sys.path.insert(0, str(REPO / "research" / "harness"))
import harness as H  # noqa: E402

H.CFG_LINE = None
HERE = Path(__file__).resolve().parent
COHORTS = ["C1_npx", "C1_xlib", "C2", "C2_hard", "C2_np"]
E006 = "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"
EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
rng = np.random.default_rng(0)


def ci90(d):
    if len(d) == 0:
        return [0.0, 0.0]
    b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]


def replay(shard):
    state, deps, cp = H.load_dump(shard)
    T = H.truths()
    mids = [m for m in state["te_mids"] if m in T]
    tk = {m: cp.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in mids}

    def ranks(st, cfg):
        out = cp.run(st, deps, **H.parse_set(cfg)) if cfg else cp.run(st, deps)
        return {m: H.rank_of(out.get(m, []), tk[m], cp) for m in mids}

    noice = copy.copy(state)
    noice["ice"] = {}
    r = dict(e006=ranks(state, E006), ctl=ranks(state, ""), noice=ranks(noice, E006),
             mf=ranks(state, "METRIC_FUSION=true,FUSION_WINDOW=40"))
    rows = []
    for m in mids:
        b = state["base"][m]
        r_base = H.rank_of(list(b[0])[:25], tk[m], cp)
        rows.append(dict(mid=m, ik14=tk[m], lib_max=float(b[2]), r_base=r_base, leak=r_base == 1,
                         **{f"r_{k}": int(v[m]) for k, v in r.items()}))
    return pd.DataFrame(rows)


def ref_split(vk_by_mid, meta_by_ik, mid, ik):
    """Instrument classes of the truth's library rows left after ban_rows (timsTOF = enveda libs)."""
    k = vk_by_mid.get(mid)
    rows = meta_by_ik.get(ik)
    if k is None or rows is None:
        return "none"
    ban = set(k["ban_rows"])
    libs = {lib for row, lib in rows if row not in ban}
    if not libs:
        return "none"
    tims = {l for l in libs if l.startswith("enveda")}
    return "tims_only" if tims == libs else ("other_only" if not tims else "both")


def main():
    vk = pickle.load(open(REPO / "val_scripts/val_key.pkl", "rb"))
    vk_by_mid = {r["molecule_id"]: r for r in vk}
    meta = pd.read_parquet(REPO / ".scratch/valgen/train_meta.parquet", columns=["row", "lib", "ik14"])
    want = {r["inchikey14"] for r in vk}
    meta = meta[meta.ik14.isin(want)]
    meta_by_ik = defaultdict(list)
    for row, lib, ik in zip(meta.row, meta.lib, meta.ik14):
        meta_by_ik[ik].append((int(row), lib))

    D = {}
    for c in COHORTS:
        df = replay(c)
        df["refs"] = [ref_split(vk_by_mid, meta_by_ik, m, ik) for m, ik in zip(df.mid, df.ik14)]
        D[c] = df
        print(c, len(df), {k: round(float(H.rr(df[f"r_{k}"].values).mean()), 4)
                           for k in ("base", "ctl", "mf", "noice", "e006")}, flush=True)

    out = {"mrr": {}, "curve": {}, "loss": {}, "refs": {}, "protect": {}}
    for c, df in D.items():
        out["mrr"][c] = {k: float(H.rr(df[f"r_{k}"].values).mean()) for k in ("base", "ctl", "mf", "noice", "e006")}
        bins = pd.cut(df.lib_max, EDGES, right=False)
        g = df.groupby(bins, observed=False)
        out["curve"][c] = [dict(bin=str(i), n=int(len(x)), p_base_r1=float(x.leak.mean()) if len(x) else None,
                                p_e006_r1=float((x.r_e006 == 1).mean()) if len(x) else None) for i, x in g]
        # Class-1 loss decomposition: where was the truth lost on molecules not at E006 rank 1
        lost = df[df.r_e006 != 1]
        cat = np.where(lost.r_base != 1, "base_wrong",
                       np.where(lost.r_ctl != 1, "base_right_ctl_demoted",
                                np.where(lost.r_noice == 1, "ctl_right_ice_demoted", "ctl_right_overlay_demoted")))
        out["loss"][c] = {k: dict(n=int(v), mrr_lost=float((1 - H.rr(lost.r_e006.values[cat == k])).sum() / len(df)))
                          for k, v in Counter(cat).items()}
        out["refs"][c] = {k: dict(n=int(len(x)), lib_max_med=float(x.lib_max.median()),
                                  e006=float(H.rr(x.r_e006.values).mean()), ctl=float(H.rr(x.r_ctl.values).mean()),
                                  noice=float(H.rr(x.r_noice.values).mean()), base_r1=float(x.leak.mean()))
                          for k, x in df.groupby("refs")}

    dec = pd.concat([D["C2"], D["C2_hard"]])
    lock = {t: float((dec.lib_max >= t).mean()) for t in TAU_GRID}
    tau_star = min([t for t in TAU_GRID if lock[t] <= 0.01], default=0.9)
    out["c2_lock_rate"] = lock
    out["tau_star"] = tau_star
    print("C2 u C2_hard lock rate", lock, "tau*", tau_star, flush=True)

    for tau in sorted({0.9, tau_star}):
        for alt in ("noice", "ctl"):
            key = f"tau{tau}_{alt}"
            out["protect"][key] = {}
            for c, df in D.items():
                hit = df.lib_max.values >= tau
                r_new = np.where(hit, df[f"r_{alt}"].values, df.r_e006.values)
                d = H.rr(r_new) - H.rr(df.r_e006.values)
                libleak = df.leak.values & hit
                out["protect"][key][c] = dict(
                    n=int(len(df)), fired=int(hit.sum()), delta=float(d.mean()), ci90=ci90(d),
                    up=int((d > 0).sum()), down=int((d < 0).sum()),
                    displaced=int(((df.r_e006.values == 1) & (r_new != 1)).sum()),
                    d_lib_leak=float(d[libleak].sum() / len(df)), d_fp_leak=float(d[df.leak.values & ~hit].sum() / len(df)),
                    d_other=float(d[~df.leak.values].sum() / len(df)))
                print(key, c, {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in out["protect"][key][c].items() if k != "ci90"},
                      [round(x, 4) for x in out["protect"][key][c]["ci90"]], flush=True)

    (HERE / "diag.json").write_text(json.dumps(out, indent=1))
    for c, df in D.items():
        df.to_csv(HERE / f"per_mol_{c}.csv", index=False)
    print("wrote", HERE / "diag.json")


if __name__ == "__main__":
    main()
