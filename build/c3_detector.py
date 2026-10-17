"""K2 c3-detector v0 (campaign 20261015-c3-gen-slot2): dump-only features, grouped CV, break-even tau, kills.

Rules: research/harness/gate_report.md, blocks `c3-gen-slot2 campaign, PRE-REGISTERED` (K2) and
`2026-10-17: K2 amendment` (v0 decisions). In short:
  * train pseudo_c3 (pos, 600) vs the leak-free "other" C1_npx/C1_xlib/C2/C2_hard molecules (neg, L40); leak-hit
    molecules are never trained on but are scored out-of-fold (GroupKFold by truth Murcko scaffold over ALL rows);
  * verdict = standardised logistic (C=1) on the CLEAN set; the FPNet-derived set is the drop-test (reported only);
  * OOF p is odds-shifted to a C3 prior pi (0.25 verdict; 0.15 / 0.35 sensitivity);
  * h_fired = mut-first top-1 hit rate on fired positives (mutator top-1 if any mutant, else decoder top-1);
  * L_fired = mean 1/r - 1/(r+1) (E006 truth rank r in 2..25) over fired "other" negatives (pooled prior if none fire);
  * tau = smallest p with p*h_fired/2 > (1-p)*L_fired; kills at tau over ALL cohort molecules:
    C1 fire > 1%, C2 u C2_hard rank-2..5 fire > 2%, or no tau.

    python build/c3_detector.py [--raw E:/casmi/c3_gen/p4_raw.pkl] [--idea <idea dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "E:/ENVEDA_casmi/research/harness")

POST = Path(HERE) / "casmi_post.py"
IDEA = "E:/ENVEDA_casmi/research/exploit/20261017-c3-detector"
E006 = dict(METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40)
POS = "pseudo_c3"
NEGS = ("C1_npx", "C1_xlib", "C2", "C2_hard")
C1 = ("C1_npx", "C1_xlib")
C2 = ("C2", "C2_hard")
CLEAN = ("lib_max", "top_pop", "margin", "ice_cov", "ice_max", "n_base", "n_eng", "n_ice", "mass", "dbe_top")
LEAKY = ("rel", "fz_top", "S", "gen1", "ngen25")
PI, PI_SENS = 0.25, (0.15, 0.35)
L_PRIOR = 0.060            # K0 cohort-pooled L (0.058/0.088/0.048/0.048 weighted by n of "other")
C1_KILL, C2_KILL = 0.01, 0.02


# ---------------------------------------------------------------- features
def _nan(v):
    return np.nan if v is None else float(v)


def dbe(formula: str) -> float:
    c = {e: int(k or 1) for e, k in re.findall(r"([A-Z][a-z]?)(\d*)", formula.split("|")[0])}
    hal = sum(c.get(x, 0) for x in ("H", "F", "Cl", "Br", "I"))
    return c.get("C", 0) - hal / 2 + c.get("N", 0) / 2 + c.get("P", 0) / 2 + 1


def features(state, m) -> dict:
    """Per-molecule dump-only features; within-molecule / relative except the cohort-free counts and mass."""
    sm, _ik, lib, sc, fo = state["base"][m]
    pc = state["pc"][m]
    ice = [v for v in state["ice"].get(m, {}).values()]
    got = [v for v in ice if v is not None]
    sc = list(sc) + [0.0, 0.0]
    return dict(
        lib_max=_nan(lib), top_pop=_nan(pc["top_pop"]), margin=float(sc[0] - sc[1]),
        ice_cov=len(got) / len(ice) if ice else 0.0, ice_max=max(got, default=0.0),
        n_base=float(len(sm)), n_eng=float(len(state["eng"].get(m, {}).get("smiles", []))), n_ice=float(len(ice)),
        mass=float(state["targets"][m]), dbe_top=dbe(fo[0]) if fo else np.nan,
        rel=_nan(pc["fz_top"]) - _nan(pc["best_pool_fz"]), fz_top=_nan(pc["fz_top"]), S=_nan(pc["S"]),
        gen1=float(bool(fo) and fo[0].endswith("|gen")), ngen25=float(sum(f.endswith("|gen") for f in fo[:25])))


# ---------------------------------------------------------------- generator top-1 (mut-first)
def top1(items, key):
    """items [(smiles, score, ...)] -> (smiles, ik14) of the best finite-score item with a valid key, else None."""
    for it in sorted((i for i in items if i[1] is not None and np.isfinite(i[1])), key=lambda i: -i[1]):
        k = key(it[0])
        if k:
            return it[0], k
    return None


def gen_top1(rec, key) -> dict:
    """Top-1 per arm. mutfirst = mutator top-1 when the molecule has any valid mutant, else decoder top-1.
    union = the one-scale union top-1 (K1)."""
    mut, dec = top1(rec.get("mut") or [], key), top1(rec.get("dec") or [], key)
    both = sorted([(s, lp) for s, lp, *_ in (rec.get("mut") or [])] + [(s, lp) for s, lp, *_ in (rec.get("dec") or [])],
                  key=lambda i: -(i[1] if i[1] is not None and np.isfinite(i[1]) else -np.inf))
    return dict(mutfirst=mut or dec, mut=mut, dec=dec, union=top1(both, key))


# ---------------------------------------------------------------- decision math
def prior_shift(p, n_pos: int, n_neg: int, pi: float):
    """Odds-shift probabilities fitted at a n_pos:n_neg mix to a class prior pi."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    odds = p / (1 - p) * (pi / (1 - pi)) / (n_pos / n_neg)
    return odds / (1 + odds)


def slot2_cost(rank: int) -> float:
    """Cost of a slot-2 insert for a truth at E006 rank r (rank 1 and misses are untouched)."""
    return 1.0 / rank - 1.0 / (rank + 1) if 2 <= rank <= 25 else 0.0


def choose_tau(p_pos, hit_pos, p_other, cost_other, l_prior=L_PRIOR):
    """Smallest p (over observed pos/other values) with p*h_fired/2 > (1-p)*L_fired, h and L on the set fired at p."""
    p_pos, hit_pos = np.asarray(p_pos, float), np.asarray(hit_pos, float)
    p_other, cost_other = np.asarray(p_other, float), np.asarray(cost_other, float)
    for t in np.unique(np.concatenate([p_pos, p_other])):
        fp = p_pos >= t
        if not fp.any():
            break
        h = float(hit_pos[fp].mean())
        fo = p_other >= t
        L = float(cost_other[fo].mean()) if fo.any() else l_prior
        if t * h / 2 > (1 - t) * L:
            return dict(tau=float(t), h_fired=h, n_pos_fired=int(fp.sum()), L_fired=L, n_other_fired=int(fo.sum()),
                        L_from_prior=not fo.any())
    return None


def verdict(tau: dict | None, c1_fire: float, c2r25_fire: float) -> tuple[str, list[str]]:
    why = []
    if tau is None:
        why.append("no tau meets p*h/2 > (1-p)*L")
    else:
        if c1_fire > C1_KILL:
            why.append(f"C1 fire {c1_fire:.4f} > {C1_KILL}")
        if c2r25_fire > C2_KILL:
            why.append(f"C2 u C2_hard rank-2..5 fire {c2r25_fire:.4f} > {C2_KILL}")
    return ("DEAD" if why else "PASS"), why


# ---------------------------------------------------------------- CV
def make_model(kind: str):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logit":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return make_pipeline(SimpleImputer(), GradientBoostingClassifier(max_depth=2, n_estimators=150, learning_rate=0.05,
                                                                     random_state=0))


def oof_scores(X, y, train_mask, groups, kind: str, folds: int = 5):
    """GroupKFold over ALL rows; each fold model trains only on its train_mask rows, scores every held-out row."""
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        tr = tr[train_mask[tr]]
        oof[te] = make_model(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def evaluate(oof, tab, pi: float) -> dict:
    from sklearn.metrics import roc_auc_score
    y, coh, other = tab["y"], tab["cohort"], tab["other"]
    train = (y == 1) | other
    n_pos, n_neg = int((y == 1).sum()), int(((y == 0) & other).sum())
    p = prior_shift(oof, n_pos, n_neg, pi)
    pos, oth = y == 1, (y == 0) & other
    tau = choose_tau(p[pos], tab["hit"][pos], p[oth], tab["cost"][oth])
    c1 = np.isin(coh, C1)
    c2r = np.isin(coh, C2) & (tab["rank"] >= 2) & (tab["rank"] <= 5)
    t = tau["tau"] if tau else np.inf
    fired = p >= t
    c1_fire, c2r_fire = float(fired[c1].mean()), float(fired[c2r].mean()) if c2r.any() else 0.0
    v, why = verdict(tau, c1_fire, c2r_fire)
    rec = float(fired[pos].mean())
    oth_rate = float(fired[oth].mean())
    ev = (pi * rec * tau["h_fired"] / 2 - (1 - pi) * oth_rate * tau["L_fired"]) if tau else None
    return dict(pi=pi, verdict=v, why=why, auc=float(roc_auc_score(y[train], oof[train])), tau=tau,
                recall_pos=rec, fire_other=oth_rate, c1_fire=c1_fire, n_c1=int(c1.sum()), c1_fired=int(fired[c1].sum()),
                c2r25_fire=c2r_fire, n_c2r25=int(c2r.sum()), c2r25_fired=int(fired[c2r].sum()),
                fire_by_cohort={c: float(fired[coh == c].mean()) for c in (POS,) + NEGS},
                ev_indicative=ev)


# ---------------------------------------------------------------- data
def build_table(raw_path: str, log=print) -> dict:
    import harness as H
    T = H.truths()
    raw = pickle.load(open(raw_path, "rb"))["raw"]
    rows = []
    for sh in (POS,) + NEGS:
        t0 = time.time()
        state, deps, cp = H.load_dump(sh, POST)
        key = cp.canonicalize_smiles_to_ik14
        mids = [m for m in state["te_mids"] if m in T]
        tk = {m: key(T[m]["smiles"]) for m in mids}
        ranks = H.ranks_for(state, deps, cp, E006, mids, tk)
        for m, r in zip(mids, ranks):
            sm = state["base"][m][0]
            leak = bool(sm) and key(sm[0]) == tk[m]
            row = dict(mid=m, cohort=sh, smiles=T[m]["smiles"], rank=int(r), leak=leak, hit=0.0, hits={},
                       **features(state, m))
            if sh == POS:
                g = gen_top1(raw.get(m) or {}, key)
                row["hits"] = {a: float(bool(v) and v[1] == tk[m]) for a, v in g.items()}
                row["hit"] = row["hits"]["mutfirst"]
            rows.append(row)
        log(f"{sh}: {len(mids)} molecules, leak-hit {sum(r['leak'] for r in rows if r['cohort'] == sh)}, "
            f"{time.time() - t0:.0f}s", flush=True)
    return rows


def to_table(rows) -> dict:
    import harness as H
    coh = np.array([r["cohort"] for r in rows])
    y = (coh == POS).astype(int)
    leak = np.array([r["leak"] for r in rows])
    return dict(rows=rows, cohort=coh, y=y, other=(y == 0) & ~leak, rank=np.array([r["rank"] for r in rows]),
                hit=np.array([r["hit"] for r in rows]), cost=np.array([slot2_cost(r["rank"]) for r in rows]),
                groups=np.array(H.scaffold_groups([r["smiles"] for r in rows])))


def ks_shift(tab) -> dict:
    from scipy.stats import ks_2samp
    pos, oth = tab["y"] == 1, tab["other"]
    out = {}
    for f in LEAKY:
        v = np.array([r[f] for r in tab["rows"]], float)
        a, b, c = v[pos], v[oth], v[tab["y"] == 0]
        a, b, c = a[~np.isnan(a)], b[~np.isnan(b)], c[~np.isnan(c)]
        out[f] = dict(ks_vs_other=float(ks_2samp(a, b).statistic), ks_vs_all_neg=float(ks_2samp(a, c).statistic),
                      med_pos=float(np.median(a)), med_other=float(np.median(b)), med_leakhit=float(np.median(
                          v[(tab["y"] == 0) & ~oth][~np.isnan(v[(tab["y"] == 0) & ~oth])])))
    return out


def run(tab) -> dict:
    train = (tab["y"] == 1) | tab["other"]
    res = {}
    for fset, cols in (("clean", CLEAN), ("clean+fpnet", CLEAN + LEAKY)):
        X = np.array([[r[c] for c in cols] for r in tab["rows"]], float)
        for kind in ("logit", "gbdt2"):
            oof = oof_scores(X, tab["y"], train, tab["groups"], kind)
            res[f"{fset}/{kind}"] = {f"pi={pi}": evaluate(oof, tab, pi) for pi in (PI,) + PI_SENS}
    pos = tab["y"] == 1
    res["h_arms_all_pos"] = {a: float(np.mean([r["hits"].get(a, 0.0) for r in tab["rows"] if r["cohort"] == POS]))
                             for a in ("mutfirst", "mut", "dec", "union")}
    res["ks_fpnet_features"] = ks_shift(tab)
    res["n"] = dict(pos=int(pos.sum()), other=int(tab["other"].sum()), leak_hit=int(((tab["y"] == 0) & ~tab["other"]).sum()),
                    by_cohort={c: int((tab["cohort"] == c).sum()) for c in (POS,) + NEGS},
                    other_by_cohort={c: int((tab["other"] & (tab["cohort"] == c)).sum()) for c in NEGS})
    prim = res["clean/logit"][f"pi={PI}"]
    drop = res["clean+fpnet/logit"][f"pi={PI}"]
    res["K2"] = prim["verdict"]
    res["leak_dependent"] = bool(prim["verdict"] == "DEAD" and drop["verdict"] == "PASS")
    return res


def report_md(res) -> str:
    def f(x, n=4):
        return "—" if x is None else f"{x:.{n}f}"
    prim = res["clean/logit"][f"pi={PI}"]
    L = [f"# K2 result (c3-detector v0, dump-only features)", "",
         f"**K2 {res['K2']}** (primary: clean set, logistic, pi = {PI}). " + ("; ".join(prim["why"]) or "all bars met"),
         f"Drop-test: clean+FPNet logistic at pi {PI} = {res['clean+fpnet/logit'][f'pi={PI}']['verdict']}"
         + (" -> LEAK-DEPENDENT" if res["leak_dependent"] else ""), "",
         f"n: pos {res['n']['pos']}, other (train neg) {res['n']['other']} {res['n']['other_by_cohort']}, "
         f"leak-hit (scored only) {res['n']['leak_hit']}", "",
         "| fit | pi | verdict | AUC | tau | h_fired (n) | L_fired (n other) | pos recall | C1 fire (n) | C2 r2-5 fire (n) | EV |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("clean/logit", "clean/gbdt2", "clean+fpnet/logit", "clean+fpnet/gbdt2"):
        for pk, e in res[k].items():
            t = e["tau"] or {}
            L.append(f"| {k} | {e['pi']} | {e['verdict']} | {e['auc']:.3f} | {f(t.get('tau'), 3)} | "
                     f"{f(t.get('h_fired'), 3)} ({t.get('n_pos_fired', '—')}) | {f(t.get('L_fired'), 3)} "
                     f"({t.get('n_other_fired', '—')}{', prior' if t.get('L_from_prior') else ''}) | {e['recall_pos']:.3f} | "
                     f"{e['c1_fire']:.4f} ({e['c1_fired']}/{e['n_c1']}) | {e['c2r25_fire']:.4f} "
                     f"({e['c2r25_fired']}/{e['n_c2r25']}) | {f(e['ev_indicative'])} |")
    L += ["", "h by arm on all 600 pseudo_c3 (top-1): " + ", ".join(f"{a} {v:.4f}" for a, v in res["h_arms_all_pos"].items()),
          "", "FPNet-derived features, pseudo_c3 vs negatives (test.parquet has no dump; KS vs the val cohorts):", "",
          "| feature | KS vs other | KS vs all neg | median pos | median other | median leak-hit |", "|---|---|---|---|---|---|"]
    for k, v in res["ks_fpnet_features"].items():
        L.append(f"| {k} | {v['ks_vs_other']:.3f} | {v['ks_vs_all_neg']:.3f} | {v['med_pos']:.2f} | {v['med_other']:.2f} | "
                 f"{v['med_leakhit']:.2f} |")
    L += ["", "EV = pi*recall*h_fired/2 - (1-pi)*fire_other*L_fired (indicative only; the val 'other' rate is not the test rate).",
          "Test fire-rate sanity: not measured in v0 (no test dump; required before any LB slot)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="E:/casmi/c3_gen/p4_raw.pkl")
    ap.add_argument("--idea", default=IDEA)
    ap.add_argument("--table", default="E:/casmi/c3_gen/k2_table.pkl", help="cache of the feature table")
    a = ap.parse_args(argv)
    if os.path.exists(a.table):
        rows = pickle.load(open(a.table, "rb"))
        print(f"table from cache {a.table}: {len(rows)} rows", flush=True)
    else:
        rows = build_table(a.raw)
        pickle.dump(rows, open(a.table, "wb"), protocol=4)
    res = run(to_table(rows))
    os.makedirs(a.idea, exist_ok=True)
    json.dump(res, open(os.path.join(a.idea, "k2_result.json"), "w"), indent=1)
    md = report_md(res)
    open(os.path.join(a.idea, "k2_result.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    return 0 if res["K2"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
