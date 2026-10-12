"""Synthetic tests for metrics.py and combiner.py.  python research/harness/test_metrics.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402
from combiner import Combiner, cv_report, rrf, truth_rank, tuned_rrf  # noqa: E402
from metrics import grouped_kfold, importance_weights, mrr_at25, paired_bootstrap, smd, stratum_signs  # noqa: E402


def test_mrr():
    assert abs(mrr_at25([1, 2, 0, 4], [1, 2, 1, 0]) - (1 + 2 * 0.5) / 4) < 1e-12
    assert abs(mrr_at25([1, 2, 0, 4]) - (1 + 0.5 + 0.25) / 4) < 1e-12
    assert mrr_at25([30, 1]) == 0.5


def test_bootstrap():
    rng = np.random.default_rng(1)
    d = 0.02 + rng.normal(0, 0.2, 2000)
    m, lo, hi = paired_bootstrap(d)
    assert lo < 0.02 < hi and abs(m - d.mean()) < 1e-12
    assert (lo, hi) == harness.bootstrap(d)                       # same draw as the harness without weights/groups
    g = np.repeat(np.arange(200), 10)                             # correlated within a group -> wider grouped CI
    dg = 0.02 + rng.normal(0, 0.2, 200)[g] + rng.normal(0, 0.05, 2000)
    _, l1, h1 = paired_bootstrap(dg)
    _, l2, h2 = paired_bootstrap(dg, groups=g)
    assert l2 < 0.02 < h2 and (h2 - l2) > 2 * (h1 - l1)
    w = rng.uniform(0.5, 2, 2000)
    m, lo, hi = paired_bootstrap(d, weights=w)
    assert abs(m - (w * d).sum() / w.sum()) < 1e-12 and lo < m < hi


def test_strata():
    r = stratum_signs([0.1, 0.2, -0.05, 0.3], ["a", "a", "b", "b"], [1, 1, 1, 1])
    assert r["per"]["a"] == dict(delta=0.15000000000000002, n=2) or abs(r["per"]["a"]["delta"] - 0.15) < 1e-12
    assert r["all_same_sign"]
    assert not stratum_signs([0.1, -0.3], ["a", "b"])["all_same_sign"]
    assert not stratum_signs([0.1, 0.0], ["a", "b"])["all_same_sign"]


def test_folds():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 137, 1000)
    folds = grouped_kfold(g, 5)
    te = np.concatenate([t for _, t in folds])
    assert len(folds) == 5 and sorted(te.tolist()) == list(range(1000))
    seen = {}
    for f, (tr, t) in enumerate(folds):
        assert not set(tr) & set(t)
        for x in set(g[t].tolist()):
            assert seen.setdefault(x, f) == f                     # no group in two folds
    sz = [len(t) for _, t in folds]
    assert max(sz) - min(sz) < 60


def test_importance():
    rng = np.random.default_rng(3)
    nv, nr = 3000, 1000
    val = dict(mass=rng.normal(400, 80, nv), ce=rng.choice([20.0, 40.0, np.nan], nv),
               adduct=rng.choice(["[M+H]+", "[M+Na]+", "[M-H]-"], nv, p=[0.5, 0.2, 0.3]))
    ref = dict(mass=rng.normal(470, 70, nr), ce=rng.choice([20.0, 40.0], nr, p=[0.7, 0.3]),
               adduct=rng.choice(["[M+H]+", "[M+Na]+", "[M-H]-"], nr, p=[0.8, 0.15, 0.05]))
    cols = ["mass", "ce", "adduct"]
    before = smd(val, ref, None, cols)
    w = importance_weights(val, ref, cols)
    after = smd(val, ref, w, cols)
    assert abs(w.mean() - 1) < 1e-9 and w.min() >= 0.1 / 1.5 and w.max() <= 10 * 1.5
    assert max(map(abs, before.values())) > 0.5 and max(map(abs, after.values())) < 0.1, (before, after)
    return before, after


def synth(n: int = 300, n_cand: int = 40, seed: int = 4) -> list[dict]:
    """'good' ranks a latent where the truth leads; 'noise' is a random order; 'sc' is a weak raw score."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lat = rng.normal(0, 1, n_cand); t = rng.integers(n_cand); lat[t] += 1.5
        good = np.argsort(np.argsort(-(lat + rng.normal(0, 0.7, n_cand)))) + 1.0
        noise = rng.permutation(n_cand) + 1.0
        good[rng.random(n_cand) < 0.1] = np.nan                   # absent from that list
        sc = lat * 0.3 + rng.normal(0, 1, n_cand)
        y = np.zeros(n_cand, bool); y[t] = True
        out.append(dict(mid=f"m{i}", cands=dict(good=good, noise=noise, sc=sc), is_truth=y,
                        group=f"g{i // 3}", stratum="C1" if i % 2 else "C2", w=float(rng.uniform(0.5, 2))))
    return out


CH = dict(good="rank", noise="rank", sc="score")


def test_combiner():
    mols = synth()
    c = Combiner(CH, p_drop=0.0).fit(mols)
    co = c.coef()
    assert c.n_params == 6 and co["good"] > abs(co["noise"]) + 0.5, co
    assert c.loss_[-1] < c.loss_[0]
    o = c.rank(mols[0]); assert sorted(o.tolist()) == list(range(40))
    rep = cv_report(mols, k=5, baseline=lambda tr: rrf(CH), make=lambda: Combiner(CH))
    assert rep["delta"] > 0.02 and rep["ci90"][0] > 0, rep
    assert len(rep["folds"]) == 5 and sum(f["delta"] > 0 for f in rep["folds"]) >= 4
    assert set(rep["strata"]["per"]) == {"C1", "C2"} and rep["lb_bar"]
    tuned = cv_report(mols, k=5, baseline=tuned_rrf(CH), make=lambda: Combiner(CH))
    assert tuned["mrr_base"] > rep["mrr_base"]                   # tuning drops the noise channel
    return rep, tuned


def test_dropout():
    mols = synth(seed=5)
    a = Combiner(CH, p_drop=0.0).fit(mols)
    b = Combiner(CH, groups=dict(rk=["good", "noise"], sc=["sc"]), p_drop=0.3, epochs=400).fit(mols)
    tail = np.array(b.loss_[-50:])
    assert tail.mean() < b.loss_[0] - 0.3 and tail.std() < 0.1, (b.loss_[0], tail.mean(), tail.std())
    assert b.coef()["good"] > abs(b.coef()["noise"]) + 0.5
    ra = mrr_at25([truth_rank(a.scores(m), m["is_truth"]) for m in mols])
    rb = mrr_at25([truth_rank(b.scores(m), m["is_truth"]) for m in mols])
    assert rb > ra - 0.03, (ra, rb)
    return ra, rb


def main() -> int:
    test_mrr(); test_bootstrap(); test_strata(); test_folds()
    before, after = test_importance()
    rep, tuned = test_combiner()
    ra, rb = test_dropout()
    print("metrics tests ok",
          dict(smd_before=round(max(map(abs, before.values())), 3), smd_after=round(max(map(abs, after.values())), 3)))
    print("combiner vs uniform RRF", dict(base=round(rep["mrr_base"], 3), comb=round(rep["mrr_comb"], 3),
                                          delta=round(rep["delta"], 3), ci90=[round(x, 3) for x in rep["ci90"]]))
    print("combiner vs tuned RRF  ", dict(base=round(tuned["mrr_base"], 3), comb=round(tuned["mrr_comb"], 3),
                                          delta=round(tuned["delta"], 4), ci90=[round(x, 4) for x in tuned["ci90"]],
                                          lb_bar=tuned["lb_bar"]))
    print("dropout p=0 / p=0.3 in-sample MRR", round(ra, 3), round(rb, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
