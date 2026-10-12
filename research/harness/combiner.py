"""Small learned combiner over per-candidate channel scores (P1.5; ≤ ~20 params, grouped CV, channel dropout).

A molecule is a dict:
    mid       str
    cands     {channel: array(n_cand)}  rank channels: 1-based rank in that channel's list, NaN = absent;
                                        score channels: raw score, NaN = not scored
    is_truth  bool array(n_cand)        candidates must already be IK14-unique (one row per IK14)
    w, group, stratum                   optional: importance weight, structure/scaffold id, stratum label
Features per channel c: rank -> (k+1)/(k+rank) (rank 1 = 1), score -> z-score within the molecule; NaN -> 0 plus a
missing indicator. Score = linear in the 2C features (C ≤ 10 -> ≤ 20 params), listwise softmax loss + L2, full-batch Adam.
Channel dropout (regulariser only, it does not fix a leak): with prob p per molecule per epoch one channel group is set
to 0 with its indicator on.

LB bar (handoff P1.5): grouped-CV Δ ≥ +0.005 over tuned RRF, CI90 lo > 0, same sign in every stratum (+ the anchor shard).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import rr  # noqa: E402
from metrics import grouped_kfold, mrr_at25, paired_bootstrap, stratum_signs  # noqa: E402

Scorer = Callable[[dict], np.ndarray]


def truth_rank(scores: np.ndarray, is_truth) -> int:
    """1-based rank of the best-placed truth in the top 25 (stable sort, input order breaks ties), else 0."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")[:25]
    hit = np.flatnonzero(np.asarray(is_truth, bool)[order])
    return int(hit[0]) + 1 if len(hit) else 0


def _within_rank(v: np.ndarray) -> np.ndarray:
    """Descending-score rank (1-based) within a molecule; NaN stays NaN."""
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    out[np.flatnonzero(ok)[np.argsort(-v[ok], kind="stable")]] = np.arange(1, ok.sum() + 1)
    return out


# ---------------------------------------------------------------- RRF baselines
def rrf(channels: dict[str, str], weights: dict[str, float] | None = None, k: float = 60) -> Scorer:
    """Weighted RRF: sum_c w_c / (k + rank_c); score channels are ranked within the molecule first."""
    def score(mol):
        s = np.zeros(len(mol["is_truth"]))
        for c, kind in channels.items():
            r = np.asarray(mol["cands"][c], float)
            r = r if kind == "rank" else _within_rank(r)
            s += (1.0 if weights is None else weights.get(c, 1.0)) * np.where(np.isnan(r), 0.0, 1.0 / (k + r))
        return s
    return score


def tuned_rrf(channels: dict[str, str], grid=(0.0, 0.5, 1.0, 2.0), k: float = 60, passes: int = 2):
    """Baseline factory: coordinate ascent of per-channel RRF weights on the training molecules' weighted MRR."""
    def factory(train: list[dict]) -> Scorer:
        w = {c: 1.0 for c in channels}
        def m(wt):
            f = rrf(channels, wt, k)
            return mrr_at25([truth_rank(f(x), x["is_truth"]) for x in train], [x.get("w", 1.0) for x in train])
        best = m(w)
        for _ in range(passes):
            for c in channels:
                for g in grid:
                    cand = {**w, c: g}
                    v = m(cand)
                    if v > best + 1e-12:
                        best, w = v, cand
        f = rrf(channels, w, k)
        f.weights = w
        return f
    return factory


# ---------------------------------------------------------------- combiner
class Combiner:
    def __init__(self, channels: dict[str, str], groups: dict[str, list[str]] | None = None, k: float = 60,
                 l2: float = 1e-3, p_drop: float = 0.3, epochs: int = 300, lr: float = 0.05, seed: int = 0):
        self.ch = list(channels)
        self.kind = channels
        self.groups = groups or {c: [c] for c in self.ch}
        self.k, self.l2, self.p_drop, self.epochs, self.lr, self.seed = k, l2, p_drop, epochs, lr, seed
        self.theta = np.zeros(2 * len(self.ch))
        self.loss_ = []

    @property
    def n_params(self) -> int:
        return len(self.theta)

    def features(self, mol: dict) -> np.ndarray:
        n = len(mol["is_truth"])
        val, ind = np.zeros((n, len(self.ch))), np.zeros((n, len(self.ch)))
        for j, c in enumerate(self.ch):
            v = np.asarray(mol["cands"].get(c, np.full(n, np.nan)), float)
            miss = np.isnan(v)
            if self.kind[c] == "rank":
                t = (self.k + 1) / (self.k + np.where(miss, np.inf, v))
            else:
                ok = v[~miss]
                sd = ok.std() if len(ok) > 1 else 0.0
                t = (v - (ok.mean() if len(ok) else 0.0)) / (sd if sd > 0 else 1.0)
            val[:, j], ind[:, j] = np.where(miss, 0.0, t), miss
        return np.hstack([val, ind])

    def coef(self) -> dict[str, float]:
        C = len(self.ch)
        return {**{c: float(self.theta[j]) for j, c in enumerate(self.ch)},
                **{f"{c}:missing": float(self.theta[C + j]) for j, c in enumerate(self.ch)}}

    def fit(self, train_mols: list[dict]) -> "Combiner":
        mols = [m for m in train_mols if len(m["is_truth"]) and np.any(m["is_truth"])]
        if not mols:
            return self
        X = np.vstack([self.features(m) for m in mols])
        y = np.concatenate([np.asarray(m["is_truth"], bool) for m in mols])
        sizes = np.array([len(m["is_truth"]) for m in mols])
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        seg = np.repeat(np.arange(len(mols)), sizes)
        w = np.array([float(m.get("w", 1.0)) for m in mols]); w /= w.sum()
        C, gidx = len(self.ch), [[self.ch.index(c) for c in g] for g in self.groups.values()]
        rng = np.random.default_rng(self.seed)
        th, mo, ve = self.theta.copy(), np.zeros_like(self.theta), np.zeros_like(self.theta)
        self.loss_ = []
        for ep in range(1, self.epochs + 1):
            Xe = X
            if self.p_drop > 0:
                drop = np.zeros((len(mols), C), bool)
                hit = np.flatnonzero(rng.random(len(mols)) < self.p_drop)
                pick = rng.integers(0, len(gidx), len(hit))
                for g, cols in enumerate(gidx):
                    drop[np.ix_(hit[pick == g], cols)] = True
                rows = drop[seg]
                Xe = X.copy()
                Xe[:, :C][rows], Xe[:, C:][rows] = 0.0, 1.0
            s = Xe @ th
            s = s - np.maximum.reduceat(s, starts)[seg]
            e = np.exp(s)
            p = e / np.add.reduceat(e, starts)[seg]
            et = np.where(y, e, 0.0)
            q = et / np.add.reduceat(et, starts)[seg]
            loss = -(w * np.log(np.add.reduceat(np.where(y, p, 0.0), starts) + 1e-300)).sum() + self.l2 * th @ th
            g = Xe.T @ (w[seg] * (p - q)) + 2 * self.l2 * th
            mo = 0.9 * mo + 0.1 * g
            ve = 0.999 * ve + 0.001 * g * g
            th -= self.lr * (mo / (1 - 0.9 ** ep)) / (np.sqrt(ve / (1 - 0.999 ** ep)) + 1e-8)
            self.loss_.append(float(loss))
        self.theta = th
        return self

    def scores(self, mol: dict) -> np.ndarray:
        return self.features(mol) @ self.theta

    def rank(self, mol: dict) -> np.ndarray:
        """Candidate indices, best first."""
        return np.argsort(-self.scores(mol), kind="stable")


def cv_report(mols: list[dict], groups=None, k: int = 5, *, baseline: Callable[[list[dict]], Scorer],
              make: Callable[[], Combiner], seed: int = 0) -> dict:
    """Grouped K-fold: combiner vs a baseline factory (tuned RRF; fit on each training fold) on the held-out fold.
    Δ is per molecule (rr combiner − rr baseline), weighted by mol['w']; the CI resamples groups."""
    groups = np.asarray(groups if groups is not None else [m.get("group", m.get("mid", i)) for i, m in enumerate(mols)])
    w = np.array([float(m.get("w", 1.0)) for m in mols])
    rc, rb, folds = np.zeros(len(mols), int), np.zeros(len(mols), int), []
    coefs = []
    for tr, te in grouped_kfold(groups, k, seed):
        train = [mols[i] for i in tr]
        comb, base = make().fit(train), baseline(train)
        coefs.append(comb.theta)
        for i in te:
            rc[i] = truth_rank(comb.scores(mols[i]), mols[i]["is_truth"])
            rb[i] = truth_rank(base(mols[i]), mols[i]["is_truth"])
        folds.append(dict(n=len(te), mrr_comb=mrr_at25(rc[te], w[te]), mrr_base=mrr_at25(rb[te], w[te])))
        folds[-1]["delta"] = folds[-1]["mrr_comb"] - folds[-1]["mrr_base"]
    d = rr(rc) - rr(rb)
    mean, lo, hi = paired_bootstrap(d, w, groups=groups, seed=seed)
    res = dict(n=len(mols), folds=folds, mrr_comb=mrr_at25(rc, w), mrr_base=mrr_at25(rb, w), delta=mean,
               ci90=[lo, hi], up=int((d > 0).sum()), down=int((d < 0).sum()),
               theta_mean=np.mean(coefs, 0).round(4).tolist(), oof_ranks=dict(comb=rc.tolist(), base=rb.tolist()))
    if all("stratum" in m for m in mols):
        res["strata"] = stratum_signs(d, [m["stratum"] for m in mols], w)
    res["lb_bar"] = bool(mean >= 0.005 and lo > 0 and res.get("strata", {"all_same_sign": True})["all_same_sign"])
    return res
