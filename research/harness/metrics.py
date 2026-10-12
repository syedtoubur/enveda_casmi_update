"""P1.5 metrics on per-molecule replay results: weighted MRR@25, grouped paired bootstrap, per-stratum sign,
grouped K-fold, test-covariate importance weights and their SMD check (G-gen3: every |SMD| < 0.1).

Ranks follow harness.rank_of: 1..25, 0 = not in the top 25. Covariate tables are anything indexable by column name
(DataFrame or dict of arrays); non-numeric columns are one-hot encoded over the union of levels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import rr  # noqa: E402


def _w(weights, n: int) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, float)


def mrr_at25(ranks, weights=None) -> float:
    """Weighted mean reciprocal rank; rank 0 (or > 25) scores 0."""
    r = np.asarray(ranks)
    r = np.where(r > 25, 0, r)
    w = _w(weights, len(r))
    return float((w * rr(r)).sum() / w.sum()) if len(r) else float("nan")


def paired_bootstrap(delta_per_mol, weights=None, n: int = 2000, seed: int = 0, groups=None):
    """(weighted mean Δ, CI90 lo, CI90 hi); resamples whole groups (structure/scaffold ids) when given.
    With no weights and no groups it draws exactly as harness.bootstrap."""
    d = np.asarray(delta_per_mol, float)
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    w = _w(weights, len(d))
    if groups is None:
        gid = np.arange(len(d))
    else:
        _, gid = np.unique(np.asarray(groups), return_inverse=True)
    G = gid.max() + 1
    swd, sw = np.bincount(gid, w * d, G), np.bincount(gid, w, G)
    idx = np.random.default_rng(seed).integers(0, G, (n, G))
    m = swd[idx].sum(1) / np.maximum(sw[idx].sum(1), 1e-300)
    return float((w * d).sum() / w.sum()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def stratum_signs(delta, strata, weights=None) -> dict:
    """{'per': {stratum: {'delta', 'n'}}, 'all_same_sign': bool}; a zero Δ in any stratum is not a sign."""
    d, s = np.asarray(delta, float), np.asarray(strata)
    w = _w(weights, len(d))
    per = {}
    for k in sorted(set(s.tolist()), key=str):
        m = s == k
        per[k] = dict(delta=float((w[m] * d[m]).sum() / w[m].sum()), n=int(m.sum()))
    sg = {np.sign(v["delta"]) for v in per.values()}
    return dict(per=per, all_same_sign=bool(len(sg) == 1 and 0 not in sg))


def grouped_kfold(groups, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_idx, test_idx)] * k; every group lands in exactly one test fold (largest groups placed first)."""
    g = np.asarray(groups)
    u, gid, cnt = np.unique(g, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(len(u)), key=lambda i: -cnt[i])
    fold_of, size = np.empty(len(u), int), np.zeros(k, int)
    for i in order:
        f = int(np.argmin(size + rng.random(k) * 1e-3))
        fold_of[i], size[f] = f, size[f] + cnt[i]
    fid = fold_of[gid]
    return [(np.flatnonzero(fid != f), np.flatnonzero(fid == f)) for f in range(k)]


# ---------------------------------------------------------------- covariate shift
def _encode(a, b, cols) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Numeric columns as is (NaN -> pooled mean), others one-hot over the union of levels."""
    xa, xb, names = [], [], []
    for c in cols:
        ca, cb = np.asarray(a[c]), np.asarray(b[c])
        try:
            fa, fb = ca.astype(float), cb.astype(float)
            mu = np.nanmean(np.concatenate([fa, fb]))
            xa.append(np.where(np.isnan(fa), mu, fa)); xb.append(np.where(np.isnan(fb), mu, fb)); names.append(c)
        except (TypeError, ValueError):
            for lv in sorted(set(map(str, ca)) | set(map(str, cb))):
                xa.append((ca.astype(str) == lv).astype(float)); xb.append((cb.astype(str) == lv).astype(float))
                names.append(f"{c}={lv}")
    return np.column_stack(xa), np.column_stack(xb), names


def importance_weights(val_cov, ref_cov, cols, clip=(0.1, 10.0), C: float = 1.0) -> np.ndarray:
    """Density ratio p_ref/p_val per val row from a logistic classifier on standardized covariates; mean 1."""
    from sklearn.linear_model import LogisticRegression
    xv, xr, _ = _encode(val_cov, ref_cov, cols)
    X = np.vstack([xv, xr])
    mu, sd = X.mean(0), X.std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1)
    y = np.r_[np.zeros(len(xv)), np.ones(len(xr))]
    clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    logit = clf.decision_function(X[:len(xv)])
    w = np.exp(logit) * len(xv) / len(xr)
    w /= w.mean()
    w = np.clip(w, *clip)
    return w / w.mean()


def smd(val_cov, ref_cov, weights=None, cols=None) -> dict[str, float]:
    """Weighted standardized mean difference (val weighted vs ref) per encoded column; G-gen3 wants all |SMD| < 0.1."""
    cols = list(cols) if cols is not None else list(val_cov.keys())
    xv, xr, names = _encode(val_cov, ref_cov, cols)
    w = _w(weights, len(xv))
    mv = (w[:, None] * xv).sum(0) / w.sum()
    vv = (w[:, None] * (xv - mv) ** 2).sum(0) / w.sum()
    mr, vr = xr.mean(0), xr.var(0)
    den = np.sqrt((vv + vr) / 2)
    return {n: float((a - b) / s) if s > 0 else 0.0 for n, a, b, s in zip(names, mv, mr, den)}
