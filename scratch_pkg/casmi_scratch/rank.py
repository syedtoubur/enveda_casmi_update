"""Two-tier ranking and slot filling (stage A -> B -> C), strictly deduplicated on ik14."""
import numpy as np


def avg_rank_desc(x):
    """1-based ranks of x in descending order, ties share their average rank (NaN ranks last, tied)."""
    x = np.where(np.isnan(x), -np.inf, np.asarray(x, np.float64))
    o = np.argsort(-x, kind="mergesort")
    xs = -x[o]
    r = np.empty(len(x), np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def fp_rrf(fps, weights, k, idx=None):
    """Sum of w_i / (k + rank(fp_i)) over the FP channels (R4), restricted to idx. fps: one array or a list of them."""
    fps = [fps] if isinstance(fps, np.ndarray) else list(fps)
    weights = [1.0] * len(fps) if weights is None else list(weights)
    out = 0.0
    for f, w in zip(fps, weights):
        f = np.asarray(f, np.float64)
        out = out + w / (k + avg_rank_desc(f if idx is None else f[idx]))
    return out


def analog_rrf(score, k, w=1.0):
    """R5 RRF term w / (k + rank) of the scored candidates (rank among them, ties averaged). Unscored (NaN)
    candidates get the mean term of the scored ones, so the list only reorders what it scores; none scored -> 0."""
    score = np.asarray(score, np.float64)
    ok = np.isfinite(score)
    out = np.zeros(len(score), np.float64)
    if ok.any():
        out[ok] = w / (k + avg_rank_desc(score[ok]))
        out[~ok] = out[ok].mean()
    return out


def rank_two_tier(lv, sv, fr, ppm, tau, bonus=None, fp=None, rrf_k=3.0, fp_w=None, an=None, an_w=1.0):
    """Order (indices) of one candidate set.

    Tier 1 = real-library match lv >= tau, by lv desc (then sv, fr, |ppm|). Tier 2 = the rest by sv (+ bonus) desc,
    fr desc, |ppm| asc. A missing sv / fr (NaN) counts 0; a missing ppm sorts last. bonus (R2 class vote) only
    touches tier 2; None = the R1 order. fp (R3, f.z per candidate, NaN = no fingerprint) replaces the tier-2 key
    by RRF: 1/(k + rank(sv + bonus)) + 1/(k + rank(f.z)), average ranks for ties, within tier 2 only.
    R4: fp may be a list of f.z arrays (one per FPNet family), each adding fp_w[i] / (k + rank(f.z_i)).
    R5: an (analogue score per candidate, NaN = not scored) adds analog_rrf(an, k, an_w) to the same tier-2 key.
    """
    lv, sv, fr = (np.nan_to_num(np.asarray(x, np.float64), nan=0.0) for x in (lv, sv, fr))
    ppm = np.nan_to_num(np.asarray(ppm, np.float64), nan=np.inf)
    s2 = sv if bonus is None else sv + np.nan_to_num(np.asarray(bonus, np.float64), nan=0.0)
    t1, t2 = np.flatnonzero(lv >= tau), np.flatnonzero(lv < tau)
    o1 = t1[np.lexsort((ppm[t1], -fr[t1], -sv[t1], -lv[t1]))]
    if (fp is not None or an is not None) and len(t2):
        key = 1.0 / (rrf_k + avg_rank_desc(s2[t2]))
        if fp is not None:
            key = key + fp_rrf(fp, fp_w, rrf_k, t2)
        if an is not None:
            key = key + analog_rrf(np.asarray(an, np.float64)[t2], rrf_k, an_w)
        o2 = t2[np.lexsort((ppm[t2], -fr[t2], -key))]
    else:
        o2 = t2[np.lexsort((ppm[t2], -fr[t2], -s2[t2]))]
    return np.concatenate([o1, o2]).astype(np.int64)


class Slots:
    """Up to `topn` (ik14, smiles) picks, never two with the same ik14."""

    def __init__(self, topn):
        self.topn, self.keys, self.smiles, self.seen = topn, [], [], set()

    @property
    def full(self):
        return len(self.smiles) >= self.topn

    def add(self, key, smi):
        if self.full or key is None:
            return False
        k14 = str(key)[:14]
        if len(k14) != 14 or k14 in self.seen:
            return False
        self.seen.add(k14)
        self.keys.append(k14)
        self.smiles.append(smi)
        return True

    def take(self, pool, idx_in_order):
        for c in idx_in_order:
            if self.full:
                break
            self.add(pool.keys[c], pool.smiles[c])

    def padded(self, pad="CCO"):
        return self.smiles[:self.topn] + [pad] * max(0, self.topn - len(self.smiles))
