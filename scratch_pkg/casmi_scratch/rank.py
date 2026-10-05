"""Two-tier ranking and slot filling (stage A -> B -> C), strictly deduplicated on ik14."""
import numpy as np


def rank_two_tier(lv, sv, fr, ppm, tau):
    """Order (indices) of one candidate set.

    Tier 1 = real-library match lv >= tau, by lv desc (then sv, fr, |ppm|). Tier 2 = the rest by sv desc, fr desc,
    |ppm| asc. A missing sv / fr (NaN) counts 0; a missing ppm sorts last.
    """
    lv, sv, fr = (np.nan_to_num(np.asarray(x, np.float64), nan=0.0) for x in (lv, sv, fr))
    ppm = np.nan_to_num(np.asarray(ppm, np.float64), nan=np.inf)
    t1, t2 = np.flatnonzero(lv >= tau), np.flatnonzero(lv < tau)
    o1 = t1[np.lexsort((ppm[t1], -fr[t1], -sv[t1], -lv[t1]))]
    o2 = t2[np.lexsort((ppm[t2], -fr[t2], -sv[t2]))]
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
