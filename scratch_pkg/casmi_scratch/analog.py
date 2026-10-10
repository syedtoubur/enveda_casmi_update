"""Analog library search with a shift-aware entropy similarity, and the NP class vote (R1 CV diagnostic, R2 candidate).

A query peak matches a library peak directly (|dmz| <= tol) or shifted by the precursor difference
(|q - (c + shift)| <= tol, shift = query neutral mass - library neutral mass): the GNPS modified-cosine idea applied
to our entropy similarity. Pairs are assigned one-to-one, greedily by intensity product.

Two candidate sets (gate CV in specs/specs_scratch/gate_report.md):
- "delta": library spectra at the query mass or one NP biosynthetic delta away (each +-ppm);
- "open":  library spectra within +-200 Da.
"""
import numpy as np
from numba import njit, prange

from .kernels import clean
from .spectra import clean_spectrum

# NP biosynthetic mass deltas (Da), fixed before the CV measurement
NP_DELTAS = {
    "CH2": 14.015650, "O": 15.994915, "H2O": 18.010565, "CH2O": 30.010565, "C2H2O acetyl": 42.010565,
    "CO2": 43.989829, "C5H8 prenyl": 68.062600, "C3H2O3 malonyl": 86.000394, "C5H8O4 pentose": 132.042259,
    "C9H6O2 coumaroyl": 146.036779, "C6H10O4 deoxyhexose": 146.057909, "C6H10O5 hexose": 162.052824,
    "C6H8O6 glucuronide": 176.032088,
}
OPEN_DA = 200.0


@njit(cache=True, fastmath=True)
def shift_entropy_sim(qmz, qp, cmz, cp, shift, tol):
    """Entropy similarity with direct and precursor-shifted peak matches (both inputs cleaned, m/z ascending)."""
    n = len(qmz)
    m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    use_shift = abs(shift) > tol
    cap = 4 * (n + m) + 16
    pi = np.empty(cap, np.int64)
    pj = np.empty(cap, np.int64)
    pw = np.empty(cap, np.float64)
    k = 0
    for pas in range(2 if use_shift else 1):
        off = 0.0 if pas == 0 else shift
        lo = 0
        for i in range(n):
            while lo < m and cmz[lo] + off < qmz[i] - tol:
                lo += 1
            j = lo
            while j < m and cmz[j] + off <= qmz[i] + tol:
                if k >= cap:
                    break
                pi[k] = i
                pj[k] = j
                # direct pairs win ties against shifted ones
                pw[k] = qp[i] * cp[j] * (1.0 + 1e-6 if pas == 0 else 1.0)
                k += 1
                j += 1
    uq = np.zeros(n, np.bool_)
    uc = np.zeros(m, np.bool_)
    mq = np.zeros(n, np.float64)        # matched partner intensity added to the query peak
    if k > 0:
        o = np.argsort(-pw[:k])
        for t in range(k):
            i = pi[o[t]]
            j = pj[o[t]]
            if uq[i] or uc[j]:
                continue
            uq[i] = True
            uc[j] = True
            mq[i] = cp[j]
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    tot = 0.0
    for x in range(n):
        tot += qp[x] + mq[x]
    for x in range(m):
        if not uc[x]:
            tot += cp[x]
    if tot <= 0:
        return 0.0
    SAB = 0.0
    for x in range(n):
        v = (qp[x] + mq[x]) / tot
        if v > 0:
            SAB -= v * np.log(v)
    for x in range(m):
        if not uc[x]:
            v = cp[x] / tot
            if v > 0:
                SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, qnm, cand, off, allmz, allin, lib_nm, tol, floor, topk, power, ent_weight):
    """Shift-aware entropy similarity of one cleaned query (neutral mass qnm) against library spectra `cand`."""
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm, cp = clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        out[k] = shift_entropy_sim(qmz, qp, cm, cp, qnm - lib_nm[c], tol)
    return out


def candidates(lib, target, mode, ppm=10.0, open_da=OPEN_DA):
    """Library spectrum indices for an analog search around target ('delta' or 'open')."""
    if mode == "open":
        return lib.window(target, open_da)
    assert mode == "delta", mode
    parts = [lib.window(target, target * ppm / 1e6)]
    for d in NP_DELTAS.values():
        for t in (target - d, target + d):
            parts.append(lib.window(t, t * ppm / 1e6))
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def analog_hits(lib, specs, target, cfg, mode, polarity=0, exclude=None, ppm=10.0, open_da=OPEN_DA):
    """Best shift-aware similarity per library ik over a molecule's spectra -> {ik: sim}.

    polarity: +1 / -1 keeps library spectra of that polarity only (0 = all). exclude: boolean mask over library
    spectra (the own-structure ban).
    """
    cand = candidates(lib, target, mode, ppm, open_da)
    if polarity and lib.pol is not None and len(cand):
        cand = cand[lib.pol[cand] == polarity]
    if exclude is not None and len(cand):
        cand = cand[~exclude[cand]]
    if len(cand) == 0:
        return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it, cfg)
        if len(qm) == 0:
            continue
        sc = search_shift(qm, qp, float(target), cand, lib.off, lib.mz, lib.it, lib.nm, cfg.mz_tol, cfg.int_floor,
                          cfg.max_peaks, cfg.int_power, cfg.ent_weight)
        for c, s in zip(cand, sc):
            kk = lib.ik[c]
            if s > agg.get(kk, -1.0):
                agg[kk] = float(s)
    return agg


# ---------------------------------------------------------------- class vote

def class_vote(hits, label_of, k=10, min_sim=0.5):
    """Top-k hits with sim >= min_sim vote their structure's label, weighted by similarity; unlabelled hits abstain.

    hits: {key: sim}; label_of: key -> label or None. -> (winning label or None, {label: weight}, n voters).
    """
    top = sorted(((s, key) for key, s in hits.items() if s >= min_sim), key=lambda x: (-x[0], x[1]))[:k]
    w = {}
    for s, key in top:
        lab = label_of(key)
        if isinstance(lab, str) and lab:
            w[lab] = w.get(lab, 0.0) + s
    if not w:
        return None, {}, 0
    win = sorted(w.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return win, w, sum(1 for s, key in top if isinstance(label_of(key), str) and label_of(key))


def vote_conf(w, win):
    """Share of the voting weight that went to the winning label (0 when nothing voted)."""
    tot = sum(w.values())
    return float(w.get(win, 0.0) / tot) if tot > 0 and win is not None else 0.0


def vote_bonus(sc, sp, names, voted, conf, beta):
    """R2 tier-2 bonus per candidate: beta * conf * P(candidate has the voted superclass).

    sc / sp: the candidates' NPClassifier top-k superclass codes (-1 = none) and probabilities. P is the probability
    of the voted superclass among the candidate's top k (0 if absent).
    """
    sc, sp = np.atleast_2d(sc), np.atleast_2d(sp)
    if voted is None or voted not in names or conf <= 0 or len(sc) == 0:
        return np.zeros(len(sc), np.float32)
    p = np.where(sc == names.index(voted), sp, 0.0).sum(axis=1)
    return (beta * conf * p).astype(np.float32)


def label_match(pred, truth):
    """A truth label may hold several NPClassifier values ('A|B' or 'A; B'): a hit on any of them counts."""
    if not isinstance(pred, str) or not isinstance(truth, str):
        return False
    parts = {p.strip() for p in truth.replace(";", "|").split("|")}
    return pred.strip() in parts or pred == truth


def vote_report(preds, truths):
    """Per-molecule predicted / truth labels -> coverage, purity, majority baseline (molecules with a truth label).

    coverage = share with a vote; purity = share of voted molecules whose vote matches; baseline = share of voted
    molecules carrying the most frequent truth label (always guessing it).
    """
    rows = [(p, t) for p, t in zip(preds, truths) if isinstance(t, str) and t]
    n = len(rows)
    voted = [(p, t) for p, t in rows if isinstance(p, str)]
    if not voted:
        return dict(n_with_truth=n, n_voted=0, coverage=0.0 if n else float("nan"), purity=float("nan"),
                    baseline=float("nan"), majority=None)
    first = [t.replace(";", "|").split("|")[0].strip() for _, t in voted]
    vals, cnt = np.unique(first, return_counts=True)
    maj = vals[cnt.argmax()]
    return dict(n_with_truth=n, n_voted=len(voted), coverage=len(voted) / n,
                purity=float(np.mean([label_match(p, t) for p, t in voted])),
                baseline=float(np.mean([label_match(maj, t) for _, t in voted])), majority=str(maj))


# ---------------------------------------------------------------- R5 analogue channel (PSSM / Tanimoto kernel)
# Open (+-200 Da) analogue hits -> their structures' raw Family-A bits -> one tier-2 RRF list over the candidates.
# Design: specs/specs_scratch/pssm.md. Train structures are not scored by this list (they get the neutral term).
MORGAN_R2 = 4096                    # the raw Family-A layout starts with Morgan r2 4096 (fpnet.Featuriser)


def molecule_analogs(lib, specs, pols, target, cfg, exclude=None):
    """Open-window analogue hits of one molecule, each spectrum searched within its own polarity -> {key: best sim}."""
    out = {}
    for pol in sorted(set(pols)):
        sp = [s for s, p in zip(specs, pols) if p == pol]
        for k, s in analog_hits(lib, sp, target, cfg, "open", polarity=pol, exclude=exclude,
                                open_da=cfg.analog_open_da).items():
            if s > out.get(k, -1.0):
                out[k] = s
    return out


def analog_set(hits, s_min, m):
    """{key: sim} -> the top m [(key, sim)] with sim >= s_min, best first (ties by key)."""
    top = sorted(((s, k) for k, s in hits.items() if s >= s_min), key=lambda x: (-x[0], str(x[1])))[:m]
    return [(k, s) for s, k in top]


def _logit(x):
    return np.log(x) - np.log1p(-x)


def pssm_weights(X, S, q, alpha=1.0, clip=3.0, q_floor=1e-5):
    """Spectrally weighted log-odds profile. X (m, D) analogue bits, S (m,) similarities, q (D,) background bit
    frequency. p~ = (sum S^4 x + alpha q) / (sum S^4 + alpha) (prior = alpha background analogues, so a bit no
    analogue carries stays near q); W = clip(logit p~ - logit q, +-clip)."""
    w = np.asarray(S, np.float64) ** 4
    q = np.clip(np.asarray(q, np.float64), q_floor, 1.0 - q_floor)
    p = (w @ np.asarray(X, np.float64) + alpha * q) / (w.sum() + alpha)
    return np.clip(_logit(p) - _logit(q), -clip, clip).astype(np.float32)


def tani_kernel(C, X, S):
    """sum_i S_i^4 T(c, a_i) / sum_i S_i^4 per candidate. C (n, D), X (m, D) binary."""
    C, X = np.asarray(C, np.float32), np.asarray(X, np.float32)
    inter = C @ X.T
    union = C.sum(1)[:, None] + X.sum(1)[None, :] - inter
    T = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    w = np.asarray(S, np.float64) ** 4
    return (T @ w / w.sum()).astype(np.float32)


def train_smiles(train_path, keymap=None):
    """Library structure key (train inchikey14 mapped through keymap) -> normalized_smiles (first row per ik14)."""
    import pyarrow.parquet as pq
    t = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    t = t.dropna().drop_duplicates("inchikey14")
    km = keymap or {}
    return {km.get(k, k): s for k, s in zip(t.inchikey14, t.normalized_smiles)}


def estimate_q(pool, feat, n=50_000, seed=0, mask=None, chunk=2_000):
    """Background bit frequency q (float32 raw bits) over a seeded sample of the pool rows in mask (default: all)."""
    idx = np.flatnonzero(np.ones(len(pool), bool) if mask is None else mask)
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    tot, cnt = np.zeros(feat.raw_bits, np.float64), 0
    for a in range(0, len(idx), chunk):
        for c in idx[a:a + chunk]:
            r = feat.raw(pool.smiles[c])
            if r is not None:
                tot += r
                cnt += 1
    return (tot / max(cnt, 1)).astype(np.float32)


class AnalogBank:
    """R5: analogue / candidate structures -> raw Family-A bits (packed, cached per key), background q, and the
    per-molecule profile ("pssm": log-odds weights W; "tani": the top-k analogues' Morgan r2 bits)."""

    def __init__(self, smiles_of, q, cfg, feat=None):
        if feat is None:
            from .fpnet import A_RAW_BITS, Featuriser
            feat = Featuriser("ahmed", np.arange(A_RAW_BITS))
        self.feat, self.smiles_of, self.cfg, self._cache = feat, smiles_of, cfg, {}
        self.q = None if q is None else np.asarray(q, np.float32)

    def bits(self, keys, smiles):
        """-> (raw bits (n, raw_bits) uint8, ok (n,) bool)."""
        nb = self.feat.raw_bits
        out, ok = np.zeros((len(keys), nb), np.uint8), np.zeros(len(keys), bool)
        for i, (k, s) in enumerate(zip(keys, smiles)):
            if k not in self._cache:
                if len(self._cache) > 300_000:
                    self._cache.clear()
                r = self.feat.raw(s)
                self._cache[k] = None if r is None else np.packbits(r)
            p = self._cache[k]
            if p is not None:
                out[i], ok[i] = np.unpackbits(p)[:nb], True
        return out, ok

    def profile(self, hits, mode=None):
        """{key: sim} -> profile dict, or None with fewer than analog_min_n analogues that have a structure."""
        c = self.cfg
        mode = mode or c.analog_channel
        m = c.analog_m if mode == "pssm" else c.analog_k
        top = [(k, s) for k, s in analog_set(hits, c.analog_smin, len(hits)) if self.smiles_of.get(k)]
        X, ok = self.bits([k for k, _ in top[:2 * m]], [self.smiles_of[k] for k, _ in top[:2 * m]])
        sel = np.flatnonzero(ok)[:m]
        if len(sel) < c.analog_min_n:
            return None
        S = np.array([top[i][1] for i in sel], np.float64)
        if mode == "pssm":
            if self.q is None:
                raise ValueError("pssm needs the background q")
            return dict(mode=mode, n=len(sel), W=pssm_weights(X[sel], S, self.q, c.analog_alpha, c.analog_clip))
        assert mode == "tani", mode
        return dict(mode=mode, n=len(sel), X=X[sel, :MORGAN_R2], S=S)

    def score(self, prof, keys, smiles):
        """Candidate scores under a profile (NaN = no fingerprint)."""
        out = np.full(len(keys), np.nan, np.float32)
        if prof is None or not len(keys):
            return out
        C, ok = self.bits(keys, smiles)
        if ok.any():
            if prof["mode"] == "pssm":
                out[ok] = C[ok].astype(np.float32) @ prof["W"]
            else:
                out[ok] = tani_kernel(C[ok, :MORGAN_R2], prof["X"], prof["S"])
        return out
