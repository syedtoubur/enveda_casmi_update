"""Spectrum cleaning and similarity with the config's single max_peaks."""
import numpy as np

from .kernels import clean, entropy_sim

EMPTY = (np.empty(0, np.float32), np.empty(0, np.float32))


def clean_spectrum(mz, it, cfg):
    return clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                 cfg.int_floor, cfg.max_peaks, cfg.int_power, cfg.ent_weight)


def prep(mz, it, prec, cfg):
    """Clean a spectrum after dropping peaks above precursor + prec_margin (the organisers' cleaning)."""
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    keep = mz <= prec + cfg.prec_margin
    return clean(mz[keep], it[keep], cfg.int_floor, cfg.max_peaks, cfg.int_power, cfg.ent_weight)


def esim(a, b, tol):
    if len(a[0]) == 0 or len(b[0]) == 0:
        return 0.0
    return float(entropy_sim(a[0], a[1], b[0], b[1], tol))


def sum_merge(specs, decimals=4):
    """Sum per-CE spectra (base peak 1 each) the way a merged acquisition does; renormalise to base 1."""
    parts = [(np.asarray(m, np.float64), np.asarray(i, np.float64)) for m, i in specs if len(m)]
    if not parts:
        return EMPTY
    mz = np.concatenate([m for m, _ in parts])
    it = np.concatenate([i / i.max() for _, i in parts])
    _, inv = np.unique(np.round(mz, decimals), return_inverse=True)
    s = np.bincount(inv, weights=it)
    m = np.bincount(inv, weights=mz) / np.bincount(inv)
    o = np.argsort(m)
    return m[o].astype(np.float32), (s[o] / s.max()).astype(np.float32)


def spectrum_sv(q, sims, cfg):
    """Simulated-library score of one prepped query spectrum against one candidate.

    sims: the candidate's raw predictions (mz, it, prec) at each of the spectrum's CEs, None where not simulated.
    max / mid: best similarity over the CEs; sum: similarity to the renormalised sum. Nothing available -> 0.
    """
    have = [s for s in sims if s is not None and len(s[0])]
    if not have or len(q[0]) == 0:
        return 0.0
    if cfg.merge_rule == "sum":
        mz, it = sum_merge([(m, i) for m, i, _ in have])
        return esim(q, prep(mz, it, have[0][2], cfg), cfg.mz_tol)
    return max(esim(q, prep(m, i, p, cfg), cfg.mz_tol) for m, i, p in have)


MERGE_RULES = ("max", "sum", "mid")


def ce_tuple(v):
    """collision_energy_ev cell (list / array / None) -> tuple of floats rounded to 0.1 eV."""
    if v is None:
        return ()
    try:
        return tuple(round(float(x), 1) for x in v)
    except TypeError:
        return ()


def spectrum_ces(ces, rule, default):
    """One spectrum's CE list -> (CEs to simulate, defaulted). CE <= 0 dropped; `mid` simulates the mean once."""
    assert rule in MERGE_RULES, rule
    v = [round(float(c), 1) for c in ces if np.isfinite(c) and c > 0]
    if not v:
        return [float(default)], True
    if rule == "mid":
        return [round(float(np.mean(v)), 1)], False
    return sorted(set(v)), False
