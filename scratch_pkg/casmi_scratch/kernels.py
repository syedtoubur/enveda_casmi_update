"""Numba kernels: spectrum cleaning and spectral entropy similarity (phase1, verbatim logic)."""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def clean(mz, it, floor, topk, power, ent_weight):
    """Drop peaks below floor * max, keep the top-k by intensity (m/z order kept), normalise to sum 1,
    and entropy-weight low-entropy spectra."""
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c):
            oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0:
                S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c):
                    oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    tot = 0.0
    buf = np.empty(n + m, np.float64)
    b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]
            i += 1
        elif d > tol:
            buf[b] = cp[j]
            j += 1
        else:
            buf[b] = qp[i] + cp[j]
            i += 1
            j += 1
        b += 1
    while i < n:
        buf[b] = qp[i]
        i += 1
        b += 1
    while j < m:
        buf[b] = cp[j]
        j += 1
        b += 1
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    """Entropy similarity of one cleaned query against library spectra `cand` (cleaned on the fly)."""
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
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out
