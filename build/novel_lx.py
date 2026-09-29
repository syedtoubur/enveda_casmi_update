"""G0.1-R2 helpers: the tie-at-the-cut flag, the parity_lx pick, peak streaming and digests.

numpy / pyarrow only. Embedded verbatim into the three G0.1-R2 notebooks (Colab GPU re-embed, Colab
CPU peak leg, Kaggle leg) by build/mk_novel_p0.py, and imported directly by build/test_novel_p1.py.
Peak selection itself stays in dreams_infer.dreams_peaks (pinned by SHA1); nothing here changes it.
"""
import hashlib
import platform

import numpy as np


def cut_tie(mz, it, prec, floor=0.002, n=100, above=2.0):
    """True if the top-n cut of dreams_peaks falls inside an intensity tie.

    Same filters as dreams_peaks (drop m/z > prec + above, drop intensity < floor * max); a tie at the
    cut means more than n peaks survive and the n-th and (n+1)-th largest intensities are equal, so
    which of them survive depends on the tie-break -- the rows G0.1-R found to differ.
    """
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    k = mz <= prec + above
    it = it[k]
    if len(it) <= n:
        return False
    mx = it.max()
    if not np.isfinite(mx) or mx <= 0:
        return False
    it = it[it >= floor * mx]
    if len(it) <= n:
        return False
    s = np.sort(it)[::-1]
    return bool(s[n - 1] == s[n])


def pick_parity(tie, lib, n_tie, n_other, seed):
    """Row indices for parity_lx: n_tie tie rows spread round-robin over libraries, then n_other
    non-tie rows drawn uniformly. Sorted, unique, deterministic for a seed."""
    tie = np.asarray(tie, bool); lib = np.asarray(lib).astype(str)
    rng = np.random.default_rng(seed)
    t_rows = np.flatnonzero(tie)
    queues = []
    for g in sorted(set(lib[t_rows])):
        q = t_rows[lib[t_rows] == g].copy()
        rng.shuffle(q)
        queues.append(list(q))
    picked = []
    while len(picked) < n_tie and any(queues):
        for q in queues:
            if q and len(picked) < n_tie:
                picked.append(q.pop())
    others = np.flatnonzero(~tie)
    picked += list(rng.choice(others, size=min(n_other, len(others)), replace=False))
    return np.sort(np.array(picked, np.int64))


def stream_rows(train_path, train_rows, fn, batch_size=100_000):
    """One pass over train.parquet's peak columns; fn(k, mz, it) for each k with train_rows[k] in the
    batch (mz, it as numpy views). Returns the number of train.parquet rows read."""
    import pyarrow.parquet as pq
    train_rows = np.asarray(train_rows, np.int64)
    pf = pq.ParquetFile(train_path)
    n_train = pf.metadata.num_rows
    pos = np.full(n_train, -1, np.int64)
    pos[train_rows] = np.arange(len(train_rows))
    base = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["ms2_mzs", "ms2_normalized_intensities"]):
        n = batch.num_rows
        hit = np.flatnonzero(pos[base:base + n] >= 0)
        if len(hit):
            mc, ic = batch.column(0), batch.column(1)
            mo, mv = mc.offsets.to_numpy(), mc.values.to_numpy(zero_copy_only=False)
            io_, iv = ic.offsets.to_numpy(), ic.values.to_numpy(zero_copy_only=False)
            for local in hit:
                fn(pos[base + local], mv[mo[local]:mo[local + 1]], iv[io_[local]:io_[local + 1]])
        base += n
    assert base == n_train, (base, n_train)
    return base


def pack_peaks(pls):
    """list of (2, k) float32 -> (mz, it, off) flat arrays for an npz."""
    off = np.concatenate([[0], np.cumsum([p.shape[1] for p in pls])]).astype(np.int64)
    mz = np.concatenate([p[0] for p in pls]).astype(np.float32) if pls else np.zeros(0, np.float32)
    it = np.concatenate([p[1] for p in pls]).astype(np.float32) if pls else np.zeros(0, np.float32)
    return mz, it, off


def unpack_peaks(mz, it, off):
    return [np.stack([mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]]).astype(np.float32)
            for i in range(len(off) - 1)]


def peaks_digest(pls, prec):
    """SHA1 over every peak list's shape, dtype and bytes plus the precursors (float64)."""
    h = hashlib.sha1()
    for p in pls:
        p = np.ascontiguousarray(p)
        h.update(f"{p.shape}{p.dtype}".encode()); h.update(p.tobytes())
    h.update(np.ascontiguousarray(np.asarray(prec, np.float64)).tobytes())
    return h.hexdigest()


def peak_diffs(a, b):
    """Indices where two lists of peak lists are not byte-identical."""
    assert len(a) == len(b), (len(a), len(b))
    return [i for i, (x, y) in enumerate(zip(a, b))
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes()]


def rowcos(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


def host_info():
    """CPU model and SIMD flags (numpy's sort kernels dispatch on them), numpy and python versions."""
    info = {"machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__}
    try:
        text = open("/proc/cpuinfo").read()
        info["cpu"] = next((ln.split(":", 1)[1].strip() for ln in text.splitlines()
                            if ln.startswith("model name")), "?")
        flags = next((ln.split(":", 1)[1].split() for ln in text.splitlines() if ln.startswith("flags")), [])
        info["avx2"], info["avx512f"] = "avx2" in flags, "avx512f" in flags
    except OSError:
        info["cpu"] = platform.processor() or "?"
    return info
