"""Reference spectral library (train.parquet) and the Class-1 direct match `lv`."""
import time
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .chem import formula_mass, neutral_mass
from .kernels import search
from .spectra import clean_spectrum


@dataclass
class Library:
    off: np.ndarray      # (n + 1,) peak offsets
    mz: np.ndarray       # flat m/z
    it: np.ndarray       # flat intensities
    nm: np.ndarray       # neutral mass per spectrum
    ik: np.ndarray       # inchikey14 per spectrum (object)
    order: np.ndarray    # spectrum indices sorted by nm (finite first)
    snm: np.ndarray      # nm[order]
    n_ok: int            # number of finite nm
    pol: np.ndarray = None   # +1 / -1 / 0 per spectrum, from the adduct's charge sign (analog search)

    def __len__(self):
        return len(self.off) - 1

    def window(self, target, tol_da):
        snm = self.snm[:self.n_ok]
        lo = np.searchsorted(snm, target - tol_da, "left")
        hi = np.searchsorted(snm, target + tol_da, "right")
        return self.order[lo:hi]


def build_library(off, mz, it, prec, adduct, ik, formula=None, struct_mass=True):
    """Library from flat arrays. With struct_mass, the label formula's exact mass replaces the precursor-derived
    neutral mass wherever it parses (the base's TIER-1 fix)."""
    nm = neutral_mass(prec, adduct)
    if struct_mass and formula is not None:
        fm = {f: formula_mass(f) for f in set(formula.tolist())}
        nm_s = np.array([fm.get(f, np.nan) for f in formula], np.float64)
        nm = np.where(np.isfinite(nm_s), nm_s, nm)
    ok = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind="mergesort")
    return Library(off=np.asarray(off, np.int64), mz=np.asarray(mz, np.float32), it=np.asarray(it, np.float32),
                   nm=nm, ik=np.asarray(ik, dtype=object), order=order, snm=nm[order], n_ok=int(ok.sum()),
                   pol=adduct_polarity(adduct))


def adduct_polarity(adduct):
    """'[M+H]+' -> 1, '[M-H]-' -> -1, anything else -> 0."""
    ad = np.asarray(adduct, dtype=object)
    return np.array([1 if isinstance(a, str) and a.endswith("+") else -1 if isinstance(a, str) and a.endswith("-")
                     else 0 for a in ad], np.int8)


def remap_keys(lib, keymap):
    """A Library view whose ik are mapped through keymap (train inchikey14 -> R1 scorer_key); unmapped keys kept.
    Peaks are shared, not copied."""
    u, inv = np.unique(lib.ik.astype(str), return_inverse=True)
    mapped = np.array([keymap.get(k, k) for k in u], dtype=object)
    return replace(lib, ik=mapped[inv])


def load_library(path, struct_mass=True, rows=None):
    """train.parquet -> Library. rows: optional row indices to keep (validation bans)."""
    t0 = time.time()
    t = pq.read_table(path, columns=["inchikey14", "adduct", "precursor_mz", "molecular_formula",
                                     "ms2_mzs", "ms2_normalized_intensities"])
    if rows is not None:
        t = t.take(pa.array(np.asarray(rows, np.int64)))
    mzc = t.column("ms2_mzs").combine_chunks()
    itc = t.column("ms2_normalized_intensities").combine_chunks()
    lib = build_library(
        off=mzc.offsets.to_numpy().astype(np.int64),
        mz=mzc.values.to_numpy(zero_copy_only=False), it=itc.values.to_numpy(zero_copy_only=False),
        prec=t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64),
        adduct=np.asarray(t.column("adduct").cast(pa.string()).to_pylist(), dtype=object),
        ik=np.asarray(t.column("inchikey14").cast(pa.string()).to_pylist(), dtype=object),
        formula=np.asarray(t.column("molecular_formula").cast(pa.string()).to_pylist(), dtype=object),
        struct_mass=struct_mass)
    # `.values` ignores a slice offset and `.offsets` index into it, so the pair stays consistent
    print(f"[library] {len(lib):,} spectra, {lib.n_ok:,} with a neutral mass ({time.time() - t0:.1f}s)", flush=True)
    return lib


def lib_sim(lib, specs, target, cfg, exclude=None):
    """Class 1: best entropy similarity per ik14 over library spectra within +-ppm_win of target.

    specs: [(mz, it)] of one molecule. exclude: optional boolean mask over library spectra to skip.
    """
    cand = lib.window(target, target * cfg.ppm_win / 1e6)
    if exclude is not None and len(cand):
        cand = cand[~exclude[cand]]
    if len(cand) == 0:
        return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it, cfg)
        if len(qm) == 0:
            continue
        sc = search(qm, qp, cand, lib.off, lib.mz, lib.it, cfg.mz_tol, cfg.int_floor, cfg.max_peaks,
                    cfg.int_power, cfg.ent_weight)
        for c, s in zip(cand, sc):
            k = lib.ik[c]
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return agg
