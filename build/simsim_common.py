"""Shared helpers for the specs_simple_ms_simulator Phase-1 notebooks.

Inlined verbatim into the generated notebooks (at the SIMSIM_COMMON placeholder) by build/mk_simple_sim_grid.py,
and imported directly by build/test_simple_sim_grid.py. Pure numpy / pandas / pyarrow: no RDKit, no
torch, no ms-pred. `prep` / `esim` call phase1's numba kernels (`_clean`, `entropy_sim`), which the
notebooks embed in an earlier cell; the tests inject them.

Lifted from build/simple_sim_gate.py (sh, sha1, formula_mass, the POOL build, to_model_ce, pb_sf, the
worker call) and from phase1 (ADDUCTS physics, the stage-A/B window logic). The gate notebook itself is
left untouched.
"""
import hashlib
import json
import os
import pickle
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]

# phase1 CFG values (spectrum cleaning and the candidate window) -- unchanged
FLOOR, TOPK, POWER, ENTW, TOL = 0.002, 64, 1.0, True, 0.01
PPM_WIN, PPM_CENTER, PPM_FALLBACK = 8.5, 1.45, 30.0
PPM_EXPAND = (8.5, 20.0, 45.0, 80.0)
TOPN = 25

# --- adduct physics: phase1 cell 3, restricted to the ten test adducts (all n = 1, z = 1) -----------
_M = dict(H=1.00782503207, N=14.0030740048, O=15.9949146196, C=12.0, Na=22.9897692809,
          K=38.96370668, Cl=34.96885268)
_E = 0.00054857990
_PROTON = _M["H"] - _E
_H2O = 2 * _M["H"] + _M["O"]
_NH4 = _M["N"] + 4 * _M["H"]
_FORMATE = _M["C"] + 2 * _M["H"] + 2 * _M["O"]
ADDUCT_DELTA = {
    "[M+H]+": _PROTON, "[M+NH4]+": _NH4 - _E, "[M+Na]+": _M["Na"] - _E, "[M+K]+": _M["K"] - _E,
    "[M-H2O+H]+": _PROTON - _H2O, "[M-2H2O+H]+": _PROTON - 2 * _H2O,
    "[M-H]-": -_PROTON, "[M-H2O-H]-": -_PROTON - _H2O, "[M+CH2O2-H]-": _FORMATE - _PROTON,
    "[M+Cl]-": _M["Cl"] + _E,
}


def neutral_mass1(mz, adduct):
    """ONE precursor m/z -> neutral mass (phase1 neutral_mass, test adducts only; NaN elsewhere).

    Not called `neutral_mass`: the submission notebook also embeds phase1's vectorised neutral_mass.
    """
    d = ADDUCT_DELTA.get(adduct)
    return float(mz) - d if d is not None else np.nan


def precursor_of(mass, adduct):
    """neutral exact mass -> theoretical precursor m/z (the inverse of neutral_mass1)."""
    return np.asarray(mass, dtype=np.float64) + ADDUCT_DELTA[adduct]


# --- label formula -> exact neutral mass (as phase1 / the gate; no RDKit) ---------------------------
_MONO = {"C": 12.0, "H": 1.0078250319, "N": 14.0030740052, "O": 15.9949146221, "S": 31.97207069,
         "P": 30.97376151, "F": 18.99840320, "Cl": 34.96885271, "Br": 78.9183376, "I": 126.904468,
         "Si": 27.9769265327, "Se": 79.9165196, "B": 11.0093055, "Na": 22.98976928, "K": 38.96370649,
         "Fe": 55.9349375, "D": 2.0141017778, "As": 74.9215965, "Li": 7.0160034, "Sn": 119.9021947}
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(f):
    if not isinstance(f, str):
        return np.nan
    s, pos = 0.0, 0
    for m in _TOK.finditer(f):
        if m.start() != pos:
            return np.nan
        pos = m.end()
        if m.group(1) not in _MONO:
            return np.nan
        s += _MONO[m.group(1)] * (int(m.group(2)) if m.group(2) else 1)
    return s if pos == len(f) else np.nan


# --- shell / files ------------------------------------------------------------------------------------
def sh(cmd, timeout=None, env=None):
    print("$", cmd if len(cmd) < 300 else cmd[:300] + " ...", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout, env=env)
    if r.stdout.strip():
        print(r.stdout[-2500:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise RuntimeError(f"command failed ({r.returncode})")
    return r.stdout


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --- ICEBERG worker call (the worker runs in its own venv; files only) -------------------------------
def to_model_ce(ce_ev, prec, mode):
    return ce_ev if mode == "ev" else ce_ev * 500.0 / prec     # ms-pred's own ev_to_nce


def write_request(req, path):
    req[["key", "smiles", "adduct", "precursor_mz", "ce", "instrument"]].to_csv(path, sep="\t", index=False)


def worker_cmd(py, worker, gen, inten, inp, out, device="cuda", batch_size=128, sparse_k=100, prec_margin=2.0):
    return [str(py), str(worker), "--gen", str(gen), "--inten", str(inten), "--inp", str(inp), "--out", str(out),
            "--device", device, "--batch-size", str(batch_size), "--sparse-k", str(sparse_k),
            "--prec-margin", str(prec_margin)]


def read_response(out, n):
    """-> spectra [(mz, it)], ok, err, timing"""
    z = np.load(out)
    # read each array ONCE: NpzFile re-reads (and re-allocates) an array on every z[key] access, and a
    # slice keeps its whole copy alive -- per-row access held n full copies (the Phase-1 Colab OOM)
    off, mz, it = z["off"], z["mz"], z["it"]
    assert len(off) == n + 1, f"{out}: {len(off) - 1} rows, expected {n}"
    spectra = [(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]) for i in range(n)]
    return spectra, z["ok"], z["err"], json.loads(str(z["timing"]))


def run_worker(req, work, tag, py, worker, gen, inten, env, **kw):
    """-> spectra, ok, err, timing, wall_s"""
    inp, out = Path(work) / f"req_{tag}.tsv", Path(work) / f"resp_{tag}.npz"
    write_request(req, inp)
    t = time.time()
    r = subprocess.run(worker_cmd(py, worker, gen, inten, inp, out, **kw), text=True, capture_output=True,
                       env=env, timeout=12 * 3600)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise RuntimeError(f"worker failed on {tag}")
    spectra, ok, err, timing = read_response(out, len(req))
    return spectra, ok, err, timing, time.time() - t


def err_counts(err):
    return pd.Series([str(e).split(":")[0] for e in err if e]).value_counts().to_dict()


# --- spectra -----------------------------------------------------------------------------------------
def prep(mz, it, prec):
    """phase1 cleaning after the precursor + 2 Da cut. Needs the phase1 kernels in globals."""
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    keep = mz <= prec + 2.0
    return _clean(mz[keep], it[keep], FLOOR, TOPK, POWER, ENTW)  # noqa: F821 (phase1 kernel cell)


def esim(a, b):
    if len(a[0]) == 0 or len(b[0]) == 0:
        return 0.0
    return float(entropy_sim(a[0], a[1], b[0], b[1], TOL))  # noqa: F821 (phase1 kernel cell)


def sum_merge(specs, decimals=4):
    """Merge per-CE predicted spectra the way a merged acquisition sums them.

    Each input is (mz, it) with base peak 1. Peaks with the same m/z (rounded to `decimals`; the same
    fragment formula gives the same m/z at every CE) are summed, then the result is renormalised to
    base 1 and sorted by m/z. Empty inputs are ignored.
    """
    parts = [(np.asarray(m, np.float64), np.asarray(i, np.float64)) for m, i in specs if len(m)]
    if not parts:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mz = np.concatenate([m for m, _ in parts])
    it = np.concatenate([i / i.max() for _, i in parts])
    key, inv = np.unique(np.round(mz, decimals), return_inverse=True)
    s = np.bincount(inv, weights=it)
    m = np.bincount(inv, weights=mz) / np.bincount(inv)
    o = np.argsort(m)
    return m[o].astype(np.float32), (s[o] / s.max()).astype(np.float32)


def rank_of_truth(sims):
    """sims[0] = truth, sims[1:] = decoys. Ties count half; a tied truth is not top-1."""
    sims = np.asarray(sims, dtype=np.float64)
    t, dec = sims[0], sims[1:]
    gt, eq = int((dec > t).sum()), int((dec == t).sum())
    return 1 + gt + eq / 2, (gt == 0 and eq == 0)


def pb_sf(ps, k):
    """P(X >= k), X = sum of independent Bernoulli(p_i) -- exact Poisson-binomial by DP."""
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for p in ps:
        d[1:] = d[1:] * (1 - p) + d[:-1] * p
        d[0] *= (1 - p)
    return float(d[k:].sum())


# --- collision-energy grid ----------------------------------------------------------------------------
def ce_tuple(v):
    """collision_energy_ev cell (list / array / None) -> tuple of floats rounded to 0.1 eV."""
    if v is None:
        return ()
    try:
        return tuple(round(float(x), 1) for x in v)
    except TypeError:
        return ()


def snap_ce(ce, grid, tol):
    """Nearest grid CE within tol (eV), else None. Ties go to the lower CE."""
    if not grid or not np.isfinite(ce):
        return None
    g = min(sorted(grid), key=lambda x: abs(x - ce))
    return float(g) if abs(g - ce) <= tol + 1e-9 else None


def spectrum_conditions(adduct, ces, grid, tol, covered):
    """One spectrum -> set of (adduct, ce, on_grid) conditions that need a simulated spectrum.

    Uncovered adducts need none (MetFrag-lite). A merged multi-CE spectrum needs each constituent CE,
    which is why merges add no grid conditions. CE <= 0 constituents (a few libraries list 0) are skipped.
    """
    if adduct not in covered:
        return set()
    out = set()
    for ce in ces:
        if not np.isfinite(ce) or ce <= 0:
            continue
        g = snap_ce(ce, grid, tol)
        out.add((adduct, g if g is not None else round(float(ce), 1), g is not None))
    return out


def choose_tolerance(median_by_delta, min_sim=0.90):
    """Largest delta (eV) whose median sim(pred@c, pred@c+delta) is >= min_sim; 0.0 if none."""
    ok = [d for d, m in median_by_delta.items() if np.isfinite(m) and m >= min_sim]
    return float(max(ok)) if ok else 0.0


def base_ce_set(ce_tuples, min_cover=0.95):
    """Distinct single CEs covering >= min_cover of single-CE spectra, plus every merged constituent."""
    singles = pd.Series([t[0] for t in ce_tuples if len(t) == 1 and t[0] > 0])
    keep = set()
    if len(singles):
        vc = singles.value_counts()
        cum = vc.cumsum() / vc.sum()
        keep = set(vc.index[: int((cum < min_cover).sum()) + 1])
    for t in ce_tuples:
        if len(t) > 1:
            keep |= {c for c in t if c > 0}
    return sorted(float(c) for c in keep)


# --- candidate pool and windows ------------------------------------------------------------------------
def build_pool(coco_dir, train_path):
    """COCONUT + train structures, deduplicated on ik14, sorted by exact neutral mass (as the gate).

    Columns: k14, smiles, mass, has_real (the structure has >= 1 training spectrum).
    """
    coco_dir = Path(coco_dir)
    cm = pickle.load(open(coco_dir / "coco_meta.pkl", "rb"))
    cmass = np.load(coco_dir / "coco_mass.npy")
    tr = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles", "molecular_formula"]).to_pandas()
    tr = tr.dropna(subset=["inchikey14", "normalized_smiles"]).drop_duplicates("inchikey14")
    fm = {f: formula_mass(f) for f in set(tr.molecular_formula.astype(object).tolist())}
    tr_mass = np.array([fm.get(f, np.nan) for f in tr.molecular_formula.astype(object)], dtype=np.float64)
    pool = pd.DataFrame({
        "k14": np.concatenate([[str(k)[:14] for k in cm["keys"]], tr.inchikey14.astype(str).to_numpy()]),
        "smiles": np.concatenate([np.asarray(cm["smiles"], dtype=object),
                                  tr.normalized_smiles.astype(object).to_numpy()]),
        "mass": np.concatenate([cmass.astype(np.float64), tr_mass]),
    })
    n_raw = len(pool)
    pool = (pool[np.isfinite(pool.mass) & pool.smiles.notna()].drop_duplicates("k14")
            .sort_values("mass", kind="stable").reset_index(drop=True))
    pool["has_real"] = pool.k14.isin(set(tr.inchikey14.astype(str)))
    print(f"pool {n_raw:,} rows -> {len(pool):,} unique ik14 with a finite mass "
          f"(COCONUT {len(cmass):,}, train {len(tr):,})")
    return pool


def window(pm, target, ppm):
    """(lo, hi) index range of the mass-sorted pool within +-ppm of target (phase1 CandidatePool.window)."""
    lo = int(np.searchsorted(pm, target * (1 - ppm / 1e6), "left"))
    hi = int(np.searchsorted(pm, target * (1 + ppm / 1e6), "right"))
    return lo, hi


def scored_range(pm, target, win=PPM_WIN, fallback=PPM_FALLBACK, expand=PPM_EXPAND, topn=TOPN):
    """Pool range whose candidates phase1's stages A + B actually score (stage C only mass-sorts).

    Stage A: the +-win window, or +-fallback if that is empty. Stage B: while fewer than topn candidates
    (pool keys are unique), every wider tier's new candidates are scored. The windows are nested, so the
    scored set is one contiguous range.
    """
    lo, hi = window(pm, target, win)
    if hi == lo:
        lo, hi = window(pm, target, fallback)
    if hi - lo < topn:
        for ppm in expand[1:]:
            a, b = window(pm, target, ppm)
            lo, hi = min(lo, a), max(hi, b)
            if hi - lo >= topn:
                break
    return lo, hi


def query_target(precs, adducts):
    """Molecule -> centred neutral target mass (phase1: median over spectra, divided by 1 + PPM_CENTER)."""
    nms = np.array([neutral_mass1(p, a) for p, a in zip(precs, adducts)], dtype=np.float64)
    nms = nms[np.isfinite(nms)]
    return float(np.median(nms)) / (1.0 + PPM_CENTER / 1e6) if len(nms) else np.nan


# --- projection and decision ---------------------------------------------------------------------------
def on_the_fly(conds, precomputed_adducts):
    """Conditions left for Kaggle: off-grid ones, and on-grid ones whose adduct was not precomputed."""
    return [c for c in conds if not (c[2] and c[0] in precomputed_adducts)]


def bootstrap_total(per_mol, n_mol=400, n_boot=2000, seed=0):
    """Total over n_mol molecules resampled from per_mol -> dict(mean, p50, p95)."""
    x = np.asarray(per_mol, dtype=np.float64)
    if len(x) == 0:
        return dict(mean=0.0, p50=0.0, p95=0.0)
    rng = np.random.default_rng(seed)
    tot = x[rng.integers(0, len(x), size=(n_boot, n_mol))].sum(1)
    return dict(mean=float(x.mean() * n_mol), p50=float(np.percentile(tot, 50)), p95=float(np.percentile(tot, 95)))


def kaggle_hours(load, rate, n_mol, t_base_s, fixed_s, simlib_gb, load_s_per_gb, worker_calls, startup_s):
    """Projected Kaggle wall time (h): fixed setup + per-molecule base cost + simlib load + on-the-fly sims."""
    sim_s = (load / rate if load > 0 else 0.0) + (worker_calls * startup_s if load > 0 else 0.0)
    return (fixed_s + n_mol * t_base_s + simlib_gb * load_s_per_gb + sim_s) / 3600.0


def decide(options, budget_h):
    """Cheapest option (colab_h, then name) whose kaggle_h <= budget_h; None if none fits (-> STOP)."""
    fit = [o for o in options if o["kaggle_h"] <= budget_h]
    return min(fit, key=lambda o: (o["colab_h"], o["name"])) if fit else None


# --- submission (Phase 4, option A: every covered condition simulated on the fly) ---------------------
MERGE_RULES = ("max", "sum", "mid")


def spectrum_ces(ces, rule, default):
    """One query spectrum's CE list -> (CEs to simulate, defaulted).

    CE <= 0 constituents are dropped (a few lists start at 0). `max` / `sum` simulate every constituent;
    `mid` simulates the mean of the list once. No usable CE -> [default], defaulted = True.
    """
    assert rule in MERGE_RULES, rule
    v = [round(float(c), 1) for c in ces if np.isfinite(c) and c > 0]
    if not v:
        return [float(default)], True
    if rule == "mid":
        return [round(float(np.mean(v)), 1)], False
    return sorted(set(v)), False


def sim_priority(lv, fr, ppm):
    """Per-candidate simulation priority (0 = first): lv desc, fr desc, |ppm| asc -- the watchdog's cap order."""
    o = np.lexsort((np.asarray(ppm, np.float64), -np.asarray(fr, np.float64), -np.asarray(lv, np.float64)))
    r = np.empty(len(o), np.int64)
    r[o] = np.arange(len(o))
    return r


REQ_COLS = ["key", "smiles", "adduct", "precursor_mz", "ce_ev", "prio", "ce", "instrument"]


def build_requests(parts, instrument="QTOF", ce_mode="ev"):
    """Molecule parts -> one worker request per distinct (key, adduct, ce_ev), sorted by priority.

    Each part is a dict with arrays key / smiles / mass / prio (one entry per candidate to simulate) and
    conds, a list of (adduct, ce_ev). The precursor is the candidate's theoretical m/z (exact mass +
    adduct delta, as the G2b gate), so a request depends on the candidate alone and is shared by every
    spectrum and molecule that needs it; a shared request keeps its best (lowest) priority.
    """
    frames = []
    for p in parts:
        if not len(p["key"]) or not p["conds"]:
            continue
        mass = np.asarray(p["mass"], np.float64)
        for adduct, ce in p["conds"]:
            frames.append(pd.DataFrame({"key": np.asarray(p["key"], dtype=object),
                                        "smiles": np.asarray(p["smiles"], dtype=object), "adduct": adduct,
                                        "precursor_mz": precursor_of(mass, adduct), "ce_ev": float(ce),
                                        "prio": np.asarray(p["prio"], np.int64)}))
    if not frames:
        return pd.DataFrame({c: [] for c in REQ_COLS})
    req = (pd.concat(frames, ignore_index=True)
           .sort_values(["prio", "key", "adduct", "ce_ev"], kind="stable")
           .drop_duplicates(["key", "adduct", "ce_ev"], keep="first").reset_index(drop=True))
    req["ce"] = to_model_ce(req.ce_ev.to_numpy(), req.precursor_mz.to_numpy(), ce_mode)
    req["instrument"] = instrument
    return req[REQ_COLS]


def rows_that_fit(remaining_s, rate, startup_s):
    """How many worker rows one more call can take before the deadline (0 if none)."""
    return max(0, int((remaining_s - startup_s) * rate))


def spectrum_sv(q, sims, rule):
    """Simulated-library score of ONE prepped query spectrum against one candidate.

    sims: the candidate's raw predictions (mz, it, prec) at each of the spectrum's CEs, None where it was
    not simulated (failed, capped). max / mid: best esim over the CEs; sum: esim against the renormalised
    sum of the available CEs (a merged acquisition). Nothing available -> 0.
    """
    have = [s for s in sims if s is not None and len(s[0])]
    if not have or len(q[0]) == 0:
        return 0.0
    if rule == "sum":
        mz, it = sum_merge([(m, i) for m, i, _ in have])
        return esim(q, prep(mz, it, have[0][2]))
    return max(esim(q, prep(m, i, p)) for m, i, p in have)


def rank_two_tier(lv, sv, fr, ppm, tau):
    """Order (indices) of one candidate set.

    Tier 1 = real-library match lv >= tau, by lv desc (then sv, fr, |ppm|). Tier 2 = the rest, by sv desc,
    fr desc, |ppm| asc. A missing sv / fr (NaN) counts 0; a missing ppm sorts last.
    """
    lv, sv, fr = (np.nan_to_num(np.asarray(x, np.float64), nan=0.0) for x in (lv, sv, fr))
    ppm = np.nan_to_num(np.asarray(ppm, np.float64), nan=np.inf)
    t1, t2 = np.flatnonzero(lv >= tau), np.flatnonzero(lv < tau)
    o1 = t1[np.lexsort((ppm[t1], -fr[t1], -sv[t1], -lv[t1]))]
    o2 = t2[np.lexsort((ppm[t2], -fr[t2], -sv[t2]))]
    return np.concatenate([o1, o2]).astype(np.int64)
