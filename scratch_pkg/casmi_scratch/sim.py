"""ICEBERG simulation: request building, the worker subprocess, and the deadline-watchdog scheduler.

The model runs in its own venv (torch / DGL / RDKit 2025.03) through `resources/iceberg_worker.py`; this module
only writes request TSVs and reads response NPZs.
"""
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import precursor_of

WORKER_SRC = Path(__file__).resolve().parent / "resources" / "iceberg_worker.py"
REQ_COLS = ["key", "smiles", "adduct", "precursor_mz", "ce_ev", "prio", "ce", "instrument"]


def to_model_ce(ce_ev, prec, mode):
    return ce_ev if mode == "ev" else ce_ev * 500.0 / prec


def sim_priority(lv, fr, ppm):
    """Per-candidate simulation priority (0 = first): lv desc, fr desc, |ppm| asc."""
    o = np.lexsort((np.asarray(ppm, np.float64), -np.asarray(fr, np.float64), -np.asarray(lv, np.float64)))
    r = np.empty(len(o), np.int64)
    r[o] = np.arange(len(o))
    return r


def build_requests(parts, instrument="QTOF", ce_mode="ev"):
    """Molecule parts -> one request per distinct (key, adduct, ce_ev), sorted by priority.

    Each part: dict with arrays key / smiles / mass / prio and conds = [(adduct, ce_ev)]. The precursor is the
    candidate's theoretical m/z, so a request is shared by every molecule that needs it (best priority kept).
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
    return max(0, int((remaining_s - startup_s) * rate))


def err_counts(err):
    return pd.Series([str(e).split(":")[0] for e in err if e], dtype=object).value_counts().to_dict()


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --- the worker ----------------------------------------------------------------------------------------
class IcebergWorker:
    """Callable(req, tag, batch_size) -> (spectra [(mz, it)], ok, err, timing, wall_s)."""

    def __init__(self, py, gen, inten, env, work, device="cuda", sparse_k=100, prec_margin=2.0, batch=128):
        self.py, self.gen, self.inten, self.env = str(py), str(gen), str(inten), env
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.device, self.sparse_k, self.prec_margin, self.batch = device, sparse_k, prec_margin, batch
        self.worker = self.work / "iceberg_worker.py"
        shutil.copyfile(WORKER_SRC, self.worker)

    def __call__(self, req, tag, batch_size=None):
        inp, out = self.work / f"req_{tag}.tsv", self.work / f"resp_{tag}.npz"
        req[["key", "smiles", "adduct", "precursor_mz", "ce", "instrument"]].to_csv(inp, sep="\t", index=False)
        cmd = [self.py, str(self.worker), "--gen", self.gen, "--inten", self.inten, "--inp", str(inp),
               "--out", str(out), "--device", self.device, "--batch-size", str(batch_size or self.batch),
               "--sparse-k", str(self.sparse_k), "--prec-margin", str(self.prec_margin)]
        t = time.time()
        r = subprocess.run(cmd, text=True, capture_output=True, env=self.env, timeout=12 * 3600)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-4000:])
            raise RuntimeError(f"worker failed on {tag}")
        z = np.load(out)
        # read each array once: NpzFile re-reads an array on every access (the Phase-1 Colab OOM)
        off, mz, it = z["off"], z["mz"], z["it"]
        assert len(off) == len(req) + 1, f"{out}: {len(off) - 1} rows, expected {len(req)}"
        spectra = [(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]) for i in range(len(req))]
        ok, err, timing = z["ok"], z["err"], json.loads(str(z["timing"]))
        for f in (inp, out):
            f.unlink(missing_ok=True)
        return spectra, ok, err, timing, time.time() - t


def setup_iceberg(input_root, work, cfg, venv="/kaggle/working/iceberg_venv"):
    """Install the offline ICEBERG wheelhouse into its own venv, verify checkpoints, return an IcebergWorker."""
    hits = [h for h in glob.glob(f"{input_root}/**/wheelhouse", recursive=True) if os.path.isdir(h)]
    if not hits:
        raise FileNotFoundError("wheelhouse not found -- attach casmi26-iceberg-offline")
    up = os.path.dirname(sorted(hits, key=len)[0])
    man = json.load(open(os.path.join(up, "manifest.json")))
    assert man["ce_mode"] == cfg.ce_mode and man["instrument"] == cfg.instrument, man
    py = f"{venv}/bin/python"
    site = os.path.join(venv, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONPATH=site, DGLBACKEND="pytorch", CUDA_VISIBLE_DEVICES="0")
    if not os.path.exists(py):
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    os.makedirs(site, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", site,
                        *sorted(glob.glob(f"{up}/wheelhouse/*.whl"))], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    gen, inten = os.path.join(up, "ckpt", "gen.ckpt"), os.path.join(up, "ckpt", "inten.ckpt")
    assert sha1(gen) == man["gen_sha1"] and sha1(inten) == man["inten_sha1"], "checkpoint SHA1 mismatch"
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    if device == "cpu":
        print("[WARNING] no GPU -- ICEBERG on CPU is ~15x slower; the watchdog will cap the simulation", flush=True)
    return IcebergWorker(py, gen, inten, env, work, device=device, sparse_k=cfg.sparse_k,
                         prec_margin=cfg.prec_margin, batch=cfg.batch)


def smoke(worker, cfg):
    """Caffeine [M+H]+ at 40 eV must predict; a 120 eV off-ladder probe is reported."""
    m = 194.0803756
    req = pd.DataFrame({"key": ["RYYVLZVUVIJVGH"] * 2, "smiles": ["Cn1cnc2c1c(=O)n(C)c(=O)n2C"] * 2,
                        "adduct": "[M+H]+", "precursor_mz": precursor_of(np.array([m, m]), "[M+H]+"),
                        "ce_ev": [40.0, 120.0]})
    req["ce"] = to_model_ce(req.ce_ev.to_numpy(), req.precursor_mz.to_numpy(), cfg.ce_mode)
    req["instrument"] = cfg.instrument
    sp, ok, err, timing, _ = worker(req, "smoke")
    assert ok[0], f"ICEBERG smoke failed: {err[0]}"
    return dict(gpu=timing.get("gpu"), n_peaks_40=int(len(sp[0][0])),
                probe_120=dict(ok=bool(ok[1]), n_peaks=int(len(sp[1][0])), err=str(err[1])))


# --- the scheduler -------------------------------------------------------------------------------------
def simulate(req, call, deadline, cfg, strict=False, log=print):
    """Run requests in priority-ordered chunks until done or the deadline.

    call: IcebergWorker-like callable. strict: re-raise a chunk that fails twice (fast commit) instead of
    degrading. Returns (SIM {(key, adduct, ce_ev): (mz, it, prec)}, stats dict).
    """
    sim, errs = {}, {}
    failed = capped = done = ci = 0
    rate, startup = cfg.rate, cfg.startup_s
    n_chunks = math.ceil(len(req) / cfg.chunk) if len(req) else 0
    while done < len(req):
        fit = rows_that_fit(deadline - time.time(), rate, startup)
        n = min(cfg.chunk, len(req) - done, fit)
        if fit < min(500, len(req) - done):
            capped = len(req) - done
            log(f"[watchdog] deadline: {capped:,} lowest-priority requests left unsimulated")
            break
        part = req.iloc[done:done + n]
        try:
            spectra, ok, err, timing, wall = call(part, f"c{ci}")
        except RuntimeError:
            try:
                spectra, ok, err, timing, wall = call(part, f"c{ci}r", cfg.batch_retry)
            except RuntimeError:
                if strict:
                    raise
                failed += n
                log(f"[WARNING] chunk {ci} ({n:,} rows) failed twice -- those candidates rank by fr/ppm")
                done, ci = done + n, ci + 1
                continue
        for row, s, o in zip(part.itertuples(), spectra, ok):
            if o:
                sim[(row.key, row.adduct, row.ce_ev)] = (s[0], s[1], row.precursor_mz)
        for k, v in err_counts(err).items():
            errs[k] = errs.get(k, 0) + v
        if timing.get("mol_per_s"):
            rate, startup = float(timing["mol_per_s"]), max(float(wall - timing["t_pred"]), 1.0)
        done, ci = done + n, ci + 1
        log(f"  sim chunk {ci}/{n_chunks}: {done:,}/{len(req):,} rows | {rate:.1f}/s | ok {len(sim):,}")
    attempted = len(req) - capped
    ok_share = len(sim) / (attempted - failed) if attempted > failed else float("nan")
    return sim, dict(n_requests=len(req), n_predicted=len(sim), sim_ok_share=ok_share, sim_errors=errs,
                     failed_rows=failed, capped_rows=capped, rate=rate, startup_s=startup)
