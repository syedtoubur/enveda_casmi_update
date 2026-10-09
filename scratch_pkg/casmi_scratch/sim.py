"""Forward simulation (ICEBERG, or GLACIER with `sim_model="glacier"`): request building, the worker subprocess, and
the deadline-watchdog scheduler.

The model runs in its own environment (torch / DGL / RDKit 2025.03: the casmi26-iceberg site, or a venv) through
`resources/iceberg_worker.py` or `resources/glacier_worker.py`; both speak the same TSV -> NPZ protocol, and this module
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
GL_WORKER_SRC = Path(__file__).resolve().parent / "resources" / "glacier_worker.py"
REQ_COLS = ["key", "smiles", "adduct", "precursor_mz", "ce_ev", "prio", "ce", "instrument"]
SIM_ADDUCTS = {"glacier": ("[M+H]+", "[M+Na]+")}     # the GLACIER MassSpecGym checkpoint's adducts (gl_runner.COV)


def sim_adducts(cfg):
    """Query adducts that get simulation requests: cfg.covered, restricted to what the model predicts."""
    allowed = SIM_ADDUCTS.get(cfg.sim_model)
    return tuple(a for a in cfg.covered if allowed is None or a in allowed)


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


def sha1(path, chunk=1 << 20, algo="sha1"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --- the worker ----------------------------------------------------------------------------------------
class IcebergWorker:
    """Callable(req, tag, batch_size) -> (spectra [(mz, it)], ok, err, timing, wall_s)."""
    src = WORKER_SRC
    batch_retry = None              # None: the scheduler retries with cfg.batch_retry

    def __init__(self, py, gen, inten, env, work, device="cuda", sparse_k=100, prec_margin=2.0, batch=128):
        self.py, self.gen, self.inten, self.env = str(py), str(gen), str(inten), env
        self._init_common(work, device, sparse_k, prec_margin, batch)

    def _init_common(self, work, device, sparse_k, prec_margin, batch):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.device, self.sparse_k, self.prec_margin, self.batch = device, sparse_k, prec_margin, batch
        self.worker = self.work / self.src.name
        shutil.copyfile(self.src, self.worker)

    def _model_args(self):
        return ["--gen", self.gen, "--inten", self.inten]

    def _cmd(self, inp, out, batch_size):
        return [self.py, str(self.worker), *self._model_args(), "--inp", str(inp), "--out", str(out),
                "--device", self.device, "--batch-size", str(batch_size or self.batch),
                "--sparse-k", str(self.sparse_k), "--prec-margin", str(self.prec_margin)]

    def __call__(self, req, tag, batch_size=None):
        inp, out = self.work / f"req_{tag}.tsv", self.work / f"resp_{tag}.npz"
        req[["key", "smiles", "adduct", "precursor_mz", "ce", "instrument"]].to_csv(inp, sep="\t", index=False)
        cmd = self._cmd(inp, out, batch_size)
        t = time.time()
        r = subprocess.run(cmd, text=True, capture_output=True, env=self.env, timeout=12 * 3600)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-4000:])
            raise RuntimeError(f"worker failed on {tag} (exit {r.returncode}):\n{r.stderr[-1500:]}")
        with np.load(out) as z:     # closed before the unlink below (Windows cannot delete an open file)
            # read each array once: NpzFile re-reads an array on every access (the Phase-1 Colab OOM)
            off, mz, it = z["off"], z["mz"], z["it"]
            ok, err, timing = z["ok"], z["err"], json.loads(str(z["timing"]))
        assert len(off) == len(req) + 1, f"{out}: {len(off) - 1} rows, expected {len(req)}"
        spectra = [(mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]]) for i in range(len(req))]
        for f in (inp, out):
            f.unlink(missing_ok=True)
        return spectra, ok, err, timing, time.time() - t


class GlacierWorker(IcebergWorker):
    """The same protocol, served by resources/glacier_worker.py on top of the casmi26-glacier gl_runner.py."""
    src = GL_WORKER_SRC

    def __init__(self, py, pkg, env, work, device="cuda", sparse_k=100, prec_margin=2.0, batch=32, batch_retry=8):
        self.py, self.pkg, self.env, self.batch_retry = str(py), str(pkg), env, batch_retry
        self._init_common(work, device, sparse_k, prec_margin, batch)

    def _model_args(self):
        return ["--pkg", self.pkg]


ICE_RDKIT = "2025.3.6"     # the RDKit release the casmi26-iceberg package pins (2026.03 kekulizes differently)


def py_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def rdkit_wheels(input_root, prefer=None):
    """Attached rdkit-*.whl built for this interpreter, the `prefer` release first (sorted otherwise)."""
    hits = sorted(w for w in glob.glob(f"{input_root}/**/rdkit-*.whl", recursive=True)
                  if f"-{py_tag()}-" in os.path.basename(w))
    if prefer:
        hits.sort(key=lambda w: not os.path.basename(w).startswith(f"rdkit-{prefer}-"))
    return hits


def _pick_ckpt(pkg, word):
    hits = sorted(p for p in glob.glob(f"{pkg}/ckpt/**/*", recursive=True)
                  if p.endswith((".ckpt", ".pt")) and word in os.path.basename(p).lower())
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one '{word}' checkpoint under {pkg}/ckpt, found {hits} "
                                f"(all: {sorted(glob.glob(f'{pkg}/ckpt/**/*', recursive=True))})")
    return hits[0]


def preseed_ice_site(pkg, site, input_root):
    """Install the casmi26-iceberg wheels into `site` (pip --target), swapping the package's RDKit wheel for the
    same release built for this interpreter (the v18 zenith notebook's fix for the cp313 Kaggle image)."""
    srcs = sorted(glob.glob(os.path.join(pkg, "wheels", "*.whl.ice")) + glob.glob(os.path.join(pkg, "wheels", "*.whl")))
    if not srcs:
        raise FileNotFoundError(f"no wheels under {pkg}/wheels -- casmi26-iceberg uploaded without sub-directories?")
    keep = [w for w in srcs if not os.path.basename(w).lower().startswith("rdkit")]
    new = rdkit_wheels(input_root, prefer=ICE_RDKIT)
    if not new:
        raise FileNotFoundError(f"no {py_tag()} rdkit wheel attached (eveliaveldrine/rdkit-wheels)")
    want = "\n".join(os.path.basename(w) for w in keep + new[:1])
    marker = os.path.join(site, ".casmi_site_ok")
    if os.path.exists(marker) and open(marker).read() == want:
        return new[0]
    shutil.rmtree(site, ignore_errors=True)
    wdir = os.path.join(site, ".wheels")
    os.makedirs(wdir, exist_ok=True)
    whl = []
    for w in keep + new[:1]:
        dst = os.path.join(wdir, os.path.basename(w)[:-4] if w.endswith(".ice") else os.path.basename(w))
        shutil.copyfile(w, dst)
        whl.append(dst)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--no-compile",
                        "--disable-pip-version-check", "--upgrade", "--target", site, *whl],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError("ICEBERG site install failed:\n" + (r.stdout or "")[-3000:])
    shutil.rmtree(wdir, ignore_errors=True)
    with open(marker, "w") as f:
        f.write(want)
    print(f"[ICEBERG] site ready: {len(keep)} package wheels + {os.path.basename(new[0])}", flush=True)
    return new[0]


def _offline_manifest(input_root):
    hits = sorted(glob.glob(f"{input_root}/**/wheelhouse", recursive=True), key=len)
    p = os.path.join(os.path.dirname(hits[0]), "manifest.json") if hits else None
    return json.load(open(p)) if p and os.path.exists(p) else None


def setup_ice_pkg(pkg, input_root, work, cfg):
    """ICEBERG from the casmi26-iceberg package (ckpt/ gen + inten .pt or .ckpt, src/, shim/, wheels/*.whl.ice), run by our own worker under
    the notebook's interpreter with shim/, src/ and the preseeded site first on its PYTHONPATH (no venv)."""
    site = os.path.join(str(work), "ice_site")
    whl = preseed_ice_site(pkg, site, input_root)
    gen, inten = _pick_ckpt(pkg, "gen"), _pick_ckpt(pkg, "inten")
    man = _offline_manifest(input_root)
    if man is not None:             # casmi26-iceberg-offline attached as well: CE convention must agree
        assert man["ce_mode"] == cfg.ce_mode and man["instrument"] == cfg.instrument, man
    print(f"[ICEBERG] casmi26-iceberg {pkg} | rdkit {os.path.basename(whl)} | gen {os.path.basename(gen)} "
          f"{sha1(gen)[:10]} | inten {os.path.basename(inten)} {sha1(inten)[:10]} | offline manifest "
          f"{'CE convention ok' if man else 'not attached'}", flush=True)
    path = [p for p in (os.path.join(pkg, "shim"), os.path.join(pkg, "src"), site) if os.path.isdir(p)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONPATH=os.pathsep.join(path), DGLBACKEND="pytorch", CUDA_VISIBLE_DEVICES="0", CASMI_ICE_PKG="1")
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    if device == "cpu":
        print("[WARNING] no GPU -- ICEBERG on CPU is ~15x slower; the watchdog will cap the simulation", flush=True)
    return IcebergWorker(sys.executable, gen, inten, env, work, device=device, sparse_k=cfg.sparse_k,
                         prec_margin=cfg.prec_margin, batch=cfg.batch)


def setup_iceberg(input_root, work, cfg, venv="/kaggle/working/iceberg_venv"):
    """casmi26-iceberg package if attached (ice_runner.py found), else the casmi26-iceberg-offline wheelhouse in
    its own venv. Verifies checkpoints, returns an IcebergWorker."""
    runner = sorted(glob.glob(f"{input_root}/**/ice_runner.py", recursive=True), key=len)
    if runner:
        return setup_ice_pkg(os.path.dirname(runner[0]), input_root, work, cfg)
    hits = [h for h in glob.glob(f"{input_root}/**/wheelhouse", recursive=True) if os.path.isdir(h)]
    if not hits:
        raise FileNotFoundError("neither ice_runner.py nor wheelhouse found -- attach casmi26-iceberg (or casmi26-iceberg-offline)")
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


def _find_pkg(input_root, marker, dataset):
    hits = sorted(glob.glob(f"{input_root}/**/{marker}", recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {input_root} -- attach {dataset}")
    return os.path.dirname(hits[0])


def setup_glacier(input_root, work, cfg):
    """GLACIER from casmi26-glacier (gl_runner.py, ckpt/glacier.pt, shim/, src/; no wheels), run by our own worker under
    the notebook's interpreter. The RDKit 2025.03 site comes from the casmi26-iceberg wheels (preseed_ice_site, cp-tag
    RDKit swap), so casmi26-iceberg and the rdkit wheels stay attached; the ICEBERG checkpoints are not loaded."""
    assert cfg.ce_mode == "ev", f"GLACIER takes the CE in eV (5 eV buckets), not ce_mode={cfg.ce_mode!r}"
    pkg = _find_pkg(input_root, "gl_runner.py", "casmi26-glacier")
    ice = _find_pkg(input_root, "ice_runner.py", "casmi26-iceberg (its wheels build GLACIER's RDKit 2025.03 site)")
    ckpt = os.path.join(pkg, "ckpt", "glacier.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} missing -- casmi26-glacier uploaded without sub-directories?")
    site = os.path.join(str(work), "ice_site")
    whl = preseed_ice_site(ice, site, input_root)
    man_p = os.path.join(pkg, "MANIFEST.json")
    want = json.load(open(man_p))["files"]["ckpt/glacier.pt"]["sha256"] if os.path.exists(man_p) else None
    got = sha1(ckpt, algo="sha256")
    assert want is None or got == want, f"glacier.pt sha256 {got[:10]} != MANIFEST {want[:10]}"
    print(f"[GLACIER] casmi26-glacier {pkg} | rdkit {os.path.basename(whl)} | glacier.pt {got[:10]} "
          f"({'MANIFEST ok' if want else 'no MANIFEST'}) | site from {ice}", flush=True)
    path = [p for p in (os.path.join(pkg, "shim"), os.path.join(pkg, "src"), site) if os.path.isdir(p)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PYTHONPATH=os.pathsep.join(path), DGLBACKEND="pytorch", CUDA_VISIBLE_DEVICES="0")
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    if device == "cpu":
        print("[WARNING] no GPU -- GLACIER on CPU is slow; the watchdog will cap the simulation", flush=True)
    return GlacierWorker(sys.executable, pkg, env, work, device=device, sparse_k=cfg.sparse_k,
                         prec_margin=cfg.prec_margin, batch=cfg.gl_batch, batch_retry=cfg.gl_batch_retry)


def setup_sim(input_root, work, cfg, venv="/kaggle/working/iceberg_venv"):
    """The worker for cfg.sim_model."""
    if cfg.sim_model == "glacier":
        return setup_glacier(input_root, work, cfg)
    assert cfg.sim_model == "iceberg", cfg.sim_model
    return setup_iceberg(input_root, work, cfg, venv=venv)


def smoke(worker, cfg):
    """Caffeine [M+H]+ at 40 eV must predict; a 120 eV off-ladder probe is reported."""
    m = 194.0803756
    req = pd.DataFrame({"key": ["RYYVLZVUVIJVGH"] * 2, "smiles": ["Cn1cnc2c1c(=O)n(C)c(=O)n2C"] * 2,
                        "adduct": "[M+H]+", "precursor_mz": precursor_of(np.array([m, m]), "[M+H]+"),
                        "ce_ev": [40.0, 120.0]})
    req["ce"] = to_model_ce(req.ce_ev.to_numpy(), req.precursor_mz.to_numpy(), cfg.ce_mode)
    req["instrument"] = cfg.instrument
    sp, ok, err, timing, _ = worker(req, "smoke")
    assert ok[0], f"{cfg.sim_model} smoke failed: {err[0]}"
    return dict(model=cfg.sim_model, gpu=timing.get("gpu"), n_peaks_40=int(len(sp[0][0])),
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
                retry = getattr(call, "batch_retry", None) or cfg.batch_retry
                spectra, ok, err, timing, wall = call(part, f"c{ci}r", retry)
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
