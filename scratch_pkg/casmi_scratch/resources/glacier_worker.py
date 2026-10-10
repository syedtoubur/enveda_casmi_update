"""GLACIER forward-simulation worker: the iceberg_worker.py protocol on top of the casmi26-glacier gl_runner.py.

Runs in a subprocess (never in the notebook's interpreter, which keeps RDKit 2026.03.3) with PYTHONPATH =
<glacier>/shim, <glacier>/src, the RDKit 2025.03 site built from the casmi26-iceberg wheels. Model loading, SMILES
canonicalisation, featurisation, batching (B * N_atoms^2 budget, OOM split, fail-fast) and the post-processing (merge
identical m/z by MAX, top 100, max 1) are gl_runner's own functions; only the I/O is ours:

    request tsv  : key, smiles, adduct (CASMI spelling), precursor_mz, ce (eV), instrument
    response npz : off (n+1,) int64, mz / it float32 flat, ok bool, err str, timing json, above_n / above_share /
                   above_off (peaks dropped above precursor_mz + --prec-margin), as iceberg_worker.py

GLACIER predicts [M+H]+ / [M+Na]+ only (gl_runner.COV); the CE is bucketed to 5 eV (min 5) as the checkpoint was used
in the A1 panel. Rejected rows get err "rdkit_parse:<why>", "adduct_unsupported:<adduct>", "ce_nan"; a failed
prediction "predict:none". A systematic failure (gl_runner fail-fast) exits non-zero, so the scheduler retries / degrades.

    python glacier_worker.py --pkg /kaggle/input/.../casmi26-glacier --inp req.tsv --out resp.npz
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", required=True, help="casmi26-glacier dir (gl_runner.py, ckpt/glacier.pt, shim/, src/)")
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--prec-margin", type=float, default=2.0)
    p.add_argument("--nondeterministic", dest="deterministic", action="store_false")
    return p.parse_args()


def to_peaks(s, k, prec, margin):
    """(mz, it) -> (mz, it, above_n, above_share, above_off): iceberg_worker.to_peaks on gl_runner's post() output.

    Drops peaks above prec + margin, then keeps the top-k by intensity, sorted by m/z, base = 1.
    """
    s = np.stack([np.asarray(s[0], np.float64), np.asarray(s[1], np.float64)], axis=1).reshape(-1, 2)
    s = s[np.isfinite(s).all(axis=1) & (s[:, 1] > 0)]
    above = s[:, 0] > prec + margin
    above_n = int(above.sum())
    above_share = float(s[above, 1].sum() / s[:, 1].sum()) if len(s) else 0.0
    above_off = float(s[above, 0].max() - prec) if above_n else np.nan
    s = s[~above]
    if len(s) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32), above_n, above_share, above_off
    s = s[np.argsort(s[:, 1])[::-1][:k]]
    s = s[np.argsort(s[:, 0])]
    return (s[:, 0].astype(np.float32), (s[:, 1] / s[:, 1].max()).astype(np.float32),
            above_n, above_share, above_off)


def main():
    args = get_args()
    if args.deterministic:                                     # before torch / CUDA initialise (gl_runner.main)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t0 = time.time()
    pkg = os.path.abspath(args.pkg)
    sys.path.insert(0, pkg)
    import gl_runner as gl                                     # noqa: E402  (the casmi26-glacier runner, as a library)
    log = gl.Log()
    # site 'none': the RDKit 2025.03 site is already on PYTHONPATH (sim.setup_glacier); gl_runner's own install_site
    # would look for its .ice_site_ok marker and reinstall the package's cp312 RDKit
    rdkit_v = gl.setup_env(SimpleNamespace(pkg=pkg, site="none", wheels=None, any_rdkit=False), log)
    import torch                                               # noqa: E402
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        log("cuda unavailable -> cpu")
        dev = "cpu"
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    model = gl.load_model(pkg, dev, log)
    t_load = time.time() - t0

    df = pd.read_csv(args.inp, sep="\t")
    n = len(df)
    err = [""] * n
    jobs = [None] * n
    for i, r in enumerate(df.to_dict("records")):
        smi = r["smiles"] if isinstance(r["smiles"], str) else ""
        ok, why = gl.smiles_ok(smi)
        ce = float(r["ce"])
        if not ok:
            err[i] = f"rdkit_parse:{why}"
        elif r["adduct"] not in gl.COV:
            err[i] = f"adduct_unsupported:{r['adduct']}"
        elif not np.isfinite(ce):
            err[i] = "ce_nan"
        else:
            jobs[i] = (smi, r["adduct"], gl.instr_token({"instr": r["instrument"]}), gl.ce_bucket(ce))

    pred = gl.Predictor(model, dev, args.batch_size, log)
    t1 = time.time()
    pred.predict({j for j in jobs if j is not None}, float("inf"))
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t_pred = time.time() - t1

    empty = (np.empty(0, np.float32), np.empty(0, np.float32))
    peaks = [empty] * n
    above_n = np.zeros(n, np.int32)
    above_share = np.zeros(n, np.float32)
    above_off = np.full(n, np.nan, np.float32)
    for i, j in enumerate(jobs):
        if j is None:
            continue
        s = pred.cache.get(j)
        if s is None:
            err[i] = "predict:none"
            continue
        mz_, it_, above_n[i], above_share[i], above_off[i] = to_peaks(s, args.sparse_k, float(df.precursor_mz.iloc[i]),
                                                                      args.prec_margin)
        peaks[i] = (mz_, it_)
        if len(mz_) == 0:
            err[i] = "empty_after_prec_cut" if above_n[i] else "empty_spectrum"
    ok = np.array([len(p[0]) > 0 for p in peaks], bool)
    lens = np.array([len(p[0]) for p in peaks], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    mz = np.concatenate([p[0] for p in peaks]) if n else np.empty(0, np.float32)
    it = np.concatenate([p[1] for p in peaks]) if n else np.empty(0, np.float32)
    n_rows = sum(j is not None for j in jobs)
    timing = dict(model="glacier", n=n, n_predicted=int(ok.sum()), t_load=t_load, t_pred=t_pred,
                  mol_per_s=float(n_rows / t_pred) if t_pred > 0 and n_rows else None,
                  n_pred=int(pred.n_pred), n_pred_fail=int(pred.n_fail), bs_final=int(pred.bs), rdkit=rdkit_v,
                  device=dev, batch_size=args.batch_size, prec_margin=args.prec_margin,
                  gpu=torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu")
    np.savez(args.out, off=off, mz=mz, it=it, ok=ok, err=np.array(err, dtype="U200"),
             above_n=above_n, above_share=above_share, above_off=above_off, timing=json.dumps(timing))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
