"""P1.5 step 1 (Colab CPU): train.parquet -> train_meta.parquet (one row per train spectrum) + per-library stats.

The lib->fold map for fpnet_0's held-out folds (np / nplib / syn / plusk) is built from these printed library names,
not guessed. Output on Drive: casmi_2026/val_own/train_meta.parquet with
row, lib, instrument_type, adduct, ionization_mode, precursor_mz, n_peaks, ce_n, ik14 (plain RDKit InChIKey[:14], as
val_key uses), scaffold (Murcko SMILES of the structure; '' when none).

usage (Colab): python gen_val_probe.py [--root /content/drive/MyDrive/casmi_2026]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def ensure_rdkit():
    """Kaggle CPU kernel: install RDKit offline from the attached wheel dataset if the image lacks it."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        import glob
        import subprocess
        import sys
        whl = sorted(glob.glob("/kaggle/input/**/rdkit-*cp%d%d*.whl" % sys.version_info[:2], recursive=True))
        assert whl, "no rdkit wheel for this python under /kaggle/input"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "-q", whl[-1]], check=True)


def ik14_and_scaffold(smiles):
    ensure_rdkit()
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out[s] = ("", "")
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        out[s] = (Chem.MolToInchiKey(m)[:14], sc)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
    ap.add_argument("--out", default="/kaggle/working")
    a, _ = ap.parse_known_args(argv)                   # a notebook kernel's argv carries its own flags
    t0 = time.time()
    if not os.path.exists(a.train):
        import glob
        hits = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
        print("train not at", a.train, "| found", hits, flush=True)
        a.train = hits[0]
    # train schema (probe v1, 10-11): ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula,
    # ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs,
    # ms2_normalized_intensities, num_peaks, base_peak_intensity, collision_energy_ev, ... (2,539,608 rows)
    cols = ["ingest_lib", "normalized_smiles", "inchikey14", "instrument_type", "adduct", "ionization_mode",
            "precursor_mz", "num_peaks", "collision_energy_ev"]
    tr = pd.read_parquet(a.train, columns=cols)
    print("train rows", len(tr), flush=True)
    km = ik14_and_scaffold(pd.unique(tr.normalized_smiles))
    print(f"structures {len(km)} in {time.time() - t0:.0f}s", flush=True)
    own = tr.normalized_smiles.map(lambda s: km[s][0]).values
    agree = own == tr.inchikey14.astype(str).values
    print(f"ik14 column vs plain RDKit IK14: agree {agree.mean():.4f} ({int((~agree).sum())} rows differ)", flush=True)
    meta = pd.DataFrame({
        "row": np.arange(len(tr)), "lib": tr.ingest_lib.astype(str).values,
        "instrument_type": tr.instrument_type.astype(str).values, "adduct": tr.adduct.astype(str).values,
        "ionization_mode": tr.ionization_mode.astype(str).values, "precursor_mz": tr.precursor_mz.astype(float).values,
        "n_peaks": tr.num_peaks.values,
        "ce_n": tr.collision_energy_ev.map(lambda x: len(x) if hasattr(x, "__len__") else 1).values,
        "ik14": tr.inchikey14.astype(str).values, "ik14_rdkit": own,
        "scaffold": tr.normalized_smiles.map(lambda s: km[s][1]).values})
    od = a.out
    os.makedirs(od, exist_ok=True)
    meta.to_parquet(os.path.join(od, "train_meta.parquet"), index=False)
    print("wrote", os.path.join(od, "train_meta.parquet"), flush=True)

    # per library: rows, structures, instruments, and structures found ONLY in this library
    libs_of = meta.groupby("ik14").lib.agg(lambda x: frozenset(x))
    only = libs_of[libs_of.map(len) == 1].map(lambda s: next(iter(s))).value_counts()
    g = meta.groupby("lib")
    rep = pd.DataFrame({"rows": g.size(), "structures": g.ik14.nunique(),
                        "only_here": only.reindex(g.size().index).fillna(0).astype(int),
                        "instruments": g.instrument_type.agg(lambda x: dict(x.value_counts().head(3))),
                        "adducts": g.adduct.agg(lambda x: dict(x.value_counts().head(3)))})
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200)
    print(rep.sort_values("rows", ascending=False).to_string(), flush=True)
    print("structures in >1 library", int((libs_of.map(len) > 1).sum()), "| empty ik14 rows", int((meta.ik14 == "").sum()))
    print(f"PROBE DONE {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
