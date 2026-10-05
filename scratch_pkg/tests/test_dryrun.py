"""Dry run of the generated R0 notebook on a fake /kaggle/input (no ICEBERG: use_sim=False, one worker).

Uses three real molecules from the local public test.parquet (skipped if absent); train / COCONUT are tiny fakes whose
masses sit inside the test molecules' windows, so the deep path, slot filling and the submission audit all run.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(PKG / "tools"))
sys.path.insert(0, str(PKG))
import mk_notebook as mk  # noqa: E402
from casmi_scratch.chem import neutral_mass  # noqa: E402

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def fake_input(d, te):
    comp, coco = Path(d) / "comp", Path(d) / "coco"
    comp.mkdir()
    coco.mkdir()
    te.to_parquet(comp / "test.parquet")
    pd.DataFrame({"molecule_id": te.molecule_id.unique(), "smiles": ""}).to_csv(comp / "sample_submission.csv",
                                                                              index=False)
    tr = te.drop(columns=["molecule_id", "spectrum_id"]).copy()
    tr["normalized_smiles"], tr["inchikey14"], tr["molecular_formula"] = CAFFEINE, "RYYVLZVUVIJVGH", "C8H10N4O2"
    tr.to_parquet(comp / "train.parquet")
    nm = neutral_mass(te.precursor_mz.to_numpy(), te.adduct.to_numpy(dtype=object))
    targets = pd.Series(nm).groupby(te.molecule_id.to_numpy()).median().to_numpy() / (1 + 1.45e-6)
    mass = np.concatenate([t * (1 + np.arange(-30, 31) * 1e-7) for t in targets])   # 61 per molecule, +-3 ppm
    keys = [f"{i:014d}" for i in range(len(mass))]
    pickle.dump(dict(keys=keys, smiles=[CAFFEINE] * len(mass), nbits=0), open(coco / "coco_meta.pkl", "wb"))
    np.save(coco / "coco_mass.npy", mass)


def test_dryrun():
    test_path = ROOT / "test.parquet"
    if not test_path.exists():
        print("test_dryrun: SKIPPED (no local test.parquet)")
        return
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    code = mk.render("r0_kaggle", {"use_sim": False, "frag_workers": 1}, tag="dry")
    with tempfile.TemporaryDirectory() as d:
        fake_input(d, te)
        work = Path(d) / "work"
        work.mkdir()
        script = Path(d) / "nb.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ, CASMI_INPUT=d, CASMI_WORK=str(work))
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        sub = pd.read_csv(work / "submission.csv")
        rep = json.loads((work / "run_report.json").read_text())
    assert len(sub) == 3 and (sub.smiles.str.split(";").map(len) == 25).all()
    assert rep["mode"] == "full" and rep["n_deep"] == 3 and rep["config"]["use_sim"] is False
    first = sub.smiles.str.split(";").str[0]
    assert (first == CAFFEINE).all(), first                      # a real pool candidate, not padding


TESTS = [test_dryrun]

if __name__ == "__main__":
    test_dryrun()
    print("test_dryrun: OK")
