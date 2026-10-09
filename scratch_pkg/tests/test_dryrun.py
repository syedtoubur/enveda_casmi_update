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


def fake_r1_pool(d):
    """casmi26-scratch-pool built from the fake COCONUT arrays: every 3rd structure is enveda-180-only."""
    from casmi_scratch import pool_build as PB
    from casmi_scratch.pool import F_COCO, F_E180, F_TRAIN
    coco = Path(d) / "coco"
    cm = pickle.load(open(coco / "coco_meta.pkl", "rb"))
    mass = np.load(coco / "coco_mass.npy")
    flags = np.where(np.arange(len(mass)) % 3 == 0, F_TRAIN | F_E180, F_COCO)
    pool = pd.DataFrame({"key": cm["keys"], "smiles": cm["smiles"], "mass": mass, "flags": flags}).sort_values("mass")
    side = pd.DataFrame({"np_superclass": None}, index=pd.Index(pool.key, name="key"))
    km = pd.DataFrame({"inchikey14": ["RYYVLZVUVIJVGH"], "scorer_key": [cm["keys"][1]]})
    PB.write_pool(pool.reset_index(drop=True), side, km, str(Path(d) / "scratch-pool"), dict(version="dry"))
    return int((flags == (F_TRAIN | F_E180)).sum())


def test_dryrun_r1b():
    test_path = ROOT / "test.parquet"
    if not test_path.exists():
        print("test_dryrun_r1b: SKIPPED (no local test.parquet)")
        return
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    code = mk.render("r0_kaggle", {"use_sim": False, "frag_workers": 1, "pool_source": "r1", "pool_drop_e180": True},
                     tag="dry-r1b")
    with tempfile.TemporaryDirectory() as d:
        fake_input(d, te)
        n_e180 = fake_r1_pool(d)
        (Path(d) / "coco" / "coco_meta.pkl").unlink()                  # R1 must not need the COCONUT arrays
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
    assert rep["dropped_e180"] == n_e180 and rep["pool_size"] == 61 * 3 - n_e180 and rep["pool_version"] == "dry"
    assert "remapped to scorer keys" in r.stdout and np.isfinite(rep["cand_median"])


def test_dryrun_r2():
    """R2 class vote on the fake R1 pool with NPClassifier labels: the vote runs, counters reach the report."""
    test_path = ROOT / "test.parquet"
    if not test_path.exists():
        print("test_dryrun_r2: SKIPPED (no local test.parquet)")
        return
    from casmi_scratch import npc
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    code = mk.render("r0_kaggle", {"use_sim": False, "frag_workers": 1, "pool_source": "r1", "pool_drop_e180": True,
                                   "class_vote": True}, tag="dry-r2")
    with tempfile.TemporaryDirectory() as d:
        fake_input(d, te)
        fake_r1_pool(d)
        n = len(np.load(Path(d) / "scratch-pool" / "pool_key.npy"))
        npc.write_npc(str(Path(d) / "scratch-pool"), pd.DataFrame({
            "sc1": np.where(np.arange(n) % 2 == 0, "Purine alkaloids", "Flavonoids"), "sc2": "Steroids", "sc3": None,
            "sp1": 0.8, "sp2": 0.1, "sp3": np.nan}))
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
    assert rep["class_vote"] is True and "[vote] NPClassifier superclass" in r.stdout
    assert all(k in rep for k in ("n_vote_mol", "n_bonus_reorders", "n_vote_first"))


TESTS = [test_dryrun, test_dryrun_r1b, test_dryrun_r2]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print("test_dryrun: OK")
