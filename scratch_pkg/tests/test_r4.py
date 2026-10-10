"""R4 FPNet family fusion: N-channel RRF key, FP-ordered stage C, the prvsiyan bank (kind "prv"), notebook dry run."""
import json
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from casmi_scratch import fpnet  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402
from casmi_scratch.pipeline import fill_mass_nearest  # noqa: E402
from casmi_scratch.pool import CandidatePool  # noqa: E402
from casmi_scratch.rank import Slots, fp_rrf, rank_two_tier  # noqa: E402


def test_config_kinds():
    c = Config()
    assert c.fp_kinds() == () and c.fp_w() == ()
    assert c.with_(fp_channel="ahmed").fp_kinds() == ("ahmed",) and c.with_(fp_channel="ahmed").fp_w() == (1.0,)
    c4 = c.with_(fp_channel="ahmed", fp_banks=("ahmed", "prv"), fp_weights=(1.0, 0.6))
    assert c4.fp_kinds() == ("ahmed", "prv") and c4.fp_w() == (1.0, 0.6)
    try:
        c.with_(fp_banks=("ahmed", "prv"), fp_weights=(1.0,)).fp_w()
        raise AssertionError("weights not matching the banks must be refused")
    except ValueError:
        pass


def test_one_family_is_r3_order():
    rng = np.random.default_rng(0)
    lv, sv, fr, ppm, b, f = (rng.random(60) for _ in range(6))
    lv[:6] = 0.9
    sv[10:30] = 0.0
    f[5] = np.nan
    r3 = rank_two_tier(lv, sv, fr, ppm, 0.75, b, f, 3.0)
    assert np.array_equal(r3, rank_two_tier(lv, sv, fr, ppm, 0.75, b, [f], 3.0, (1.0,)))
    assert np.array_equal(r3, rank_two_tier(lv, sv, fr, ppm, 0.75, b, [f], 3.0))


def test_family_key_tier2_only():
    lv = np.array([0.9, 0, 0, 0, 0])
    sv = np.zeros(5)
    fr = np.zeros(5)
    fa = np.array([9.0, 1.0, 3.0, 2.0, np.nan])
    fp = np.array([9.0, 3.0, 2.0, 1.0, 5.0])
    o = rank_two_tier(lv, sv, fr, np.ones(5), 0.75, fp=[fa, fp], fp_w=(1.0, 0.6))
    assert o[0] == 0                                                # tier 1 untouched by FP
    t2 = np.array([1, 2, 3, 4])
    key = 1 / (3 + 2.5) + fp_rrf([fa, fp], (1.0, 0.6), 3.0, t2)    # all sv tie at the average rank 2.5
    assert o[1:].tolist() == t2[np.argsort(-key, kind="stable")].tolist()
    # candidate 4: no family-A bits (NaN, last in A) but best in P -> ahead of 3, which is last in P
    assert o.tolist() == [0, 2, 4, 1, 3], o                         # worked by hand
    # P's weight matters: with P alone, 4 leads tier 2
    assert rank_two_tier(lv, sv, fr, np.ones(5), 0.75, fp=[fp])[1] == 4


def test_stage_c_order():
    pool = CandidatePool([100.0, 100.0001, 100.0002, 100.0003], ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC",
                                                                "DDDDDDDDDDDDDD"], ["C", "CC", "CCC", "CCCC"])
    s = Slots(4)
    fill_mass_nearest(s, pool, 100.00021, 120.0)
    assert s.keys == ["CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD", "BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"]   # mass only
    fz = {0: [5.0, 1.0], 1: [np.nan, 0.0], 2: [1.0, 0.5], 3: [5.0, 2.0]}
    fp_fn = lambda idx: [np.array([fz[int(i)][0] for i in idx]), np.array([fz[int(i)][1] for i in idx])]  # noqa: E731
    s = Slots(4)
    fill_mass_nearest(s, pool, 100.00021, 120.0, fp_fn, (1.0, 0.6), 3.0)
    assert s.keys == ["DDDDDDDDDDDDDD", "AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", "BBBBBBBBBBBBBB"], s.keys
    s = Slots(4)
    fill_mass_nearest(s, pool, 100.00021, 120.0, lambda idx: None, (1.0, 0.6), 3.0)
    assert s.keys[0] == "CCCCCCCCCCCCCC"                           # no logits -> mass order


def prv_tree(d, nbits=64):
    """Own + prvsiyan + full1 checkpoints under d; the prvsiyan set gets its merged model and the COCONUT arrays a
    coco_fp.npy, plus a same-length decoy fp_bits.npy without coco_fp.npy next to it."""
    import torch
    from test_r3 import make_ckpts
    cdir, bdir = make_ckpts(d, "own", nbits=nbits)
    make_ckpts(d, "ahmed", nbits=nbits + 32)
    ck = torch.load(Path(d) / "casmi26-fp-models-v2" / "fp_single_s2.pt", weights_only=False)
    torch.save(ck, Path(d) / "casmi26-fp-models-v2" / "fp_merged_m1.pt")
    np.save(bdir / "coco_fp.npy", np.zeros((3, (nbits + 7) // 8), np.uint8))
    (Path(d) / "a").mkdir(exist_ok=True)
    np.save(Path(d) / "a" / "fp_bits.npy", np.arange(nbits))
    return bdir


def test_prv_bank():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("test_r4: torch missing, bank test SKIPPED")
        return
    with tempfile.TemporaryDirectory() as d:
        bdir = prv_tree(d)
        paths, bits, ref = fpnet.find_bank_files("prv", d)
        assert [Path(p).name for p in paths] == ["fp_merged_m1.pt", "fp_single_s2.pt"], paths
        assert Path(bits).parent == bdir and Path(ref) == bdir     # the coco-adjacent one, not the decoy in a/
        bank = fpnet.load_bank("prv", paths, bits, device="cpu")
        assert bank.kind == "prv" and bank.merged == [True, False] and bank.feat.raw_bits == fpnet.P_RAW_BITS
        from test_r3 import SMI, spectra
        z = bank.molecule_z(spectra())
        assert z.shape == (64,) and np.isfinite(bank.fz(z, ["K"], [SMI[0]])).all()
        own_paths, _, _ = fpnet.find_bank_files("own", d)          # the own export still excludes prvsiyan's set
        assert all("casmi26-fp-models-v2" not in p for p in own_paths)
        assert len(fpnet.find_bank_files("ahmed", d)[0]) == 1


def test_dryrun_r4():
    test_path = ROOT / "test.parquet"
    try:
        import torch  # noqa: F401
    except ImportError:
        test_path = None
    if test_path is None or not test_path.exists():
        print("test_dryrun_r4: SKIPPED")
        return
    import mk_notebook as mk
    import test_dryrun as TD
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    over = dict(use_sim=False, frag_workers=1, pool_source="r1", pool_drop_e180=True, fp_channel="ahmed",
                fp_banks=("ahmed", "prv"), fp_weights=(1.0, 0.6), fp_stagec=True, ppm_win=0.5, ppm_expand=(0.5,),
                backfill_ppm=500.0)
    code = mk.render("r0_kaggle", over, tag="dry-r4")
    with tempfile.TemporaryDirectory() as d:
        TD.fake_input(d, te)
        tr = pq.read_table(Path(d) / "comp" / "train.parquet").to_pandas()
        tr["ingest_lib"] = "enveda-np-examples"
        tr.to_parquet(Path(d) / "comp" / "train.parquet")
        TD.fake_r1_pool(d)
        prv_tree(d)
        work = Path(d) / "work"
        work.mkdir()
        script = Path(d) / "nb.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ, CASMI_INPUT=d, CASMI_WORK=str(work))
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        sub = pd.read_csv(work / "submission.csv")
        rep = json.loads((work / "run_report.json").read_text())
    assert len(sub) == 3 and (sub.smiles.str.split(";").map(len) == 25).all()
    assert rep["fp_banks"] == ["ahmed", "prv"] and rep["fp_weights"] == [1.0, 0.6]
    assert rep["n_fp_z_by"] == [3, 3] and rep["n_fp_mol_by"] == [3, 3]
    assert [b["kind"] for b in rep["fp_bank"]] == ["ahmed", "prv"]
    assert [c["kind"] for c in rep["fp_check"]["banks"]] == ["ahmed", "prv"]
    assert rep["n_stagec_mol"] > 0 and rep["n_stagec_slots"] > 0      # tight window -> stage C fills, FP-ordered
    assert r.stdout.count("[fp] featuriser parity") == 2
    return rep


TESTS = [test_config_kinds, test_one_family_is_r3_order, test_family_key_tier2_only, test_stage_c_order,
         test_prv_bank, test_dryrun_r4]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print("test_r4: OK")
