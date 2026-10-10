"""R3 FPNet channel: RRF fusion, featurisers, banks (tiny real checkpoints), audit helpers, notebook dry runs."""
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
from casmi_scratch.rank import avg_rank_desc, rank_two_tier  # noqa: E402

SMI = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O", "OC1=CC=CC=C1"]


def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        print("test_r3: torch missing, bank tests SKIPPED")
        return False


def test_avg_rank_desc():
    assert avg_rank_desc(np.array([3, 1, 3, np.nan, 2.0])).tolist() == [1.5, 4.0, 1.5, 5.0, 3.0]


def test_fp_none_is_r2_order():
    rng = np.random.default_rng(0)
    lv, sv, fr, ppm, b = (rng.random(40) for _ in range(5))
    lv[:5] = 0.9
    assert np.array_equal(rank_two_tier(lv, sv, fr, ppm, 0.75, b), rank_two_tier(lv, sv, fr, ppm, 0.75, b, None))


def test_rrf_tier2_only_and_ties():
    lv = np.array([0.9, 0, 0, 0])
    sv = np.array([0, 0, 0, 0.5])
    fr = np.array([0, 0.1, 0.2, 0])
    o = rank_two_tier(lv, sv, fr, np.ones(4), 0.75, fp=np.array([0, 5, 1, 0]))
    assert o.tolist() == [0, 1, 3, 2]                               # tier 1 first; sv=0 tie broken by FP
    o = rank_two_tier(np.zeros(3), np.zeros(3), np.array([0.3, 0.2, 0.1]), np.ones(3), 0.75,
                      fp=np.array([np.nan, 1.0, 2.0]))
    assert o.tolist() == [2, 1, 0]                                  # all sv tie -> FP order, NaN last


def test_merge_peaks_variants():
    a = (np.array([100.0, 100.003, 200.0]), np.array([1.0, 2.0, 1.0]))
    mz, it = fpnet._merge_peaks([a])
    assert len(mz) == 2 and it[0] == 1.0
    mz2, _ = fpnet._merge_peaks([a], chain=True)
    assert len(mz2) == 2


def test_adduct_lists():
    assert len(fpnet.P_ADDUCTS) == 26 and fpnet.P_AD_IX["<unk>"] == 25
    assert fpnet.A_ADDUCTS[:10] == fpnet.A_TEST_ADDUCTS and fpnet.A_ADDUCTS[-1] == "<unk>"
    assert len(set(fpnet.A_ADDUCTS)) == len(fpnet.A_ADDUCTS) == 60


def test_featurisers():
    for kind, raw in (("own", fpnet.P_RAW_BITS), ("ahmed", fpnet.A_RAW_BITS)):
        bits = np.arange(0, raw, 7)
        f = fpnet.Featuriser(kind, bits)
        b, ok = f(SMI + ["not a smiles", None])
        assert b.shape == (5, len(bits)) and ok.tolist() == [True, True, True, False, False]
        assert b[0].sum() > 0 and not b[3].any()
        full = f.raw(SMI[0])
        assert len(full) == raw and np.array_equal(full[bits], b[0])
        ref = np.packbits(b[:3], axis=1)
        assert fpnet.featuriser_parity(f, SMI, ref) == 3
    try:
        fpnet.Featuriser("own", np.array([fpnet.P_RAW_BITS]))
        raise AssertionError("bits past the raw layout must be refused")
    except ValueError:
        pass


def make_ckpts(d, kind, nbits=64, seed=0):
    """Tiny real checkpoints of one family (+ fp_bits.npy) under d -> (ckpt dir, bits dir)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    raw = fpnet.P_RAW_BITS if kind == "own" else fpnet.A_RAW_BITS
    bits = np.sort(rng.choice(raw, nbits, replace=False))
    bdir = Path(d) / ("coco" if kind == "own" else "v2pool")
    cdir = Path(d) / ("casmi-fpnet" if kind == "own" else "casmi26-fpnet-full1")
    bdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    np.save(bdir / "fp_bits.npy", bits)
    sha = fpnet.bits_sha1(bits)
    names = ["fp_single_s0.pt", "fp_merged_m0.pt"] if kind == "own" else ["fpnet_full1.pt"]
    for n in names:
        net = fpnet.p_model(nbits, d=16, layers=1) if kind == "own" else fpnet.a_model(nbits, d=16, layers=1)
        ck = dict(model=net.state_dict(), nbits=nbits, d=16, layers=1, heads=8)
        if kind == "own":
            ck["fp_bits_sha1"] = sha
        torch.save(ck, cdir / n)
    if kind == "own":
        (cdir / "fp_bits_sha1.txt").write_text(sha)
        decoy = Path(d) / "casmi26-fp-models-v2"                    # prvsiyan's set: no fp_bits_sha1.txt
        decoy.mkdir(exist_ok=True)
        torch.save(dict(ck, fp_bits_sha1=None), decoy / "fp_single_s2.pt")
    return cdir, bdir


def spectra(n=3, mode=1):
    rng = np.random.default_rng(1)
    return [dict(mz=np.sort(rng.uniform(50, 190, 30)), it=rng.random(30), prec=195.0877, adduct="[M+H]+",
                 mode=mode, instr="timsTOF", ce=20.0 + 10 * i, ce_signed=20.0 + 10 * i, ce_n=1) for i in range(n)]


def test_banks():
    if not _torch_ok():
        return
    with tempfile.TemporaryDirectory() as d:
        for kind in ("own", "ahmed"):
            make_ckpts(d, kind)
            paths, bits, ref = fpnet.find_bank_files(kind, d)
            bank = fpnet.load_bank(kind, paths, bits, device="cpu")
            assert bank.nbits == 64 and bank.describe()["kind"] == kind
            if kind == "own":
                assert bank.merged == [True, False]                  # sorted: fp_merged_m0, fp_single_s0
                assert all("casmi-fpnet" + os.sep in p or "casmi-fpnet/" in p for p in paths), paths
            z = bank.molecule_z(spectra() + [dict(spectra(1, -1)[0], adduct="[M-H]-")])
            assert z.shape == (64,) and np.isfinite(z).all()
            assert bank.molecule_z([]) is None
            fz = bank.fz(z, ["K1", "K2", "K3"], [SMI[0], "bad", SMI[1]])
            assert np.isfinite(fz[0]) and np.isnan(fz[1]) and np.isfinite(fz[2])
            b, _ = bank.feat([SMI[0]])
            assert np.isclose(fz[0], b[0].astype(np.float32) @ z, atol=1e-3)
            bank.release()
            assert bank.models == []
        # own bank with a single checkpoint type is refused; a foreign fp_bits sha is refused
        p, b, _ = fpnet.find_bank_files("own", d)
        try:
            fpnet.load_bank("own", [x for x in p if "single" in x], b, device="cpu")
            raise AssertionError("own bank without a merged model must be refused")
        except ValueError:
            pass
        np.save(Path(d) / "coco" / "fp_bits.npy", np.arange(64))
        try:
            fpnet.find_bank_files("own", d)
            raise AssertionError("own fp_bits with a foreign sha1 must not be picked")
        except FileNotFoundError:
            pass


def test_molecule_spectra_and_ce():
    sub = pd.DataFrame({"ms2_mzs": [[50.0, 60.0]], "ms2_normalized_intensities": [[1.0, 0.5]], "precursor_mz": [100.0],
                        "adduct": ["[M-H]-"], "ionization_mode": ["negative"], "collision_energy_ev": [[-20.0, -40.0]],
                        "instrument_type": ["timsTOF"]})
    s = fpnet.molecule_spectra(sub)[0]
    assert s["mode"] == -1 and s["ce"] == 30.0 and s["ce_signed"] == -30.0 and s["ce_n"] == 2
    s = fpnet.molecule_spectra(sub.drop(columns="instrument_type").assign(collision_energy_ev=[None]))[0]
    assert s["instr"] is None and np.isnan(s["ce"]) and s["ce_n"] == 0


def toy_world():
    """Toy library (test_r1) + 12-candidate pool around 600 Da with real SMILES; the truth is the last one."""
    from test_r1 import toy_analog_lib
    from casmi_scratch import chem
    from casmi_scratch.pool import CandidatePool
    target = 600.0
    lib, hexose = toy_analog_lib(target)
    spec = dict(adduct="[M+H]+", precursor_mz=chem.precursor_of(target, "[M+H]+") * (1 + 1.45e-6),
                ionization_mode="positive", collision_energy_ev=np.array([20.0]), instrument_type="timsTOF",
                ms2_mzs=np.array([100.0, 200.0, 300.0 + hexose]), ms2_normalized_intensities=np.array([0.2, 0.3, 0.5]))
    keys = [f"DECOY{i:09d}" for i in range(11)] + ["TRUTHAAAAAAAAA"]
    smis = [SMI[i % 3] + "C" * i for i in range(11)] + ["OC1C(O)C(O)C(CO)OC1Oc1ccccc1"]
    pool = CandidatePool([target * (1 + p * 1e-6) for p in np.linspace(-3, 3, 12)], keys, smis)
    return lib, pool, [("TRUTHAAAAAAAAA", pd.DataFrame([spec]))], (lambda ik: ik)


def test_fp_audit_toy():
    """fp_discrimination / summarize_fp / fp_self_rank on a toy pool + library with a tiny bank."""
    if not _torch_ok():
        return
    from casmi_scratch import audit
    from casmi_scratch.config import Config
    lib, pool, mols, truth_of = toy_world()
    no_frag = lambda smis, *a: np.zeros(len(smis), np.float32)  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        make_ckpts(d, "own")
        bank = fpnet.load_bank("own", *fpnet.find_bank_files("own", d)[:2], device="cpu")
        cfg = Config().with_(frag_workers=1)
        rows = audit.fp_discrimination([("M0", ik, sub) for ik, sub in mols], lib, pool, truth_of, bank, cfg,
                                       frag_fn=no_frag, log=lambda s: None)
        r = rows[0]
        assert r["truth_in_scored"] and r["fp_ok"] and r["rank0"] >= 10 and 1 <= r["fz_rank"] <= 12
        assert r["rank1"] <= r["rank0"]                              # sv/fr all tie -> FP decides
        s = audit.summarize_fp(rows)
        assert s["n"] == 1 and s["n_fp_ok"] == 1 and np.isclose(s["delta"], s["mrr_after"] - s["mrr_before"])
        hit, rr = audit.fp_self_rank(bank, mols, pool, truth_of, cfg, n=5, k=25)
        assert hit == 1 and rr[0]["n_window"] == 12
    m = audit.ban_mask(lib, [0, 10 ** 9], [lib.ik[1]], keymap={})
    assert m[0] and m[1] and m.sum() == 2


def _dry(fp_kind, tag):
    test_path = ROOT / "test.parquet"
    if not test_path.exists() or not _torch_ok():
        print(f"test_dryrun_{tag}: SKIPPED")
        return None
    import mk_notebook as mk
    import test_dryrun as TD
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    code = mk.render("r0_kaggle", {"use_sim": False, "frag_workers": 1, "pool_source": "r1", "pool_drop_e180": True,
                                   "fp_channel": fp_kind}, tag=f"dry-{tag}")
    with tempfile.TemporaryDirectory() as d:
        TD.fake_input(d, te)
        tr = pq.read_table(Path(d) / "comp" / "train.parquet").to_pandas()
        tr["ingest_lib"] = "enveda-np-examples"
        tr.to_parquet(Path(d) / "comp" / "train.parquet")
        TD.fake_r1_pool(d)
        make_ckpts(d, fp_kind)
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
    assert rep["fp_channel"] == fp_kind and rep["n_fp_z"] == 3 and rep["n_fp_mol"] == 3
    assert rep["fp_bank"]["kind"] == fp_kind and "fp_z_s" in rep["times"] and "[fp] featuriser parity" in r.stdout
    return rep


def test_dryrun_r3_own():
    _dry("own", "r3own")


def test_dryrun_r3_ahmed():
    _dry("ahmed", "r3ahmed")


TESTS = [test_avg_rank_desc, test_fp_none_is_r2_order, test_rrf_tier2_only_and_ties, test_merge_peaks_variants,
         test_adduct_lists, test_featurisers, test_banks, test_molecule_spectra_and_ce, test_fp_audit_toy,
         test_dryrun_r3_own, test_dryrun_r3_ahmed]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print("test_r3: OK")
