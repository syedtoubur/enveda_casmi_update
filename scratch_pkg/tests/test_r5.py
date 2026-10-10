"""R5 analogue channel: PSSM weights, Tanimoto kernel, neutral RRF term, train exclusion, notebook dry run."""
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
from casmi_scratch import analog as A  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402
from casmi_scratch.pipeline import analog_scores  # noqa: E402
from casmi_scratch.pool import F_COCO, F_TRAIN, CandidatePool  # noqa: E402
from casmi_scratch.rank import analog_rrf, rank_two_tier  # noqa: E402

SMI = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
       "CC(=O)Oc1ccccc1C(=O)O", "OC1=CC=CC=C1"]


def test_pssm_weights():
    X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0]], np.uint8)
    q = np.array([0.2, 0.3, 0.001, 0.2])
    W = A.pssm_weights(X, np.ones(3), q, alpha=1.0, clip=3.0)
    assert W[0] > 0                                   # conserved, enriched -> reward
    assert -3.0 <= W[1] < 0                           # common bit no analogue carries -> bounded penalty
    assert W[2] == np.float32(3.0)                    # rare conserved bit -> capped at the clip
    assert abs(W[3]) < W[0]                           # variable bit near background -> muted
    Wlow = A.pssm_weights(X, np.full(3, 0.5), q)      # weak analogues -> pulled to the background prior
    assert np.all(np.abs(Wlow) <= np.abs(W) + 1e-6)


def test_analog_set():
    hits = {"a": 0.9, "b": 0.4, "c": 0.7, "d": 0.7}
    assert A.analog_set(hits, 0.5, 10) == [("a", 0.9), ("c", 0.7), ("d", 0.7)]
    assert A.analog_set(hits, 0.5, 2) == [("a", 0.9), ("c", 0.7)]


def test_analog_rrf_neutral():
    s = np.array([3.0, np.nan, 1.0, 2.0, np.nan])
    t = analog_rrf(s, 3.0)
    assert np.allclose(t[[0, 3, 2]], [1 / 4, 1 / 5, 1 / 6])
    assert np.allclose(t[[1, 4]], np.mean([1 / 4, 1 / 5, 1 / 6]))
    assert not analog_rrf(np.full(3, np.nan), 3.0).any()


def test_rank_an_none_and_tier1():
    rng = np.random.default_rng(1)
    lv, sv, fr, ppm, fp = (rng.random(30) for _ in range(5))
    lv[:4] = 0.9
    base = rank_two_tier(lv, sv, fr, ppm, 0.75, None, fp)
    assert np.array_equal(base, rank_two_tier(lv, sv, fr, ppm, 0.75, None, fp, an=None))
    an = rng.random(30)
    o = rank_two_tier(lv, sv, fr, ppm, 0.75, None, fp, an=an)
    assert np.array_equal(o[:4], base[:4])            # tier 1 untouched
    assert not np.array_equal(o, base)
    # an alone (no FP channel) also enters the tier-2 RRF; all-NaN an = the sv order
    nan = rank_two_tier(lv, sv, fr, ppm, 0.75, an=np.full(30, np.nan))
    assert np.array_equal(nan, rank_two_tier(lv, sv, fr, ppm, 0.75))


def _bank(q=None, mode="pssm", **kw):
    cfg = Config().with_(analog_channel=mode, **kw)
    smiles_of = {f"K{i}": s for i, s in enumerate(SMI)}
    b = A.AnalogBank(smiles_of, q, cfg)
    if q is None and mode == "pssm":
        b.q = np.full(b.feat.raw_bits, 0.05, np.float32)
    return b, cfg


def test_profile_min_n_and_modes():
    b, cfg = _bank()
    assert b.profile({"K0": 0.9, "K1": 0.8}) is None                      # 2 < analog_min_n
    assert b.profile({"K0": 0.9, "K1": 0.8, "K2": 0.3}) is None           # one below s_min
    p = b.profile({"K0": 0.9, "K1": 0.8, "K2": 0.7, "KX": 0.99})          # KX has no structure
    assert p["mode"] == "pssm" and p["n"] == 3 and p["W"].shape == (b.feat.raw_bits,)
    s = b.score(p, ["K0", "K3", "bad"], [SMI[0], SMI[3], "not a smiles"])
    assert s[0] > s[1] and np.isnan(s[2])                                 # xanthine family beats aspirin
    t = b.profile({"K0": 0.9, "K1": 0.8, "K2": 0.7}, mode="tani")
    assert t["mode"] == "tani" and t["X"].shape == (3, A.MORGAN_R2)


def test_tani_matches_rdkit():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as g
    b, _ = _bank(mode="tani")
    gen = g.GetMorganGenerator(radius=2, fpSize=4096)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMI]
    S = np.array([0.9, 0.8, 0.6])
    p = b.profile({"K0": 0.9, "K1": 0.8, "K2": 0.6}, mode="tani")
    got = b.score(p, ["K3", "K4"], SMI[3:])
    for j, c in enumerate((3, 4)):
        T = np.array(DataStructs.BulkTanimotoSimilarity(fps[c], fps[:3]))
        assert abs(got[j] - (T @ S ** 4) / (S ** 4).sum()) < 1e-5


def test_train_exclusion_only_in_analog_list():
    b, cfg = _bank()
    pool = CandidatePool(np.array([194.0, 194.1, 194.2]), ["P0", "P1", "P2"], [SMI[0], SMI[1], SMI[3]],
                         flags=np.array([F_TRAIN, F_COCO, F_TRAIN | F_COCO]))
    plan = dict(an_prof=b.profile({"K0": 0.9, "K1": 0.8, "K2": 0.7}), an_free=set())
    s = analog_scores(plan, np.arange(3), pool, cfg, b)
    assert np.isnan(s[0]) and np.isfinite(s[1]) and np.isnan(s[2])      # any train flag -> not scored
    plan["an_free"] = {"P0"}                                              # validation: the held-out truth
    s = analog_scores(plan, np.arange(3), pool, cfg, b)
    assert np.isfinite(s[0]) and np.isnan(s[2])
    assert analog_scores(dict(an_prof=None), np.arange(3), pool, cfg, b) is None
    assert analog_scores(plan, np.arange(3), pool, cfg.with_(analog_channel=""), b) is None
    plan["an_free"] = set()                                               # R5b: exclusion off -> train scored too
    assert np.isfinite(analog_scores(plan, np.arange(3), pool, cfg.with_(analog_exclude_train=False), b)).all()


def test_estimate_q():
    b, _ = _bank()
    pool = CandidatePool(np.arange(4.0), list("abcd"), [SMI[0], SMI[0], SMI[3], "bad"])
    q = A.estimate_q(pool, b.feat, n=10, mask=np.array([True, True, False, True]))
    assert q.shape == (b.feat.raw_bits,) and set(np.unique(q)) <= {0.0, 1.0}   # two caffeines, one unparsable
    assert np.array_equal(q, b.feat.raw(SMI[0]).astype(np.float32))


def test_audit_rows():
    """analog_discrimination on a toy library: three analogues 14 Da up share the query's peaks; the truth is a train
    structure counted as non-train, so the analogue arms score it while the other train decoy stays neutral."""
    from casmi_scratch import audit
    from casmi_scratch.library import build_library
    mz = np.array([91.054, 105.07, 119.05, 150.0], np.float64)
    it = np.array([0.4, 1.0, 0.6, 0.3], np.float64)
    prec_q = 195.0877                                                      # caffeine [M+H]+
    precs = [prec_q, prec_q + 14.0157, prec_q + 14.0157, prec_q + 14.0157]
    keys = ["TRUTH", "K0", "K1", "K2"]
    lib = build_library(np.arange(0, 4 * len(precs) + 1, 4), np.tile(mz, len(precs)), np.tile(it, len(precs)),
                        np.array(precs), np.array(["[M+H]+"] * len(precs), object), np.array(keys, object),
                        struct_mass=False)
    sub = pd.DataFrame(dict(precursor_mz=[prec_q], adduct=["[M+H]+"], ms2_mzs=[mz], ms2_normalized_intensities=[it],
                            ionization_mode=["positive"], collision_energy_ev=[None]))
    tm = 194.0804
    pool = CandidatePool(tm * (1 + np.array([-1, 0, 1, 2]) * 1e-6), ["DECOY_T", "TRUTH", "D1", "D2"],
                         [SMI[1], SMI[0], SMI[3], SMI[4]], flags=np.array([F_TRAIN, F_TRAIN, F_COCO, F_COCO]))
    b, cfg = _bank()
    b.smiles_of = {"K0": SMI[0], "K1": SMI[1], "K2": SMI[2], "TRUTH": SMI[0]}
    frag = lambda smis, specs, mode, c: np.zeros(len(smis), np.float32)  # noqa: E731
    rows = audit.analog_discrimination([("TRUTH", sub)], lib, pool, lambda k: k, None, b,
                                       cfg.with_(use_sim=False, frag_workers=1), frag_fn=frag, log=lambda s: None)
    r = rows[0]
    assert r["truth_in_scored"] and r["n_pssm"] == 3 and r["n_tani"] == 3 and r["pssm_scored"] == 3, r
    assert r["tani_rank"] == 1 and r["pssm_rank"] == 1, r                 # truth = caffeine, analogues = xanthines
    assert "fp0_pssm_v0" in r and "fp1_tani_v0" in r and "fp0_none_v1" not in r
    s = audit.summarize_analog(rows)
    assert s["base"] == "fp1_none_v0" and s["n_in_range"] == 1 and s["pssm_coverage"] == 1.0
    assert s["mrr"]["fp0_tani_v0"] >= s["mrr"]["fp0_none_v0"]


def test_dryrun_r5():
    """R5 PSSM notebook on the fake R1 pool: library spectra are the test spectra under 3 structures, so every
    molecule with >= 3 spectra finds >= 3 analogues; the channel builds profiles and reports its counters."""
    test_path = ROOT / "test.parquet"
    if not test_path.exists():
        print("test_dryrun_r5: SKIPPED (no local test.parquet)")
        return
    import mk_notebook as mk
    from test_dryrun import fake_input, fake_r1_pool
    te = pq.read_table(test_path).to_pandas()
    te = te[te.molecule_id.isin(te.molecule_id.unique()[:3])].reset_index(drop=True)
    for mode, excl in (("pssm", True), ("tani", True), ("pssm", False)):     # R5b: exclusion off (q over all rows)
        code = mk.render("r0_kaggle", {"use_sim": False, "frag_workers": 1, "pool_source": "r1",
                                       "pool_drop_e180": True, "struct_mass": False, "analog_channel": mode,
                                       "analog_q_n": 50, "analog_exclude_train": excl}, tag=f"dry-r5-{mode}-{excl}")
        with tempfile.TemporaryDirectory() as d:
            fake_input(d, te)
            tr_path = Path(d) / "comp" / "train.parquet"
            tr = pd.read_parquet(tr_path)
            i = np.arange(len(tr)) % 3
            tr["normalized_smiles"] = np.array(SMI[:3], object)[i]
            tr["inchikey14"] = np.array(["RYYVLZVUVIJVGH", "ZFXYFBGIUFBOJW", "YAPQBXQYLJRXSA"], object)[i]
            tr.to_parquet(tr_path)
            fake_r1_pool(d)
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
        # a molecule with < 3 spectra sees < 3 structures and is dropped (analog_min_n)
        assert rep["analog_channel"] == mode and rep["n_analog_mol"] >= 1, rep
        assert rep["n_analog_mol"] + rep["n_analog_dropped"] == 3 and rep["n_analog_ok"] == rep["n_analog_mol"], rep
        assert rep["analog_n_median"] == 3 and "analog_s" in rep["times"] and f"[analog] {mode}" in r.stdout
        assert (rep["analog_info"]["q_source"] == "sample") if mode == "pssm" else rep["analog_info"]["q_source"] is None
        assert rep["analog_info"]["exclude_train"] is excl, rep["analog_info"]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for f in TESTS:
        f()
    print(f"test_r5: {len(TESTS)} OK")
