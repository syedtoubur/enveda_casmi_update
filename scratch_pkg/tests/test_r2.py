"""R2 class vote: NPClassifier fingerprint / decoding / pool_npc I/O, the tier-2 bonus and its pipeline wiring.

Run: python scratch_pkg/tests/run.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from casmi_scratch import analog, npc, pipeline  # noqa: E402
from casmi_scratch import pool_build as PB  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402
from casmi_scratch.pool import CandidatePool, load_pool, npc_top1  # noqa: E402
from casmi_scratch.rank import rank_two_tier  # noqa: E402

CFG = Config()
NAMES = ["Flavonoids", "Steroids", "Triterpenoids"]


def test_bonus_none_is_r1_order():
    rng = np.random.default_rng(0)
    lv, sv, fr, ppm = rng.random(50) * 0.9, rng.random(50), rng.random(50), rng.random(50) * 8
    a = rank_two_tier(lv, sv, fr, ppm, 0.75)
    assert np.array_equal(a, rank_two_tier(lv, sv, fr, ppm, 0.75, None))
    assert np.array_equal(a, rank_two_tier(lv, sv, fr, ppm, 0.75, np.zeros(50)))


def test_bonus_reorders_tier2_only():
    lv = np.array([0.9, 0.8, 0.1, 0.1, 0.1])
    sv = np.array([0.1, 0.5, 0.50, 0.45, 0.2])
    fr, ppm = np.zeros(5), np.zeros(5)
    assert list(rank_two_tier(lv, sv, fr, ppm, 0.75)) == [0, 1, 2, 3, 4]
    bonus = np.array([0.0, 1.0, 0.0, 0.08, 0.0])               # tier-1 bonus ignored; 3 passes 2 (0.53 > 0.50)
    assert list(rank_two_tier(lv, sv, fr, ppm, 0.75, bonus)) == [0, 1, 3, 2, 4]
    bonus = np.array([0.0, 0.0, 0.0, 0.0, 0.08])               # 0.28 < 0.45: a clear sv gap holds
    assert list(rank_two_tier(lv, sv, fr, ppm, 0.75, bonus)) == [0, 1, 2, 3, 4]


def test_vote_conf_and_soft_bonus():
    assert analog.vote_conf({"A": 3.0, "B": 1.0}, "A") == 0.75
    assert analog.vote_conf({}, None) == 0.0
    sc = np.array([[0, 1, 2], [1, 0, -1], [2, 1, -1], [-1, -1, -1]], np.int16)
    sp = np.array([[0.9, 0.05, 0.01], [0.6, 0.3, 0.0], [0.7, 0.2, 0.0], [0, 0, 0]], np.float32)
    b = analog.vote_bonus(sc, sp, NAMES, "Flavonoids", 0.5, 0.10)
    assert np.allclose(b, [0.045, 0.015, 0.0, 0.0])
    assert not analog.vote_bonus(sc, sp, NAMES, None, 0.5, 0.10).any()
    assert not analog.vote_bonus(sc, sp, NAMES, "Unknown", 0.5, 0.10).any()


def test_npc_fingerprint_shape_and_counts():
    f, b = npc.fingerprint("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert f.shape == (2048,) and b.shape == (4096,)
    assert f.sum() == 24                       # caffeine with explicit H: 24 atoms, one radius-0 env each
    assert npc.fingerprint("not a smiles") is None
    F, B, ok = npc.fingerprints(["CCO", "xx"])
    assert list(ok) == [True, False] and F[1].sum() == 0 and F[0].sum() == 9


def test_npc_rdkit_bits_guard():
    """NP-Classifier V1 needs the Morgan bits of RDKit <= 2021.03 (R2.0 FAIL): a newer RDKit must not feed it."""
    assert npc.rdkit_bits_ok("2020.09.5") and npc.rdkit_bits_ok("2021.03.5")
    assert not npc.rdkit_bits_ok("2021.09.5") and not npc.rdkit_bits_ok("2026.03.3")
    names = dict(pathway=["P0", "P1"], superclass=["S0", "S1", "S2"], **{"class": ["C0", "C1"]})
    models = {h: _FakeModel(len(v)) for h, v in names.items()}
    if not npc.rdkit_bits_ok():
        try:
            npc.predict(["CCO"], models, names)
            raise AssertionError("predict ran on post-2021.09 Morgan bits without an explicit fp_fn")
        except RuntimeError:
            pass
    assert len(npc.predict(["CCO"], models, names, npc.fingerprints)) == 1


def test_npc_fp_worker_matches_in_process():
    """The side-env worker computes the same arrays as npc.fingerprints on the same RDKit, keeps row order through its
    process pool (> 1000 rows), flags unparsable rows, and refuses a new RDKit unless told otherwise."""
    base = ["CCO", "xx", "c1ccccc1O", "", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CCCCCCCCCCCCCCCC(=O)O", "OCC1OC(O)C(O)C(O)C1O"]
    smis = (base * 150)[:1003]
    cmd = [sys.executable, "-I", str(npc.FP_WORKER), "--any-rdkit"]
    F, B, ok, meta = npc.fp_subprocess(smis, cmd=cmd, workers=2)
    F0, B0, ok0 = npc.fingerprints(smis)
    assert np.array_equal(ok, ok0) and np.array_equal(F, F0) and np.array_equal(B, B0)
    assert meta["parity"] == -1 and not ok[1] and not ok[3]
    if not npc.rdkit_bits_ok():
        try:
            npc.fp_subprocess(base, cmd=[sys.executable, "-I", str(npc.FP_WORKER)], workers=1)
            raise AssertionError("worker accepted post-2021.09 RDKit")
        except RuntimeError as e:
            assert "need <= 2021.03" in str(e)


def test_npc_names_and_decode():
    idx = {"Pathway": {"P0": 0, "P1": 1}, "Superclass": {"S0": 0, "S1": 1, "S2": 2, "S1dup": 1},
           "Class": {"C0": 0, "C1": 1}}
    names = npc.label_names(idx)
    assert names["superclass"] == ["S0", "S1", "S2"] and names["pathway"] == ["P0", "P1"]
    preds = dict(superclass=np.array([[0.1, 0.8, 0.3], [0.5, 0.1, 0.2]]), pathway=np.array([[0.2, 0.9], [0.6, 0.1]]),
                 **{"class": np.array([[0.4, 0.1], [0.1, 0.2]])})
    df = npc.decode(preds, names, np.array([True, False]))
    assert list(df.loc[0, ["sc1", "sc2", "sc3"]]) == ["S1", "S2", "S0"]
    assert np.isclose(df.loc[0, "sp1"], 0.8) and df.loc[0, "pathway"] == "P1" and df.loc[0, "class"] == "C0"
    assert pd.isna(df.loc[1, "sc1"]) and pd.isna(df.loc[1, "pathway"]) and np.isnan(df.loc[1, "sp1"])


def _toy_pool_dir(d):
    pool = pd.DataFrame({"key": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"], "smiles": ["CCO", "CCN", "CCC"],
                         "mass": [100.0, 100.0005, 100.001], "flags": [1, 2, 3]})
    side = pd.DataFrame({"np_superclass": None}, index=pd.Index(pool.key, name="key"))
    km = pd.DataFrame({"inchikey14": ["A"], "scorer_key": ["AAAAAAAAAAAAAA"]})
    PB.write_pool(pool, side, km, d, dict(version="t"))
    lab = pd.DataFrame({"sc1": ["Steroids", "Flavonoids", None], "sc2": ["Flavonoids", "Steroids", None],
                        "sc3": ["Triterpenoids", None, None], "sp1": [0.7, 0.9, np.nan], "sp2": [0.2, 0.05, np.nan],
                        "sp3": [0.05, np.nan, np.nan], "pathway": ["Terpenoids", "Shikimates", None],
                        "pathway_p": [0.9, 0.9, np.nan], "class": ["x", "y", None], "class_p": [0.5, 0.5, np.nan]})
    npc.write_npc(d, lab)


def test_pool_npc_io_alignment():
    with tempfile.TemporaryDirectory() as d:
        _toy_pool_dir(d)
        sc, sp, names = npc.load_npc(d)
        assert names == ["Flavonoids", "Steroids", "Triterpenoids"] and sc.shape == (3, 3)
        assert list(sc[0]) == [1, 0, 2] and list(sc[2]) == [-1, -1, -1] and sp[1, 2] == 0.0
        pool, _ = load_pool(d, with_npc=True)
        assert pool.npc is not None and list(pool.npc[0][:, 0]) == [1, 0, -1]
        sub = pool.without(["BBBBBBBBBBBBBB"])
        assert list(sub.keys) == ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC"] and list(sub.npc[0][:, 0]) == [1, -1]
        assert npc_top1(d, ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC", "ZZ"]) == {"AAAAAAAAAAAAAA": "Steroids"}
        assert load_pool(d)[0].npc is None


def test_pool_npc_survives_mass_sort():
    sc = np.array([[0, -1, -1], [1, -1, -1]], np.int16)
    sp = np.array([[0.9, 0, 0], [0.8, 0, 0]], np.float32)
    p = CandidatePool([200.0, 100.0], ["HEAVY", "LIGHT"], ["a", "b"], None, (sc, sp, NAMES))
    assert list(p.keys) == ["LIGHT", "HEAVY"] and list(p.npc[0][:, 0]) == [1, 0]


def test_class_bonus_and_counters():
    sc = np.array([[0, -1, -1], [1, -1, -1], [0, 1, -1]], np.int16)
    sp = np.array([[0.9, 0, 0], [0.8, 0, 0], [0.5, 0.4, 0]], np.float32)
    pool = CandidatePool([1.0, 2.0, 3.0], ["A" * 14, "B" * 14, "C" * 14], ["a", "b", "c"], None, (sc, sp, NAMES))
    cfg = CFG.with_(class_vote=True)
    plan = dict(vote=("Steroids", 1.0))
    assert np.allclose(pipeline.class_bonus(plan, np.arange(3), pool, cfg), [0.0, 0.08, 0.04])
    assert pipeline.class_bonus(plan, np.arange(3), pool, CFG) is None                  # vote off: R1 order
    c = pipeline.vote_counters({1: dict(voted=True, reordered=True, vote_first=False),
                                2: dict(voted=False, reordered=False, vote_first=False), 3: dict(top_is_tier1=True)})
    assert c == dict(n_vote_mol=1, n_bonus_reorders=1, n_vote_first=0)


def test_run_requires_labels():
    try:
        pipeline.run([], None, CandidatePool([], [], []), CFG.with_(class_vote=True))
    except ValueError:
        return
    raise AssertionError("class_vote without labels must fail loudly")


class _FakeModel:
    """Two inputs given in 4096 / 2048 order (model_inputs must match on width); output = fixed logits."""
    def __init__(self, n_out):
        self.inputs = [np.zeros((1, 4096)), np.zeros((1, 2048))]
        self.n = n_out

    def predict(self, xs, batch_size=None, verbose=0):
        assert xs[0].shape[1] == 4096 and xs[1].shape[1] == 2048
        P = np.zeros((len(xs[0]), self.n), np.float32)
        P[:, 0] = xs[1].sum(axis=1) / 100.0          # radius-0 counts (atoms incl. H) drive output 0
        P[:, -1] = 0.5
        return P


def test_label_all_chunked_resumes():
    names = dict(pathway=["P0", "P1"], superclass=["S0", "S1", "S2"], **{"class": ["C0", "C1"]})
    models = {h: _FakeModel(len(v)) for h, v in names.items()}
    smis = ["CCO", "c1ccccc1", "bad", "CCCCCCCCCCCCCCCCCC(=O)O"]
    with tempfile.TemporaryDirectory() as d:
        a = npc.label_all(smis, models, names, d, fp_fn=npc.fingerprints, chunk=3)
        b = npc.label_all(smis, {}, names, d, fp_fn=npc.fingerprints, chunk=3)            # resumes from checkpoints: models unused
    assert a.equals(b) and len(a) == 4 and pd.isna(a.sc1[2])
    assert list(a.sc1[[0, 1]]) == ["S2", "S2"] and a.sc1[3] == "S0"  # 9 / 12 atoms < 0.5 ; stearic acid 56 > 0.5


def test_discrimination_verdict():
    from casmi_scratch.audit import summarize_discrimination
    r = lambda v, tp, mp, r0, r1, ch: dict(voted=v, truth_in_scored=True, truth_p=tp, window_median_p=mp,  # noqa
                                           rank0=r0, rank1=r1, changed=ch, fr_gap=0.1)
    rows = [r("A", 0.9, 0.1, 5, 2, True), r("A", 0.8, 0.2, 3, 1, True), r("A", 0.1, 0.5, 2, 4, True),
            r(None, None, None, 7, 7, False)]
    out = summarize_discrimination(rows)
    assert out["n_voted_in_range"] == 3 and out["up"] == 2 and out["down"] == 1
    assert np.isclose(out["truth_above_median"], 2 / 3) and out["pass"]
    assert np.isclose(out["mrr_after"], (1 / 2 + 1 + 1 / 4 + 1 / 7) / 4) and out["reach"] == 0.75
    rows[1]["truth_p"] = 0.2                                        # ties do not count as discrimination
    assert not summarize_discrimination(rows)["pass"]


def test_audit_discrimination_end_to_end():
    """Toy slice: the hexose analog votes Flavonoids; the truth (+3 ppm, the worst |ppm|) is the only Flavonoid."""
    from test_r1 import toy_analog_lib
    from casmi_scratch import audit, chem
    target = 600.0
    lib, hexose = toy_analog_lib(target)
    spec = dict(adduct="[M+H]+", precursor_mz=chem.precursor_of(target, "[M+H]+") * (1 + 1.45e-6),
                ionization_mode="positive", collision_energy_ev=np.array([20.0]),
                ms2_mzs=np.array([100.0, 200.0, 300.0 + hexose]), ms2_normalized_intensities=np.array([0.2, 0.3, 0.5]))
    mols = [("TRUTHAAAAAAAAA", pd.DataFrame([spec]))]
    sc = np.array([[1, -1, -1], [1, -1, -1], [0, -1, -1]], np.int16)       # names: Flavonoids=0, Steroids=1
    sp = np.array([[0.9, 0, 0], [0.9, 0, 0], [0.9, 0, 0]], np.float32)
    pool = CandidatePool([target * (1 + p * 1e-6) for p in (-2, 0, 3)],
                         ["DECOYAAAAAAAAA", "DECOYBBBBBBBBB", "TRUTHAAAAAAAAA"], ["a", "b", "t"], None, (sc, sp, NAMES))
    no_frag = lambda smis, specs, mode, cfg: np.zeros(len(smis), np.float32)  # noqa: E731
    rows = audit.discrimination(mols, lib, pool, lambda ik: ik, {"HEXHITAAAAAAAA": "Flavonoids"}, CFG,
                                frag_fn=no_frag, log=lambda s: None)
    r = rows[0]
    assert r["voted"] == "Flavonoids" and r["conf"] == 1.0 and r["rank0"] == 3 and r["rank1"] == 1 and r["changed"]
    assert np.isclose(r["truth_p"], 0.9) and r["window_median_p"] == 0.0
    out = audit.summarize_discrimination(rows)
    assert out["pass"] and out["mrr_before"] < out["mrr_after"] == 1.0


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    t0 = time.time()
    for f in TESTS:
        f()
    print(f"test_r2: {len(TESTS)} OK ({time.time() - t0:.1f}s)")
