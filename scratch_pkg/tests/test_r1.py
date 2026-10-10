"""R1 unified pool, NP labels, analog search and class vote (specs/specs_scratch/r1_pool_plan.md).

Run: python scratch_pkg/tests/run.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi_scratch import analog, audit, chem, pool_build as PB  # noqa: E402
from casmi_scratch.config import Config  # noqa: E402
from casmi_scratch.kernels import entropy_sim  # noqa: E402
from casmi_scratch.library import build_library, remap_keys  # noqa: E402
from casmi_scratch.np_labels import assign_labels  # noqa: E402
from casmi_scratch.pool import (F_BIO, F_COCO, F_E180, F_NPT, F_TRAIN, F_TRAIN_OTHER, F_V2, CandidatePool,  # noqa: E402
                                e180_only, load_keymap, load_np_labels, load_pool)

CFG = Config()


def toy_rows():
    """Five sources; keys are given by a fake props table (no RDKit)."""
    return pd.concat([
        PB._rows(["s_tr_a", "s_tr_b"], ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"], [F_TRAIN | F_E180, F_TRAIN | F_TRAIN_OTHER],
                 "train"),
        PB._rows(["s_v2_a", "s_v2_c"], ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC"], F_V2, "v2"),
        PB._rows(["s_co_c", "s_co_bad"], ["CCCCCCCCCCCCCC", "ZZZZZZZZZZZZZZ"], F_COCO, "coco"),
        PB._rows(["s_bio_d"], ["DDDDDDDDDDDDDD"], F_BIO, "bio"),
        PB._rows(["s_np_d", "s_np_enol"], ["DDDDDDDDDDDDDD", "XXXXXXXXXXXXXX"], F_NPT, "np_table"),
    ], ignore_index=True)


def toy_props():
    rows = {"s_tr_a": ("AAAAAAAAAAAAAA", 300.0), "s_v2_a": ("AAAAAAAAAAAAAA", 300.1),
            "s_tr_b": ("BBBBBBBBBBBBBB", 200.0), "s_v2_c": ("CCCCCCCCCCCCCC", 250.0),
            "s_co_c": ("CCCCCCCCCCCCCC", 250.2), "s_co_bad": (None, np.nan),
            "s_bio_d": ("DDDDDDDDDDDDDD", 400.0), "s_np_d": ("DDDDDDDDDDDDDD", 400.5),
            "s_np_enol": ("CCCCCCCCCCCCCC", 250.4)}     # a tautomer of C: its source key differs from the scorer key
    return pd.DataFrame([(s, k, m, 1, False) for s, (k, m) in rows.items()],
                        columns=["smiles", "key", "mass", "nfrag", "charged"]).set_index("smiles")


def test_merge_priority_flags_hygiene():
    pool, st = PB.merge_rows(toy_rows(), toy_props())
    assert list(pool.key) == ["BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD"]   # mass-sorted
    assert np.all(np.diff(pool.mass) >= 0) and not pool.key.duplicated().any()
    p = pool.set_index("key")
    assert p.loc["AAAAAAAAAAAAAA", "smiles"] == "s_tr_a" and p.loc["AAAAAAAAAAAAAA", "mass"] == 300.0  # train wins
    assert p.loc["CCCCCCCCCCCCCC", "smiles"] == "s_v2_c"                                             # v2 > coco
    assert p.loc["DDDDDDDDDDDDDD", "smiles"] == "s_bio_d"                                            # bio > NP table
    assert p.loc["AAAAAAAAAAAAAA", "flags"] == F_TRAIN | F_E180 | F_V2
    assert p.loc["CCCCCCCCCCCCCC", "flags"] == F_V2 | F_COCO | F_NPT                                 # tautomer merged
    assert st["dropped_unkeyable"] == {"coco": 1} and st["key_disagree"].get("np_table") == 1
    assert st["e180_only"] == 1 and st["pool_size"] == 4


def test_e180_only():
    f = np.array([F_TRAIN | F_E180, F_TRAIN | F_E180 | F_V2, F_TRAIN | F_E180 | F_TRAIN_OTHER, F_E180 | F_COCO,
                  F_E180 | F_NPT, F_V2, F_TRAIN | F_TRAIN_OTHER])
    assert list(e180_only(f)) == [True, True, False, False, False, False, False]


def test_superset_check():
    props = toy_props()
    props.loc["s_r0_unkeyable"] = (None, 123.0, 1, False)          # parses (has a mass) but no key: lost
    r = PB.superset_check(["s_tr_a", "s_co_c", "s_co_bad", "s_r0_unkeyable"], props,
                          ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC"])
    assert r["lost_unkeyable"] == 1 and r["missing_keys"] == 0 and r["missing"] == 1 and r["r0_parsed"] == 3
    r = PB.superset_check(["s_tr_a", "s_co_c"], props, ["AAAAAAAAAAAAAA"])
    assert r["missing"] == 1 and r["examples"] == ["CCCCCCCCCCCCCC"]


def test_scorer_key_tautomers_rdkit():
    a, b = PB.smiles_props("CC(C)=O"), PB.smiles_props("C=C(C)O")
    assert a[0] == b[0] == "CSCPPACGZOOCGX" and abs(a[1] - 58.0418648) < 1e-6
    salt = PB.smiles_props("[Na+].CC(=O)[O-]")
    assert salt[2] == 2 and salt[3] is True                          # kept verbatim, counted
    assert PB.smiles_props("not a smiles")[0] is None and PB.smiles_props(None)[0] is None


def test_run_chunked_resumes():
    calls = []

    def fn(items):
        calls.append(len(items))
        return [(x * 2,) for x in items]
    with tempfile.TemporaryDirectory() as d:
        out = PB.run_chunked(list(range(25)), fn, d, "t", ["v"], chunk=10, workers=1, log=lambda s: None)
        assert list(out.v) == [x * 2 for x in range(25)] and calls == [10, 10, 5]
        out2 = PB.run_chunked(list(range(25)), fn, d, "t", ["v"], chunk=10, workers=1, log=lambda s: None)
        assert calls == [10, 10, 5] and out2.equals(out)              # resumed from the checkpoints
        PB.run_chunked(list(range(26)), fn, d, "t", ["v"], chunk=10, workers=1, log=lambda s: None)
        assert calls[3:] == [10, 10, 6]                               # a changed input never reuses stale chunks


def test_compute_props_injected():
    with tempfile.TemporaryDirectory() as d:
        p = PB.compute_props(["b", "a", "b", None], d, chunk=1, workers=1,
                             props_fn=lambda s: [(x.upper() * 14, 1.0, 1, False) for x in s], log=lambda s: None)
    assert list(p.index) == ["a", "b"] and p.loc["b", "key"] == "B" * 14


def toy_np_table():
    return pd.DataFrame({"np_pathway": ["Terpenoids", None, "Shikimates"], "np_superclass": ["Sesquiterpenoids", None,
                         "Flavonoids"], "np_class": ["x", None, "y"], "kingdoms": ["Plantae", None, None],
                         "n_organisms": [3, 1, 2], "smiles": ["t1", "t2", "t3"]},
                        index=pd.Index(["DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE", "FFFFFFFFFFFFFF"], name="key"))


def test_assign_labels():
    keys = ["DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE", "GGGGGGGGGGGGGG", "HHHHHHHHHHHHHH"]

    def nearest(q, bank):
        assert bank == ["t1", "t3"] and q == ["q_e", "q_g", "q_h"]    # unlabelled table rows also get a kNN try
        return pd.DataFrame({"bank_j": [0, 1, -1], "sim": [0.5, 0.9, 0.0]})
    lab = assign_labels(keys, ["q_d", "q_e", "q_g", "q_h"], toy_np_table(), nearest)
    assert list(lab.label_source) == [1, 0, 2, 0]
    assert lab.loc["DDDDDDDDDDDDDD", "np_superclass"] == "Sesquiterpenoids"
    assert lab.loc["GGGGGGGGGGGGGG", "np_superclass"] == "Flavonoids" and abs(lab.loc["GGGGGGGGGGGGGG", "knn_sim"] -
                                                                            0.9) < 1e-6
    assert lab.loc["EEEEEEEEEEEEEE", "np_superclass"] is None or pd.isna(lab.loc["EEEEEEEEEEEEEE", "np_superclass"])
    assert list(assign_labels(keys, ["a"] * 4, toy_np_table()).label_source) == [1, 0, 0, 0]


def test_write_load_roundtrip():
    pool, st = PB.merge_rows(toy_rows(), toy_props())
    km = PB.keymap_from_rows(toy_rows(), toy_props())
    side = assign_labels(pool.key, pool.smiles, toy_np_table())
    side["np_score"] = np.float16(0.5)
    with tempfile.TemporaryDirectory() as d:
        man = PB.write_pool(pool, side, km, d, dict(version="test", stats=st))
        assert set(man["files"]) == {"pool_mass.npy", "pool_key.npy", "pool_flags.npy", "pool_smiles.parquet",
                                     "pool_np.parquet", "train_keymap.parquet"}
        p, info = load_pool(d)
        assert len(p) == 4 and list(p.keys) == list(pool.key) and info["pool_version"] == "test"
        assert isinstance(p.keys[0], str) and p.smiles[0] == "s_tr_b"
        pb, info_b = load_pool(d, drop_e180=True)
        assert len(pb) == 3 and info_b["dropped_e180"] == 1 and "AAAAAAAAAAAAAA" not in set(pb.keys)
        assert load_keymap(d) == {"AAAAAAAAAAAAAA": "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB": "BBBBBBBBBBBBBB"}
        lab = load_np_labels(d)
        assert lab.loc["DDDDDDDDDDDDDD", "np_superclass"] == "Sesquiterpenoids" and len(lab) == 4


def test_remap_keys():
    lib = build_library(np.array([0, 1, 2]), np.array([100.0, 100.0]), np.array([1.0, 1.0]),
                        prec=np.array([301.0, 401.0]), adduct=np.array(["[M+H]+", "[M-H]-"], dtype=object),
                        ik=np.array(["PLAINKEYAAAAAA", "OTHERKEYBBBBBB"], dtype=object), struct_mass=False)
    assert list(lib.pol) == [1, -1]
    lib2 = remap_keys(lib, {"PLAINKEYAAAAAA": "SCORERKEYAAAAA"})
    assert list(lib2.ik) == ["SCORERKEYAAAAA", "OTHERKEYBBBBBB"] and lib2.mz is lib.mz and list(lib.ik)[0] == \
        "PLAINKEYAAAAAA"


def _f32(*x):
    return np.array(x, np.float32)


def test_shift_entropy_kernel():
    lib_m, lib_p = _f32(100.0, 200.0, 300.0), _f32(0.2, 0.3, 0.5)
    hexose = analog.NP_DELTAS["C6H10O5 hexose"]
    q_m, q_p = _f32(100.0, 200.0, 300.0 + hexose), _f32(0.2, 0.3, 0.5)
    direct = float(entropy_sim(q_m, q_p, lib_m, lib_p, 0.01))
    shifted = float(analog.shift_entropy_sim(q_m, q_p, lib_m, lib_p, hexose, 0.01))
    assert shifted > 0.999 and direct < 0.6, (shifted, direct)
    same = float(analog.shift_entropy_sim(lib_m, lib_p, lib_m, lib_p, 0.0, 0.01))
    assert abs(same - 1.0) < 1e-6
    a_m, a_p = _f32(100.0, 150.0, 210.0), _f32(0.5, 0.3, 0.2)
    b_m, b_p = _f32(100.0, 151.0, 210.0), _f32(0.4, 0.4, 0.2)
    assert abs(float(analog.shift_entropy_sim(a_m, a_p, b_m, b_p, 0.0, 0.01)) -
               float(entropy_sim(a_m, a_p, b_m, b_p, 0.01))) < 1e-5     # no shift == plain entropy
    assert float(analog.shift_entropy_sim(_f32(50.0), _f32(1.0), _f32(70.0), _f32(1.0), 5.0, 0.01)) < 1e-6
    # one-to-one: a library peak used directly is not reused by a shifted match
    s = float(analog.shift_entropy_sim(_f32(100.0, 110.0), _f32(0.5, 0.5), _f32(100.0), _f32(1.0), 10.0, 0.01))
    assert 0.2 < s < 0.99


def toy_analog_lib(target):
    hexose = analog.NP_DELTAS["C6H10O5 hexose"]
    nms = [target - hexose, target - hexose, target - 37.0, target - 150.0]    # delta hit, wrong polarity, off-list, open
    peaks = [([100.0, 200.0, 300.0], [0.2, 0.3, 0.5])] * 4
    off = np.cumsum([0] + [len(p[0]) for p in peaks])
    adducts = ["[M+H]+", "[M-H]-", "[M+H]+", "[M+H]+"]
    prec = np.array([chem.precursor_of(m, a) for m, a in zip(nms, adducts)])
    return build_library(off, np.concatenate([p[0] for p in peaks]), np.concatenate([p[1] for p in peaks]),
                         prec=prec, adduct=np.array(adducts, dtype=object),
                         ik=np.array(["HEXHITAAAAAAAA", "NEGPOLAAAAAAAA", "OFFLISTAAAAAAA", "OPENAAAAAAAAAA"],
                                     dtype=object), struct_mass=False), hexose


def test_analog_hits_modes():
    target = 600.0
    lib, hexose = toy_analog_lib(target)
    spec = [(np.array([100.0, 200.0, 300.0 + hexose]), np.array([0.2, 0.3, 0.5]))]
    h = analog.analog_hits(lib, spec, target, CFG, "delta", polarity=1)
    assert set(h) == {"HEXHITAAAAAAAA"} and h["HEXHITAAAAAAAA"] > 0.99
    assert set(analog.analog_hits(lib, spec, target, CFG, "delta", polarity=0)) == {"HEXHITAAAAAAAA", "NEGPOLAAAAAAAA"}
    o = analog.analog_hits(lib, spec, target, CFG, "open", polarity=1)
    assert set(o) == {"HEXHITAAAAAAAA", "OFFLISTAAAAAAA", "OPENAAAAAAAAAA"}
    ban = lib.ik == "HEXHITAAAAAAAA"
    assert "HEXHITAAAAAAAA" not in analog.analog_hits(lib, spec, target, CFG, "open", polarity=1, exclude=ban)


def test_class_vote_and_report():
    labels = {"A": "Flavonoids", "B": "Flavonoids", "C": "Coumarins", "D": None}
    win, w, n = analog.class_vote({"A": 0.6, "B": 0.55, "C": 0.9, "D": 0.95, "E": 0.3}, labels.get)
    assert win == "Flavonoids" and abs(w["Flavonoids"] - 1.15) < 1e-9 and n == 3
    assert analog.class_vote({"A": 0.4}, labels.get)[0] is None
    assert analog.class_vote({"A": 0.6, "C": 0.9}, labels.get, k=1)[0] == "Coumarins"
    assert analog.label_match("Flavonoids", "Coumarins|Flavonoids") and not analog.label_match(None, "x")
    r = analog.vote_report(["Flavonoids", None, "Coumarins", "Flavonoids", "x"],
                           ["Flavonoids", "Flavonoids", "Flavonoids", "Coumarins", None])
    assert r["n_with_truth"] == 4 and r["n_voted"] == 3 and abs(r["coverage"] - 0.75) < 1e-9
    assert abs(r["purity"] - 1 / 3) < 1e-9 and r["majority"] == "Flavonoids" and abs(r["baseline"] - 2 / 3) < 1e-9


def test_audit_coverage_growth():
    t = 500.0
    sub = pd.DataFrame([dict(adduct="[M+H]+", precursor_mz=chem.precursor_of(t, "[M+H]+") * (1 + 1.45e-6),
                             ionization_mode="positive")])
    mols = [("TRUTHPLAINKEYA", sub)]
    p0 = CandidatePool([t * (1 + 3e-6), t * (1 + 50e-6)], ["OTHERAAAAAAAAA", "OTHERBBBBBBBBB"], ["a", "b"])
    p1 = CandidatePool([t * (1 + 3e-6), t, t * (1 + 50e-6)], ["OTHERAAAAAAAAA", "TRUTHSCORERKEY", "OTHERBBBBBBBBB"],
                       ["a", "t", "b"])
    r0 = audit.coverage(mols, p0, lambda ik: ik, CFG)
    r1 = audit.coverage(mols, p1, {"TRUTHPLAINKEYA": "TRUTHSCORERKEY"}.get, CFG)
    assert not r0[0]["in_window"] and r1[0]["in_window"] and r1[0]["in_pool"] and r1[0]["in_scored"]
    assert r0[0]["n_window"] == 1 and r1[0]["n_window"] == 2
    g = audit.growth(r0, r1)
    assert g["recovered"] == 1 and g["lost"] == 0 and g["ratio_median"] == 2.0
    assert audit.summarize_coverage(r1)["recall_window"] == 1.0
    assert audit.sim_budget_per_mol(CFG) > 100


def test_audit_truth_in_sim_and_cv():
    target = 600.0
    lib, hexose = toy_analog_lib(target)
    spec = dict(adduct="[M+H]+", precursor_mz=chem.precursor_of(target, "[M+H]+") * (1 + 1.45e-6),
                ionization_mode="positive", collision_energy_ev=np.array([20.0, 40.0]),
                ms2_mzs=np.array([100.0, 200.0, 300.0 + hexose]), ms2_normalized_intensities=np.array([0.2, 0.3, 0.5]))
    mols = [("TRUTHAAAAAAAAA", pd.DataFrame([spec]))]
    pool = CandidatePool([target * (1 + p * 1e-6) for p in (-2, 0, 3)], ["DECOYAAAAAAAAA", "TRUTHAAAAAAAAA",
                                                                        "DECOYBBBBBBBBB"], ["a", "t", "b"])
    no_frag = lambda smis, specs, mode, cfg: np.zeros(len(smis), np.float32)  # noqa: E731
    r = audit.truth_in_sim(mols, lib, pool, lambda ik: ik, CFG, req_per_mol=4.0, frag_fn=no_frag, log=lambda s: None)
    assert r["n"] == 1 and r["truth_in_sim"] == 1.0                 # rank 0 (|ppm| 0) < 4 requests / 2 CEs
    r = audit.truth_in_sim(mols, lib, pool, lambda ik: ik, CFG, req_per_mol=2.0, frag_fn=no_frag, log=lambda s: None)
    assert r["truth_in_sim"] == 1.0
    labels = {"HEXHITAAAAAAAA": "Flavonoids", "TRUTHAAAAAAAAA": "Flavonoids", "OPENAAAAAAAAAA": "Coumarins",
              "OFFLISTAAAAAAA": "Coumarins"}
    cv = audit.class_vote_diag(mols, lib, lambda ik: ik, lambda k, lvl: labels.get(k), CFG, log=lambda s: None)
    assert cv["delta"]["np_superclass"]["purity"] == 1.0 and cv["delta"]["np_superclass"]["coverage"] == 1.0
    assert cv["open"]["np_superclass"]["n_voted"] == 1 and not cv["delta"]["pass"]   # baseline 1.0 on one molecule


def fixture_sources(d, workers_note=""):
    """Tiny real-SMILES versions of the five build inputs."""
    import pickle
    d = Path(d)
    acetone_enol, acetone, quercetin = "C=C(C)O", "CC(C)=O", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
    caffeine, aspirin = "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O"
    pd.DataFrame({"inchikey14": ["RYYVLZVUVIJVGH", "RYYVLZVUVIJVGH", "BSYNRYMUTXBXSQ"],
                  "normalized_smiles": [caffeine, caffeine, aspirin],
                  "ingest_lib": ["enveda-180", "gnps", "enveda-180"]}).to_parquet(d / "train.parquet")
    pd.DataFrame({"key": ["RYYVLZVUVIJVGH", "REFJWTPEDVJJIY"], "smiles": [caffeine, quercetin],
                  "mass": [194.08, 302.04]}).to_parquet(d / "pool_meta.parquet")
    (d / "coco").mkdir()
    pickle.dump(dict(keys=["REFJWTPEDVJJIY-UHFFFAOYSA-N", "BADBADBADBADBA"], smiles=[quercetin, "not_a_smiles"],
                     nbits=0), open(d / "coco" / "coco_meta.pkl", "wb"))
    (d / "bio").mkdir()
    pickle.dump(dict(keys=["NARVIWMVBMUEOG"], smiles=[acetone_enol]), open(d / "bio" / "bio_meta.pkl", "wb"))
    pd.DataFrame({"ik14": ["REFJWTPEDVJJIY", "CSCPPACGZOOCGX"], "scorer_key": ["REFJWTPEDVJJIY", "CSCPPACGZOOCGX"],
                  "smiles": [quercetin, acetone], "in_lotus": [True, False], "in_npatlas": [False, True],
                  "np_pathway": ["Shikimates and Phenylpropanoids", None], "np_superclass": ["Flavonoids", None],
                  "np_class": ["Flavonols", None], "kingdoms": ["Archaeplastida", None], "n_organisms": [120, 1]}
                 ).to_parquet(d / "natural_products.parquet")
    return dict(train=str(d / "train.parquet"), v2=str(d / "pool_meta.parquet"), coco=str(d / "coco"),
                bio=str(d / "bio"), np_table=str(d / "natural_products.parquet"))


def test_build_end_to_end_rdkit():
    from casmi_scratch.pool import F_LOTUS
    with tempfile.TemporaryDirectory() as d:
        inputs = fixture_sources(d)
        out, ck = str(Path(d) / "pool"), str(Path(d) / "ckpt")
        man = PB.build(inputs, out, ck, "t1", workers=1, log=lambda s: None)
        st = man["stats"]
        assert st["pool_size"] == 4 and st["dropped_unkeyable"] == {"coco": 1} and st["e180_only"] == 1
        assert man["R1_1"]["missing"] == 0 and man["np_parity"]["table_key_differs"] == 0
        assert st["key_disagree"].get("bio") == 1                         # enol: source key != scorer key
        p, _ = load_pool(out)
        k = dict(zip(p.keys, p.flags))
        assert k["CSCPPACGZOOCGX"] & F_BIO and k["CSCPPACGZOOCGX"] & F_NPT       # enol (BIO) merged with acetone
        assert k["REFJWTPEDVJJIY"] & F_V2 and k["REFJWTPEDVJJIY"] & F_COCO and k["REFJWTPEDVJJIY"] & F_LOTUS
        assert k["RYYVLZVUVIJVGH"] & F_TRAIN_OTHER and set(p.keys[e180_only(p.flags)]) == {"BSYNRYMUTXBXSQ"}
        assert load_keymap(out)["RYYVLZVUVIJVGH"] == "RYYVLZVUVIJVGH"
        lab = load_np_labels(out)
        assert lab.loc["REFJWTPEDVJJIY", "label_source"] == 1
        assert lab.loc["REFJWTPEDVJJIY", "np_superclass"] == "Flavonoids"
        assert lab.loc["RYYVLZVUVIJVGH", "label_source"] == 0               # caffeine: no labelled NP neighbour
        assert np.isfinite(lab.loc["REFJWTPEDVJJIY", "np_score"]) and lab.loc["REFJWTPEDVJJIY", "n_organisms"] == 120
        n_ck = len(list(Path(ck).rglob("*.parquet")))
        man2 = PB.build(inputs, out, ck, "t1", workers=1, log=lambda s: None)      # resumes from checkpoints
        assert len(list(Path(ck).rglob("*.parquet"))) == n_ck and man2["stats"] == man["stats"]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    t0 = time.time()
    for f in TESTS:
        f()
    print(f"test_r1: {len(TESTS)} OK ({time.time() - t0:.1f}s)")
