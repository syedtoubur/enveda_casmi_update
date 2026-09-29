"""Offline checks for specs_novel P1 (build/novel_cohort.py), on synthetic data. No Drive, no Colab.

1. scaffold_key: benzene derivatives share a key; stereo/isotopes ignored; acyclic and unparsable
   structures are singleton keys
2. formula_counts / structure_record: in_scope, SELFIES length, formula mismatch flag
3. label_ok: ppm floor, the 0.0075 Da rounding floor at low mass, mass-range edges, bad adduct/formula
4. eligible_molecules: msg exclusion, NP libraries only, timsTOF first, src_lib tie-break
5. choose_gmax: largest G within the budget; smallest G (met = False) when none fits
6. sample_cohorts: timsTOF first (all of them when <= 600), never above G_MAX, c2 outside c3 groups,
   deterministic per seed
7. pick_spectra: count, CE-known first, distinct (adduct, CE) before repeats
8. assign_splits: one label per ik14, c3 groups whole, dec_val ~3 % of whole groups, disjoint holdouts
9. dreams_ft_risk and check_invariants (each invariant can fail)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import novel_cohort as nc
import simsim_common as ssc


def test_scaffold_key():
    k = nc.scaffold_key
    assert k("Cc1ccccc1", "A") == k("Oc1ccccc1", "B") == k("c1ccccc1", "C") == "c1ccccc1"
    assert k("C[C@H](N)c1ccccc1", "D") == k("CC(N)c1ccccc1", "E")                 # stereo ignored
    assert k("[13CH3]c1ccccc1", "F") == "c1ccccc1"                                # isotope ignored
    assert k("CCCCCC(=O)O", "IKACYCLIC") == "acyclic:IKACYCLIC"
    assert k("CCO", "X1") != k("CCN", "X2")                                       # acyclic: singletons
    assert k("not a smiles", "IKBAD") == "unparsable:IKBAD" and k(None, "IKNONE") == "unparsable:IKNONE"
    assert k("O=C1CCCN1c1ccccc1", "G") == "O=C1CCCN1c1ccccc1"                      # keeps ring-attached =O
    print("  scaffold_key: OK")


def test_structure_record():
    assert nc.formula_counts("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert nc.formula_counts("CH4") == {"C": 1, "H": 4}
    assert nc.formula_counts("C6H5+") is None and nc.formula_counts("") is None and nc.formula_counts(None) is None
    r = nc.structure_record(("IK", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", 171))
    assert r["status"] == "ok" and r["in_scope"] and r["ntok"] > 0 and r["formula_mismatch"] is False
    assert r["scaffold_key"] == "c1ccccc1" and r["can"] and r["selfies"]
    assert nc.structure_record(("IK", "CC(=O)Oc1ccccc1C(=O)O", "C9H10O4", 171))["formula_mismatch"] is True
    short = nc.structure_record(("IK", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", 5))
    assert short["status"] == "ok" and not short["in_scope"]
    multi = nc.structure_record(("IK", "CCO.Cl", "C2H7ClO", 171))
    assert multi["status"] == "multi" and not multi["in_scope"] and multi["formula_mismatch"] is None
    si = nc.structure_record(("IKSI", "C[Si](C)(C)C", "C4H12Si", 171))
    assert si["status"] == "element" and not si["in_scope"] and si["scaffold_key"] == "acyclic:IKSI"
    print("  structure_record: OK")


def test_label_ok():
    f = "C20H30O5"                                                                 # ~350 Da
    m = ssc.formula_mass(f)
    h = ssc.ADDUCT_DELTA["[M+H]+"]
    na = ssc.ADDUCT_DELTA["[M+Na]+"]
    # at 350 Da, 10 ppm = 0.0035 Da < 0.0075, so the rounding floor rules
    mz = np.array([m + h, m + h + 0.0074, m + h + 0.0076, m + na - 0.007, m + h])
    ad = np.array(["[M+H]+", "[M+H]+", "[M+H]+", "[M+Na]+", "[M+2H]2+"], object)
    ok, neutral, fm = nc.label_ok(mz, ad, np.array([f] * 5, object))
    assert list(ok) == [True, True, False, True, False], ok
    assert np.allclose(neutral[:1], [m]) and np.allclose(fm, m)
    # at ~1000 Da, 10 ppm = 0.010 Da > 0.0075, so ppm rules
    f2 = "C50H80O18"
    m2 = ssc.formula_mass(f2)
    assert 900 < m2 < 1159
    ok2, *_ = nc.label_ok(np.array([m2 + h + 0.0095, m2 + h + 0.0115]), np.array(["[M+H]+"] * 2, object),
                          np.array([f2] * 2, object))
    assert list(ok2) == [True, False], ok2
    # mass-range edges on the neutral mass: formulas just inside and just outside [157, 1159]
    for f3, inside in (("C8H15NO2", True), ("C6H6", False), ("C60H102O20", True), ("C62H104O21", False)):
        mm = ssc.formula_mass(f3)
        assert (157.0 <= mm <= 1159.0) == inside, (f3, mm)
        o, *_ = nc.label_ok(np.array([mm + h]), np.array(["[M+H]+"], object), np.array([f3], object))
        assert o[0] == inside, (f3, mm)
    # unusable formulas are never ok
    o, *_ = nc.label_ok(np.array([300.0, 300.0]), np.array(["[M+H]+"] * 2, object), np.array([None, "C6H5+"], object))
    assert not o.any()
    print("  label_ok: OK")


def test_eligible():
    rows = pd.DataFrame({
        "inchikey14": ["A", "A", "A", "B", "B", "B", "C", "D", "E", "E", "F", "F"],
        "ingest_lib": ["gnps", "enveda-np-examples", "riken", "mona", "gnps", "enveda-180",
                       "enveda-180", "riken", "mona", "gnps", "riken", "massbank"],
        "ok": [True, True, True, True, True, True, True, True, False, True, True, True]})
    e = nc.eligible_molecules(rows, np.array(["D"])).set_index("inchikey14")
    assert set(e.index) == {"A", "B", "E", "F"}                                    # C: enveda-180 only; D: msg
    assert e.loc["A", "src_lib"] == "enveda-np-examples" and e.loc["A", "timstof"]
    assert e.loc["B", "src_lib"] == "gnps" and not e.loc["B", "timstof"]           # tie mona/gnps -> gnps
    assert e.loc["E", "src_lib"] == "gnps"                                         # mona row not ok
    assert e.loc["F", "src_lib"] == "massbank" and e.loc["F", "n_ok"] == 1
    print("  eligible_molecules: OK")


def test_choose_gmax():
    gs = np.array([1] * 800 + [30] * 100 + [80] * 60 + [150] * 40)
    U = 200_000
    g, met, means = nc.choose_gmax(gs, U)
    cap = 0.07 * U - 400 - 0.03 * U
    assert met and 600 * means[g] <= cap
    bigger = [c for c in nc.GMAX_CHOICES if c > g]
    assert all(600 * means[c] > cap for c in bigger), (g, means)
    g2, met2, _ = nc.choose_gmax(np.array([25] * 10), 1000)
    assert (g2, met2) == (25, False)
    g3, met3, _ = nc.choose_gmax(np.array([1] * 10), U)
    assert (g3, met3) == (200, True)
    print(f"  choose_gmax: OK (G_MAX {g}, m(G) {means[g]:.2f})")


def synthetic_elig(n=3000, n_ts=40, seed=0):
    rng = np.random.default_rng(seed)
    ik = [f"IK{i:012d}" for i in range(n)]
    scaf = np.array([f"S{int(x)}" for x in rng.zipf(1.6, n) % 400], object)
    acyc = rng.random(n) < 0.2
    scaf[acyc] = [f"acyclic:{k}" for k in np.array(ik, object)[acyc]]
    s = pd.Series(scaf, index=ik)
    gsz = nc.group_sizes(s)
    ts = np.zeros(n, bool); ts[rng.choice(n, n_ts, replace=False)] = True
    elig = pd.DataFrame({"inchikey14": ik, "src_lib": np.where(ts, nc.TIMSTOF_LIB, "gnps"), "timstof": ts,
                         "n_ok": 1, "group_size": gsz.to_numpy(), "scaffold_key": scaf})
    return elig, s


def test_sample_cohorts():
    elig, scaf = synthetic_elig()
    G = 25
    c3, c2 = nc.sample_cohorts(elig, G, seed=7, n_c3=600, n_c2=400)
    e = elig.set_index("inchikey14")
    assert len(c3) == 600 and len(c2) == 400 and len(set(c3)) == 600
    assert (e.loc[c3, "group_size"] <= G).all()
    ts_ok = set(e.index[e.timstof & (e.group_size <= G)])
    assert ts_ok <= set(c3), "every eligible timsTOF molecule within G_MAX must be in pseudo_c3"
    c3_scaf = set(e.loc[c3, "scaffold_key"])
    assert not set(e.loc[c2, "scaffold_key"]) & c3_scaf and not set(c2) & set(c3)
    assert (c3, c2) == nc.sample_cohorts(elig, G, seed=7, n_c3=600, n_c2=400)
    assert c3 != nc.sample_cohorts(elig, G, seed=8, n_c3=600, n_c2=400)[0]
    # more timsTOF than n_c3: all slots go to timsTOF
    elig2 = elig.copy(); elig2["timstof"] = True
    c3b, _ = nc.sample_cohorts(elig2, G, seed=1, n_c3=50, n_c2=10)
    assert len(c3b) == 50
    # fewer eligible than asked: take them all
    small = elig.head(100)
    c3s, c2s = nc.sample_cohorts(small, 1_000_000, seed=1)
    assert len(c3s) == 100 and c2s == []
    print("  sample_cohorts: OK")


def test_pick_spectra():
    rng = np.random.default_rng(3)
    g = pd.DataFrame({"row": np.arange(10),
                      "adduct": ["[M+H]+"] * 4 + ["[M+Na]+"] * 2 + ["[M+H]+"] * 4,
                      "ce": [20.0, 20.04, 40.0, 40.0, 20.0, np.nan, np.nan, np.nan, 60.0, 20.0]})
    for n in range(1, 13):
        p = nc.pick_spectra(g, n, rng)
        assert len(p) == min(n, 10) and len(set(p)) == len(p) and set(p) <= set(range(10))
        known = set(g.row[g.ce.notna()])
        if n <= len(known):
            assert set(p) <= known, (n, p)                                         # CE-known first
    # the distinct (adduct, CE) pairs: (H,20) (H,40) (Na,20) (H,60) -> 4 distinct among known rows
    for trial in range(20):
        p = nc.pick_spectra(g, 4, rng)
        pairs = {(g.adduct[r], round(g.ce[r], 1)) for r in p}
        assert len(pairs) == 4, p
    only_unknown = pd.DataFrame({"row": [5, 6], "adduct": ["[M+H]+"] * 2, "ce": [np.nan, np.nan]})
    assert sorted(nc.pick_spectra(only_unknown, 5, rng)) == [5, 6]
    assert nc.draw_n([1, 1, 40], np.random.default_rng(0)) in (1, 16)
    assert all(nc.draw_n([40], rng) == 16 for _ in range(3))
    print("  pick_spectra: OK")


def test_assign_splits():
    elig, scaf = synthetic_elig(n=5000, n_ts=30, seed=1)
    G = 25
    c3, c2 = nc.sample_cohorts(elig, G, seed=2, n_c3=200, n_c2=100)
    split = nc.assign_splits(scaf, c3, c2, G, seed=2)
    assert split.index.is_unique and set(split.index) == set(scaf.index) and split.isin(nc.SPLITS).all()
    df = pd.DataFrame({"scaf": scaf, "split": split})
    c3_scaf = set(scaf.loc[c3])
    assert df[df.scaf.isin(c3_scaf)].split.eq("heldout_c3").all()
    assert df[~df.scaf.isin(c3_scaf)].split.ne("heldout_c3").all()
    assert split.loc[c2].eq("heldout_c2").all() and split.eq("heldout_c2").sum() == len(c2)
    dv = df[df.split == "dec_val"]
    assert 0.025 * len(scaf) <= len(dv) <= 0.03 * len(scaf), len(dv)
    per = df.groupby("scaf").split.nunique()
    assert (per.loc[list(set(dv.scaf))] == 1).all(), "a dec_val scaffold group is split"
    assert (dv.scaf.map(scaf.value_counts()) <= G).all()
    assert split.equals(nc.assign_splits(scaf, c3, c2, G, seed=2))
    print(f"  assign_splits: OK (dec_val {len(dv) / len(scaf):.3%}, heldout_c3 {split.eq('heldout_c3').mean():.2%})")
    return scaf, c3, c2, split


def test_ts_dec():
    """A-G1.1: ts_dec = ICEBERG-seen timsTOF molecules outside every c3 / c2 group, whole groups held out."""
    elig, scaf = synthetic_elig(n=4000, n_ts=30, seed=4)
    G = 25
    c3, c2 = nc.sample_cohorts(elig, G, seed=3, n_c3=150, n_c2=80)
    ik = np.array(scaf.index)
    rng = np.random.default_rng(5)
    ts_rows = rng.choice(ik, 500, replace=False)
    rows = pd.DataFrame({"inchikey14": np.concatenate([ts_rows, ik[:300]]),
                         "ingest_lib": [nc.TIMSTOF_LIB] * 500 + ["gnps"] * 300,
                         "ok": np.r_[np.arange(500) % 10 != 0, np.ones(300, bool)]})
    msg = np.concatenate([ts_rows[:400], ik[:300]])                  # 100 timsTOF molecules not in msg
    ts, fun = nc.sample_ts_dec(rows, msg, scaf, c3, c2, G)
    size = scaf.value_counts()
    c3_scaf, c2_scaf = set(scaf.loc[c3]), set(scaf.loc[c2])
    okts = set(rows.inchikey14[rows.ok & (rows.ingest_lib == nc.TIMSTOF_LIB)])
    want = sorted(k for k in okts & set(msg) if scaf[k] not in c3_scaf | c2_scaf and size[scaf[k]] <= G)
    assert ts == want and len(ts) > 20, (len(ts), len(want))
    assert fun["candidates"] == len(okts & set(msg)) >= fun["after_cohort_exclusions"] >= fun["within_gmax"] == len(ts)
    assert not set(ts) & (set(c3) | set(c2))
    split = nc.assign_splits(scaf, c3, c2, G, seed=3, ts=ts)
    ts_scaf = set(scaf.loc[ts])
    df = pd.DataFrame({"scaf": scaf, "split": split})
    assert df[df.scaf.isin(ts_scaf)].split.eq("heldout_ts").all() and split.eq("heldout_ts").sum() >= len(ts)
    assert df[df.scaf.isin(c3_scaf)].split.eq("heldout_c3").all() and split.loc[c2].eq("heldout_c2").all()
    assert not set(df[df.split == "dec_val"].scaf) & (ts_scaf | c3_scaf | c2_scaf)
    base = nc.assign_splits(scaf, c3, c2, G, seed=3)
    assert base.eq("heldout_c3").equals(split.eq("heldout_c3")) and "heldout_ts" not in set(base)
    # invariants with ts
    allk = list(c3) + list(c2) + list(ts)
    cohort_rows = {k: [i] for i, k in enumerate(allk)}
    src = {**{k: "gnps" for k in list(c3) + list(c2)}, **{k: nc.TIMSTOF_LIB for k in ts}}
    row_lib = np.array([src[k] for k in allk], object)
    args = dict(split=split, scaf=scaf, c3=c3, c2=c2, cohort_rows=cohort_rows, row_ok=np.ones(len(allk), bool),
                row_lib=row_lib, src_lib_of=src, msg_ik14=np.array(ts), ts=ts)
    inv = nc.check_invariants(**args)
    assert all(inv.values()), inv
    s2 = split.copy(); s2.loc[ts[0]] = "train"
    assert not nc.check_invariants(**{**args, "split": s2})["ts_groups_not_in_train_or_dec_val"]
    assert not nc.check_invariants(**{**args, "msg_ik14": np.array(ts[1:])})["ts_all_in_msg"]
    assert not nc.check_invariants(**{**args, "ts": list(ts) + [c3[0]], "msg_ik14": np.array(list(ts) + [c3[0]])}
                                   )["ts_disjoint_from_c3_c2"]
    src2 = {**src, ts[0]: "gnps"}
    assert not nc.check_invariants(**{**args, "src_lib_of": src2})["ts_src_lib_timstof"]
    print(f"  ts_dec: OK ({len(ts)} molecules, funnel {fun})")


def test_ft_risk_and_invariants(scaf, c3, c2, split):
    assert nc.dreams_ft_risk("mona", ["[M+H]+", "[M+Na]+"])
    assert not nc.dreams_ft_risk("mona", ["[M+Na]+"]) and not nc.dreams_ft_risk("gnps", ["[M+H]+"])
    ik = list(c3) + list(c2)
    cohort_rows = {k: [2 * i, 2 * i + 1] for i, k in enumerate(ik)}
    n_rows = 2 * len(ik)
    row_ok = np.ones(n_rows, bool)
    row_lib = np.array(["gnps"] * n_rows, object)
    src = {k: "gnps" for k in ik}
    args = dict(split=split, scaf=scaf, c3=c3, c2=c2, cohort_rows=cohort_rows, row_ok=row_ok,
                row_lib=row_lib, src_lib_of=src, msg_ik14=np.array(["ZZZ"]))
    inv = nc.check_invariants(**args)
    assert all(inv.values()), inv
    bad = row_ok.copy(); bad[0] = False
    assert not nc.check_invariants(**{**args, "row_ok": bad})["cohort_rows_quality_ok_from_src_lib"]
    lib = row_lib.copy(); lib[1] = "riken"
    assert not nc.check_invariants(**{**args, "row_lib": lib})["cohort_rows_quality_ok_from_src_lib"]
    assert not nc.check_invariants(**{**args, "msg_ik14": np.array([c2[0]])})["no_cohort_ik14_in_msg"]
    s2 = split.copy(); s2.loc[c3[0]] = "train"
    assert not nc.check_invariants(**{**args, "split": s2})["c3_groups_not_in_train_or_dec_val"]
    s3 = split.copy(); s3.loc[c2[0]] = "dec_val"
    assert not nc.check_invariants(**{**args, "split": s3})["c2_not_in_train_or_dec_val"]
    assert not nc.check_invariants(**{**args, "c2": list(c2) + [c3[0]]})["cohorts_disjoint"]
    assert not nc.check_invariants(**{**args, "split": split.iloc[1:]})["split_covers_every_ik14_once"]
    print("  dreams_ft_risk + invariants: OK")


def test_notebook():
    """casmi_novel_p1_colab: fresh, parses, embeds the sources verbatim, and uses the G1 thresholds."""
    import ast
    import json
    import mk_novel_p0 as mk
    src, dst, tags = next(b for b in mk.BUILDS if b[0].name == "novel_p1_colab.py")
    before = dst.read_text(encoding="utf-8") if dst.exists() else None
    mk.build(src, dst, tags)
    assert before is None or before == dst.read_text(encoding="utf-8"), f"{dst.name} was stale -- re-run"
    code = ["".join(c["source"]) for c in json.loads(dst.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]
    for i, c in enumerate(code):
        ast.parse(c, filename=f"{dst.name}[code {i}]")
    for tag in tags:
        var, path = mk.SOURCES[tag]
        lines = [ln for c in code for ln in c.split("\n") if ln.startswith(f"{var} = ")]
        assert len(lines) == 1 and ast.literal_eval(lines[0][len(var) + 3:]) == path.read_text(encoding="utf-8"), var
    text = src.read_text(encoding="utf-8")
    c = {t.id: ast.literal_eval(a.value) for a in ast.parse(text).body if isinstance(a, ast.Assign)
         for t in a.targets if isinstance(t, ast.Name) and isinstance(a.value, ast.Constant)}
    assert (c["MIN_C3"], c["MIN_TS_DEC"], c["MIN_C2"], c["MIN_TRAIN_SHARE"]) == (300, 100, 300, 0.90)   # A-G1.1
    assert "MIN_C3_TIMSTOF" not in c and "\"G1.1'\"" in text and "\"G1.4'\"" in text
    assert "nc.sample_ts_dec(" in text and "ts=TS)" in text and '"ts_dec.parquet"' in text
    assert c["MAX_LEN"] == 171 and c["N_CORPUS_ROWS"] == 1_323_878 and c["N_CORPUS_IK14"] == 194_668
    used = set(__import__("re").findall(r"\bnc\.([A-Za-z_]\w*)", text))
    assert used <= set(dir(nc)), used - set(dir(nc))
    assert 'OUT = ROOT / "_novel" / "p1"' in text and "emb_" not in text.split("# %%", 2)[2].replace(
        "emb_dbec3a0", ""), "P1 must not read embeddings"
    print(f"  notebook: OK ({dst.name}, {len(code)} code cells)")


def main():
    test_notebook()
    test_scaffold_key()
    test_structure_record()
    test_label_ok()
    test_eligible()
    test_choose_gmax()
    test_sample_cohorts()
    test_pick_spectra()
    test_ft_risk_and_invariants(*test_assign_splits())
    test_ts_dec()
    print("test_novel_p1: OK")


if __name__ == "__main__":
    main()
