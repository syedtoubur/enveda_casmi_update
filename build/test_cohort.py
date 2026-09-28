"""Dry-run Stage C v2's cohort + assembly cells on synthetic data, using the shipped code.

Checks the parts that only fail at runtime on Kaggle:
  * the four-cell cohort (FPNet-seen / FPNet-unseen x library visible / banned)
  * Stage-D structures never appearing as Stage-C queries
  * the fallback branch when the `es` pool is too small (this is where v1's rewrite had a bug:
    `E` had already been filtered, so relatives could never be picked up)
  * ban_vis vs ban_all row sets
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

NB_NAME = sys.argv[1] if len(sys.argv) > 1 else "casmi_rank_matrix.ipynb"
NB = Path(__file__).resolve().parent.parent / NB_NAME
SRC = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]
       if c["cell_type"] == "code"]
COHORT = next(s for s in SRC if "BOTH model-strength regimes" in s)
# the real formula_mass, lifted out of the verbatim phase1 library cell
_libcell = next(s for s in SRC if "def load_library" in s)
_a, _b = _libcell.index("import re as _re"), _libcell.index("def neutral_mass")
FORMULA_MASS_SRC = _libcell[_a:_b]
assert "def formula_mass" in FORMULA_MASS_SRC
ASSEMBLY = next(s for s in SRC if "compute both ban sets" in s or "then both ban settings" in s)
# stop before the parquet peak read -- unchanged from the version that ran fine
_cut = ASSEMBLY.find("# read the peak lists")
if _cut < 0: _cut = ASSEMBLY.find("sel_rows = np.array")
ASSEMBLY = ASSEMBLY[:_cut]
print(f"### testing cohort + assembly cells from {NB_NAME}\n")

TMP = Path(tempfile.mkdtemp())
NP_LIBS = ['riken', 'mona', 'msdial', 'massbank', 'gnps', 'enveda-np-examples', 'masaryk']
ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
           "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


# 'riken' stands in for a source library that reports m/z to too few decimals. Its rows
# are chemically fine and must still be dropped: they cannot retrieve their own structure.
LOW_PRECISION_LIB = "riken"
PROTON = 1.00728


def synth(n_struct, n_es, n_banned, spec_per=6, seed=0):
    rng = np.random.default_rng(seed)
    keys = np.array([f"K{i:013d}"[:14] for i in range(n_struct)], dtype=object)
    rows = []
    for i, k in enumerate(keys):
        nC = int(rng.integers(12, 55)); nH = int(rng.integers(12, 90))
        formula = f"C{nC}H{nH}"
        exact = 12.0 * nC + 1.0078250319 * nH
        for _ in range(rng.integers(2, spec_per + 1)):
            lib = rng.choice(NP_LIBS)
            mz = exact + PROTON
            # one decimal place -> up to 0.05 Da, >50 ppm: outside its own window, always
            if lib == LOW_PRECISION_LIB: mz = round(mz, 1)
            rows.append(dict(
                inchikey14=k, normalized_smiles="CCO", molecular_formula=formula,
                ingest_lib=lib, adduct=rng.choice(ADDUCTS),
                ionization_mode=rng.choice(["positive", "negative"]),
                instrument_type="timsTOF", precursor_mz=float(mz),
                num_peaks=int(rng.integers(5, 60)),
                collision_energy_ev=(np.nan if rng.random() < 0.3
                                     else np.array([float(rng.uniform(10, 45))]))))
    md = pd.DataFrame(rows)
    order = rng.permutation(len(keys))
    banned = keys[order[:n_banned]]
    es = keys[order[n_banned:n_banned + n_es]]
    seen = keys[order[n_banned + n_es:]]
    U = lambda a: np.asarray([str(x) for x in a], dtype="U14")
    np.savez(TMP / "fpnet_split_keys.npz", seen_keys=U(seen), es_keys=U(es),
             banned_keys=U(banned))
    # val_key: the first 40 banned structures are the actual Stage-D queries; the rest are
    # scaffold relatives swept in by C2_hard
    vq = banned[:40]
    vk = [dict(molecule_id=f"VAL{i:05d}", inchikey14=k, cohort="C2_hard", cls=2,
               ban_keys=[k] + list(banned[40:45])) for i, k in enumerate(vq)]
    pickle.dump(vk, open(TMP / "val_key.pkl", "wb"))
    return md, set(vq)


class FakeTbl:
    def __init__(s, df): s.df = df
    def to_pandas(s): return s.df


def make_globals(md_df, n_seen, n_unseen):
    class SC:
        SHARD, N_SHARDS = 0, 1
        N_SEEN, N_UNSEEN = n_seen, n_unseen
        NP_LIBS = globals()["NP_LIBS"]
        MASS_LO, MASS_HI = 157.0, 1159.0
        MIN_PEAKS = 5
        MAX_ROWS_PER_QUERY = 400
        SEED = 1
        OUT = str(TMP)
        QUERY_PPM_MAX = 8.5

    class CFG:
        PPM_WIN = 8.5
        PPM_CENTER = 1.45

    class FakePq:
        @staticmethod
        def read_table(path, columns=None):
            if "test" in str(path):
                return FakeTbl(pd.DataFrame({"molecule_id":
                                             np.repeat(np.arange(400), 3)}))
            return FakeTbl(md_df[[c for c in columns if c in md_df.columns]].copy())

    def neutral_mass(mz, adduct):
        return np.asarray(mz, float) - PROTON

    g = {"np": np, "pd": pd, "pq": FakePq, "pickle": pickle, "print": print,
         "SC": SC, "CFG": CFG, "TRAIN": "train.parquet", "TEST": "test.parquet",
         "neutral_mass": neutral_mass}
    exec(compile(FORMULA_MASS_SRC, "formula_mass", "exec"), g)
    return {**g,
            "find_file": lambda n: str(TMP / n),
            "DRIVE_ROOT": str(TMP), "STAGE": str(TMP)}


def check(tag, n_struct, n_es, n_banned, n_seen_q, n_unseen_q, expect_fallback):
    print("=" * 72)
    print(f"{tag}")
    print("=" * 72)
    md_df, val_query = synth(n_struct, n_es, n_banned)
    g = make_globals(md_df, n_seen_q, n_unseen_q)
    exec(compile(COHORT, "cohort", "exec"), g)
    exec(compile(ASSEMBLY, "assembly", "exec"), g)

    recs = g["records"]
    regimes = pd.Series([r["regime"] for r in recs]).value_counts().to_dict()
    print(f"\n  records: {len(recs)}  regimes: {regimes}")

    assert recs, "no records produced"
    assert "seen" in regimes and "unseen" in regimes, \
        f"both model-strength regimes must be present, got {regimes}"

    # Stage-D queries must never be Stage-C queries, fallback or not
    leaked = [r["ik"] for r in recs if r["ik"] in val_query]
    assert not leaked, f"Stage-D query molecules leaked into Stage C: {leaked[:3]}"
    print(f"  no Stage-D query molecule used as a Stage-C query  [ok]")

    # every record carries both ban settings, and 'all' is a superset of 'visible'
    for r in recs:
        assert set(r["ban_vis"]) <= set(r["ban_all"]), "ban_vis must be a subset of ban_all"
        assert set(r["rows"]) <= set(r["ban_vis"]), "the query's own rows must always be banned"
    bigger = sum(1 for r in recs if len(r["ban_all"]) > len(r["ban_vis"]))
    print(f"  ban_vis subset of ban_all for all {len(recs)} records; "
          f"{bigger} have strictly more rows banned under M=1  [ok]")

    E = g["E"]
    vc = E.ingest_lib.value_counts()
    lo = int(vc.get(LOW_PRECISION_LIB, 0))
    hi = float(np.median([vc.get(l, 0) for l in NP_LIBS if l != LOW_PRECISION_LIB]))
    # a few coarse rows round close enough to survive; the bulk must not
    assert lo < 0.25 * hi, (f"{LOW_PRECISION_LIB} reports m/z too coarsely to retrieve its own "
                            f"structure, yet {lo} of its rows survived (others ~{hi:.0f})")
    assert hi > 0, "the filter over-fired: no high-precision rows survive either"
    print(f"  low-precision source library cut to {lo} rows vs ~{hi:.0f} for the others  [ok]")

    seen_k, unseen_k = set(g["seen_pool"]), set(g["unseen_pool"])
    assert not (seen_k & unseen_k), "cohorts overlap"
    print(f"  seen_pool {len(seen_k)} / unseen_pool {len(unseen_k)}, disjoint  [ok]")

    used_fb = len(unseen_k & set(np.load(TMP / "fpnet_split_keys.npz")["banned_keys"])) > 0
    assert used_fb == expect_fallback, \
        f"fallback used={used_fb}, expected {expect_fallback}"
    print(f"  fallback branch used = {used_fb} (expected {expect_fallback})  [ok]\n")


check("A. healthy es pool - fallback must NOT fire",
      n_struct=3000, n_es=900, n_banned=300, n_seen_q=200, n_unseen_q=200,
      expect_fallback=False)

check("B. es pool too small - fallback MUST fire and must still exclude Stage-D queries",
      n_struct=3000, n_es=60, n_banned=600, n_seen_q=200, n_unseen_q=200,
      expect_fallback=True)

print("=" * 72)
print("COHORT LOGIC OK: four cells built, Stage-D holdout respected, fallback path works.")
