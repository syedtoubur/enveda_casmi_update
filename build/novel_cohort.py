"""specs_novel P1 helpers: honest benchmark cohorts and the decoder training split.

The definitions are pre-registered in specs_novel/gate_report.md -> G1 (design: phase1_plan.md §3-§4).
numpy / pandas plus simsim_common; RDKit and selfies are imported lazily (structure pass only).
Embedded verbatim into casmi_novel_p1_colab.ipynb by build/mk_novel_p0.py and imported directly by
build/test_novel_p1.py.
"""
import re

import numpy as np
import pandas as pd

import simsim_common as ssc

NP_LIBS = ("riken", "mona", "msdial", "massbank", "gnps", "enveda-np-examples", "masaryk")
TIMSTOF_LIB = "enveda-np-examples"
DREAMS_FT_LIB, DREAMS_FT_ADDUCT = "mona", "[M+H]+"   # the DreaMS head's fine-tuning data (leakage flag)
MASS_LO, MASS_HI = 157.0, 1159.0
PPM_FLOOR, ROUND_TOL_DA = 10.0, 0.0075
GMAX_CHOICES = (200, 100, 50, 25)
HELDOUT_BUDGET = 0.07
DEC_VAL_SHARE = 0.03
N_C3, N_C2 = 600, 400
MAX_SPEC = 16
SEED = 20260929
SPLITS = ("train", "dec_val", "heldout_c3", "heldout_c2")


# --------------------------------------------------------------------------------------------- #
# Per structure (RDKit, lazily)
# --------------------------------------------------------------------------------------------- #

def scaffold_key(smiles, ik14):
    """Murcko scaffold canonical SMILES of the stereo/isotope-stripped structure.
    Empty scaffold -> 'acyclic:<ik14>' (a group of one); unparsable -> 'unparsable:<ik14>'."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    if m is None:
        return f"unparsable:{ik14}"
    Chem.RemoveStereochemistry(m)
    for a in m.GetAtoms():
        a.SetIsotope(0)
    try:
        s = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return f"unparsable:{ik14}"
    return s if s else f"acyclic:{ik14}"


_FTOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(f):
    """'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}; None if the string is not a plain element formula."""
    if not isinstance(f, str) or not f:
        return None
    out, pos = {}, 0
    for m in _FTOK.finditer(f):
        if m.start() != pos:
            return None
        pos = m.end()
        out[m.group(1)] = out.get(m.group(1), 0) + (int(m.group(2)) if m.group(2) else 1)
    return out if pos == len(f) else None


def structure_record(args):
    """(ik14, normalized_smiles, molecular_formula, max_len) -> dict for one corpus structure.

    prep via novel_decoder.prep_structure; `in_scope` = prep ok and SELFIES length <= max_len;
    `formula_mismatch` = the RDKit formula of the prepped structure differs from the label formula
    (None when prep did not produce a formula)."""
    import selfies as sf
    import novel_decoder as nd
    ik, smi, label_formula, max_len = args
    status, can, selfies, counts = nd.prep_structure(smi)
    ntok = sf.len_selfies(selfies) if selfies else None
    mismatch = None
    if counts is not None:
        lab = formula_counts(label_formula)
        mismatch = lab is None or {k: v for k, v in lab.items() if v} != {k: v for k, v in counts.items() if v}
    return dict(inchikey14=ik, status=status, can=can, selfies=selfies, ntok=ntok,
                in_scope=bool(status == "ok" and ntok is not None and ntok <= max_len),
                formula_mismatch=mismatch, scaffold_key=scaffold_key(smi, ik))


# --------------------------------------------------------------------------------------------- #
# Per row: label quality
# --------------------------------------------------------------------------------------------- #

def label_ok(precursor_mz, adduct, formula):
    """Vectorised G1 label-quality filter -> (ok bool array, neutral mass, truth mass).

    ok = |neutral - truth| <= max(PPM_FLOOR ppm of truth, ROUND_TOL_DA) and neutral in [MASS_LO, MASS_HI].
    Adducts outside the ten test adducts and unparsable formulas are not ok."""
    mz = np.asarray(precursor_mz, np.float64)
    delta = pd.Series(np.asarray(adduct, object)).map(ssc.ADDUCT_DELTA).to_numpy(np.float64)
    f = pd.Series(np.asarray(formula, object))
    uf = f.dropna().unique()
    fm = f.map(dict(zip(uf, (ssc.formula_mass(x) for x in uf)))).to_numpy(np.float64)
    neutral = mz - delta
    tol = np.maximum(PPM_FLOOR * 1e-6 * fm, ROUND_TOL_DA)
    with np.errstate(invalid="ignore"):
        ok = (np.abs(neutral - fm) <= tol) & (neutral >= MASS_LO) & (neutral <= MASS_HI)
    return ok & np.isfinite(neutral) & np.isfinite(fm), neutral, fm


# --------------------------------------------------------------------------------------------- #
# Per molecule: eligibility, group sizes, G_MAX
# --------------------------------------------------------------------------------------------- #

def eligible_molecules(rows, msg_ik14):
    """rows: DataFrame with inchikey14, ingest_lib, ok. -> one row per eligible ik14:
    inchikey14, src_lib, timstof, n_ok (quality-OK rows in src_lib).

    Eligible: ik14 not in msg_ik14 and >= 1 quality-OK row in an NP library. src_lib is the timsTOF
    library when the molecule has quality-OK rows there, else its NP library with the most
    quality-OK rows (ties -> the name that sorts first)."""
    msg = set(np.asarray(msg_ik14).astype(str))
    r = rows.loc[rows.ok.astype(bool) & rows.ingest_lib.isin(NP_LIBS), ["inchikey14", "ingest_lib"]]
    r = r[~r.inchikey14.astype(str).isin(msg)]
    cnt = r.groupby(["inchikey14", "ingest_lib"]).size().rename("n_ok").reset_index()
    cnt["is_ts"] = cnt.ingest_lib == TIMSTOF_LIB
    cnt = cnt.sort_values(["inchikey14", "is_ts", "n_ok", "ingest_lib"], ascending=[True, False, False, True])
    out = cnt.drop_duplicates("inchikey14").rename(columns={"ingest_lib": "src_lib", "is_ts": "timstof"})
    return out[["inchikey14", "src_lib", "timstof", "n_ok"]].reset_index(drop=True)


def group_sizes(scaf):
    """scaf: Series ik14 -> scaffold_key (one entry per corpus ik14). -> Series ik14 -> group size."""
    return scaf.map(scaf.value_counts()).astype(np.int64)


def choose_gmax(elig_group_size, n_universe, n_c3=N_C3, n_c2=N_C2, choices=GMAX_CHOICES,
                budget=HELDOUT_BUDGET, dec_val=DEC_VAL_SHARE):
    """Largest G in choices with n_c3 * m(G) + n_c2 + dec_val * U <= budget * U, where m(G) is the mean
    group size of eligible molecules with group size <= G. -> (G, met, {G: m(G)}); if none meets the
    budget, the smallest G with met = False."""
    gs = np.asarray(elig_group_size, np.float64)
    means = {int(g): (float(gs[gs <= g].mean()) if (gs <= g).any() else 0.0) for g in choices}
    cap = budget * n_universe - n_c2 - dec_val * n_universe
    for g in sorted(choices, reverse=True):
        if n_c3 * means[g] <= cap:
            return int(g), True, means
    return int(min(choices)), False, means


# --------------------------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------------------------- #

def _draw(keys, n, rng):
    a = np.array(sorted(keys), dtype=object)
    if len(a) <= n:
        return list(a)
    return list(a[np.sort(rng.choice(len(a), size=n, replace=False))])


def sample_cohorts(elig, gmax, seed=SEED, n_c3=N_C3, n_c2=N_C2):
    """elig: eligible_molecules() plus group_size and scaffold_key. -> (pseudo_c3 ik14s, c2_displace ik14s).

    pseudo_c3: every eligible timsTOF molecule with group_size <= gmax (a uniform n_c3 of them if
    more), then a uniform draw from the other eligible molecules with group_size <= gmax to n_c3.
    c2_displace: a uniform n_c2 of the eligible molecules in no pseudo_c3 scaffold group (no cap)."""
    rng = np.random.default_rng(seed)
    cap = elig[elig.group_size <= gmax]
    ts = set(cap.inchikey14[cap.timstof.astype(bool)])
    c3 = _draw(ts, n_c3, rng)
    rest = set(cap.inchikey14) - set(c3)
    c3 += _draw(rest, n_c3 - len(c3), rng)
    c3_scaf = set(elig.set_index("inchikey14").scaffold_key.loc[c3])
    c2_pool = set(elig.inchikey14[~elig.scaffold_key.isin(c3_scaf)]) - set(c3)
    c2 = _draw(c2_pool, n_c2, rng)
    return sorted(c3), sorted(c2)


def draw_n(spec_counts, rng, cap=MAX_SPEC):
    """Spectra per molecule, drawn from the test's per-molecule counts, capped at `cap`."""
    return int(min(cap, rng.choice(np.asarray(spec_counts))))


def pick_spectra(g, n, rng):
    """g: candidate rows of one molecule (src_lib, quality-OK) with columns row, adduct, ce (NaN =
    unknown). -> min(n, len(g)) row ids: CE-known rows first -- distinct (adduct, CE rounded to 0.1)
    pairs, then the other CE-known rows -- and CE-unknown rows only to fill."""
    g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    ce = g.ce.to_numpy(np.float64)
    known = g[np.isfinite(ce)]
    unknown = g[~np.isfinite(ce)]
    n = int(min(n, len(g)))
    pick, seen = [], set()
    for r in known.itertuples():
        t = (r.adduct, round(float(r.ce), 1))
        if t not in seen:
            seen.add(t); pick.append(int(r.row))
        if len(pick) == n:
            return pick
    chosen = set(pick)
    for r in list(known.itertuples()) + list(unknown.itertuples()):
        if len(pick) == n:
            break
        if int(r.row) not in chosen:
            chosen.add(int(r.row)); pick.append(int(r.row))
    return pick


def dreams_ft_risk(src_lib, adducts):
    """True when the molecule's spectra come from the DreaMS head's fine-tuning data (MoNA [M+H]+)."""
    return bool(src_lib == DREAMS_FT_LIB and DREAMS_FT_ADDUCT in set(adducts))


# --------------------------------------------------------------------------------------------- #
# Decoder training split
# --------------------------------------------------------------------------------------------- #

def assign_splits(scaf, c3, c2, gmax, seed=SEED, dec_val=DEC_VAL_SHARE):
    """scaf: Series ik14 -> scaffold_key over every corpus ik14. -> Series ik14 -> split label.

    heldout_c3: every member of a pseudo_c3 scaffold group; heldout_c2: the c2_displace structures;
    dec_val: whole random scaffold groups with size <= gmax that hold no cohort structure, added in
    random order while the total stays <= dec_val * U; train: the rest."""
    rng = np.random.default_rng(seed + 1)
    size = scaf.value_counts()
    c3_scaf = set(scaf.loc[list(c3)])
    split = pd.Series("train", index=scaf.index, dtype=object)
    split[scaf.isin(c3_scaf)] = "heldout_c3"
    c2 = list(c2)
    assert not split.loc[c2].eq("heldout_c3").any(), "a c2_displace structure is in a pseudo_c3 group"
    split.loc[c2] = "heldout_c2"
    c2_scaf = set(scaf.loc[c2])
    cand = np.array(sorted(k for k, s in size.items() if s <= gmax and k not in c3_scaf and k not in c2_scaf),
                    dtype=object)
    rng.shuffle(cand)
    target, total, take = dec_val * len(scaf), 0, []
    for k in cand:
        s = int(size[k])
        if total + s <= target:
            take.append(k); total += s
        if total >= target - 0.5:
            break
    split[scaf.isin(set(take))] = "dec_val"
    return split


def check_invariants(split, scaf, c3, c2, cohort_rows, row_ok, row_lib, src_lib_of, msg_ik14):
    """G1.4 -> {invariant: bool}.

    cohort_rows: {ik14: [corpus rows]} over both cohorts; row_ok / row_lib: per corpus row;
    src_lib_of: {ik14: src_lib}."""
    msg = set(np.asarray(msg_ik14).astype(str))
    c3, c2 = set(c3), set(c2)
    c3_members = set(scaf.index[scaf.isin(set(scaf.loc[list(c3)]))])
    ok = {
        "split_labels_valid": bool(split.isin(SPLITS).all()),
        "split_covers_every_ik14_once": bool(split.index.is_unique and set(split.index) == set(scaf.index)),
        "c3_groups_not_in_train_or_dec_val": bool(split.loc[list(c3_members)].eq("heldout_c3").all()),
        "c2_not_in_train_or_dec_val": bool(split.loc[list(c2)].eq("heldout_c2").all()),
        "cohort_rows_quality_ok_from_src_lib": all(
            len(rs) > 0 and all(bool(row_ok[r]) and row_lib[r] == src_lib_of[ik] for r in rs)
            for ik, rs in cohort_rows.items()),
        "no_cohort_ik14_in_msg": not ((c3 | c2) & msg),
        "cohorts_disjoint": not (c3 & c2),
    }
    return ok
