"""Class-3 detection for the mutator channel (specs/track1_retrieval_fusion §4).

The mutator used to fire on a fixed share of the field (`MUT_FRAC = 0.30`, the weakest
`best_pool_fz`).  Three things were wrong with that, and all three are fixed here:

1. A **rank** gate always fires on exactly 30% of the molecules, even when none is novel.  It
   cannot express "nothing qualifies today".  This gate is a conjunction of thresholds, so firing
   on zero molecules is a legal outcome.
2. `best_pool_fz` scores the wrong database.  It is the best `f.z` over train + COCONUT, while
   Class 3 means *absent from PubChem*.  Missing from the pool but present in PubChem is the
   textbook definition of Class **2**, which retrieval already handles at MRR ~0.50-0.65.
3. It was negative evidence only -- "everything scores badly" -- which cannot separate a novel
   molecule from an uninformative spectrum.  A sparse spectrum depresses every `f.z` in the
   pipeline, and those molecules are exactly where the ICEBERG/GLACIER blend is least reliable.

The signal that fixes (3) is **positive**: a strong *mass-shifted* spectral analog together with a
weak *exact-mass* match -- "I recognise this spectral family at a mass offset, but nothing at this
exact mass explains it".  `casmi_engine.molecule_channels` already computes both.  A bad spectrum
cannot produce a high analog similarity, so requiring one *is* the spectrum-quality control.

It carries a second meaning that turns out to matter more.  Measured on `p1_novel/pseudo_c3.parquet`
(600 leak-free Class-3 natural products), **319 of 600 are the only structure in the 194,668-molecule
universe carrying their Murcko scaffold** -- for those, no retrieval list can ever supply the
scaffold the mutator needs, and 47 of the 72 oracle-conditions hits fall in that group.  Cutting to
molecules whose scaffold is actually harvestable drops recovery from the headline 17.2% to **4.2%**
of the cohort.  But cutting *further* to deltas reachable within **2** transformations gives
**28.2%** (20 of 71) -- 20 of the 25 production-viable hits live in that narrow band.

A strong analog is the inference-time evidence that a scaffold relative exists *and was found*, and
the analog's mass shift gives the step count.  So the gate and the mutator's sweet spot are the same
object, which is the whole reason this is worth gating on rather than tuning.

`MAX_STEPS` is 1, not 2, and that is measured rather than cautious.  Promotion needs the truth in
the *top 3* of the mutator list, not merely inside it, and shallow deltas generate far fewer
candidates to choose between.  Over `pseudo_c3`, restricted to harvestable scaffolds, with `q3` the
chance of landing a recovered truth in the top 3 under a random screen:

    band        fires on   truth generated   q3 | fired   candidates (median)
    depth <= 1      2.5%            80.0%        31.4%                     16
    depth <= 2     11.8%            28.2%         7.6%                     24
    depth <= 6     40.0%            10.4%         2.3%                     75

Depth 1 buys the same expected gain as depth 2 from a quarter of the exposure, so the downside if
the gate is wrong is a quarter as large.  `C3_MAX_STEPS=2` remains available for a later probe once
GC.2 has measured what the pipeline actually scores on the fired set.

Everything here is a similarity in [0, 1] or a count.  Nothing is an `f.z` value: `mut_runner`
chose a rank gate originally because "the f.z scale is not comparable between the visible dummy
test and the hidden test", and a commit run scores a slice of `train.parquet`, so no absolute `f.z`
cut can be calibrated in advance.  The thresholds below are provisional pending GC.3 and are
overridable from the environment so the notebook can move them without a code edit.
"""
from __future__ import annotations

import collections
import os

# --- thresholds.  Provisional pending GC.3 (fire rate on pseudo_c3 vs c2_displace + C1).
LIB_TAU = float(os.environ.get("C3_LIB_TAU", 0.90))        # the pipeline's own Class-1 cut
ANA_MIN = float(os.environ.get("C3_ANA_MIN", 0.50))        # family recognised at all
ANA_SURE = float(os.environ.get("C3_ANA_SURE", 0.70))      # family recognised strongly
MARGIN_MIN = float(os.environ.get("C3_MARGIN_MIN", 0.15))  # analog beats every same-mass candidate
MARGIN_SURE = float(os.environ.get("C3_MARGIN_SURE", 0.30))
MAX_STEPS = int(os.environ.get("C3_MAX_STEPS", 1))         # measured: see the table below
ICE_MAX = float(os.environ.get("C3_ICE_MAX", 0.60))        # a candidate simulated this well explains the spectrum

SURE, LIKELY = "sure", "likely"

# The ranks handed to the mutator per tier, for `merge(base, base_keys, mut, mut_keys, slots)`.
# Nearly all displacement risk sits in slot 1: a retrieval truth at rank 1 pushed to rank 4 costs
# 0.75 RR, while every deeper shift costs an order of magnitude less.  So SURE takes slot 1 only
# once GC.2 has measured the pipeline's top-1 accuracy on the fired set at <= 0.05; until then
# set SLOTS[SURE] to the LIKELY pattern.  See §6 of the plan.
SLOTS = {SURE: [1, 2, 3], LIKELY: [2, 3, 4]}


def classify(lib_max, exact_max, ana_max, delta_steps, ice_max=None, reasons=None):
    """-> "sure" | "likely" | None, naming the veto in `reasons` when it declines.

    lib_max      best spectral *library* match for this molecule (BASE[m][2], [0,1])
    exact_max    best *exact-mass* library similarity in the +-10 ppm window (max of
                 `rec['lib']`, [0,1]); 0.0 when the window was empty
    ana_max      best *mass-shifted* analog similarity over +-200 Da (`rec['ana'][0][1]`,
                 [0,1]); 0.0 when no analog was found
    delta_steps  minimum palette steps realising the analog -> target formula delta, or None
                 when the delta is unreachable, negative, or there was no analog
    ice_max      best ICEBERG/GLACIER agreement over the retrieved candidates, or None when not
                 yet known.  The mutator cell runs before ICE/GL so it passes None; the fusion
                 cell re-runs the gate with it.  None never vetoes.

    Terms are ordered cheapest-first and the first failure wins, so `reasons` reads as a funnel
    rather than a set of overlapping counts.
    """
    r = reasons if reasons is not None else collections.Counter()

    if lib_max is not None and lib_max >= LIB_TAU:
        r["class1_library_match"] += 1           # solved elsewhere at >90%; never touch it
        return None
    if ana_max is None or ana_max < ANA_MIN:
        r["no_analog_evidence"] += 1             # also the bad-spectrum veto: see the docstring
        return None
    if delta_steps is None:
        r["delta_unreachable"] += 1
        return None
    if delta_steps > MAX_STEPS:
        r["delta_too_deep"] += 1                 # >=3 steps recovered 3.7%, >=5 recovered 0%
        return None

    margin = ana_max - (exact_max or 0.0)
    if margin < MARGIN_MIN:
        r["exact_match_explains_it"] += 1        # something at this mass matches: Class 1 or 2
        return None
    if ice_max is not None and ice_max >= ICE_MAX:
        r["forward_sim_explains_it"] += 1        # a retrieved candidate reproduces the spectrum
        return None

    if ana_max >= ANA_SURE and margin >= MARGIN_SURE:
        r["fired_sure"] += 1
        return SURE
    r["fired_likely"] += 1
    return LIKELY


# ================================================================== v2 (Track 4 c3v2 plan, I0/I4)
#   Stage 1 runs in mut_runner after generation; stage 2 runs in the fusion cell once ICEBERG and
#   GLACIER scores exist; `cap_fires` bounds the exposure across the whole run.  The v1 `classify`
#   above is untouched.
#
#   The f.z terms here are *within-molecule* comparisons -- every number shares the query's own
#   predicted-fingerprint vector z -- so none of them is an absolute cut.  That is the difference
#   from the bio2 gate (`best_pool_fz < 300`, -0.064): its threshold compared f.z ACROSS molecules,
#   whose scale moved by +172 between the commit slice and the hidden test.

MAX_STEPS_V2 = int(os.environ.get("C3_MAX_STEPS_V2", 2))   # anchored + exact-mass delta
SEP_SURE = float(os.environ.get("C3_SEP_SURE", 1.0))       # best edit stands out from its regioisomers
ICE_SLACK = float(os.environ.get("C3_ICE_SLACK", 0.05))
SURE_FRAC = float(os.environ.get("C3_SURE_FRAC", 0.03))
FIRE_FRAC = float(os.environ.get("C3_FIRE_FRAC", 0.12))
DISABLE_FRAC = float(os.environ.get("C3_DISABLE_FRAC", 0.25))
ALLOW_RANK1 = os.environ.get("C3_ALLOW_RANK1", "1") == "1"


def stage1_v2(lib_exact, exact_max, ana_max, delta_steps, fz_best, fz_anchor, best_pool_fz,
              fz_top=None, fz_sep=None, reasons=None):
    """Stage-1 tier after generation -> "sure" | "likely" | None.

    lib_exact     best exact-window library similarity from engine A ([0,1]); the real v4n
                  lib_max is not built yet, so stage 2 re-checks with it
    exact_max     best exact-mass similarity ([0,1]); margin = ana_max - exact_max
    ana_max       similarity of the anchor analog ([0,1])
    delta_steps   signed moves realising the anchor's exact mass delta, None if no match
    fz_best       f.z of the best generated candidate        } all with the SAME query z,
    fz_anchor     f.z of the anchor it was grown from        } so only differences matter
    best_pool_fz  best f.z over the pool's exact-mass window }
    fz_top        best f.z over the PubChem-only list (raw, NOT the popularity-mixed pc_fz)
    fz_sep        (fz_best - median f.z of all candidates) / their std; None if < 3 candidates

    The first failure wins, so `reasons` reads as a funnel.
    """
    r = reasons if reasons is not None else collections.Counter()
    if lib_exact is not None and lib_exact >= LIB_TAU:
        r["class1_library_match"] += 1
        return None
    if ana_max is None or ana_max < ANA_MIN:
        r["no_analog_evidence"] += 1
        return None
    if delta_steps is None:
        r["delta_unmatched"] += 1
        return None
    if delta_steps > MAX_STEPS_V2:
        r["delta_too_deep"] += 1
        return None
    margin = ana_max - (exact_max or 0.0)
    if margin < MARGIN_MIN:
        r["exact_match_explains_it"] += 1
        return None
    if fz_best is None or fz_anchor is None:
        r["no_candidates"] += 1
        return None
    if fz_best <= fz_anchor:
        r["no_fp_improvement"] += 1               # I0: the edit must move toward the query's fp
        return None
    if best_pool_fz is not None and fz_best < best_pool_fz:
        r["pool_explains_fp"] += 1                # a database structure fits the fp better
        return None
    sure = (ana_max >= ANA_SURE and margin >= MARGIN_SURE
            and (fz_top is None or fz_best >= fz_top)
            and fz_sep is not None and fz_sep >= SEP_SURE)
    r["s1_sure" if sure else "s1_likely"] += 1
    return SURE if sure else LIKELY


def classify_v2(tier1, lib_max, top1_agree, gen_ice=None, ret_ice=None, gen_gl=None, ret_gl=None,
                reasons=None, ice_unit=True):
    """Stage 2, in the fusion cell -> "sure" | "likely" | None.

    tier1       stage-1 tier
    lib_max     the real v4n library match for this molecule (BASE[m][2])
    top1_agree  v4n top-1 and engine-A top-1 are the same structure (IK14)
    gen_ice     best ICEBERG similarity over the generated candidates;  ret_ice: of the retrieved
    gen_gl      the same for GLACIER;                                    ret_gl:  top-1
                (None when not scored -- ICE runs under a time budget, GLACIER on [M+H]+ only)
    ice_unit    the run's ICE scores were verified to lie in [0,1].  Only then is the absolute
                ICE_MAX veto meaningful; the gen-vs-retrieved comparisons are scale-free either way.

    `sure` needs FPNet (stage 1), ICEBERG and, where it ran, GLACIER to prefer the generated
    structure, plus two retrieval engines that disagree.  Anything short of that is at most
    `likely`, which never touches rank 1.
    """
    r = reasons if reasons is not None else collections.Counter()
    if tier1 is None:
        return None
    if lib_max is not None and lib_max >= LIB_TAU:
        r["s2_class1_library_match"] += 1
        return None
    if ice_unit and ret_ice is not None and ret_ice >= ICE_MAX:
        r["s2_forward_sim_explains_it"] += 1
        return None
    if gen_ice is not None and ret_ice is not None and gen_ice < ret_ice - ICE_SLACK:
        r["s2_ice_prefers_retrieved"] += 1
        return None
    tier = tier1
    if tier == SURE:
        ice_ok = gen_ice is not None and ret_ice is not None and gen_ice > ret_ice
        gl_ok = gen_gl is None or ret_gl is None or gen_gl > ret_gl
        if top1_agree or not ice_ok or not gl_ok or not ALLOW_RANK1:
            r["s2_sure_demoted"] += 1
            tier = LIKELY
    r["s2_" + tier] += 1
    return tier


def cap_fires(rows, n_total, sure_frac=None, fire_frac=None, disable_frac=None):
    """Bound the run-wide exposure.  rows: [(mid, tier, confidence)].

    -> ({mid: tier or None}, stats).  If more than `disable_frac` of the run would fire, the gate
    is miscalibrated for this test set and the channel switches itself off (bio2 fired on enough
    molecules to cost -0.064); otherwise the excess over `sure_frac` is demoted to likely and the
    excess over `fire_frac` is declined, lowest confidence first.  Fractions are of `n_total`, so
    the caps are scale-free."""
    sure_frac = SURE_FRAC if sure_frac is None else sure_frac
    fire_frac = FIRE_FRAC if fire_frac is None else fire_frac
    disable_frac = DISABLE_FRAC if disable_frac is None else disable_frac
    st = collections.Counter()
    fired = [(m, t, c) for m, t, c in rows if t is not None]
    out = {m: None for m, _, _ in rows}
    st["requested"] = len(fired)
    if n_total and len(fired) > disable_frac * n_total:
        st["disabled"] = 1
        return out, st
    fired.sort(key=lambda x: -x[2])
    n_sure = n_fire = 0
    max_sure, max_fire = int(sure_frac * n_total), int(fire_frac * n_total)
    for m, t, c in fired:
        if n_fire >= max_fire:
            st["cap_declined"] += 1
            continue
        if t == SURE:
            if n_sure < max_sure:
                n_sure += 1
            else:
                t = LIKELY
                st["cap_demoted"] += 1
        out[m] = t
        n_fire += 1
        st[t] += 1
    return out, st


def delta_of(target_vec, seed_vec):
    """`target - seed` in formula space, elementwise; None when either side is unknown."""
    if target_vec is None or seed_vec is None:
        return None
    return tuple(t - s for t, s in zip(target_vec, seed_vec))


def heavy_negative(delta, elements):
    """True when `delta` needs a non-hydrogen atom removed -- unreachable by an additive palette.

    Hydrogen may go negative (`-2H` is in the palette); any other negative element means the seed
    is not a formula-space substructure of the target and no additive route exists.
    """
    return any(v < 0 for v, e in zip(delta, elements) if e != "H")


def min_steps(delta, reach, elements=None):
    """Minimum palette steps realising `delta`, or None if unreachable.

    `reach` is `delta_m_mutator.build_reach()`: {delta vector: (decomposition, ...)}, shortest
    first.  Kept here so one place defines "depth" and `classify` stays free of RDKit.
    """
    if delta is None:
        return None
    if elements is not None and heavy_negative(delta, elements):
        return None
    decs = reach.get(tuple(delta))
    return None if not decs else min(len(d) for d in decs)
