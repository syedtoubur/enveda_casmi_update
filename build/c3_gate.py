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
