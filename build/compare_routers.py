"""Compare the two Class-3 routers locally: an absolute f.z threshold vs the scale-free gate.

Two notebooks now route the mutator on a Class-3 judgement:

* `casmi-v4n-mutator-bio2-submit.ipynb` fires on `best_pool_fz < 300.0` and promotes to ranks
  1-5.  The threshold is annotated "calibrated: known mols >= 456.0, novel mols <= 265.8".
* `casmi26-v4n-mutator-c3-submit.ipynb` fires on `c3_gate`: a strong mass-shifted spectral analog,
  a weak exact-mass match, and a formula delta within `C3_MAX_STEPS` transforms of that analog.
  Every term is a similarity in [0,1] or a count.

What this script can and cannot settle, stated plainly: an MRR comparison needs the engine's
per-molecule channels, which are a Kaggle/Colab run.  What is decidable offline is **how each
gate's fire count behaves when the f.z scale moves**, and that is the risk that matters here,
because the only artefact the absolute threshold could have been calibrated against is a commit
run -- and a commit run scores a slice of `train.parquet`, not the test set
(`casmi-saved-outputs-are-placeholder`).  `mut_runner`'s own docstring is explicit that it chose a
rank gate because "the f.z scale is not comparable between the visible dummy test and the hidden
test".

A gate reading only [0,1] similarities is invariant to that shift by construction; it is not
immune to being *wrong*, but it cannot silently change how many molecules it touches.

usage: python build/compare_routers.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1 = os.path.join(ROOT, "specs", "track1_retrieval_fusion")
BIO2_THRESHOLD = 300.0
BIO2_SLOTS = 5          # ranks 1-5
C3_SLOTS = 3            # ranks 1-3 (sure) / 2-4 (likely)


def load(name):
    p = os.path.join(T1, name)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    v = np.array([float(x["best_pool_fz"]) for x in d.values()
                  if x.get("best_pool_fz") is not None and np.isfinite(x["best_pool_fz"])])
    return d, v


def rr_loss(slots):
    """RR lost when a correct rank-1 candidate is pushed down by `slots` reserved ranks."""
    return 1.0 - 1.0 / (slots + 1)


def main():
    full = load("pc_lists.json")
    smoke = load("pc_lists_bio.json")
    if full is None:
        print("pc_lists.json not found under specs/track1_retrieval_fusion -- nothing to compare")
        return 1
    d_full, v_full = full

    print("=" * 78)
    print("1. the two f.z distributions this threshold could have been calibrated against")
    print("=" * 78)
    rows = [("pc_lists.json (the real 400)", v_full)]
    if smoke is not None:
        rows.insert(0, ("pc_lists_bio.json (commit artefact)", smoke[1]))
    print("  %-38s%6s%9s%9s%9s%9s" % ("source", "n", "min", "median", "max", "< 300"))
    for lab, v in rows:
        print("  %-38s%6d%9.1f%9.1f%9.1f%8d%s"
              % (lab, len(v), v.min(), np.median(v), v.max(),
                 int((v < BIO2_THRESHOLD).sum()),
                 " (%.1f%%)" % (100 * (v < BIO2_THRESHOLD).mean())))
    if smoke is not None:
        shift = float(np.median(v_full) - np.median(smoke[1]))
        print("\n  median moves by %+.1f between the two, against a threshold of %.1f." % (shift, BIO2_THRESHOLD))
        print("  A commit run has %d molecules (SMOKE_N) and scores a slice of train.parquet, so it"
              % len(smoke[1]))
        print("  cannot contain a novel molecule at all -- the 'novel mols <= 265.8' half of that")
        print("  calibration is measuring training structures.")

    print()
    print("=" * 78)
    print("2. fire count when the f.z scale moves (the hidden test is a different distribution)")
    print("=" * 78)
    q = float((v_full < BIO2_THRESHOLD).mean())      # the quantile the threshold picks out today
    print("  shifting every best_pool_fz by delta, as a scale change would:")
    print("  %8s%26s%26s%16s" % ("delta", "absolute < 300 (bio2)", "same quantile (q=%.3f)" % q, "c3_gate"))
    for delta in (-300, -200, -100, 0, 100, 200, 300):
        v = v_full + delta
        n_abs = int((v < BIO2_THRESHOLD).sum())
        n_q = int(round(q * len(v)))                 # a quantile gate is invariant by construction
        print("  %+8d%20d (%4.1f%%)%20d (%4.1f%%)%16s"
              % (delta, n_abs, 100 * n_abs / len(v), n_q, 100 * n_q / len(v), "invariant"))
    lo = int((v_full - 300 < BIO2_THRESHOLD).sum())
    hi = int((v_full + 300 < BIO2_THRESHOLD).sum())
    print()
    print("  The absolute gate swings between %d and %d molecules over that range. A quantile"
          % (min(lo, hi), max(lo, hi)))
    print("  gate holds at %d by construction, and c3_gate never reads f.z at all, so no scale"
          % int(round(q * len(v_full))))
    print("  change can move it.")

    print()
    print("=" * 78)
    print("3. displacement exposure: fire count x reserved ranks")
    print("=" * 78)
    print("  A reserved rank costs nothing on a true Class-3 molecule -- its reciprocal rank is")
    print("  already 0. The cost is paid when the gate is wrong, so exposure is what to compare.")
    print("  `a` is the pipeline's top-1 accuracy on the fired set, unmeasured until GC.2.\n")
    print("  %-34s%8s%8s%12s%12s%12s"
          % ("router", "fires", "slots", "a=0.02", "a=0.10", "a=0.30"))
    scenarios = [
        ("bio2, absolute < 300", int((v_full < BIO2_THRESHOLD).sum()), BIO2_SLOTS),
        ("c3_gate, depth <= 1 (default)", int(round(0.025 * len(v_full))), C3_SLOTS),
        ("c3_gate, depth <= 2 (probe)", int(round(0.118 * len(v_full))), C3_SLOTS),
    ]
    for lab, n_fire, slots in scenarios:
        losses = ["%+.4f" % (-n_fire * a * rr_loss(slots) / len(v_full)) for a in (0.02, 0.10, 0.30)]
        print("  %-34s%8d%8d%12s%12s%12s" % (lab, n_fire, slots, *losses))
    print("\n  For reference the lineage's worst single regression, forward lambda 2.0, was -0.034.")

    print()
    print("=" * 78)
    print("4. upside, from the pseudo_c3 measurement")
    print("=" * 78)
    print("  q3 = chance a fired molecule ends with the truth in the mutator's top 3, measured on")
    print("  600 leak-free Class-3 natural products under a random screen (the f.z screen should")
    print("  beat random, so these are floors).\n")
    print("  %-34s%8s%8s%10s%12s" % ("band", "fires", "slots", "q3", "gain"))
    for lab, frac, q3 in (("c3_gate, depth <= 1 (default)", 0.025, 0.314),
                          ("c3_gate, depth <= 2 (probe)", 0.118, 0.076)):
        n_fire = int(round(frac * len(v_full)))
        gain = n_fire * q3 * 0.61 / len(v_full)       # 0.61 = mean RR over slots 1,2,3
        print("  %-34s%8d%8d%10.1f%%%+12.4f" % (lab, n_fire, C3_SLOTS, 100 * q3, gain))
    print("\n  Net is gain minus the loss column above at the real `a`. depth <= 1 carries the same")
    print("  upside as depth <= 2 from a quarter of the exposure, which is why it is the default.")

    print()
    print("=" * 78)
    print("verdict")
    print("=" * 78)
    print("  Offline this cannot say which router scores better -- that needs the engine channels.")
    print("  It does say the absolute threshold is the riskier of the two: its fire count is a")
    print("  function of a scale it cannot observe, the only artefact available to calibrate it is")
    print("  a 12-molecule commit run over training structures, and it reserves 5 ranks rather than")
    print("  3. If that gate is kept, making the cut a quantile of the current run's own")
    print("  distribution costs one line and removes the scale dependence entirely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
