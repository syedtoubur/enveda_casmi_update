# Roadmap

| Step | Content | Status |
|---|---|---|
| S0 | Spec folder, `casmi_scratch` package, tests (`scratch_pkg/tests/run.py`, incl. a notebook dry run), generator | done 10-05 |
| S1a | **R0** Kaggle twin `scratch_pkg/notebooks/casmi_scratch_r0.ipynb` (+ `-nosim` control) → LB | ready to submit |
| S1b | Validation harness (slices I + C, ban arms) on Colab, R0 baseline readings | design discussion pending |
| S2 | **R1** unified pool build (Colab → Kaggle dataset), set K, harness + paired LB | todo |
| S3 | R2+ chosen from the R1 miss breakdown (`kb_index.md`) | todo |

## R0 — port with fixes
Behaviour of `build/simple_sim_submit.py` (LB 0.289) with `MAX_PEAKS=100` (LB 0.296), except:
1. `MAX_PEAKS` drives library similarity, MetFrag-lite **and** the ICEBERG similarity `sv` (was hard-coded 64).
2. No global state; all settings in `casmi_scratch.config.Config`.
3. The pool skips the fingerprint arrays nobody used (mass from RDKit `ExactMolWt` for train structures, as before).
4. Fast commit runs 20 molecules deep (was 30).
5. Stage-C top-up deduplicates on ik14 like stages A/B.

## Harness — open design questions
Settled: two slices (I: 245 enveda-np-examples timsTOF molecules; C: ~400 NP-library molecules, CE known, ≥ 2 spectra,
adduct-matched, MSG-unseen preferred), per-query arms C1/C2/C3 by banning, 40/35/25 weighting at report time,
pool recall + truth-in-simulated + MRR per arm + lever reach + MSG-seen split, ~400 molecules ≤ 1.5 h T4, cache by
(pool, simulator) version. Still to decide with the user before building: slice C's library mix, how a slice-C query's
non-timsTOF spectra are corrected (ppm offset), and the pass/fail rule per arm.
