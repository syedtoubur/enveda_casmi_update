# Hypothesis: nl-tiebreak-r1frozen

- revives: research/antigravity-exploit/nl-sim-rda
- cause: c1c2-harm. The global λ-weighted z-term inside `gl_multi` reordered confident rank-1 calls: C1_npx Δ −0.0161
  [−0.0283, −0.0055], 8 rank-1 truths displaced; C2 7 displaced (research/antigravity-exploit/nl-sim-rda/necropsy.md:4-5,10,13,
  as quoted in research/ideas/20261018-revivals.md:21-22).
- fix: new mechanism at a new fusion site. Drop the z-term from `gl_multi`. After final fusion, take each group of
  candidates with the same molecular formula inside ranks 2..25 and reorder only within that group by
  `nl_stoich_rda_score`. Rank 1 is frozen and no candidate crosses a formula boundary, so rank-1 displacement is 0 by
  construction. The target stratum moves from "all candidates" to "same-formula isomers below rank 1", where the signal
  discriminated (A1 same-formula pairwise win 58.5%, necropsy.md:7).
- novelty: variant (same signal, new site and stratum)
- source: research/ideas/20261018-revivals.md §1 (necropsy-synth round 1, ranked first)
- branch: post-fusion-tiebreak

## Claim
A rank-1-frozen, within-formula reorder of ranks 2..25 by the stoichiometry/RDA neutral-loss score lifts truths that
sit behind same-formula isomers (C2_hard rank 2-5 bucket: 92/200, loss_map.md:38) without touching any rank-1 call.

## Target cohort
C2_hard (target), "other" stratum Δ is the judged number. C2, C1_npx, C1_xlib, pseudo_c3 are veto cohorts; C2_np advisory.

## Mechanism (code cites)
- Remove: the `nl_stoich_rda_score` z-term added to the fused score in `gl_multi` (dead run, necropsy.md:4-5).
- Add: a default-off post-scoring knob in `build/casmi_post.py` (CLAUDE.md §4: post-scoring changes are casmi_post
  knobs), e.g. `NL_TIEBREAK=false`, applied after final fusion and IK14 dedup: stable sort inside each
  (formula, rank ≥ 2) group by score; candidate positions occupied by a group are reused by the same group, so
  inter-formula order is unchanged. Exact line cites are /spec's job (not openable under this brief).
- Counter (R05): `nl_groups_reordered` (groups whose order changed) must print > 0 on C2_hard.
- Port: the dead run was on the antigravity-exploit line; /spec must confirm whether it was V22 or E006 and port to V22
  (charter Scope). Prerequisite from necropsy.md:12,17: resolve the A4 patcher inconsistency (`b2_patch_runner_text`
  missing) before the build.

## Prior
ASSUMED (source: revivals.md:39-42): the old mechanism read C2_hard "other" +0.0206 (necropsy.md:8), inflated by
cross-formula and rank-1 moves now removed. Estimate C2_hard +0.002..+0.006, C1/C2 0 ± 0.002. LB Δ likely < 0.004
alone.

## Falsifier (pre-registered)
DEAD if C2_hard Δ ≤ 0 or C2_hard "other" Δ ≤ 0; or any non-advisory cohort's CI upper bound < −0.004; or rank-1
displaced > 0 on any cohort (a bug, rank 1 is frozen); or `nl_groups_reordered` ≤ 0 on C2_hard.

## Cheapest measurement
One Kaggle CPU kernel: `harness.py scorecard` on the existing dumps via the harness kit (no new data if
`nl_stoich_rda_score` is recomputable from the dumps, as in the dead run).

## EV
reach C2_hard rank 2-25 same-formula subset (≤ 52% of C2_hard; same-formula share unknown) × +0.004 ASSUMED ×
P(survive) 0.5 ÷ cost 2 (pipeline: port + patcher fix) ≈ 0.0010 — highest in the pool
(#2 taut-rekey ≈ 0.0003, #4 xformula-c3 ≈ 0.0004, #3 netstoich / #5 ce-band ≈ 0, both expected to STOP on power).
