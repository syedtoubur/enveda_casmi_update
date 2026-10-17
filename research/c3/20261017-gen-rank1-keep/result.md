# Result: gen-rank1-keep (GEN1_KEEP)

Scorecard (a = V22, b = V22 + GEN1_KEEP=true), target pseudo_c3, `harness_result.json`; per-cohort detail `gen1_report.json`
(script `build/gen1_report.py` in the idea worktree). Verdict from the scorecard: QUEUE, composite +0.0038, 0 displaced.

| cohort | n | V22 MRR | Delta | 90% CI | up/dn | fires | gen1_moved | rank-1 displaced |
|---|---|---|---|---|---|---|---|---|
| pseudo_c3 (target) | 600 | 0.0562 | +0.0100 | [+0.0045, +0.0163] | 10/0 | 37 | 24 | 0 |
| C2 | 400 | 0.5907 | +0.0029 | [0, +0.0088] | 2/0 | 3 | 3 | 0 |
| C1_npx | 245 | 0.9285 | 0 | [0, 0] | 0/0 | 0 | 0 | 0 |
| C2_hard | 200 | 0.5707 | 0 | [0, 0] | 0/0 | 1 | 1 | 0 |
| C1_xlib | 300 | 0.9153 | 0 | [0, 0] | 0/0 | 0 | 0 | 0 |
| C2_np (advisory) | 245 | 0.5432 | 0 | [0, 0] | 0/0 | | | 0 |

## Kept generated BASE #1 on pseudo_c3
- Generated BASE #1: 138 molecules, 55 correct (matches K0).
- Correct generated BASE #1 at final rank 1: V22 11, GEN1_KEEP 21. So 10 are recovered, not the ~44 that post-scoring
  moved away.
- Fires 37, of which 24 actually move g (13 were already rank 1). 10 of the 24 moves are wins, 14 are on wrong g
  (no change in MRR, truth was not at rank 1 either way). Zero molecules lose a rank-1 truth.
- Leak-hit stratum (BASE #1 = truth, n 56) Delta +0.107, other stratum 0.

## Interpretation
- The ICE-agreement guard is very safe (0 rank-1 displacements on every cohort, C1 untouched, 4 C1/C2 fires) but recovers
  only 10 of 44. ICE rarely ranks the generated structure at or above the other top-5 candidates when it is correct; the
  loss is mostly in the guard, not in the keep step.
- Gain is +0.0100 on pseudo_c3 with CI lower bound above 0; C2 +0.0029 (2 up, 0 down, CI includes 0), all within noise
  for C1/C2. pseudo_c3 gain is small in absolute terms (MRR 0.056 to 0.066).
- A2 (MassSpecGym split of C1/C2 fires) is UNTESTED: no msg ik14 list in the repo. Proxy: 4 C1/C2 fires in total, 2 gain.
- A5: on the hidden test FPNet has not seen the spectra, so generated-#1 precision will be lower than on pseudo_c3.

## Tests
`build/test_casmi_post.py`: 5 new GEN1_KEEP tests pass (knob off identical; fire/no-fire conditions incl. tie and no
comparison; dedupe and length; no FPNet read; non-vacuous on the pseudo_c3 dump, 138 gen BASE #1 > 50).
`test_e2r2_row_pure_invariants` fails, inherited from the base (unchanged).
