# Digest: 20261017-c3-iso, cycle 6 (2026-10-18 20:31)

- idea: `research/c3/20261018-iso-rank` (branch idea/20261018-iso-rank) closed as `dead`
- reason: -
- override: none
- harness: none -; flags: no harness result
- model calls so far: {'sonnet': 1}

## Loop metrics
- hit_rate: 0/1 submitted kept
- calibration_pairs: 2
- novelty: 0/6 ideas tagged novel
- unattended: 4/6 cycles closed without a human

## All cycles
| # | idea | closed as | now | LB | reason |
|---|---|---|---|---|---|
| 1 | 20261017-c3-slot-curve | instrument-ready | instrument-ready |  | no fixed slot pays (s*=7, -0.0009) |
| 2 | 20261017-isomer-mut-sim | dead | dead |  | I1 reach_20 0.073 < 0.10 (reach_all 0.167; A5 pre-screen loses it) |
| 3 | 20261017-gen-rank1-keep | dead | dead | 0.446 (ref 57325719, 2026-10-18; < keep bar 0.451 vs V22 0.447 unpaired) -> DEAD; harness gain was leak-hit only | G4 waived changed=0 (gen1_checked 12, kept 0); submitted ref 57325719 PENDING; K |
| 4 | 20261017-isomer-mut-ice | needs-human | needs-human |  | I1b.0 PASS reach_150 0.140; I1b.1 NOT DEAD marginal reach_20 0.1017 CI 0.083-0.1 |
| 5 | 20261018-iso-rank | spec | dead |  | human grill 2026-10-18: rank isomer children properly; fast-ship in parallel wit |
| 6 | 20261018-iso-rank | dead | dead |  |  |

Submit decisions: `research/submit_queue.md`. Lesson and rule drafts: `research/proposals/`.
