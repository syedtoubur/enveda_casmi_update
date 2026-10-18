# Campaign report: 20261017-c3-iso

Stopped 2026-10-18 20:48: replan picked no idea 2 times

Cycles 3, probes pushed 1, submitted 1, model calls {'sonnet': 1, 'claude:sonnet': 1, 'claude:opus': 2}, closed branches none.

| idea | closed as | now | LB | branch | reason | override |
|---|---|---|---|---|---|---|
| research/c3/20261017-c3-slot-curve | instrument-ready | instrument-ready |  | c3-route | no fixed slot pays (s*=7, -0.0009) |  |
| research/c3/20261017-isomer-mut-sim | dead | dead |  | c3-iso | I1 reach_20 0.073 < 0.10 (reach_all 0.167; A5 pre-screen loses it) |  |
| research/c3/20261017-gen-rank1-keep | dead | dead | 0.446 (ref 57325719, 2026-10-18; < keep bar 0.451 vs V22 0.447 unpaired) -> DEAD; harness gain was leak-hit only | c3-route | G4 waived changed=0 (gen1_checked 12, kept 0); submitted ref 57325719 PENDING; KEEP iff LB>=0.451 and hidden gen1_kept>0 | g4_min_changed waived by orchestrator |
| research/c3/20261017-isomer-mut-ice | needs-human | needs-human |  | c3-iso | I1b.0 PASS reach_150 0.140; I1b.1 NOT DEAD marginal reach_20 0.1017 CI 0.083-0.124 vs bar 0.10; I3 pending (trigger non-discriminating, C1/C2 untested) |  |
| research/c3/20261018-iso-rank | spec | dead |  | c3-iso | human grill 2026-10-18: rank isomer children properly; fast-ship in parallel with Stage A |  |
| research/c3/20261018-iso-rank | dead | dead |  | idea/20261018-iso-rank |  |  |

## Agenda for the next grill
- New preflight rules since the charter: `python research/bin/preflight.py --selftest` (veto any).
- Open questions: every questions.md of this campaign's ideas.
- LB results: `python research/bin/poll_lb.py` DECIDE lines for the pushed probes.
