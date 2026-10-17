# I1b.0 result: isomer-mut-ice, CPU reach with cap 600 and a top-150 freeze (pseudo_c3, n=600)

Bar (pre-registered, fixed): DEAD iff reach_150 < 0.12. **Verdict: not dead. reach_150 = 0.140 -> STATUS `needs-human` (GPU go for I1b.1).**

| metric | value (Wilson 90% CI) |
|---|---|
| reach_all (truth in all <=600 children, plain IK14; I1 cap 300: 16.7%) | 111/600 = 18.5% [16.0, 21.2] |
| **reach_150** (tautomer IK14, frozen frag top 150) | 84/600 = 14.0% [11.8, 16.5] |
| reach_50 (frozen top 50) | 64/600 = 10.7% [8.8, 12.9] |
| reach_20 (frozen top 20; I1 same metric: 7.3%) | 36/600 = 6.0% [4.6, 7.8] |
| truth not in V22 top 25 (n=512): reach_all / 150 / 50 / 20 | 21.5% / 16.2% / 12.3% / 6.8% |
| acyclic truth (n=129): reach_all / 150 / 50 / 20 | 14.7% / 13.2% / 11.6% / 7.8% |
| ring truth (n=471): reach_all / 150 / 50 / 20 | 19.5% / 14.2% / 10.4% / 5.5% |
| timsTOF (n=15): reach_all / 150 / 50 / 20 | 13.3% / 6.7% / 0.0% / 0.0% |

Truth frag-rank among all children (n=111 reachable truths): {'<=20': 36, '21-50': 28, '51-150': 20, '151-300': 17, '>300': 10}, median 39.

Children/molecule: mean 471, capped at 600: 344/600, zero-child molecules 0; deadline hits 24; formula drift 0.

CPU 21.9 s/molecule (gen 19.9, frag 2.0), max 135 s; wall 4915 s for 600 on 3 procs.

## I1b.1 ICEBERG request size

Per molecule: (frozen children, <=150, mean 143.5, plus the V22 fused top 10) x distinct (adduct, collision_energy_orig) settings of the query spectra (NaN CE counts as one value; mean 1.27 settings per molecule: 1 setting 480 molecules, 2: 90, 3: 21, 4: 5, 5: 3, 6: 1).

- **Total requests: 117,660** (86,074 child-molecule pairs + 6,000 fused-top-10 pairs, before the settings multiplication). Adduct-only variant (CE ignored): 111,316.
- The fused-top-10 part (about 5% of the requests) may already be in the base's ICEBERG cache.

## Interpretation (honest)

- The bar is passed on the point estimate (0.140 vs 0.12), but the 90% CI [11.8, 16.5] straddles 0.12: a pass, not a comfortable one. The rule was a point rule, so STATUS is `needs-human`.
- Cap 600 adds little: reach_all 18.5% vs 16.7% (+11 truths). 344/600 molecules still hit the cap, so reach_all stays a lower bound; the operators reach the truth for about 1 in 5 molecules at best.
- The ceiling of any re-ranker of the frozen 150 is 14.0% (84 truths). The I1b.1 bar is reach_20 by ICE >= 10%, so ICEBERG must put about 71% of those 84 into its top 20 (the CPU frag order puts 36/84 = 43% there). That is a demanding ask for a scorer not trained for isomer discrimination; the 150 cut has already lost 27 truths that frag ranks beyond 150.
- Leak effect: 88/600 truths are already in the V22 top 25 and cannot count as reach. On the 512 not-in-top-25 molecules reach_150 is 16.2% and reach_20 6.8%; this is the more informative figure for the C3 population.
- timsTOF 1/15 at reach_150 (uninformative). Hit operators (reach_all): subst_shift 65, chain_iso 16, group_move 11, subst_shift+subst_shift 8, db_shift 4 (+2 double), methyl_swap 3.
- Same operators as I1; the differences are cap, top-N and the freeze. reach_20 by frag is lower than I1 (6.0% vs 7.3%): the deeper child list dilutes the frag order, which also shows frag is a weak ranker here.

## Deviations and caveats
- reach_all uses plain IK14 on all children, reach_150/50/20 use tautomer IK14 on the frozen list; plain IK14 is stricter, so reach_all is slightly conservative.
- Pool 3 (not 4), chunked output (20 chunks of 30). The deadline is enforced per move inside generation and in the selection pass (30 s grace after the deadline so the dedupe/cap pass can finish what was generated); the slowest molecule ran 126 s.
- The first full process exited silently after writing the frozen file (empty .err), so the measurement ran in a separate process (`E:/casmi/c3_iso/i1b_measure_run.py`) on the frozen file with the same code. Cause unknown (likely memory pressure); the frozen data are unaffected.
- CE strings (for example '40(NCE)') are the settings keys; the base worker may merge or split some, so the request total is approximate.
- V22 MRR 0.0562 was not re-measured here (dump and V22 rows are unchanged from I1, which reproduced it).

## Cross-check (orchestrator, 2026-10-18)
The same frozen file was measured independently on Kaggle CPU (`syedtoubur/c3iso-i1b-measure-run` v1, RDKit 2026.03.3):
identical numbers (reach_all 111, reach_150 84, reach_50 64, reach_20 36 of 600). The local measure run above predates
the human rule of 2026-10-17 (heavy compute on Kaggle); future measurements run on Kaggle only. The human's go for
I1b.1 was given 2026-10-17 ("ship the final notebook"); the I1b.1 builder is running `syedtoubur/c3iso-i1b1-ice-run`.
