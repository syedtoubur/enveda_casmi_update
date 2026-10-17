# I1b.1 result (isomer-mut-ice, ICEBERG ranking), pseudo_c3 n=600

State: measured. Verdict per gate block: **not DEAD** (bar: DEAD iff reach_20 by ICE < 0.10). STATUS -> needs-human.
The point estimate clears the bar only barely: reach_20 0.1017, CI90 [0.083, 0.124] straddles 0.10.

Kernels: ICE `syedtoubur/c3iso-i1b1-ice-run` v2 (GPU T4, ice_runner, msg_all MassSpecGym weights); measure `syedtoubur/c3iso-i1b1-measure-run` v1
(CPU, rdkit 2026.03.3 tautomer IK14). Truth was loaded only after ICE scores existed; the ICE kernel has no truth. Query spectra = train.parquet rows
of pseudo_c3 train_rows. Candidates = frozen I1b.0 top-150 children + V22 fused top 10 per molecule.

## Reach among the ICE-ranked children (truth tautomer-IK14)
| rate | all 600 (bar rate; 118 uncovered molecules fall back to CPU-frag order) | ICE-covered 482 |
|---|---|---|
| reach_20 by ICE | 0.1017 (61/600, CI90 0.083-0.124) | 0.1120 (54/482, CI90 0.091-0.138) |
| reach_50 by ICE | 0.1317 (79/600, CI90 0.111-0.156) | 0.1432 (69/482, CI90 0.119-0.171) |
| reach_150 (ceiling) | 0.1400 (84/600, CI90 0.118-0.165) | 0.1494 (72/482, CI90 0.125-0.178) |
| reach_20 by CPU frag (I1b.0) | 0.0600 (36/600, CI90 0.046-0.078) | 0.0602 (29/482, CI90 0.045-0.081) |

- ICE lifts reach_20 from 0.060 to 0.102 (all 600), 36 -> 61 hits; ICE-covered 0.112.
- Truth ICE rank among the 84 reachable: median 11; <=5: 28, <=20: 61, <=50: 79, <=150: 84.
- Truth not in fused top 10 (n 519): reach_20 0.1156 (60/519, CI90 0.094-0.141).
- Hit operators (ICE top 20, 61 hits): {'subst_shift': 31, 'db_shift': 4, 'chain_iso': 13, 'db_shift+db_shift': 1, 'subst_shift+subst_shift': 4, 'methyl_swap': 2, 'group_move': 5, 'chain_iso+chain_iso': 1}.

## I3 fire signal (best child vs best fused-top-10 candidate, by ICE score), pseudo_c3
The best child scores above the best fused-top-10 candidate in 356/482 covered molecules (74%); among the 61 reach_20 hits 44 (72%).
Same rate on hits as on all molecules: the raw "best child beats best retrieved" comparison does not discriminate; a margin or rank threshold is needed (not measured here).
C2 and C1_npx 100-molecule samples: **UNTESTED** (not run this round).

## ICE counters / cost
600 molecules, 482 covered, 92,074 candidates in, scored 73620, null 18454 (all in the 118 uncovered molecules), molecules scored 482/482,
n_pred 238092, n_pred_fail 0, bad smiles 2 (InChI round trip), status ok.
GPU 205 min, 5.99 candidate scores/s (21.99 predictions/s), ICE rdkit 2025.03.6.
Projected hidden test (about 400 molecules): roughly 2.3 h of GPU for ICE alone, plus CPU generation (about 20 s/molecule).

## Audit
The unknown agent's kaggle_run (pruned operators, cap 50, no ICE call, truth used as fallback parent, truth columns read) was not used; kept as *.orig.
Code of record: worktree E:/casmi/lines/c3-isomer-mut-ice commit 9c1ef2b (build/kaggle/).
