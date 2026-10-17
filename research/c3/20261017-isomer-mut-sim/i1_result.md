# I1 result: isomer-mut-sim reach (pseudo_c3, n=600) - VERDICT DEAD

Bar (pre-registered, fixed): DEAD iff reach_20 < 0.10 on pseudo_c3.

| metric | value (Wilson 90% CI) |
|---|---|
| reach_all (truth in all <=300 children, plain IK14) | 100/600 = 16.7% [14.3, 19.3] |
| **reach_20** (tautomer IK14 in frozen pre-screen top 20) | **44/600 = 7.3% [5.8, 9.3]** |
| reach_20, truth not in fused top 25 (n=512) | 43/512 = 8.4% [6.6, 10.6] |
| acyclic truth | 12/129 = 9.3% [5.9, 14.4] |
| ring truth | 32/471 = 6.8% [5.1, 9.0] |
| timsTOF (15) | 0/15 = 0.0% [0.0, 15.3] (reach_all 1/15) |
| mean truth rank within the 20 (44 hits) | 8.3 |

reach_20 upper CI bound 9.3% is below 10%: dead even on the optimistic edge, and also dead on the 512 molecules with no truth in the fused list (8.4%).

## Interpretation (honest)
- A5 applies: reach_all 16.7% >= 10% but reach_20 7.3% < 10%. The operator set reaches the truth for 100/600, the CPU fragment
  pre-screen (MetFrag-lite, query measured spectrum) keeps only 44 in its top 20 (mean rank 8.3 of 20). A stronger pre-screen
  (GPU ICEBERG on all ~270 children) could in principle recover up to the 16.7% reach_all; that is the human's call at the checkpoint.
  Ceiling of any scorer on this child set: 16.7% (capped at 300 children; 474/600 molecules hit the cap, 76 hit the 60 s deadline,
  so reach_all is itself a lower bound on a larger child set).
- Operators of the 44 reach_20 hits: {'subst_shift': 25, 'chain_iso': 10, 'db_shift+db_shift': 1, 'methyl_swap': 2, 'db_shift': 2, 'group_move': 4}. subst_shift (regio-isomers) and chain_iso dominate; reach_all hits: {'subst_shift': 61, 'chain_iso': 16, 'db_shift+db_shift': 2, 'methyl_swap': 3, 'subst_shift+subst_shift': 3, 'db_shift': 2, 'group_move': 11, 'chain_iso+chain_iso': 1, 'methyl_swap+group_move': 1}.
- A2 flag: the V22 pseudo_c3 rows carry the truth in the fused top 10 for 81/600 molecules (top 25: 88),
  i.e. the K0 ban does not remove the FPNet/popularity leak (known "leak-hit", MRR 0.0562). Those molecules cannot count as reach (a child equal
  to a retrieved candidate is dropped), so they only lower reach; the 512-molecule not-in-top-25 row is the cleaner number. No child was
  generated from a truth parent *and* counted as the truth.
- Other cohorts (100-molecule seeded samples): C2 truth in fused top 10 for 93/100 (top 25: 96), reach on the 4 outside: 0/4. C1_npx: truth in top 10
  for 100/100, nothing to reach (3 db_shift children counted as reach_20 are tautomer-canonical duplicates of the retrieved truth, not new information). C1_xlib and C2_hard, C2_np not run. These cohorts give little signal: the truth is nearly always already retrieved.
- A6: timsTOF split has only 15 molecules, 0/15 reach_20 (CI up to 15%): uninformative. A7: see runtime.

## Runtime (A7)
CPU per molecule on pseudo_c3: generation 16.7 s + frag pre-screen 1.2 s = 17.9 s
(max 68 s; the 60 s deadline is only checked between parents/L2 expansions, so a molecule can overrun). 400 hidden-test molecules:
about 2.0 h single core, about 0.5 h on 4 cores. Feasible inside 9 h, but moot given the verdict.
Wall for the 600-molecule run on 4 procs: 45 min. Dry run (30 mol): 17.6 s/mol, 4 timeouts.

## Provenance
Query spectra: train.parquet (Kaggle competition file, downloaded to E:/casmi/data/comp_raw/) rows `train_rows` of p1_novel/pseudo_c3.parquet,
the same rows the dump's query came from. timsTOF = pseudo_c3.parquet `timstof`. acyclic = ring count of truth == 0. Frozen top-20 lists were written to
E:/casmi/c3_iso/frozen_pseudo_c3_full.pkl before truths were loaded (generate() vs measure(); a test checks generate() has no truth access).
Sanity: V22 MRR@25 on pseudo_c3 = 0.0562 reproduced. Formula drift: 0 children (all returned children re-verified, plus 0 internal drops).
