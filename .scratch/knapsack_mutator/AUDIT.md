# AUDIT 2026-10-16 — Knapsack + MetFrag claim RETRACTED

Reproduce: `python .scratch/knapsack_mutator/audit_honest.py` and `audit_zero_delta.py` (run from repo root).

| config (600 pseudo_c3) | MRR@25 |
|---|---|
| base (`state['base']`, E006 BASE, not "Engine 1") | 0.0996 |
| backfill: base top-25 then knapsack (generation order) | **0.1000 (+0.0004)** |
| backfill, random knapsack order (50 reps) | 0.0997 |
| knapsack #1 inserted at slot 1 / 2 | 0.0679 / 0.0988 |
| reported "Union" | 0.1320 — invalid |

1. **Truth leak in MetFrag** — `full_batch_benchmark.py:_score_molecule_job` and `metfrag_eval.py` score candidates against
   `fragment_masses(target_smi)` (the TRUE structure's in-silico fragments), not the real MS2. The truth explains itself → rank 1.
2. **Oracle union** — `union_ranks` picks base rank if base hit, else knapsack rank; that choice needs the truth.
   Only 6/28 "blind gains" have a base list < 25, the proposed inference trigger.
3. **Multi-step premise never exercised** — seeds `fused_top[:5]` sit at the query mass; every one of the 28 gains has
   seed ΔM = 0.0000 and comes from a cancelling depth-2 pair (hydroxyl+dehydroxyl, methoxy+demethoxy, ...). It is a regioisomer shuffler.
4. **Truth already in seeds** for 68/600 (56 of the 83 knapsack reaches, 62 of the 80 base hits).
5. Minor: docs say 20 basis vectors / list -H2O, -CO (code: 24 rules, neither implemented); plain InChIKey not tautomer-canonical;
   target not timsTOF-corrected; runtime 0.79 s/mol, not < 0.5.

Untested and still open: knapsack basis on **Engine 2 banned-pool anchors** (`eng_lists.json.c3.json[mid]['ana']`), reach only.
See plan: `C:\Users\h\.claude\plans\scratch-biomet-md-and-scratch-handoffs-smooth-moler.md` (Step 1).
