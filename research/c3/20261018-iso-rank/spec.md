# Spec: iso-rank (c3-iso campaign, human grill 2026-10-18)

Plan: `~/.claude/plans/scratch-handoffs-handoff-20261018-c3-ov-floating-zebra.md`. Gate block: gate_report
`### 2026-10-18: c3-iso iso-rank, PRE-REGISTERED`. Predecessor `../20261017-isomer-mut-ice` (reach_150 0.140; reach_20 by
ICE 0.1017; I3 trigger "best child > best retrieved" fires on 74% of molecules, so it doesn't discriminate).
Worktree `E:/casmi/lines/c3-iso-rank`, branch `idea/20261018-iso-rank` (off 9c1ef2b). Knob `ISO_MUT` (default off).

## Claim
The isomer children already reach Class-3 truths. Ranking them by a truth-blind RRF of orthogonal signals, and inserting
the best one **directly below its own parent** only when it beats that parent on ICE AND frag, gains on pseudo_c3. It also
costs little on C1/C2: there the parent is usually the truth at rank 1, and a child placed below it displaces only the
candidates below.

## Components (all truth-blind; truth only scores)
- **Children:** `build/isomer_mut.py` unchanged (`mutate_many`, cap 600, 120 s deadline) on the V22 fused top 10 (the
  parents). Plain-IK14 dedupe; drop children whose IK14 is already in the V22 list. CPU frag (`build/frag_rescore.py`,
  measured spectrum) cuts to the top K (offline K = 150).
- **Parent map:** child IK14 → set of (parent fused rank, operator). parent(child) = the parent with the best fused rank;
  multiplicity = number of distinct (parent, operator) pairs.
- **Lists** (each ranks every child of one molecule; a list missing for a child, e.g. no ICE coverage or no GLACIER adduct,
  is simply absent from that child's RRF sum):
  1. ICE cosine vs the measured spectra, aggregated as the base does (`build/iceberg_worker.py`);
  2. CPU frag score;
  3. multiplicity (higher is better);
  4. parent fused rank (lower is better);
  5. NL: `score_molecule(spectra, cand_smiles)` from `research/antigravity-exploit/nl-sim-rda/nl_stoich_rda.py`, ported as
     `build/iso_nl.py` (exact-mass diagnostic neutral losses + 6-ring RDA; measured spectrum only, R10);
  6. GLACIER cosine on `[M+H]+`/`[M+Na]+` spectra (`gl_runner.py`, dataset `ahmedberatozer/casmi26-glacier`).
- **RRF:** score(child) = Σ_lists 1/(k + rank_list(child)), k = 10, ranks 1-based, ties take the average rank.
- **Fire rule** (per molecule, on the RRF top child c with parent p): fire iff ICE(c) and ICE(p) both exist AND
  ICE(c) − ICE(p) > m_ice AND frag(c) > frag(p). If p is not in the final top 25 row, there is no fire.
- **Placement:** when fired, c goes at final position pos(p) + 1, the rest shift down, and the row is cut at 25.
  Always (fired or not), ranks 21–25 are replaced by the next RRF children not already in the row (tail ride-along).
  Dedupe by IK14; never rank 1 (a parent at rank 1 puts c at rank 2 at best).
- **Counters:** `iso_children` (molecules with ≥ 1 child), `iso_fired`, `iso_tail`, `iso_skipped_time`. Wall-clock guard:
  stop processing new molecules once elapsed time > 8.3 h; their rows stay unchanged.

## Configs
- **Fast-ship (frozen a priori):** lists 1–6 (GLACIER where available), m_ice = 0, k = 10, K = 150 if the projection
  ≤ 8.5 h (V22 ≈ 4.5 h + mutation ≈ 0.6 h + ICE ≈ 2.3 h + GLACIER from a smoke), else K = 50.
- **Offline (Stage A/B):** list set forward-selected on scaffold half A (keep a list iff within-pool MRR@20 on half A
  rises), then frozen; m_ice chosen on half A from the grid {0, 0.01, 0.02, 0.05, 0.1}, then frozen; verdict on half B.

## Out of scope
FPNet anywhere in the child path; a separate C3 detector; rank-1 inserts; retuning anything against the LB.
