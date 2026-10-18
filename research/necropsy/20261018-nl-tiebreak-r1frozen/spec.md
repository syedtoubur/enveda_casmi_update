# Spec: <idea title>

## Problem Statement
## Solution
## User Stories
## Implementation Decisions
- Line / base: <exploit: patch on casmi-notebook-0-433 | explore: scratch_pkg module>
- Files touched: <paths>
## Testing Decisions
- Seam(s): <the highest seam where a test proves the change>

## CASMI section (required; the judge rejects a spec without it)
- **Mechanism**: <what changes in which ranked list>
- **Counters that must move**: <printed counters that prove the change fired, with expected ranges>
- **Arithmetic ceiling**: <best-case ΔMRR, as in spike_result.md>
- **Leakage audit**: <which data trained which model; why no query is FPNet-seen (L01, L02)>
- **Hard rules**: ppm centre 1.45 / InChIKey14 dedupe / offline wheels by cp tag / ≤ 9 h T4 / fast-commit
- **Pre-registered gates** (fill before any measurement):
  | gate | measure | PASS if | BLOCK if |
  |---|---|---|---|
  | G1 | compare.py diff | only cells <list> change | other cells change |
  | G2 | pytest + CPU dry-run | all pass | any fail |
  | G3a | Colab smoke 20 mol | runs, counters move, time projection < 8 h | crash / no-op |
  | G3b | Colab val cohort <name> | no drop > 0.02 vs base | drop > 0.02 |
  | G4 | Kaggle push | kernel COMPLETE, counters printed | error |
- **KEEP bar (LB)**: <paired control + 0.004 (L05)>
- **Rollback**: <how to turn it off: flag / patch revert>

## Out of Scope
