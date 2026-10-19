# Production Plan & Handoff Dossier: Integrating Knapsack Mutator + MetFrag-lite into Base Notebook

**Generated:** 2026-10-16 17:20:00
**Source Conversation ID:** 3db54b36-e3c8-4589-9c3f-7dac07dcd83a
**Target Notebook:** `E:\ENVEDA_casmi\casmi26-e006-window-40-ice-fallback-true.ipynb`
**Destination Notebook:** `casmi26-e006-knapsack-metfrag-fallback.ipynb`
**Comprehensive Master Dossier:** [`e:\ENVEDA_casmi\.scratch\biomet.md`](file:///e:/ENVEDA_casmi/.scratch/biomet.md)

---

## 1. Executive Summary & Verified Empirical Justification
In our full-batch 600-molecule benchmark on `pseudo_c3` validation queries:
- **Baseline Engine 1 Alone:** 50 Top-1 (8.3%), 80 Top-25 (13.3%), MRR@25 = 0.0996.
- **Ensembled Union (Engine 1 + Knapsack Fallback):**
  - **Top-1 Hits:** 50 $\to$ **64 (+14 hits, +28.0% relative increase)**.
  - **Top-25 Hits:** 80 $\to$ **108 (+28 hits, +35.0% relative increase)**.
  - **MRR@25:** 0.0996 $\to$ **0.1320 (+0.0325 absolute MRR gain)**.
  - **Zero Regressions:** Because Knapsack Mutator acts as an **additive fallback** on queries where Engine 1 missed, **0 baseline hits are lost**.
  - **Compute Speed:** 100% offline CPU execution running in $<0.5$s per molecule (~3.5 minutes total inference for all 400 test molecules).

---

## 2. Step-by-Step Architectural Plan for Claude Sonnet Subagent

### Phase 1: Notebook Cloning & Seeding
1. Duplicate `casmi26-e006-window-40-ice-fallback-true.ipynb` to `casmi26-e006-knapsack-metfrag-fallback.ipynb`.
2. Ensure offline compliance: verify all imports utilize only local modules (`rdkit`, `numba`, `numpy`, `pandas`).

### Phase 2: Embedding the Lightweight Knapsack Engine
In the candidate generation utility cell (Cell C6 or helper module):
1. **Define the Compact Biological Basis Set:**
   - 20 essential additions and chops:
     - Additions: `+O` (+15.994915), `+CH2` (+14.015650), `+Hexose` (+162.052823), `+Deoxyhexose` (+146.057909), `+Pentose` (+132.042259), `+Acetyl` (+42.010565), `+Methoxy` (+30.010565), `+H2` (+2.015650).
     - Chops: `-H2` (-2.015650), `-CH2` (-14.015650), `-H2O` (-18.010565), `-CO` (-27.994915), `-CO2` (-43.989829), `-O` (-15.994915), `-Acetyl` (-42.010565), `-Demethoxy` (-30.010565).
   - Precompute depth $\le 3$ integer combinations into a sorted `np.ndarray` for sub-microsecond `np.searchsorted` lookup.
2. **Implement Seed Mutation Function:**
   - Function: `generate_knapsack_fallback(seeds, target_mass, max_mutants=100, ppm=10.0)`
   - Iterates through top retrieved seeds (`fused_top[:5]`), matches mass deficits $\Delta M = M_{target} - M_{seed} \pm 10\text{ ppm}$, applies RDKit reaction SMARTS, sanitizes molecules, and deduplicates on canonical InChIKey14.

### Phase 3: Embedding MetFrag-lite Forward Physics Scorer
In the physics scoring / reranking cell (Cell C8 / C10):
1. **In-Silico Cleavage:**
   - Implement `fragment_masses(smiles, max_breaks=2)` using RDKit bond cuts.
2. **Explained Intensity Scoring:**
   - Implement `explain_score(frag_mass, peak_mz, peak_int, mode, tol=0.01)` to calculate the fraction of experimental daughter ion intensity explained by each candidate.
3. **Candidate Ordering & Splicing:**
   - For any query molecule where the retrieved candidate count is $< 25$ or where top candidate confidence is low:
     - Generate mutants via `generate_knapsack_fallback`.
     - Score mutants via `explain_score`.
     - Sort descending by explained intensity.
     - Backfill vacated slots (up to 25 total candidates) with the highest-scoring mutants, ensuring no duplicate InChIKey14s are emitted.

### Phase 4: Fast-Commit Guardrail
1. Wrap deep execution in `KAGGLE_IS_COMPETITION_RERUN` check:
   - If draft/commit mode: run quick smoke test on first 10 molecules to ensure zero crashes and instant feedback.
   - If hidden competition rerun: execute across 100% of test molecules.

---

## 3. Multi-Agent Delegation & Verification Workflow
In the fresh session:
1. **Delegation (Claude Sonnet Subagent via Claude CLI):**
   - Claude Sonnet is tasked with reading the plan and modifying the notebook cells in `casmi26-e006-knapsack-metfrag-fallback.ipynb`.
2. **Analysis & Verification (Antigravity):**
   - Antigravity runs `python .agents/skills/analyze-notebook/scripts/compare.py` to inspect cell-by-cell semantic diffs between the baseline and updated notebook.
   - Runs syntax checks and dry-run validation on the modified cells.
   - Verifies that offline execution constraints, memory limits, and timing thresholds are strictly satisfied.
3. **Human Review & Push:**
   - Antigravity presents the verified changes and bug-free status to the user.
   - User reviews, commits, and pushes to Kaggle for submission.

---

## 4. Boot Command for Fresh Session
To begin the building and verification process with maximum reasoning context:
> **`@.scratch/handoffs/latest.md continue`**
