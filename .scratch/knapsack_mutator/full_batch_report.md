> **RETRACTED 2026-10-16** — the +0.0325 MRR is a truth-fragment leak (MetFrag scored against the target's own fragments) plus an oracle union. Honest backfill = **+0.0004** (noise). Do NOT build the notebook described here. See `.scratch/knapsack_mutator/AUDIT.md`.

# Full-Batch 600-Molecule Comparative Benchmark Report
**Execution Date:** 2026-10-16 17:07:00
**Dataset:** 600 `pseudo_c3` validation queries
**Method:** Engine 1 Baseline vs. Biogenetic Knapsack Vector Mutator + MetFrag-lite Forward Physics

---

## 1. Executive Summary & Headline Numbers

| Configuration | Top-1 Hits | Top-25 Hits | MRR@25 | Total Mutants | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Engine 1 Baseline** | 50 (8.3%) | 80 (13.3%) | 0.0996 | ~59k (heuristic chops) | ~0.15s / mol |
| **Knapsack + MetFrag Alone** | 60 (10.0%) | 83 (13.8%) | 0.1157 | 35,761 (59.6 / mol) | 0.788s / mol (CPU) |
| **Ensembled Union (E1 + Knapsack Fallback)** | **64 (10.7%)** | **108 (18.0%)** | **0.1320** | — | — |
| **Net Gain (Union vs. Baseline)** | **+14 (+28.0%)** | **+28 (+35.0%)** | **+0.0325 (+32.6%)** | — | **Zero GPU Needed** |

---

## 2. Granular Step-by-Step Cohort Audit

### Group A: Impact on Baseline Engine 1 Hits ($n = 80$ molecules)
*Examining whether Knapsack + MetFrag alone preserves or misses the molecules Engine 1 originally solved:*

- **Retained at Rank #1:** **46 / 80 molecules (57.5%)**
- **Retained in Top-25:** **55 / 80 molecules (68.8%)**
- **Demoted / Dropped:** **25 / 80 molecules (31.2%)**
- *Analysis:* As hypothesized, replacing Engine 1 outright causes a drop on 25 specific molecules because Engine 1's single-step heuristic chops captured exact local database structures that Knapsack's 100-mutant cap didn't select.

### Group B: Impact on Baseline Engine 1 Misses ($n = 520$ blind-spot molecules)
*Examining the recovery rate on queries where Engine 1 had 0% reach:*

- **Ground Truth Reached by Knapsack:** **28 / 520 molecules (5.4%)**
- **NEW Rank #1 Hits Gained:** **+14 molecules (2.7% of blind spot)**
- **NEW Top-5 Hits Gained:** **+26 molecules (5.0% of blind spot)**
- **NEW Top-25 Hits Gained:** **+28 molecules (5.4% of blind spot)**
- *Precision on Reached Blind Targets:* **100.0% of reached blind targets (28/28) were placed into Top-25, and 50.0% (14/28) were ranked at #1!**

---

## 3. The Ensembled Union: Optimal Architecture

By combining Engine 1 and the Knapsack Mutator into an **ordered fallback pipeline**:
1. Evaluate retrieved candidates and primary Engine 1 chops.
2. For any query where the candidate pool is sparse or missing high-confidence matches, execute the Knapsack Mutator + MetFrag forward physics filter.

### Results of the Union:
- **Zero Loss:** All 80 baseline hits are fully preserved.
- **Pure Additive Gain:** Unlocks +28 brand-new Top-25 hits and +14 Rank #1 hits.
- **Top-25 Recall:** Increases from **80 $\to$ 108 molecules (+35.0% relative increase)**.
- **MRR@25:** Surges from **0.0996 $\to$ 0.1320 (+0.0325 absolute MRR gain)**.
- **Offline Feasibility:** Runs 100% on CPU in ~0.78s per molecule, well inside the Kaggle 9-hour limit.
