# Transductive Molecular Networking: Test-Train Scaffold Architecture & Empirical Falsifier

> **Target Competition**: Enveda CASMI 2026 — Molecule ID From Mass Spectra  
> **Official Metric**: Mean Reciprocal Rank at 25 (`MRR@25`) on tautomer-canonical `InChIKey14` (RDKit 2026.03.3)  
> **Primary Goal**: Break through the 0.000 MRR ceiling on **Class 3 (completely novel natural products)** without cannibalizing Class 1 and Class 2 retrieval precision.  
> **Core Concept**: Pre-cluster the 2.54M training spectra in `train.parquet` by unique Bemis-Murcko scaffolds into a compact centroid library. Use Shift-Aware Modified Cosine & Neutral Loss to detect which training scaffold cluster an unknown test query belongs to, then synthesize novel candidates using a constrained biogenetic side-chain palette.

---

## Part 1: Empirical Evaluation Results (`.scratch/kaggle_scaffold_eval/`)

To test the physical foundation of this idea, we designed and executed an empirical falsification benchmark directly on the official competition training spectra in Kaggle ([`syedtoubur/casmi26-scaffold-network-eval`](https://www.kaggle.com/code/syedtoubur/casmi26-scaffold-network-eval)).

### 1.1 Dataset & Scale
- **Dataset**: `train.parquet` (2,539,608 spectra, 275,810 structures).
- **Execution Platform**: Kaggle Environment (CPU, 76.7 s total execution time).
- **Cohort Scale**: **10,457 pre-stratified pairs** across **13,512 unique spectra**.
- **The 4 Treatment Arms**:
  1. **Arm 1 (Same Scaffold General, N=3,000)**: Pairs of different molecules sharing the same Murcko scaffold with unconstrained mass difference.
  2. **Arm 2 (Same Scaffold + Biogenetic Delta, N=1,500)**: Pairs sharing the same scaffold where $\Delta m = |m_A - m_B|$ matches one of 17 known biogenetic mass defects ($\pm 14.0157\text{ Da } [\text{CH}_2]$, $\pm 15.9949\text{ Da } [\text{O}]$, $\pm 162.0528\text{ Da } [\text{hexose}]$, etc.) within $\pm 5\text{ ppm}$.
  3. **Arm 3 (Different Scaffold Isobaric Decoys, N=2,957)**: Precursor mass within $\pm 20\text{ ppm}$ with different Murcko scaffolds (direct retrieval competitors).
  4. **Arm 4 (Different Scaffold Random Decoys, N=3,000)**: Random background decoy pairs.

---

### 1.2 Summary of Empirical Findings

| Metric / Question | Measured Value | Standard Benchmark | Verdict |
| :--- | :---: | :---: | :---: |
| **Separation AUC (Mod-Cosine vs. Isobaric Decoys)** | **0.9483** | $\ge 0.80$ required | **MASSIVE WIN (PASS)** |
| **Cohen's $d$ Effect Size** | **3.0957** | $\ge 1.20$ required | **EXTREME SEPARATION** |
| **Shift-Aware Lift ($\Delta$AUC over Direct Entropy)** | **+0.1103** | $\ge +0.10$ required | **PROVES SHIFT-AWARE IS MANDATORY** |
| **Biogenetic Congener Recall at $\tau = 0.70$** | **51.87%** | $\ge 30.0\%$ required | **PASS** |
| **Isobaric Decoy False Lock Rate at $\tau = 0.70$** | **0.44%** | $\le 2.0\%$ required | **PASS (< 0.5% contamination)** |
| **Biogenetic Edge Precision at $\tau = 0.70$** | **98.36%** | $\ge 75.0\%$ required | **PASS (FDR = 1.64%)** |

---

### 1.3 Quantile Distribution Table (10,457 Pairs)

| Treatment Arm | Similarity Metric | p05 | p10 | p25 | p50 (Median) | p75 | p90 | p95 | p99 | Mean | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 2: Same Scaffold + Bio-Delta** | **Modified Cosine** | **0.0341** | **0.1531** | **0.4542** | **0.7114** | **0.8617** | **0.9451** | **0.9720** | **0.9934** | **0.6325** | **1.0000** |
| Arm 2: Same Scaffold + Bio-Delta | Raw Direct Cosine | 0.0000 | 0.0121 | 0.1228 | 0.3852 | 0.6856 | 0.8564 | 0.9211 | 0.9898 | 0.4188 | 1.0000 |
| Arm 2: Same Scaffold + Bio-Delta | Spectral Entropy Similarity | 0.0000 | 0.0000 | 0.0624 | 0.3170 | 0.5453 | 0.6876 | 0.7683 | 0.9158 | 0.3267 | 0.9999 |
| Arm 2: Same Scaffold + Bio-Delta | Neutral Loss Cosine | 0.0000 | 0.0000 | 0.0440 | 0.1997 | 0.4994 | 0.7253 | 0.8516 | 0.9605 | 0.2912 | 0.9948 |
| **Arm 1: Same Scaffold General** | Modified Cosine | 0.0000 | 0.0000 | 0.1003 | 0.4002 | 0.7138 | 0.8738 | 0.9345 | 0.9842 | 0.4196 | 1.0000 |
| **Arm 3: Diff Scaffold Isobaric** | **Modified Cosine** | **0.0000** | **0.0000** | **0.0000** | **0.0118** | **0.0633** | **0.1546** | **0.2309** | **0.5078** | **0.0535** | **1.0000** |
| Arm 3: Diff Scaffold Isobaric | Spectral Entropy Similarity | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0293 | 0.1355 | 0.2126 | 0.4210 | 0.0383 | 1.0000 |
| **Arm 4: Diff Scaffold Random** | Modified Cosine | 0.0000 | 0.0000 | 0.0000 | 0.0109 | 0.0648 | 0.1440 | 0.2145 | 0.3718 | 0.0477 | 0.9078 |

---

### 1.4 Operating Characteristics Across Thresholds ($\tau$)

| Threshold ($\tau$) | Arm 2 Bio Recall (%) | Isobaric Decoy FLR (%) | Bio-Delta Precision (%) | Direct Entropy Recall (%) | Direct Entropy Iso FLR (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.90** | 18.33% | 0.17% | **98.21%** | 1.53% | 0.10% |
| **0.85** | 27.27% | 0.20% | **98.55%** | 2.13% | 0.14% |
| **0.80** | 36.67% | 0.27% | **98.57%** | 3.53% | 0.17% |
| **0.75** | 44.40% | 0.34% | **98.52%** | 5.60% | 0.17% |
| **0.70 (Proposed Cut)** | **51.87%** | **0.44%** | **98.36%** | **8.80%** | **0.24%** |
| **0.65** | 58.73% | 0.51% | **98.33%** | 13.67% | 0.24% |
| **0.60** | 64.47% | 0.64% | **98.07%** | 18.33% | 0.37% |
| **0.50** | 72.20% | 1.05% | **97.22%** | 29.27% | 0.71% |

---

## Part 2: The Proposed Test-Train Network Plan

### 2.1 The Concept
Instead of only searching within the ~400 test queries (which misses scaffolds if no relative was sampled in the test set):
1. **Offline Phase**: Group `train.parquet` into its **30,732 multi-molecule Murcko scaffolds**. Distill each scaffold cluster into **1–2 representative consensus centroid spectra** ($[M+H]^+$, high SNR). This yields a compact **Scaffold Centroid Library** (~50,000 spectra, $< 35\text{ MB}$).
2. **Inference Phase**: For every test query, calculate Modified Cosine and Neutral Loss against the 50,000 centroid spectra.
3. **Scaffold Identification**: If a query has Modified Cosine $\ge 0.70$ and precursor difference $\Delta m$ matches a biogenetic mass defect, assign it that core scaffold.
4. **Biogenetic Side-Chain Decoration**: Compute $\Delta \text{Formula} = \text{Formula}(Q) - \text{Formula}(S_{\text{core}})$. Synthesize candidate regioisomers by attaching standard biogenetic moieties ($-\text{OH}$, $-\text{OMe}$, $-\text{Me}$, $-\text{Hexose}$, etc.).
5. **Class 3 Injection**: Forward-simulate the generated isomers (ICEBERG/GLACIER) and inject them into candidate lists.

---

## Part 3: Forensic Analysis of Problems, Bottlenecks & Pitfalls

Past experiments in this repository (`v4n-mutator-submit` row 22, `v4n-mutator-bio` row 23, `v4n-mutator-bio2` row 25, `c3v2-tau8` row 34) failed or caused regressions due to four critical bottlenecks:

### 1. The Combinatorial Regioisomer Explosion ($MRR@25$ Bottleneck)
- **Problem**: Knowing the Murcko scaffold does not specify *where* substituents attach.
- For example, quercetin has 5 phenolic hydroxyls. Adding 1 glucose and 1 methyl generates $> 20$ regioisomers. At depth $k=3$, it generates $100\text{--}300$ isomers.
- **Metric Consequence**: The official metric is strictly `MRR@25`. If 100 isomers are generated and ranked randomly, the expected rank of the true 2D structure is ~50. Ranks $> 25$ receive an exact score of **0.000 MRR**.
- **Required Fix**: Enforce a strict shallow-depth constraint: **$k \le 2$ transformations only**. (Data from `validate_mutator.md`: depth $\le 1$ achieves **80% recovery** with only 16 isomers; depth $\ge 3$ collapses recovery to 10% with $> 100$ isomers).

### 2. The Acyclic & Singleton Scaffold Wall
- **Problem**: Measured data on 600 novel natural products (`p1_novel/pseudo_c3.parquet`) revealed:
  - **21.5% of natural products have NO Murcko scaffold** (acyclic lipids, linear peptides).
  - **53.2% of natural products are unique singletons** whose scaffold appears nowhere else.
- **Consequence**: Scaffold harvesting has an intrinsic reach ceiling of **~30–40% of novel queries**. It cannot solve 100% of Class 3.

### 3. The Fatal "Friendly Fire" / Cannibalization Risk (The `bio2` Lesson)
- **Problem 1 (Cannibalization)**: In Row 25 (`bio2`), an uncalibrated gate misfired on Class 1 and Class 2 queries, pushing true Rank 1 database hits down to Rank 6 and **dropping LB score by -0.064** (0.399 $\to$ 0.335).
- **Problem 2 (Friendly Fire)**: In Row 23 (`mutator-bio`), generated candidates had the same formula as the target. Uncalibrated ICEBERG/GLACIER noise in the same-formula group re-ranked a generated decoy above the true database candidate at Rank 1 (**-0.002 LB drop**).
- **Required Fix**:
  1. Only fire the generator if `lib_max < 0.70` AND `best_pool_fz` is in the lowest 15% quantile.
  2. Tag generated candidates with a formula suffix `'|gen'` (e.g., `'C15H10O5|gen'`). This isolates them from the database formula groups so forward simulators cannot swap them with true database hits.

### 4. Kaggle 9-Hour Offline Runtime & Memory Constraint
- **Problem**: Raw `train.parquet` is 3.03 GB (2.54M spectra). Matching 400 queries against 2.54M spectra requires **1 billion comparisons** ($> 2.5\text{ hours}$ on CPU) and risks 13 GB RAM OOM.
- **Required Fix**: The training graph **must be pre-clustered offline into a compact centroid file** ($< 40\text{ MB}$, $\le 50,000$ spectra). At test time, matching 400 queries against 50,000 centroids takes **$< 15\text{ seconds}$** in Numba.

---

## Part 4: The 4 Non-Negotiable Engineering Rules for Implementation

1. **Rule 1 (Offline Centroid Compression)**: Precompute `train_scaffold_centroids.npz` on Colab/local. Never read raw `train.parquet` during Kaggle inference.
2. **Rule 2 (Shallow Depth Cap $k \le 2$)**: Limit biogenetic decoration to $\le 2$ reaction steps ($+\text{OH}, +\text{OMe}, +\text{Me}, +\text{Hexose}, +\text{Glucuronide}, +\text{CO}, +\text{H}_2\text{O}$). Cap generated isomers per query at $\le 25$.
3. **Rule 3 (Formula Isolation `'|gen'`)**: Every generated candidate must carry the `'|gen'` formula tag to prevent friendly-fire rank swaps in Cell 19 forward simulation.
4. **Rule 4 (Quantile Scale-Free Gating)**: The generator must only fire on the weakest 15% of queries by pool likelihood. Never use fixed absolute $f \cdot z$ thresholds.

---

## Part 5: Benchmark & Artifact Registry

- **Kernel Code**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\casmi_scaffold_network_eval.py`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/casmi_scaffold_network_eval.py)
- **Kernel Metadata**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\kernel-metadata.json`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/kernel-metadata.json)
- **Distribution CSV**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\output_v2\scaffold_quantile_distribution.csv`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/output_v2/scaffold_quantile_distribution.csv)
- **Operating Curves CSV**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\output_v2\scaffold_operating_curves.csv`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/output_v2/scaffold_operating_curves.csv)
- **Report JSON**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\output_v2\scaffold_eval_report.json`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/output_v2/scaffold_eval_report.json)
- **Visual Plots**: [`E:\ENVEDA_casmi\.scratch\kaggle_scaffold_eval\output_v2\scaffold_network_curves.png`](file:///E:/ENVEDA_casmi/.scratch/kaggle_scaffold_eval/output_v2/scaffold_network_curves.png)
