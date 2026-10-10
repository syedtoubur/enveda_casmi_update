# specs_scratch R5: Spectral Analogue PSSM (Position-Specific Scoring Matrix) Plan

**Status**: v2, SETTLED in the 2026-10-08 grilling and implemented (`scratch_pkg`, probes `r5_*`). **§0 overrides the proposal text below** wherever they differ.  
**Date**: 2026-10-08  
**Base Lineage**: `casmi_scratch_r3_full1_r2gl` (LB **0.346**, best scratch run)  
**Preceding Negative Result**: `casmi_scratch_r4_famfuse` (LB **0.345**, dropped; proved a 2nd FPNet family is redundant)

---

## 0. Settled design (v2, grilling 2026-10-08)

| Topic | Decision |
|---|---|
| Role | Score-only. The list re-ranks the existing R1b pool and injects no structures. |
| Search | Open ±200 Da, `shift_entropy_sim`, full train library **including enveda-180** (its shared substructures are signal). Each spectrum is searched in its own polarity. Max S per analogue structure, S ≥ 0.5. No runtime fallback. |
| Scorers | Both are built and both go to the LB. **PSSM:** top M = 100. **Tanimoto kernel:** Σ S⁴·T / Σ S⁴ over the top K = 20, with T on the Morgan r2 4096 slice. Fewer than 3 analogues with a structure means no list for that molecule. |
| PSSM math | Replaces §3 step 2. **16,520 raw Family-A bits** (not the 10,226 subset, which was chosen on the older v2 pool). w = S⁴, p̃ = (Σw·x + α·q)/(Σw + α) with α = 1, W = clip(logit p̃ − logit q, ±3), s = C·W. With the §3 form `log((p+ε)/(q+ε))`, ε = 1e-4, any bit no analogue carries scored about −9. The true molecule usually carries such bits (its modification), so the pseudocount prior and the full Bernoulli log-odds keep that penalty bounded. The score then has the same form as f·z. |
| Background q | Bit frequency over the structures the list scores: the non-train R1b pool. It is sampled at runtime (seeded, 50k rows), or taken from an attached `analog_q.npy`; the audit notebook writes one. |
| Synthetic analogues | Drug-only bits get a large W, so they could boost the wrong chemistry. Handled two ways. First, **train structures are excluded from the analogue list only**: they get the neutral term, which is the mean RRF term of the scored candidates. The pool, tier 1, sv and f·z still see them, and Class-1 answers are kept. Second, the W clip. The exclusion also stops a train candidate from matching its own library spectrum (T = 1) and echoing lv. |
| Fusion | Tier-2 key `1/(3+r_sv+bonus) + 1/(3+r_fz) + 1.0/(3+r_an)` in stages A and B; tier 1 is untouched. w = 1.0 (not 0.4–0.8). |
| Probes | `r5_pssm`, `r5_tani` (class vote on) and `r5_pssm_novote`, `r5_tani_novote`. No control. **KEEP iff LB ≥ 0.354.** |
| Gates | Superseded by `gate_report.md` R5 (R5.0 fast commit, R5.audit, R5.LB). The 0.15 s/molecule budget in §7 does not apply: the open search measured about 10 s per molecule. |
| Offline | Slice I, run alongside the LB (`casmi_scratch_r5_audit_colab`). The own structure is banned, and **the truth counts as non-train**. The audit explains the LB result and does not decide it. |

---

## 1. Executive Summary & Why This Lever

In **R4** ([EXPERIMENTS.md](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/EXPERIMENTS.md) row 50, LB 0.345), fusing a second neural fingerprint model (prvsiyan Family P next to Ahmed Family A) yielded **-0.001 (neutral drop)**. As formulated in the Graveyard (line 219):
> *"Orthogonality has to be in the **evidence**, not the model: a second model of the same spectrum $\to$ FP task is a near-copy. Look for channels that read different information (popularity, PubChem candidates, library analogs)."*

All public SOTA submissions above 0.399 (v4n 0.399, v29 0.411, Sovereign Zenith 0.425) rely on **Engine A's AFIX analogue propagation index** ($N_{\text{analog}} = 200$). 

In `casmi_scratch`:
- [scratch_pkg/casmi_scratch/analog.py](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/scratch_pkg/casmi_scratch/analog.py) already contains Numba-accelerated shift-aware entropy matching (`shift_entropy_sim`), but currently uses it only for **coarse NPClassifier superclass voting** (R2: +0.008 LB).
- **R5 proposes direct substructural fingerprint propagation** from these spectral analogues, structured as a **Position-Specific Scoring Matrix (PSSM) / Empirical Log-Odds Profile**.
- This acts as an **empirical, library-grounded counterpart to FPNet**: while FPNet predicts bits through a neural black-box, the Analogue PSSM computes consensus bits directly from real, experimentally verified reference spectra across the metabolic neighbourhood.

---

## 2. Core Concept: Analogue Search + PSSM Consensus

### 2.1 Shift-Aware Analogue Retrieval
For a query spectrum $Q$ with neutral precursor mass $M_Q$:
1. Search the reference library within a $\pm 200$ Da open precursor window (or along defined biosynthetic mass deltas $\Delta M \in \text{NP\_DELTAS}$, within $\pm 10$ ppm).
2. Compute shift-aware entropy similarity $S_i = \text{shift\_entropy\_sim}(Q, A_i)$ considering both direct peak matches ($m/z_{\text{frag}} \approx m/z_{\text{lib}}$) and precursor-shifted matches ($m/z_{\text{frag}} \approx m/z_{\text{lib}} + \Delta M$).
3. Collect the top $M$ analogues ($M \approx 50\text{--}200$) satisfying $S_i \ge S_{\text{min}}$ (e.g. $0.50$).

### 2.2 The Analogue Matrix
Each retrieved library analogue $A_i$ has a known chemical structure with a precomputed binary fingerprint $\mathbf{x}_i \in \{0, 1\}^D$ (where $D = 10{,}226$ matching Ahmed's bit basis, or $D = 2{,}048$ Morgan radius 2).
Stacking them yields an analogue matrix:
$$\mathbf{X} \in \{0, 1\}^{M \times D}$$

---

## 3. Mathematical Formulation: Avoiding the Sparsity & Background Traps

A naive bit-entropy calculation fails because high-dimensional fingerprints are **~98% sparse**:
* An average natural product activates only 50 to 150 bits out of 10,226.
* Consequently, ~10,000 bits have $p_j \approx 0$ across all 200 analogues simply because no analogue carries those esoteric fragments.
* Candidate pool decoys with the same precursor formula *also* have zeros at those 10,000 positions. Evaluating raw bit entropy would flood the score with uninformative zero-matches.

To solve this, we adapt bioinformatics PSSM mathematics into a **Spectrally-Weighted Log-Odds Profile (Relative Entropy)**:

### Step 1: Spectrally-Weighted Consensus Frequency
Do not treat all analogues equally. Weight each analogue by its 4th-power spectral similarity so high-confidence spectral matches dominate:
$$w_i = (S_i)^4, \quad \text{where } S_i \ge 0.50$$
The weighted frequency of bit $j$ across the retrieved analogues is:
$$p_j = \frac{\sum_{i=1}^M w_i \cdot X_{i,j}}{\sum_{i=1}^M w_i} \in [0, 1]$$

* **Conserved bits ($p_j \approx 1$)**: Core structural scaffold fragments shared across the biogenetic family.
* **Variable bits ($p_j \approx 0.5$)**: Peripheral modification sites (variable $+OH, +CH_3, +sugar$).
* **Absent bits ($p_j \approx 0$)**: Features absent in this family.

### Step 2: Background Log-Odds Weighting (Information Content)
Let $q_j \in [0, 1]$ be the precomputed background frequency of bit $j$ across the entire candidate pool (778,095 structures).
Define the position-specific scoring weight $W_j$:
$$W_j = \log \left(\frac{p_j + \epsilon}{q_j + \epsilon}\right)$$
*(with Laplace smoothing $\epsilon \approx 10^{-4}$)*

**Chemical Behavior**:
* **$p_j \gg q_j \implies W_j > 0$ (Reward)**: The bit is heavily enriched in this analogue family relative to random chemistry (true scaffold identifier).
* **$p_j \ll q_j \implies W_j < 0$ (Penalty)**: Ubiquitous chemical features that are conspicuously absent in this analogue family.
* **$p_j \approx q_j \approx 0 \implies W_j \approx 0$ (Neutral)**: The ~10,000 sparse empty bits contribute zero and do not distort the score.
* **$p_j \approx 0.5 \implies$** Variable decoration bits naturally carry muted weights compared to rigidly conserved core bits ($p_j \to 1.0$).

### Step 3: Ultra-Fast Candidate Scoring ($O(K)$ Complexity)
For all $K$ candidates in the query mass window with binary fingerprint matrix $\mathbf{C} \in \{0, 1\}^{K \times D}$:
$$\mathbf{s}_{\text{pssm}} = \mathbf{C} \cdot \mathbf{W}$$

* **Complexity**: Matrix-vector product in NumPy takes **$< 1$ ms** per molecule for $K \approx 100$ candidates, compared to $O(K \times M)$ pairwise Tanimotos.

---

## 4. Pipeline & RRF Integration in `casmi_scratch`

### 4.1 Strict Protection of Tier 1
* Analogue PSSM evidence must **never demote confident Tier 1 library matches** ($lv \ge 0.75$).
* It operates strictly inside **Tier 2** (unmatched candidates).

### 4.2 Fused Tier-2 Key in `rank.py`
In [scratch_pkg/casmi_scratch/rank.py](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/scratch_pkg/casmi_scratch/rank.py), add $\mathbf{s}_{\text{pssm}}$ as a 3rd orthogonal channel in the Tier 2 RRF:
$$\text{Key}_{\text{Tier 2}} = \frac{1}{k + r_{sv + \text{bonus}}} + \frac{1}{k + r_{\text{fp}}} + \frac{w_{\text{pssm}}}{k + r_{\text{pssm}}}$$
* Tie-breakers: $fr$ (MetFrag-lite), then $|ppm|$.
* Default hyperparameters: $k = 3.0$, $w_{\text{pssm}} \in [0.4, 0.8]$, $S_{\text{min}} = 0.50$, $N_{\text{analog}} \in [50, 200]$.

---

## 5. Comparison: Empirical PSSM vs. Other Pipeline Channels

| Channel | Input Evidence | Model Type | Orthogonality |
|---|---|---|---|
| **$sv$ (GLACIER / ICEBERG)** | Candidate 2D structure $\to$ Query spectrum | Forward Physics / Quantum GNN Simulator | High (evaluates forward fragmentation) |
| **$fz$ (FPNet full1)** | Query spectrum $\to$ Predicted bit probabilities | Deep Convolutional / Attention Neural Encoder | Baseline neural channel |
| **Class Vote (R2)** | Analogue hits $\to$ Superclass categorical prior | Discrete NPClassifier consensus | Coarse structural prior (+0.008 LB) |
| **Analogue PSSM (R5)** | Library analogue spectra ($\pm 200$ Da) $\to$ Bit consensus | **Empirical Library Substructure Consensus** | **High** (reads real reference library networks, not neural representations) |

---

## 6. Pre-Flight Checks & Implementation Guards

1. **Self-Match Leak Guard (Critical for Validation)**:
   * During offline validation on `train.parquet` or validation holdouts, the query molecule itself exists in the library!
   * Always pass `exclude=lib_exclude` into `analog_hits` so the query cannot match itself at $\Delta M = 0$ with $S = 1.0$.
2. **Kaggle 9-Hour Offline Runtime Budget**:
   * Computing shift-aware entropy similarity against the library:
     - `mode="delta"` (biosynthetic deltas within $\pm 10$ ppm) is 10× faster than full `mode="open"` ($\pm 200$ Da).
     - Test whether `mode="delta"` captures $\ge 90\%$ of high-scoring analogues vs `mode="open"`.
3. **Background Frequencies $q_j$ Storage**:
   * Precompute $q_j$ once across `pool.keys` / `pool.smiles` (shape: `float32[10226]`, file size: ~40 KB) and ship as part of the pool bundle.

---

## 7. Proposed Pre-Registered Gates (Gate R5)

* **R5.0 Fast-Commit Gate**:
  - $N_{\text{analog}} \ge 1$ retrieved on $\ge 90\%$ of test molecules.
  - $\mathbf{s}_{\text{pssm}}$ finite, non-trivial variance across candidates.
  - Runtime overhead $\le 0.15$ s per molecule.
* **R5.audit Offline Gate**:
  - Evaluate on `slice_I` (Class 2 / unmeasured holdouts).
  - Condition: $\Delta \text{MRR@25} \ge +0.010$ over R3 baseline, with up > down reorderings.
* **R5.LB Submission Gate**:
  - Paired submission against `casmi_scratch_r3_full1_r2gl` (0.346).
  - **Pass Threshold**: $\text{LB} \ge 0.352$ ($\ge +0.006$ over 0.346).
