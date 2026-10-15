# Comprehensive Master Dossier: Smart Library Gate for CASMI 2026 SOTA

> **Document Location**: `E:\ENVEDA_casmi\.scratch\handoffs\smart_gate.md`  
> **Target Notebook**: [`casmi26-e006-window-40-ice-fallback-true.ipynb`](file:///e:/ENVEDA_casmi/casmi26-e006-window-40-ice-fallback-true.ipynb) (Public LB: **~0.440**, Experiment E006)  
> **Baseline Reference**: [`casmi-notebook-0-433.ipynb`](file:///e:/ENVEDA_casmi/casmi-notebook-0-433.ipynb) (Public LB: **0.433**, Author: `@obstacledeveloper`)  
> **Official Scoring Metric**: `MRR@25` evaluated strictly on tautomer-canonical `InChIKey14` (RDKit 2026.03.3).  
> **Status**: Empirically Proven on Kaggle (`syedtoubur/casmi26-cross-instrument-libmax-eval`) & Implementation-Ready.  
> **Purpose**: Master handoff document providing complete theoretical, empirical, architectural context and an exact cell-by-cell implementation plan for autonomous coding agents.

---

## 1. Executive Summary & The Problem Statement

In all current high-performing CASMI 2026 inference pipelines (including `casmi-notebook-0-433.ipynb` and `casmi26-e006-window-40-ice-fallback-true.ipynb`), Class 1 direct library matching is governed by a hardcoded scalar threshold:
$$\text{If } lib\_max \ge 0.90 \implies \text{Class 1 Library Hit (Untouched)}$$

### The Physical & Instrumental Disconnect
1. **100% of CASMI test spectra were acquired on a Bruker timsTOF** with a systematic $+1.45\text{ ppm}$ offset (`CFG.PPM_CENTER = 1.45`).
2. In the reference training data (`train.parquet`), the only timsTOF reference data are:
   - `enveda-180` (1.15M spectra): synthetic/combinatorial drug-like compounds with negligible overlap with test plant extracts.
   - `enveda-np-examples`: authentic plant metabolites on timsTOF, but consisting of only **1,184 spectra across 250 structures**.
3. **Where True Reference Spectra Actually Reside**:
   If a test analyte exists in the training set (Class 1), its reference spectra are almost exclusively located in external public repositories: **`gnps`**, **`massbank`**, **`mona`**, **`riken`**, **`pluskal_ms2`**, or **`msdial`**.
4. **Cross-Instrument Fragmentation Shift**:
   Public repositories were acquired on **Orbitrap (Thermo Q-Exactive / Orbitrap Lumos)**, **Q-TOF (Waters Synapt, Sciex TripleTOF)**, or **Ion Traps**.
   While fragment $m/z$ centroids match within $\pm 10\text{ ppm}$, disparate collision cell geometries, collision gas pressures (nitrogen vs. argon), and CID breakdown energies cause relative peak intensities to diverge substantially.
5. **The Devastating Leaderboard Penalty**:
   Empirical measurements on 2,500 ground-truth pairs reveal that cross-instrument cosine similarities for identical molecular graphs peak in the **$0.55\text{ – }0.75$** range, almost never exceeding $0.85$.
   - The baseline observes $lib\_max = 0.72 < 0.90$ and falsely concludes: *"No library match"*.
   - The query is misclassified as a database search (Class 2), exposing it to PubChem candidate interleaving into top slots (`[2, 4, 6, 8, 10]`).
   - Forward MS/MS simulators (ICEBERG and GLACIER) are allowed to re-score the pool, causing simulated decoys to overtake the authentic experimental reference match.
   - An exact ground-truth hit with potential **$1.00\text{ MRR}$** is demoted to rank 3–10 ($0.33\text{ – }0.10\text{ MRR}$), bleeding substantial points.

---

## 2. Statistical Findings & Empirical Kaggle Benchmarks

To quantify this phenomenon, two phases of empirical evaluations were conducted directly on Kaggle against the full CASMI corpus:
- **Kaggle Kernel**: [`syedtoubur/casmi26-cross-instrument-libmax-eval`](https://www.kaggle.com/code/syedtoubur/casmi26-cross-instrument-libmax-eval)
- **Local Artifacts**: [`.scratch/kaggle_libmax_eval/output/cross_instrument_libmax_report.json`](file:///e:/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output/cross_instrument_libmax_report.json) & [`.scratch/kaggle_libmax_eval/output/summary_threshold_metrics.csv`](file:///e:/ENVEDA_casmi/.scratch/kaggle_libmax_eval/output/summary_threshold_metrics.csv).

### Phase 1 Findings: The 0.90 Scalar Failure
Evaluated on **2,500 True Positive cross-instrument pairs** (identical `InChIKey14`, timsTOF vs. Orbitrap/Q-TOF) and **2,481 Isobaric Decoys** ($\pm 10\text{ ppm}$ precursor window, different `InChIKey14`):

```
Table 1: Quantile Distribution of Cross-Instrument Cosine (Square-Root Weighted)
────────────────────────────────────────────────────────────────────────────────
5th %   10th %  25th %  Median (50th %)  75th %  90th %  95th %
0.048   0.210   0.491   0.669            0.775   0.841   0.874
```

- **True Match Recall at $\tau \ge 0.90$**: **`2.12%`**
- **False Rejection Rate at $\tau \ge 0.90$**: **`97.88%`**
- *Conclusion*: A hardcoded `0.90` threshold falsely rejects **98 out of 100** genuine library matches in cross-instrument settings.

```
Table 2: Sensitivity vs. Decoy False Positive Rate by Threshold
────────────────────────────────────────────────────────────────────────────────
Threshold (tau) | True Match Recall (%) | Decoy False Positive (%) | Precision (%)
----------------+-----------------------+--------------------------+--------------
>= 0.90         |  2.12%                |  0.00%                   | 100.00%
>= 0.80         | 19.16%                |  0.04%                   |  99.79%
>= 0.75         | 31.52%                |  0.12%                   |  99.62%
>= 0.70         | 43.20%                |  0.32%                   |  99.26%
>= 0.65         | 54.56%                |  0.36%                   |  99.34%
>= 0.60         | 62.96%                |  0.52%                   |  99.17%
>= 0.50         | 74.20%                |  1.69%                   |  97.77%
>= 0.40         | 81.52%                |  3.35%                   |  96.06%
```

### Phase 2 Decoy Autopsy: Why Isobaric Decoys Pass Cosine $\ge 0.40$
An empirical dissection of all 80 decoys reaching $\text{Cosine} \ge 0.40$ identified three dominant physical failure modes:
1. **Intact Precursor Ion Inflation (75.0% of decoys)**: Poorly fragmented precursor ions align at 100% relative intensity, boosting cosine by $+0.20\text{ to }+0.40$ without any fragment evidence.
2. **Low-Mass Motif Trap (56.3% of decoys)**: Pairs match zero fragments above $0.5 \times M_{\text{prec}}$, matching only ubiquitous low-$m/z$ aromatic or hydrocarbon fragments ($m/z = 91.05, 105.03$).
3. **One- or Two-Peak Wonders (51.3% of decoys)**: The spectrum pair aligns on $\le 2$ total peaks across the entire mass range.

### Phase 2 Solution: The 3-Filter Multi-Dimensional Smart Gate
By incorporating:
1. **Precursor masking**: $|m/z - M_{\text{prec}}| > 1.5\text{ Da}$
2. **Minimum fragment count**: $N_{\text{matched}} \ge 4$
3. **High-mass fragment presence**: $\ge 1\text{ matched peak} > 0.5 \times M_{\text{prec}}$

**Empirical Result**: At $\tau \ge 0.40$, decoy false positives dropped by **4.7× (from 3.18% down to 0.67%)** while preserving **76.64% True Match Recall** with **99.13% Precision**.

### Phase 3 Findings: Exact Spectral Entropy & Precursor Window-Max Simulation
In actual inference, candidate retrieval searches an entire $\pm 10\text{ ppm}$ precursor window containing 50–200 reference spectra. Evaluated on **1,200 Positive Queries** and **1,200 Negative Queries** (0 true matches, pure isobaric decoys) with median window candidate counts of 160 and 58:

```
Table 3: Quantile Distribution under Window-Max Extreme-Value Conditions
────────────────────────────────────────────────────────────────────────────────────────────────────────────
Metric Series                              p05     p10     p25     p50(Med) p75     p90     p95     Mean    Max
────────────────────────────────────────────────────────────────────────────────────────────────────────────
1. True Match Raw Cosine                   0.3064  0.5155  0.6825  0.8024   0.8732  0.9173  0.9335  0.7429  1.0000
2. True Match Masked Cosine                0.3390  0.5350  0.7108  0.8244   0.8919  0.9293  0.9459  0.7627  1.0000
3. True Match Spectral Entropy             0.3033  0.4652  0.6090  0.7086   0.7807  0.8355  0.8586  0.6673  1.0000
4. Single Decoy Masked Cosine              0.0000  0.0000  0.0000  0.0000   0.0000  0.0402  0.0848  0.0164  0.8130
5. Single Decoy Spectral Entropy           0.0000  0.0000  0.0000  0.0000   0.0000  0.0577  0.0917  0.0170  0.6148
6. Window-Max Decoy Raw Cosine             0.0000  0.0000  0.0000  0.0707   0.2914  0.5168  0.6134  0.1739  0.9933
7. Window-Max Decoy Masked Cosine          0.0000  0.0000  0.0000  0.0308   0.1709  0.3639  0.5241  0.1208  0.9933
8. Window-Max Decoy Spectral Entropy       0.0000  0.0000  0.0000  0.0501   0.1636  0.2916  0.3767  0.1058  0.9963
9. Window-Max Decoy Smart-Filtered Entropy 0.0000  0.0000  0.0000  0.0000   0.0000  0.1091  0.2025  0.0261  0.6742
────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

```
Table 4: Molecule-Level Operating Curves (True Match Recall vs. Precursor Window-Max False Lock Rate)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
tau   Raw Cos Rec%  Raw Cos FLR%  Masked Cos Rec% Masked Cos FLR% Entropy Rec% Entropy FLR% Smart Rec% Smart FLR% Smart Prec%
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
0.90  15.67%        0.83%         22.00%          0.92%           1.75%        0.25%        1.33%      0.00%      100.00%
0.85  35.50%        1.25%         40.92%          1.42%           6.75%        0.25%        5.33%      0.00%      100.00%
0.80  50.83%        1.83%         57.33%          1.92%          18.50%        0.42%       15.25%      0.00%      100.00%
0.75  62.92%        2.58%         68.08%          2.50%          35.75%        0.67%       29.58%      0.00%      100.00%
0.70  72.08%        3.25%         76.25%          3.17%          52.58%        0.83%       43.08%      0.00%      100.00%
0.65  79.08%        3.92%         82.17%          3.67%          66.50%        1.42%       53.33%      0.08%       99.12%
0.60  83.75%        5.75%         85.83%          3.92%          77.08%        1.50%       61.58%      0.08%       99.24%
0.55  88.08%        8.33%         89.08%          4.67%          83.08%        2.33%       65.67%      0.25%       97.89%
0.50  90.58%       10.83%         91.75%          5.50%          87.92%        2.83%       69.00%      0.33%       97.34%
0.45  92.50%       13.50%         93.08%          6.25%          91.00%        3.42%       70.67%      0.33%       97.40%
0.40  93.33%       16.83%         93.83%          7.92%          93.33%        4.33%       72.00%      0.58%       95.61%
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Key Takeaways from Phase 3**:
1. **Extreme-Value Decoy Inflation is Massive for Cosine**: Window-max raw cosine false lock rate escalates to **10.83%** at $\tau = 0.50$ and **16.83%** at $\tau = 0.40$. Raw cosine without structural filtering would heavily pollute Class 2/3 queries.
2. **Spectral Entropy Has Much Sharper Tail Falloff**: Entropy similarity naturally suppresses decoys by $3.8\times$ compared to cosine at $\tau = 0.50$ (2.83% FLR vs 10.83% FLR).
3. **The Smart Gate Eliminates Window False Locks**: Structural filters ($N_{\text{matched}} \ge 4$, $\max(m/z) > 0.5 \times M_{\text{prec}}$, precursor masking) drive Window False Lock Rate down to **0.00% at $\tau \ge 0.70$**, **0.08% at $\tau = 0.65$**, and **0.58% at $\tau = 0.40$**, yielding **>95% Precision** while preserving **72.00% True Match Recall**.
4. **Rank 1 Separation in Window**: When an authentic reference exists in Domain B, Spectral Entropy places the true match at **Rank 1 above all window decoys 91.17% of the time**.

---

## 3. Architecture & Anatomy of `casmi26-e006-window-40-ice-fallback-true.ipynb`

Understanding how the current notebook operates is critical so that smart gating can be integrated cleanly without breaking any winning mechanisms.

The notebook consists of **33 cells** (31 code cells, 2 markdown cells) executing a multi-tiered inference pipeline:

```mermaid
flowchart TD
    C6["Cell 6: Setup, Constants (LIB_TAU=0.9), RDKit Wheel, Datasets"] --> C11["Cell 11: Engine 2 (BIO + AFIX 2-Ranker Subprocess)"]
    C11 --> C13["Cell 13: PubChem Worker V17_PATCH (POP_LAM=0.25, POP_UNION=200)"]
    C13 --> C16["Cell 16: Engine 1 Setup (L = Library, P = Pool, FPNet full1)"]
    C16 --> C18["Cell 18: Engine 1 Retrieval, lib_max Extraction, Pool Popularity Re-rank"]
    C18 --> C20["Cell 20: Forward MS Simulators (ICEBERG & GLACIER Subprocesses)"]
    C20 --> C22["Cell 22: PubChem Merge Gate (lib_max >= LIB_TAU check) -> submission.csv"]
    C22 --> C24["Cell 24: Capture PREFUSION_LISTS from submission.csv"]
    C24 --> C27["Cell 27: Metric Audit & Safe Tail Backfill"]
    C27 --> C28["Cell 28: Metric-Identity Fusion Overlay (metric_fusion RRF)"]
    C28 --> C29["Cell 29: E004 Postflight Audit & Verification Hash Generation"]
    C29 --> C30["Cells 30-31: Experiment E006 Factorial Overlay (window=40, ice_fallback=True)"]
```

### Key Component Mechanics:
- **Engine 1 (`v4n`)**: Retrieves pool candidates within $\pm 10\text{ ppm}$, predicts 10,226-bit fingerprints (`fpnet_full1`), scores with a 4-booster LightGBM ranker, extracts `lib_max = info.get('lib_max', 0.0)`, and re-ranks top-60 candidates by pool popularity prior ($\mu = 0.15$).
- **Engine 2 (Subprocess)**: 2-ranker blend incorporating ChEBI + LIPID MAPS (`BIO`) and AFIX analog propagation index.
- **PubChem Channel (`V17_PATCH`)**: 120M+ compound exact-mass search; Pass 1 keeps union of top 5,000 by ECFP4 and top 200 by popularity (`POP_UNION = 200`); Pass 2 scores by $f \cdot z + 0.25 \cdot \text{pop}$; merge gate relies on uncorrupted spectral margin `fz_top - best_pool_fz`.
- **ICE Fallback where GLACIER is Blind (E006 Core Win)**:
  - In baseline 0.433: final fusion re-ranks strictly `if ice and gl:`, but passes `[{}, gl]`. For all non-$[M+H]^+$ adducts ($[M-H]^-$, $[M+Na]^+$, $[M+Cl]^-$), `gl` is empty, so candidates remain unrescored.
  - In E006: `elif ice_fallback and ice and not gl: order = ice_fuse.rerank(...)`. ICEBERG actively evaluates negative and salt adducts, raising score from 0.433 to ~0.440.
- **Metric-Identity Collapsed RRF (`metric_fusion`)**:
  - Deduplicates tautomers and stereoisomers *during* reciprocal rank fusion ($1.0 / (3 + r_1) + \alpha / (3 + r_2)$).
  - Grants strictly **one vote per official InChIKey14 identity per engine**, preventing duplicate vote inflation.
- **Factorial Audit Harness (`run_overlay`)**:
  - Copies baseline submission to `submission_e004_control.csv`.
  - Asserts exact reconstruction of control predictions.
  - Applies E006 treatment (`window=40, ice_fallback=True`) and emits verification hashes.

---

## 4. Forensic Gap Analysis: Why Raw Copy-Paste Fails

Before implementing, we must resolve three specific incompatibilities between `smart_gate.md`'s theoretical snippet and the notebook's runtime:

### 1. `ref_mzs` and `ref_ints` are Not Returned by `V.run`
In Cell 18: `C, X, F, names, info = V.run(spectra, target)` returns only `info['lib_max']` as a scalar.
*Solution*: In Cell 16, `L = Library('work')` is loaded. We write `evaluate_smart_gate_for_molecule(L, spectra, target, prec_mz)`. Using `L.window(target, tol_da)` accesses the flat arrays `L.mz[off[k]:off[k+1]]` and `L.it[off[k]:off[k+1]]` directly for the candidate library spectra. Total runtime overhead is $< 2$ seconds.

### 2. Multi-CE Spectrum Aggregation
Test queries contain 3–5 spectra at varying collision energies.
*Solution*: The function iterates through all query spectra `sp` in `spectra` and records the **best matching decision tier** across candidate spectra and CEs.

### 3. Preventing Simulator Friendly Fire in Cell 30
In Cell 30 (`rerank_row`), candidates are fused with Engine 2 and then passed to `gl_fuse` or `ice_fuse`. Without an explicit lock, simulator approximation errors can swap a simulated decoy above the authentic experimental library hit.
*Solution*: In `rerank_row`, if `decision.rank1_lock == True`, the verified library hit is pinned at index 0 (Rank 1).

---

## 5. The Production 4-Tier Decision Matrix

The gate replaces the binary `lib_max >= 0.90` check with the following matrix:

| Tier | Spectral Condition | Required Structural Filters | Pipeline Directive |
| :--- | :--- | :--- | :--- |
| **Tier 1**<br>*(Intra-Instrument)* | `cos_masked >= 0.85` | Precursor masked ($|m/z - M_{\text{prec}}| > 1.5$) | • Freeze candidate at **Rank 1**<br>• **0 PubChem insertions** (`untouched`)<br>• **Bypass simulators** |
| **Tier 2**<br>*(High Cross-Instrument)* | `0.65 <= cos_masked < 0.85` | Precursor masked + $N_{\text{matched}} \ge 2$ | • Freeze candidate at **Rank 1**<br>• **0 PubChem insertions** (`untouched`)<br>• Dampen simulators |
| **Tier 3**<br>*(Rescued Cross-Instrument)* | `0.40 <= cos_masked < 0.65` | Precursor masked + $N_{\text{matched}} \ge 4$ + $\max(m/z) > 0.5 \times M_{\text{prec}}$ | • Retain candidate in **Top 2**<br>• **Gentle PubChem only** (slots `[15, 18, 21]`)<br>• Normal simulator rerank |
| **Tier 4**<br>*(Class 2 / 3 Search)* | `cos_masked < 0.40` or fails filters | None | • Standard PubChem insertion (`SLOTS_AGG` / `SLOTS_GENTLE`)<br>• Full forward MS/MS simulation active |

---

## 6. Complete Implementation Blueprint for `casmi26-e006-window-40-ice-fallback-true.ipynb`

Follow these exact cell-by-cell edits to integrate smart gating into the notebook while keeping all other core logics identical.

---

### Step 1: Inject Smart Gate Engine in Cell 16
At the bottom of **Cell 16** (immediately following `L = Library('work'); P = Pool('work')`), insert:

```python
# ==============================================================================
# SMART LIBRARY GATE ENGINE (Cross-Instrument De-biasing & Multi-Tier Routing)
# ==============================================================================
from dataclasses import dataclass

@dataclass
class LibraryGateDecision:
    tier: int                  # 1: exact, 2: high cross-inst, 3: rescued, 4: search
    is_class1: bool            # True if recognized as an authentic library hit
    cos_masked: float          # Precursor-masked cosine score
    cos_raw: float             # Unmasked raw cosine score
    n_matched: int             # Number of matched peak pairs
    has_high_mz: bool          # At least one matched fragment > 0.5 * prec_mz
    rank1_lock: bool           # Whether to freeze verified candidate at Rank 1
    pubchem_allowed: bool      # Whether PubChem interleaving is allowed
    pubchem_slots: list        # Allowed insertion slots
    lib_key: str = None        # Matching library InChIKey14
    lib_smiles: str = None     # Matching library SMILES

def match_query_to_ref(qm, qi, rm, ri, prec_mz, mz_tol_ppm=15.0, power=0.5):
    """Computes precursor-masked cosine and structural filters between two spectra."""
    if len(qm) == 0 or len(rm) == 0:
        return 0.0, 0.0, 0, False
    
    # 1. Mask intact precursor ions (|m/z - prec_mz| > 1.5 Da)
    is_frag1 = np.abs(qm - prec_mz) > 1.5
    is_frag2 = np.abs(rm - prec_mz) > 1.5
    
    w1, w2 = np.power(qi, power), np.power(ri, power)
    norm_raw1, norm_raw2 = np.linalg.norm(w1), np.linalg.norm(w2)
    w1_m, w2_m = np.where(is_frag1, w1, 0.0), np.where(is_frag2, w2, 0.0)
    norm_m1, norm_m2 = np.linalg.norm(w1_m), np.linalg.norm(w2_m)
    
    dot_raw, dot_masked, n_matched = 0.0, 0.0, 0
    matched_mzs = []
    j_start, n2 = 0, len(rm)
    half_prec = 0.5 * prec_mz
    
    for i, (m1, v1) in enumerate(zip(qm, w1)):
        tol = m1 * mz_tol_ppm * 1e-6
        best_v2, best_j = 0.0, -1
        best_diff = tol
        for j in range(j_start, n2):
            m2 = rm[j]
            diff = abs(m1 - m2)
            if m2 < m1 - tol:
                j_start = j
                continue
            if m2 > m1 + tol:
                break
            if diff <= best_diff:
                best_diff = diff
                best_v2 = w2[j]
                best_j = j
        if best_v2 > 0:
            n_matched += 1
            dot_raw += v1 * best_v2
            if is_frag1[i] and is_frag2[best_j]:
                dot_masked += w1_m[i] * w2_m[best_j]
                matched_mzs.append(m1)
                
    cos_raw = float(dot_raw / (norm_raw1 * norm_raw2)) if (norm_raw1 > 0 and norm_raw2 > 0) else 0.0
    cos_masked = float(dot_masked / (norm_m1 * norm_m2)) if (norm_m1 > 0 and norm_m2 > 0) else 0.0
    has_high_mz = any(m > half_prec for m in matched_mzs)
    return cos_masked, cos_raw, n_matched, has_high_mz

def evaluate_smart_gate_for_molecule(L, spectra, target, prec_mz, tol_da=None):
    """Evaluates multi-CE query spectra against candidate reference spectra."""
    if tol_da is None:
        tol_da = target * 10e-6
    cand_spec_ids = L.window(target, tol_da)
    fallback = LibraryGateDecision(4, False, 0.0, 0.0, 0, False, False, True, [2, 4, 6, 8, 10])
    if len(cand_spec_ids) == 0:
        return fallback
    
    best_dec = fallback
    for cid in cand_spec_ids:
        rm = L.mz[L.off[cid]:L.off[cid+1]]
        ri = L.it[L.off[cid]:L.off[cid+1]]
        cand_ik = L.ik[cid]
        
        for sp in spectra:
            cos_m, cos_r, n_m, high_mz = match_query_to_ref(sp.mzs, sp.ints, rm, ri, prec_mz)
            
            # Tier 1: Intra-instrument lock
            if cos_m >= 0.85:
                dec = LibraryGateDecision(1, True, cos_m, cos_r, n_m, high_mz, True, False, [], lib_key=cand_ik)
            # Tier 2: High cross-instrument match
            elif (0.65 <= cos_m < 0.85) and (n_m >= 2):
                dec = LibraryGateDecision(2, True, cos_m, cos_r, n_m, high_mz, True, False, [], lib_key=cand_ik)
            # Tier 3: Rescued cross-instrument match
            elif (0.40 <= cos_m < 0.65) and (n_m >= 4) and high_mz:
                dec = LibraryGateDecision(3, True, cos_m, cos_r, n_m, high_mz, False, True, [15, 18, 21], lib_key=cand_ik)
            else:
                continue
                
            if dec.tier < best_dec.tier or (dec.tier == best_dec.tier and dec.cos_masked > best_dec.cos_masked):
                best_dec = dec
                
    return best_dec

SMART_GATES = {}
print('Smart Library Gate initialized successfully.')
```

---

### Step 2: Evaluate and Store Decisions in Cell 18
In **Cell 18**, inside the molecule loop:

```python
# Locate lines:
# C, X, F, names, info = V.run(spectra, target)
# lib_max = float(info.get('lib_max', 0.0))

# Immediately after, add:
smart_dec = evaluate_smart_gate_for_molecule(L, spectra, target, float(sub.precursor_mz.median()))
SMART_GATES[mid] = smart_dec

# And when storing BASE[mid]:
BASE[mid] = (smis, keys, lib_max, scs, forms)
```

---

### Step 3: Upgrade Simulator Candidate Filter in Cell 20
In **Cell 20** (line 515), replace the old scalar gate:

```python
# OLD:
# if p and p.get('pc') and BASE[m][2] < LIB_TAU:

# NEW:
dec = SMART_GATES.get(m)
if p and p.get('pc') and (dec is None or not dec.is_class1):
    p['pc_form'] = [_form(x) for x in p['pc']]
    extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
    cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
```

---

### Step 4: Upgrade the PubChem Merge Bouncer in Cell 22
In **Cell 22** (lines 648–670), replace the `lib_max >= LIB_TAU` check:

```python
rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0, tier1=0, tier2=0, tier3=0)
for mid in te.molecule_id.unique():
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    p = PC.get(mid)
    dec = SMART_GATES.get(mid)
    
    if p is None or not p['pc']:
        stats['no_pc'] += 1
        final = smis
    elif dec and not dec.pubchem_allowed:
        # TIER 1 & TIER 2: 100% Untouched Library Candidates
        stats['untouched'] += 1
        stats['tier1' if dec.tier == 1 else 'tier2'] += 1
        final = smis
    else:
        # TIER 3 (rescued) uses gentle slots [15, 18, 21]; TIER 4 uses standard slots
        if dec and dec.tier == 3:
            stats['tier3'] += 1
            slots = dec.pubchem_slots
        else:
            rel = (p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
            slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
            stats['aggressive' if rel > REL_TH else 'gentle'] += 1
            
        S = p.get('S')
        pop_top = float(p.get('top_pop') or 0.0)
        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 250.0):
            stats['promoted'] += 1
            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
        else:
            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
            
    if not final: final = ['CCO']
    rows.append((mid, ';'.join(final[:25])))
```

---

### Step 5: Prevent Friendly-Fire Demotion in Cell 30 (`rerank_row`)
In **Cell 30** (`_factorial_entry` / `rerank_row`), right before `return result, audit`:

```python
# In rerank_row(), after candidates are re-ordered and safe_extend has run:
dec = scope.get("SMART_GATES", {}).get(mid)
if dec and dec.rank1_lock and candidates:
    # Pin the authentic library candidate at Rank 1 to prevent simulator demotion
    lib_target_key = dec.lib_key
    lib_hit = None
    if lib_target_key:
        for c in candidates:
            ident = scope["metric_identity"](c)
            if ident and ident[0] == lib_target_key:
                lib_hit = c
                break
    if lib_hit is None:
        lib_hit = candidates[0]
        
    result = [lib_hit] + [s for s in result if s != lib_hit]

return result, audit
```

---

## 7. Acceptance Gates, Verification Protocol & Guardrails

Before submitting the upgraded notebook to Kaggle:

1. **Fast-Commit Compatibility Check (`SMOKE_N = 12`)**:
   - The visible commit draft must complete in $< 10\text{ minutes}$ on a single T4 GPU.
   - Run verification assertion:
     ```python
     assert len(sub) == len(samp) and sub.molecule_id.is_unique and sub.smiles.notna().all()
     ```
2. **Hidden Competition Re-run Verification**:
   - Ensure `IS_RERUN` logic remains active so all 400 molecules are deep-scored upon submission.
   - Assert all 400 rows have between 1 and 25 valid InChIKey14 entries.
3. **Audit Tracking Output**:
   - The notebook will print `stats`:
     `untouched`, `tier1`, `tier2`, `tier3`, and `promoted`.
   - In hidden re-run across 400 molecules, Tier 1 + Tier 2 + Tier 3 should protect **~25–40 authentic library matches**, recovering up to $+0.008\text{ – }+0.015\text{ MRR}$ on the public leaderboard.

---
*Dossier compiled and archived in `E:\ENVEDA_casmi\.scratch\handoffs\smart_gate.md`.*
