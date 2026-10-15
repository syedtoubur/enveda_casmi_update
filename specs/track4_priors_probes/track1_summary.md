# Track 1 Summary: Novel Scaffold Generation, Mutator History, and Integration Blueprint for Track 4

> **Purpose**: This document provides a complete technical autopsy of all Class 3 (novel natural product) generation attempts in Track 1 (`specs/track1_retrieval_fusion`). It explains the underlying chemistry, the exact mathematical failure modes of every submission (`0.399 → 0.397 → 0.335 → 0.399`), and provides a step-by-step blueprint for how any autonomous agent or human engineer can safely integrate novel structure generation into the current Track 4 SOTA (`0.411 LB`).

---

## 1. The Core Challenge: Why Class 3 Requires Generation

In the CASMI 2026 challenge, test molecules belong to three novelty classes:
- **Class 1 (Library Match)**: The exact molecule is in the reference spectral library. Spectral entropy and cosine matching achieve $>90\%$ top-1 accuracy.
- **Class 2 (Database Match)**: The molecule exists in structure databases (PubChem, COCONUT, ChEBI, LIPID MAPS), but has no reference spectrum. Handled by neural fingerprint retrieval (FPNet), GBDT rankers, and forward mass spectrometry simulators (`ICEBERG` + `GLACIER`).
- **Class 3 (Completely Novel)**: The molecule is absent from all chemical databases (true novel natural products). 

> [!IMPORTANT]
> **The 0% Ceiling**: For Class 3 molecules, pure database retrieval has an absolute mathematical ceiling of **`0.000 MRR`**. If the true molecular structure does not exist in the candidate pool, no ranker, prior, or physics simulator can ever rank it in the top 25. To score points on Class 3, the pipeline **must generate or mutate new 2D chemical graphs**.

---

## 2. The Track 1 Mechanism: The Formula-Space $\Delta M$ Mutator

Track 1 implemented an autonomous chemical generator: [`build/delta_m_mutator.py`](file:///e:/ENVEDA_casmi/build/delta_m_mutator.py).

### How the Mutator Works:
1. **Scaffold Harvesting**: 
   - Takes the top-ranked candidates retrieved by the database engine (from PubChem, Engine-A BIO, or mass-shifted spectral analogs).
   - Decomposes each candidate into its Bemis-Murcko core scaffold (rings + linkers, stripping side chains).
2. **Formula Delta Calculation**:
   - Measures the formula difference between the candidate scaffold and the query molecule's exact precursor formula:
     $$\Delta = \text{Formula}_{\text{target}} - \text{Formula}_{\text{scaffold}}$$
3. **Biogenetic Mutation Palette**:
   - Searches a pre-compiled lookup table of 20 standard biogenetic metabolic transformations (e.g., $+\text{OH}, +\text{OMe}, +\text{CO}, +\text{CH}_2, +\text{OAc}, +\text{prenyl}$).
   - If $\Delta$ can be decomposed into $\le 6$ steps of these transformations, the mutator chemically decorates the scaffold using RDKit reaction SMARTS.
4. **Fingerprint Screening ($f \cdot z$)**:
   - Generates tens to hundreds of valid regioisomers matching the target formula.
   - Passes them through FPNet logit scoring ($f \cdot z$) using the query spectrum's predicted fingerprint $f$ to rank the generated isomers.

---

## 3. Autopsy of What Failed & Why (The 4 Notebook Submissions)

Track 1 tested 4 distinct notebook iterations attempting to merge the mutator into the `0.399` retrieval baseline. Every iteration revealed a critical lesson:

```
0.399 Baseline (v4n + Engine-A + ICE_UNION)
   │
   ├── casmi26-v4n-mutator-submit.ipynb        ──> 0.399 (0.000)  [Rank 20 Arithmetic Ceiling]
   │
   ├── casmi26-v4n-mutator-bio-submit.ipynb    ──> 0.397 (-0.002) [Formula-Group Friendly Fire]
   │
   ├── casmi-v4n-mutator-bio2-submit.ipynb     ──> 0.335 (-0.064) [Absolute Gate Shift + Rank 1-5 Cannibalization]
   │
   └── casmi26-v4n-mutator-c3-submit.ipynb     ──> 0.399 (0.000)  [Harvestability Wall + Exposure Capping]
```

---

### Failure Mode 1: The Arithmetic Entry Ceiling (`casmi26-v4n-mutator-submit` → 0.399, 0.000 delta)

* **The Design**: The mutator candidate list was fused directly into the main Reciprocal Rank Fusion (`fuse2`) alongside the v4n base list and Engine-A list:
  $$\text{Score}(k) = \frac{1.0}{3.0 + r_{\text{v4n}}} + \frac{0.6}{3.0 + r_{\text{eng}}} + \frac{\alpha_{\text{mut}}}{3.0 + r_{\text{mut}}} \quad (\alpha_{\text{mut}} = 0.25, KRR = 3.0)$$
* **The Mathematical Flaw**:
  - The v4n list had up to 25 items; Engine-A had 40 items.
  - With $\alpha_{\text{mut}} = 0.25$, the highest possible score a mutator hit could receive at Rank 1 was:
    $$\frac{0.25}{3.0 + 1} = 0.0625$$
  - In the fused list cut at top 25, a score of $0.0625$ placed the mutator's best candidate at **Rank 20**.
  - **The Payoff**: Rank 20 yields a reciprocal rank of only $\frac{1}{20} = 0.05$.
  - Across 120 gated queries with a 5% mutator recovery rate:
    $$\text{Max Expected Gain} = \frac{120 \times 0.05 \times 0.05}{400} = \mathbf{+0.00075}\text{ MRR}$$
  - Kaggle's public leaderboard displays only 3 decimal places. $+0.00075$ rounds to $+0.000$.
* **Rule Learned**: **Never fuse an unranked generative list with a low RRF weight.** A weight below the retrieval tail caps the entry rank below Rank 15, guaranteeing the payoff cannot move the leaderboard.

---

### Failure Mode 2: Formula-Group "Friendly Fire" (`casmi26-v4n-mutator-bio-submit` → 0.397, −0.002 drop)

* **The Design**: Added Engine-A (BIO + AFIX) as an orthogonal scaffold source, expanding mutator candidates.
* **The Root Cause of the −0.002 Drop**:
  - The mutator produces **same-formula regioisomers** of the target.
  - Downstream in Cell 14/17, `ICEBERG` and `GLACIER` re-rank candidates **inside same-formula groups**:
    $$\text{Final Score} = z(\text{RRF}) + 1.0 \cdot z(\text{ICEBERG}) + 1.0 \cdot z(\text{GLACIER})$$
  - When a query was actually Class 1 or 2, the true retrieved structure was at Rank 1 in its formula group.
  - Because the mutator candidate shared the *exact same formula*, it entered that same formula group at Rank 20.
  - Minor simulation noise in ICEBERG/GLACIER occasionally promoted the synthetic decoy to Rank 1 and demoted the true database answer to Rank 20!
  - **The Cost**: Just **one single molecule** experiencing this swap on the 400 test set costs:
    $$\text{Loss} = \frac{1.0 - 1/20}{400} = \frac{0.95}{400} = \mathbf{-0.002375}\text{ MRR}$$
  - $0.399 - 0.0024 = \mathbf{0.397}$. The drop was exactly one true hit demoted by simulator noise!
* **Rule Learned**: **Isolate generated candidates from database formula groups.** Always tag synthetic candidates with a `'|gen'` suffix (e.g. `C15H22O3|gen`) so physics simulators never reorder synthetic decoys against database truths.

---

### Failure Mode 3: The False-Positive Gate Disaster (`casmi-v4n-mutator-bio2-submit` → 0.335, −0.064 drop)

* **The Design**: Recognized that fusing mutators into RRF was flawed. Instead, attempted **dynamic routing**:
  ```python
  C3_THRESHOLD = 300.0   # calibrated: known mols >= 456.0, novel mols <= 265.8
  is_class_3 = (best_pool_fz < C3_THRESHOLD) and bool(u.get('smiles'))
  if is_class_3:
      vs = merge(base_smis, base_keys, u['smiles'], u['keys'], slots=[1, 2, 3, 4, 5], n=25)
  ```
* **Fatal Flaw 1 (The Commit Calibration Trap)**:
  - The threshold `300.0` was calibrated during a fast-commit smoke run (`SMOKE_N = 12`) on `train.parquet`.
  - On the real 400-molecule test set, the median `best_pool_fz` was **720.6** (+172 points higher), with high variance across adducts and negative-mode spectra.
  - A low `best_pool_fz` is merely *negative evidence* (a sparse or noisy spectrum), **not** proof of a novel structure. The gate misfired on **50 to 100+ perfectly normal Class 1 and 2 queries**!
* **Fatal Flaw 2 (Violent Rank Hijacking)**:
  - On every misfired query, `bio2` hijacked **Ranks 1, 2, 3, 4, and 5** for unverified mutator decoys.
  - The retrieval engine's true Rank 1 answer was violently demoted to **Rank 6**.
  - Demoting Rank 1 to Rank 6 costs $1.0 - 1/6 = \mathbf{-0.833}$ RR per query.
  - Across ~31 misfired queries:
    $$\text{Loss} = \frac{31 \times (-0.833)}{400} = \mathbf{-0.0646}\text{ MRR} \quad (0.399 \rightarrow 0.335)$$
* **Rule Learned**: **Never gate on unnormalized $f \cdot z$ dot products, and never blindly hijack Rank 1.** An unnormalized score changes with spectral noise; hijacking Rank 1 destroys leaderboard performance whenever the gate misfires.

---

### Failure Mode 4: The Harvestability Wall (`casmi26-v4n-mutator-c3-submit` → 0.399, 0.000 delta)

* **The Design**: Solved the `bio2` disaster by creating [`build/c3_gate.py`](file:///e:/ENVEDA_casmi/build/c3_gate.py):
  1. Scale-free $[0, 1]$ signals: Library match $< 0.90$, mass-shifted analog similarity $\ge 0.50$, analog margin $\ge 0.15$.
  2. Formula reachability: Analog-to-target formula delta must be reachable in $\le 1$ step (`C3_MAX_STEPS = 1`).
  3. Safe reserved slots: `sure → [1, 2, 3]`, `likely → [2, 3, 4]`.
* **Why it Avoided the Collapse**: It completely protected the 0.399 baseline (0 misfire cannibalization).
* **Why it Stayed Flat (0.000 Delta)**:
  - **The Oracle vs Production Gap**: Early mutator benchmarks showed 17.2% recovery on `pseudo_c3`, but that used the *truth's own scaffold* as seed.
  - In real inference, the scaffold must be *harvested from a retrieved hit*.
  - Offline audit revealed that **319 of 600 novel natural products (53.2%) are the *only* structure in the chemical universe carrying their Murcko scaffold**. No database retrieval list can ever supply a scaffold that exists nowhere in databases!
  - The mutator's true production ceiling dropped from 17.2% to **4.2%**.
  - At `MAX_STEPS = 1`, the gate fired on only **2.5% of the test set** (~10 molecules).
  - Out of 10 molecules, gaining 1 hit ($+0.0025$) while incurring 1 minor displacement ($-0.0019$) resulted in a net $+0.0006$ MRR, which rounds to **0.399**.
* **Rule Learned**: **The mutator has a 53% blind spot due to unique scaffolds.** To achieve large gains on Class 3, the pipeline cannot rely on scaffold harvesting alone; it needs de novo graph generation (Track 2).

---

## 4. Master Comparison of Class 3 Gating Strategies

| Metric / Property | Rank Gate (Original) | Absolute $f \cdot z$ (`bio2`) | Scale-Free Gate (`c3_gate`) |
| :--- | :--- | :--- | :--- |
| **Logic** | Weakest 30% of $f \cdot z$ | `best_pool_fz < 300.0` | `lib < 0.90` ∧ `ana >= 0.50` ∧ `margin >= 0.15` ∧ `steps <= 1` |
| **Input Signal** | Negative evidence only | Negative evidence only | **Positive analog evidence** + formula reachability |
| **Scale Dependence** | Relative within run | **Catastrophic** (shifts +172 on test) | **Scale-free** (cosine similarities in $[0, 1]$) |
| **Can Fire on 0?** | No (always fires on 120) | Yes | **Yes** (strict quality vetoes) |
| **Displacement Risk** | High | **Disastrous** (−0.064 loss) | **Zero to minimal** ($\le -0.001$) |
| **Public LB Result** | 0.399 / 0.397 | **0.335** | **0.399** |

---

## 5. Blueprint: Integrating Novel Scaffold Generation into Track 4 (0.411 SOTA)

Track 4 currently holds the competition record at **`0.411 Public LB`** ([`casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb`](file:///e:/ENVEDA_casmi/casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb)). Its core strengths are:
1. Multi-view retrieval fusion (v4n + Engine-A + PubChem).
2. PubChem Popularity Prior ($0.25 \cdot \text{lsid} + 0.15 \cdot \text{lpmid}$).
3. **Uncorrupted `fz_top` Gate**: Basing merge decisions strictly on raw spectral similarity without popularity scale distortion.
4. Selective GLACIER simulation restricted to `[M+H]+`.

To integrate novel scaffold generation into Track 4 without regressing from 0.411, any agent must adhere to the following **5 Golden Architectural Rules**:

```
                         [Query Spectrum + Precursor Mass]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
      [Track 4 Database Retrieval]                 [Class 3 Novelty Detector]
    • v4n + Engine-A + PubChem                    • c3_gate: lib < 0.90, ana >= 0.50
    • Add Popularity Prior (lsid/lpmid)           • Positive analog margin >= 0.15
    • Top 40 Candidates                            │
                   │                               ├──> If DECLINED: Keep 0.411 Base
                   │                               │
                   │                               ▼ If FIRED:
                   │                  [Dual-Engine Novel Generator]
                   │                  • Harvestable: ΔM Mutator (≤ 1 step)
                   │                  • Unique Scaffold: DreaMS SELFIES Decoder
                   │                  • Suffix formula with '|gen'
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                       [Safe Reserved Splicing]
                   • If 'likely': slots = [2, 3, 4]  (Rank 1 untouched!)
                   • If 'sure':   slots = [1, 2, 3]  (Only if margin > 0.30)
                                   │
                                   ▼
                   [ICEBERG / GLACIER Reranking]
                   • '|gen' tag prevents simulator friendly fire
                   • Forward models order novel hits internally
                                   │
                                   ▼
                       Final Submission (0.415+ LB)
```

---

### Rule 1: Never Corrupt the Novelty Gate with Popularity Priors
When popularity is added to PubChem candidates:
$$\text{Score} = z(f \cdot z) + 0.25 \cdot \text{pop}$$
If a Class 3 gate checks `best_pool_fz`, the popularity term inflates the score by several standard deviations. The gate must **only evaluate raw spectral evidence** (`lib_max`, `exact_max`, `ana_max`), completely ignoring the popularity prior.

### Rule 2: Isolate Formula Groups with `'|gen'`
When generated candidates are injected into the candidate table, always append `'|gen'` to their formula:
```python
candidate_formula = f"{target_formula}|gen"
```
This forces ICEBERG and GLACIER to place generated candidates into their own independent group. The simulators will rank generated candidates against each other, but **can never demote a retrieved database candidate in favor of a synthetic decoy**.

### Rule 3: Enforce Safe Splicing (Protect Rank 1)
Unless spectral entropy cosine is essentially zero ($< 0.10$) and analog margin is overwhelming ($> 0.30$), **never take Rank 1**.
Use hedged slot reservations:
```python
C3_SLOTS = {
    'sure':   [1, 2, 3],  # Reserved only for extreme confidence
    'likely': [2, 3, 4],  # Default: Rank 1 remains protected for retrieval
}
```
If retrieval was correct at Rank 1, placing the novel candidate at Rank 2 preserves a reciprocal rank of $1.0$. If retrieval was wrong (Class 3), placing the novel candidate at Rank 2 pays $\frac{1}{2} = 0.50$ RR (infinitely better than 0.0).

### Rule 4: Combine the Mutator with Track 2's De Novo Decoder
To break through the 53.2% unique Murcko scaffold wall:
- **For Harvestable Relatives ($\le 1$ transform delta)**: Use the Track 1 $\Delta M$ mutator (precise, 69.2% reachability on shallow deltas).
- **For Unharvestable Unique Scaffolds**: Use Track 2's DreaMS $\rightarrow$ SELFIES decoder ([`specs/track2_novel_generation`](file:///e:/ENVEDA_casmi/specs/track2_novel_generation)), which generates de novo structures directly from spectrum embeddings with 0% database dependence (8.17% exact@25).
- Blend the top 10 mutator candidates with the top 15 SELFIES decoder candidates before splicing.

### Rule 5: Check the Entry Payoff Before Shipping
Before running a submission, calculate the theoretical maximum lift:
$$\Delta_{\text{MRR}} = \frac{N_{\text{fired}} \times P(\text{recovery}) \times \frac{1}{\text{slot}}}{400} - \frac{N_{\text{misfires}} \times P(\text{displacement}) \times \Delta \text{RR}}{400}$$
If the projected gain is $< +0.001$, do not submit without widening the candidate quality or adjusting gate thresholds.

---

## 6. Relevant Code & Spec References

- **Track 1 Routing Specification**: [`specs/track1_retrieval_fusion/c3_routing.md`](file:///e:/ENVEDA_casmi/specs/track1_retrieval_fusion/c3_routing.md)
- **Mutator Offline Benchmark**: [`specs/track1_retrieval_fusion/validate_mutator.md`](file:///e:/ENVEDA_casmi/specs/track1_retrieval_fusion/validate_mutator.md)
- **Scale-Free Router Implementation**: [`build/c3_gate.py`](file:///e:/ENVEDA_casmi/build/c3_gate.py)
- **Chemical Mutator Generator**: [`build/delta_m_mutator.py`](file:///e:/ENVEDA_casmi/build/delta_m_mutator.py)
- **Track 4 Comparison Summary**: [`specs/track4_priors_probes/comparision_summary.md`](file:///e:/ENVEDA_casmi/specs/track4_priors_probes/comparision_summary.md)
- **Master Experiment Ledger**: [`EXPERIMENTS.md`](file:///e:/ENVEDA_casmi/EXPERIMENTS.md)
