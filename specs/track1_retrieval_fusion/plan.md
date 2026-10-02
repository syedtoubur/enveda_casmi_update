# Implementation Plan: PubChem & Analogue-Derived Combinatorial $\Delta M$ Scaffold Mutation Engine

**Document Path:** `specs_v4n_update/plan.md`  
**Target Baseline:** `casmi26-v4n-inference (1).ipynb` (Current Public Leaderboard Score: **0.384**)  
**Objective:** Upgrade candidate generation and isomer resolution to break the **Class 3 (Novel Natural Products)** ceiling and correct synthetic-regioisomer mismatches in Class 2, lifting the score significantly beyond 0.384.

---

## 1. Executive Summary & Problem Diagnosis

### The SOTA Baseline Status (v4n - 0.384 LB)
The current solution uses a multi-stage hybrid architecture:
1. **PubChem Channel:** Precursor mass window $\pm 10$ ppm over 100M+ structures, screened via 4,096-bit ECFP4 (Pass 1 depth $N_1 = 5,000$), refined by FPNet Bayes log-likelihood dot product ($f \cdot z$), producing top 25 candidate structures.
2. **Main Pool Retrieval:** 710,701 structures (COCONUT 2.0 + Training set).
3. **160-Feature LightGBM Ranker:** Evaluates structural, fragmentation, and derivation features.
4. **Dual Forward Physics Simulators:** Equal-weighted within-formula rescoring via **ICEBERG** (GNN fragmentation) and **GLACIER** (MassSpecGym Transformer):
   $$\text{Score}_{\text{formula}} = z(\text{LightGBM}) + 1.0 \cdot z(\text{ICEBERG}) + 1.0 \cdot z(\text{GLACIER})$$

### The Critical Bottlenecks in v4n
1. **The Class 3 Dead End (MRR = 0.000):**
   Class 3 molecules are novel natural products absent from PubChem, COCONUT, and the Training library. For these molecules, both the Main Pool window and the PubChem channel return zero valid candidates. The notebook falls back to dummy ethanol (`['CCO']`), yielding an exact score of **0.000**.
2. **The Primitive Baseline Generator (`EngineCfg(generate=True)`):**
   The on-the-fly generator inside the v1 engine attempts naive single-step functional group chops (e.g., stripping a single terminal methyl or hydroxyl). Because natural product mass deficits ($\Delta M$) are typically multi-step ($+46\text{ Da}$, $+162\text{ Da}$, etc.), single-step chops almost never land within the mandatory $\pm 10\text{ ppm}$ tolerance and are discarded.
3. **The "Synthetic Regiochemistry" Flaw in PubChem:**
   PubChem contains over 100M compounds, but the vast majority are synthetic screening libraries. When FPNet matches a spectrum against PubChem, it correctly identifies molecules that share the **core ring scaffold** (since ring fragments dominate the MS/MS peaks). However, the PubChem molecule often has the **wrong decoration pattern** (e.g. synthetic protecting groups, ester linkages, or substituents at the wrong ring positions).

---

## 2. The Core Innovation: Dual-Source Scaffold Harvesting & $\Delta M$ Combinatorial Mutation

Instead of treating PubChem as a static "take-it-or-leave-it" database, we treat the top 25 PubChem hits and top training analogues as **Scaffold Generators**.

```
[Query Spectrum + Target Mass M_query]
       |
       +---> Source A: Top 25 PubChem Hits (from Stage 2 FPNet dot product)
       |
       +---> Source B: Top Spectral Analogues (from Library modified cosine > 0.70)
       |
       v
[RDKit Bemis-Murcko Scaffold Extraction]
  - Strip all terminal side chains and synthetic functional groups
  - Deduplicate core ring frames (typically 25 hits -> 3 to 8 unique scaffolds)
       |
       v
[Mass Deficit Calculation]
  - Delta_M = M_query - M_scaffold
       |
       v
[Biogenetic Knapsack Solver (Tolerance <= 10 ppm)]
  - Decompose Delta_M into exact biological transform vectors:
    (+O: 15.9949 Da, +CH2: 14.0157 Da, +Hexose: 162.0528 Da, -2H: -2.0157 Da, etc.)
       |
       v
[RDKit Chemical Reaction SMARTS Execution]
  - Enumerate natural regioisomers across open C-H / O-H sockets
       |
       v
[Quality Gating & Strict Validation]
  - Exact monoisotopic mass verification (<= 10 ppm)
  - Valence sanitization & metric canonicalization (InChIKey14)
       |
       v
[Injection into Candidate Stream]
  - Featurized by fe_v4 (derivation priors)
  - Forward-simulated by ICEBERG + GLACIER for isomer disambiguation
```

---

## 3. Detailed Algorithmic Specifications

### Step 1: Scaffold Harvesting & Deduplication
* **Input:**
  1. Top 25 PubChem candidate SMILES returned by `pc_runner.py` (Cell 2).
  2. Top spectral analogues from `Library('work')` with modified cosine $> 0.70$.
* **Algorithm:**
  ```python
  from rdkit import Chem
  from rdkit.Chem.Scaffolds import MurckoScaffold

  unique_scaffolds = {}
  for smi in candidate_smiles:
      m = Chem.MolFromSmiles(smi)
      if m is None or m.GetNumAtoms() == 0:
          continue
      scaf = MurckoScaffold.GetScaffoldForMol(m)
      if scaf is None or scaf.GetNumAtoms() == 0:
          continue  # Skip acyclic molecules (safeguard)
      can_scaf_smi = Chem.MolToSmiles(scaf)
      if can_scaf_smi not in unique_scaffolds:
          unique_scaffolds[can_scaf_smi] = scaf
  ```
* **Performance:** Processing 25 SMILES takes $<5\text{ ms}$.

---

### Step 2: Exact Mass Deficit ($\Delta M$) Computation
For each unique scaffold $S_j$:
$$M_{\text{scaffold}} = \text{ExactMolWt}(S_j)$$
$$\Delta M_j = M_{\text{query}} - M_{\text{scaffold}}$$
* If $\Delta M_j < 0$: The scaffold itself is heavier than the target molecule (cannot be decorated; discard or prune).
* If $\Delta M_j \ge 0$: Proceed to knapsack decomposition.

---

### Step 3: Biogenetic Transformation Palette & Knapsack Solver

#### Biological Transformation Palette
| Transformation | Stoichiometry | Exact Monoisotopic Mass ($\Delta m_i$) | Biological Mechanism |
| :--- | :--- | :--- | :--- |
| **Hydroxylation** | $+\text{O}$ | $+15.994915\text{ Da}$ | Cytochrome P450 monooxygenase |
| **Methylation** | $+\text{CH}_2$ | $+14.015650\text{ Da}$ | $S$-adenosylmethionine (SAM) methyltransferase |
| **Methoxylation** | $+\text{OCH}_2$ | $+30.010565\text{ Da}$ | Direct aromatic methoxylation |
| **Carbonylation** | $+\text{CO}$ | $+27.994915\text{ Da}$ | Oxidation to ketone/quinone |
| **Acetylation** | $+\text{C}_2\text{H}_2\text{O}$ | $+42.010565\text{ Da}$ | Acetyl-CoA transferase |
| **Glucosylation / Hexose** | $+\text{C}_6\text{H}_{10}\text{O}_5$ | $+162.052823\text{ Da}$ | UDP-glucosyltransferase |
| **Rhamnosylation** | $+\text{C}_6\text{H}_{10}\text{O}_4$ | $+146.057909\text{ Da}$ | Deoxyhexosyl transfer |
| **Prenylation** | $+\text{C}_5\text{H}_8$ | $+68.062600\text{ Da}$ | Dimethylallyl diphosphate (DMAPP) transfer |
| **Dehydrogenation** | $-2\text{H}$ | $-2.015650\text{ Da}$ | Desaturase (double bond / ring formation) |

#### Integer Subset-Sum Knapsack Solver
Solve for integer coefficients $c_i \in [-2, 4]$ satisfying:
$$\left| \Delta M_j - \sum_{i=1}^K c_i \cdot \Delta m_i \right| \le M_{\text{query}} \times 10^{-5} \quad (\pm 10\text{ ppm})$$
Subject to:
$$\sum_{i=1}^K |c_i| \le 4 \quad (\text{Max mutation depth})$$

---

### Step 4: RDKit Combinatorial Reaction Execution
Apply the transformation vector $\vec{c} = (c_1, \dots, c_K)$ to the scaffold using RDKit `rdChemReactions`:

1. **Aromatic Hydroxylation:**
   ```python
   rxn_hydroxyl = AllChem.ReactionFromSmarts('[cH:1]>>[c:1][OH]')
   ```
2. **O-Methylation (Ether Formation):**
   ```python
   rxn_methyl_o = AllChem.ReactionFromSmarts('[#6:1][OX2H:2]>>[#6:1][OX2:2]C')
   ```
3. **C-Methylation:**
   ```python
   rxn_methyl_c = AllChem.ReactionFromSmarts('[cH:1]>>[c:1]C')
   ```
4. **Dehydrogenation (Desaturation):**
   ```python
   rxn_dehydro = AllChem.ReactionFromSmarts('[CH2:1][CH2:2]>>[CH:1]=[CH:2]')
   ```
5. **Glycosylation:** Attach $\beta$-D-glucopyranosyl onto available phenolic/hydroxyl oxygens.

* **Regioisomer Generation:** When multiple open carbons or oxygens exist, RDKit generates all distinct substitution patterns.

---

### Step 5: Validation, Deduplication & Quality Filtering
1. **Exact Mass Check:**
   $$\frac{|\text{ExactMolWt}(m) - M_{\text{query}}|}{M_{\text{query}}} \times 10^6 \le 10.0\text{ ppm}$$
2. **Chemical Sanity Filter:**
   * Reject structures with invalid valences.
   * Reject bridgehead double bonds (Bredt's rule violations).
   * Reject unstable peroxides (`[O][O]`) or reactive radicals.
3. **Metric Canonicalization & Deduplication:**
   ```python
   from rdkit.Chem import MolToInchiKey
   key14 = MolToInchiKey(mol)[:14]
   ```
   Deduplicate on `InChIKey14`. Keep the top 20–30 highest natural product likelihood structures.

---

### Step 6: Pipeline Rescoring & Isomer Disambiguation
Generated candidates are injected directly into the query's candidate list before downstream scoring:
1. **Feature Extraction (`fe_v4`):**
   * The `derivation` feature family (already loaded in `v4n`) featurizes the generation priors and distance to seed analogues.
2. **LightGBM Ranking:**
   * The 4-booster ensemble scores the generated molecules alongside Main Pool candidates.
3. **Forward MS/MS Simulation (ICEBERG + GLACIER):**
   * The generated isomers are passed to ICEBERG and GLACIER in Cell 6.
   * Theoretical MS/MS spectra are simulated and compared against experimental peaks.
   * The correct regioisomer achieves the highest forward simulation score and ranks into the top positions of `submission.csv`.

---

## 4. Execution Roadmap & Milestones

### Phase 1: Standalone Mutator Module (`delta_m_mutator.py`)
- [ ] Implement `MurckoScaffold` extraction and deduplication helper.
- [ ] Implement the `DeltaM_KnapsackSolver` using pre-calculated exact isotope weights.
- [ ] Implement RDKit Reaction SMARTS decoration engine with regioisomer enumeration.
- [ ] Unit test on 5 diverse CASMI validation molecules (flavonoid, alkaloid, terpenoid, peptide, lipid).
- [ ] Verify execution time per molecule is $<50\text{ ms}$.

### Phase 2: Offline Validation & Benchmark
- [ ] Run benchmark against validation queries with known Class 3 ground truth.
- [ ] Measure candidate recall recovery (% of cases where the true structure or its top-25 isomer is generated).
- [ ] Benchmark ICEBERG and GLACIER simulation throughput on generated candidates.

### Phase 3: Integration into `casmi26-v4n-inference (1).ipynb`
- [ ] Embed `delta_m_mutator.py` cleanly between Cell 2 (PubChem runner) and Cell 5 (Base candidate ranking).
- [ ] Wire the generated candidates into `BASE[mid]` candidate arrays.
- [ ] Ensure seamless handoff to Cell 6 (ICEBERG/GLACIER rerankers).
- [ ] Run full 400-molecule test pass to confirm:
  * Runtime $\le 5,500\text{ seconds}$ (well under the 9-hour Kaggle ceiling).
  * `submission.csv` strictly compliant: (400, 2), no NaNs, exactly 25 candidates per row.

---

## 5. Risk Assessment & Safeguards

| Potential Risk | Failure Mode | Mitigation Safeguard |
| :--- | :--- | :--- |
| **Acyclic Molecules** | Fatty acids / linear peptides have no rings; `MurckoScaffold` returns empty mol. | Check `mol.GetNumAtoms() > 0`. If empty, skip scaffold mutation and retain raw SMILES. |
| **Combinatorial Explosion** | Scaffolds with 10+ open positions and 4+ groups could produce $>500$ isomers. | Cap maximum generated isomers to **$N \le 20$** per scaffold, prioritized by chemical natural product heuristics. |
| **Exact Mass Drift** | Rounding errors in floating point arithmetic could violate 10 ppm. | Compute exact masses using 64-bit IEEE monoisotopic isotope weights. Enforce hard filter `tol <= 10.0 ppm`. |
| **GPU / Timeout Overhead** | Adding too many candidates to ICEBERG/GLACIER risks 9-hour Kaggle timeout. | Hard budget of maximum 25 novel generated candidates per query molecule entering forward simulation. |
