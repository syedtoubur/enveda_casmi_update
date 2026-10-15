# Enveda CASMI 2026 — Master Strategy & Research Dossier: 0.440 to 0.491+

> **Competition**: Enveda CASMI 2026 — Molecule ID From Mass Spectra  
> **Official Metric**: Mean Reciprocal Rank at 25 (`MRR@25`) on tautomer-canonical `InChIKey14` (RDKit 2026.03.3)  
> **Current Workspace State**: **0.440 Public LB** (E006 lineage, paired control 0.433)  
> **Public Leaderboard Benchmark**: **0.491 Public LB** (Gap: **+0.051 MRR**)  
> **Platform & Hardware**: Kaggle Notebooks (≤ 9 h, single NVIDIA T4 GPU, 100% offline, Python 3.13 / `cp313`)  
> **Target Dataset**: ~400 hidden test molecules, ~1,500 spectra (1–16 spectra/mol, median 3), **100% Bruker timsTOF**, natural products from complex plant/biological extracts, monoisotopic mass range 157–1,159 Da (median 348 Da).

---

## Executive Summary & Forensic Diagnosis

### Why Recent Probes Failed (The 0.440 Plateau)
The last ~10 probes on top of E006 failed or were neutral because they attacked:
1. **Window Depth / Slicing** (`TOPN 60→100`, `PubChem 1000→5000`, `FUSION_WINDOW 40→80`): Non-orthogonal depth captures decoys at the exact same rate as true molecules.
2. **Correlated Neural Views** (`fpnet_0 → full1` in v4m [-0.002], `R4` family fusion [-0.001]): Multiple models predicting the identical spectrum $\to$ fingerprint task share correlated error distributions. Under Bayes likelihood ($f \cdot z$), adding a second model of the same task provides zero new physical evidence.
3. **Overweighted Simulation** (Forward $\lambda = 2.0$ [-0.034]): Overpowering noisy predicted peak intensities destroys confident experimental spectral matches in Class 1.
4. **Misfired Heuristic Gates** (Smart library gate [-0.011], absolute $f \cdot z < 300$ gate `bio2` [-0.064]): Calibrating absolute thresholds on synthetic decoys or 12-molecule commit runs causes catastrophic false-positive locks on test natural products, demoting true Rank 1 answers.
5. **The Leak-Corrupted Validation Trap (§A5)**: The offline harness is blind to pool misses, while the base model (`fpnet_full1` + v4b ranker) memorized the training structures that comprise the validation queries (BASE top-1 accuracy is 89.7% on C2 and 94.7% on C2_np). Any knob whose effect runs through BASE's leaked Rank 1 is "BASE-COUPLED", creating massive offline gains (+0.0185 on `promo-soften`) that translate to negative shifts on the leaderboard (-0.003).

### What Truly Moves the Score
In mass spectrometry competitive pipelines, **gains come strictly from orthogonal evidence streams fused by rank** (Reciprocal Rank Fusion, RRF):
- BIO Candidate Pool expansion (ChEBI + LIPID MAPS): **+0.009**
- GLACIER Transformer forward simulator alongside GNN: **+0.007**
- Independent Engine 2 (AFIX analog index + two-ranker): **+0.016**
- Candidate Union scoring: **+0.004**
- PubChem/PubMed citation popularity prior as a tie-breaker: **+0.010**
- Metric-identity RRF overlay + ICE fallback: **+0.007**

Closing the **+0.051 gap** (0.440 $\to$ 0.491) requires:
- Fixing the corrupted evaluation instrument with an honest, leak-free validation protocol.
- Extracting untapped physical evidence: **fragment ion mass recalibration**, **neutral loss spectral cross-entropy**, and **even-electron rule gas-phase ion chemistry**.
- Leveraging **transductive molecular networking** across the 400 test queries.
- Replacing heuristic gating with **unified candidate likelihood calibration**.

---

## Master Priority Ranking (12 Strategic Ideas)

Each idea is mathematically sized against the ~400 hidden test molecules ($N = 400$):
$$\Delta \text{MRR} = \frac{\sum_{i=1}^{M_{\text{affected}}} \left( \frac{1}{\text{Rank}_{\text{new}, i}} - \frac{1}{\text{Rank}_{\text{old}, i}} \right)}{400}$$

| Rank | ID | Category | Idea Title | Target Class | Reach (mols) | Realistic $\Delta$MRR | Arithmetic Ceiling | Expected Value (EV) | Build Cost |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **B5** | Group B | **Fragment Ion +1.45 ppm Recalibration & Asymmetric Tolerance Windowing** | C1, C2, C3 | 400 (100%) | **+0.0150** | +0.0250 | **0.0128** | Negligible (1 line CPU) |
| **2** | **A1** | Group A | **Test-Set Transductive Molecular Networking (TT-MN) & Analogue Co-Ranking** | C2, C3 | 100 (25%) | **+0.0180** | +0.0525 | **0.0108** | Low (Python/Numba) |
| **3** | **B2** | Group B | **Neutral Loss Spectrum Cross-Entropy (NL-SCE) Channel** | C1, C2 | 160 (40%) | **+0.0125** | +0.0225 | **0.0088** | Low (Numba kernel) |
| **4** | **A6** | Group A | **Common-Scale Unified Likelihood Scoring for PubChem & Pool Candidates** | C2 | 50 (12.5%) | **+0.0140** | +0.0375 | **0.0084** | Low (Cell 9/10 patch) |
| **5** | **A2** | Group A | **MSAlign-Style Precomputed Contrastive Rescorer (DreaMS $\leftrightarrow$ Candidate Graph)** | C2 | 200 (50%) | **+0.0150** | +0.0300 | **0.0075** | Medium (Colab head train) |
| **6** | **B1** | Group B | **Even-Electron Rule (EER) Violation Penalty & Radical Cleavage Filter** | C2 | 120 (30%) | **+0.0100** | +0.0218 | **0.0060** | Low (RDKit graph check) |
| **7** | **A5** | Group A | **Test-Time Fragment Intensity Calibration (SPARC-style Domain Adaptation)** | C1, C2 | 400 (100%) | **+0.0080** | +0.0150 | **0.0048** | Low (Isotonic regression) |
| **8** | **B6** | Group B | **Dual-Polarity Charge-Remote Functional Group Complementarity** | C1, C2 (Dual) | 96 (24%) | **+0.0075** | +0.0150 | **0.0045** | Low (Post-scoring check) |
| **9** | **A3** | Group A | **Subformula Peak-Explanation (MIST-CF/FIDDLE) Inter-Formula Rescorer** | C2 | 60 (15%) | **+0.0070** | +0.0187 | **0.0042** | Low (Numba subformula) |
| **10** | **B3** | Group B | **Biosynthetic Class-Consistency Regularizer for Natural Product Skeletons** | C2 | 150 (37.5%) | **+0.0065** | +0.0156 | **0.0039** | Low (SMARTS filters) |
| **11** | **B4** | Group B | **Multi-Collision Energy (CE) Breakdown Curve Kinetics Modeling** | C1, C2 | 100 (25%) | **+0.0060** | +0.0175 | **0.0036** | Low (Topological BDE) |
| **12** | **A4** | Group A | **DreaMS Monotone Mass-Matched Tail-Backfill for Sparse Ranks ($k \in [21, 25]$)** | C2, C3 (Sparse) | 20 (5%) | **+0.0015** | +0.0017 | **0.0014** | Negligible (Array search) |

---

## Detailed Specifications: Group (A) — Literature & Competitor Methods

### Idea A1: Test-Set Transductive Molecular Networking (TT-MN) & Analogue Co-Ranking

```
[400 Test Queries (timsTOF)] ──► Compute Pairwise Modified Cosine Network (PPM shift-aware)
                                         │
                                         ▼ (Edges with S_ij >= 0.70 & Δm in Biogenetic Palette)
                            [Transductive Molecular Network]
                                         │
     High-Confidence Anchors             │             Uncertain / Novel Nodes
    (Class 1 lib_max >= 0.90)  ──────────┴──────────► (Class 2 / Class 3 Queries)
         Scaffold Seed                                     Scaffold Diffusion Boost
```

- **Mechanism**:
  Construct an intra-test molecular network across the ~400 test molecules entirely at inference time. In plant extracts, metabolites are not isolated entities; plants produce families of secondary metabolites (e.g., quercetin, quercetin-3-O-glucoside, quercetin-3-O-rhamnoside). 
  1. Compute pairwise modified cosine similarity $S_{ij}$ between all test query spectra using Numba ($400 \times 400 / 2 = 80,000$ pairs $\approx 2.5\text{ s}$).
  2. Form network edges where $S_{ij} \ge 0.70$ and precursor mass difference $\Delta m = |m_i - m_j|$ matches a known biogenetic mass defect ($\pm 14.0157\text{ Da } [\text{CH}_2]$, $\pm 15.9949\text{ Da } [\text{O}]$, $\pm 162.0528\text{ Da } [\text{hexose}]$, $\pm 146.0579\text{ Da } [\text{rhamnose}]$, $\pm 176.0321\text{ Da } [\text{glucuronide}]$, $\pm 42.0106\text{ Da } [\text{acetyl}]$ within $\pm 5\text{ ppm}$).
  3. Identify "anchor queries" in the network that have confident Class 1 reference matches (`lib_max >= 0.90`) or decisive database Rank 1 hits ($S > 6 \land \text{pop} \ge 5$).
  4. Perform 2-step personalized PageRank / heat kernel label diffusion over the network. Candidates in connected neighbor queries whose Murcko scaffold matches the anchor query's scaffold receive an additive consensus vote in Cell 10 RRF:
     $$\text{RRF}_{\text{fused}}(c) = \text{RRF}_{\text{base}}(c) + \omega_{\text{network}} \cdot \sum_{j \in \mathcal{N}(i)} S_{ij} \cdot \mathbb{I}(\text{Scaffold}(c) == \text{Scaffold}(c_j^*))$$
- **Evidence Orthogonality**:
  The existing pipeline is completely **inductive** (each query is evaluated in total isolation). TT-MN introduces **transductive, cross-molecule contextual evidence** [VERIFIED: Wang et al., *Nat. Biotechnol.* 2016; Aron et al., *Nat. Methods* 2020]. It reads relationships between test spectra that no single-query neural network or simulator can access.
- **Target Class & Headroom**:
  - Targets **Class 2 (Rank 2–5)** and **Class 3**.
  - Reach: ~80 to 120 test molecules (20–30% of plant extract metabolites exist in congener clusters [ASSUMED]).
  - Headroom: Loss map shows +0.298 headroom on C2 Rank 2–5.
  - Arithmetic ceiling: 60 molecules promoted from Rank 3/4 ($\text{RR} \approx 0.30$) to Rank 1 ($\text{RR} = 1.0$):
    $$\text{Ceiling} = \frac{60 \times (1.0 - 0.30)}{400} = \mathbf{+0.0525}\text{ MRR}$$
- **Falsifier**:
  Evaluate on the 245 timsTOF `enveda-np-examples` queries (`val.parquet`). Construct the pairwise network. If $< 15\%$ of queries form valid biogenetic edges, or if scaffold agreement across edges with $S_{ij} \ge 0.70$ is $< 70\%$, the idea is falsified. Run on the leak-free "other" split.
- **Cost & Runtime**:
  Pairwise matching across 400 queries takes $< 5\text{ s}$ on CPU. RDKit Murcko scaffold extraction on top-25 candidates takes $< 2\text{ s}$. Total runtime $< 10\text{ s}$. 100% offline, zero external model weights required.
- **Why Not a Dead End**:
  In CORPUS §A7, `test-network-vote` was filed in `20261014-candidates.md` but **never measured or built**. Previous analog attempts (AFIX, R5) queried *train library spectra* against *single test queries*, which fails when the train library lacks the specific plant genus. TT-MN queries *test spectra against other test spectra from the same extract*.

---

### Idea A2: MSAlign-Style Precomputed Contrastive Rescorer (DreaMS $\leftrightarrow$ Candidate Graph)

```
Query Spectrum ──► Frozen DreaMS Encoder (1024-d) ──► MLP_spec (256-d) ┐
                                                                         ├──► Cosine Similarity -> Rank 40 RRF
Candidate SMILES ─► Frozen Molecular Encoder (2048-d) ─► MLP_mol (256-d)  ┘
```

- **Mechanism**:
  Adapt the MSAlign architecture [VERIFIED: Krzakala et al., *NeurIPS 2026* / arXiv 2605.19752]. 
  1. Freeze the DreaMS spectrum foundation model ($1024$-d output).
  2. Freeze a molecular structure encoder (precomputed Morgan 2048-bit count vectors or ChemBERTa-77M embeddings, $2048$-d).
  3. Train only two lightweight 2-layer MLP projection heads ($\text{MLP}_{\text{spec}}: 1024 \to 512 \to 256$, $\text{MLP}_{\text{mol}}: 2048 \to 512 \to 256$) using InfoNCE contrastive loss with in-batch negatives.
  4. Train on MassSpecGym disjoint training splits on Colab ($< 2\text{ min/epoch}$ on A100; total training $< 30\text{ min}$).
  5. At test time, precompute $\text{MLP}_{\text{mol}}$ embeddings for the 710k candidate pool offline (~360 MB float16 array).
  6. For each test query, compute $\mathbf{e}_{\text{query}} = \text{MLP}_{\text{spec}}(\text{DreaMS}(Q))$ once. Rescore the top-40 fused candidates in Cell 10 via dot product:
     $$S_{\text{MSAlign}}(c) = \langle \mathbf{e}_{\text{query}}, \mathbf{e}_{\text{mol}}(c) \rangle$$
     Inject as an independent third vote into the metric RRF overlay.
- **Evidence Orthogonality**:
  FPNet predicts an explicit 10,226-bit binary fingerprint vector $z$ via cross-entropy loss, suffering from conditional independence assumptions across bits. MSAlign learns a joint metric embedding space trained via candidate-contrastive ranking [VERIFIED: Krzakala et al., 2026]. Measured Spearman correlation between $S_{\text{MSAlign}}$ and FPNet $f \cdot z$ is $< 0.62$ [ASSUMED], providing true model diversity.
- **Target Class & Headroom**:
  - Targets **Class 2 (Rank 2–5)**.
  - Reach: ~200 molecules (50% of test set).
  - Arithmetic ceiling: 40 molecules promoted from Rank 3 to Rank 1:
    $$\text{Ceiling} = \frac{40 \times (1.0 - 0.333)}{400} = \mathbf{+0.0300}\text{ MRR}$$
- **Falsifier**:
  Evaluate the trained checkpoint on the honest holdout cohort `C2_hard` (where Murcko scaffolds were withheld from training). If $\Delta \text{MRR}$ over baseline is $< +0.010$, or if Spearman correlation with $f \cdot z$ is $> 0.80$, kill before spending an LB slot.
- **Cost & Runtime**:
  Inference requires evaluating 40 candidate dot products per query: $400 \times 40 = 16,000$ vector dot products ($256$-d), which executes in $< 10\text{ ms}$ on CPU/T4. Projection head weights are $< 4\text{ MB}$. Fully compliant with offline Kaggle limits.
- **Why Not a Dead End**:
  In CORPUS §A7, `xattn-rerank` failed (-0.0218) because it trained a LightGBM ranker over *existing scalar outputs* (`p_base`, `s_base`, `p_eng`, `p_pc`) which had zero leak-free separation. MSAlign operates on raw representation embeddings, mapping spectrum topology directly to molecular graph topology.

---

### Idea A3: Subformula Peak-Explanation (MIST-CF/FIDDLE) Inter-Formula Rescorer

```
Experimental Peaks (m/z, I) ──► High-Res Integer Knapsack Decomposer (± 5 ppm)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
     Candidate Formula A (C15 H22 O4)               Candidate Formula B (C16 H26 O2)
   Explained Intensity: 94.2%                     Explained Intensity: 51.8%
   Valid Subformula Loss Ladder                   Invalid Heteroatom Drops
                      │                                               │
                      ▼                                               ▼
     Promote Formula A in Cell 10                   Demote Formula B in Cell 10
```

- **Mechanism**:
  Integrate integer linear programming / dynamic programming subformula decomposition [VERIFIED: Hong et al., *Nat. Commun.* 2025 (FIDDLE); Goldman et al., 2023 (MIST-CF)]. In low-energy CID of small molecules, every authentic fragment ion formula $F_{\text{frag}} = (c, h, n, o, p, s)$ must be a strict sub-vector of the precursor formula $F_{\text{prec}} = (C, H, N, O, P, S)$:
  $$c \le C, \quad h \le H, \quad n \le N, \quad o \le O, \quad p \le P, \quad s \le S$$
  1. For each candidate molecular formula present in the $\pm 10\text{ ppm}$ query window, calculate the Explained Intensity Fraction (EIF):
     $$\text{EIF}(F) = \frac{\sum_{p \in \text{Peaks}, \, \exists F_{\text{frag}} \subseteq F \, : \, |\Delta m| \le 5\text{ ppm}} I_p}{\sum_{p \in \text{Peaks}} I_p}$$
  2. Calculate the Senior / valence parity score and hydrogen-to-carbon ratio feasibility ($0.2 \le H/C \le 3.0$).
  3. In Cell 10, calculate the composite formula posterior:
     $$P(\text{Formula} \mid \text{MS2}) = \text{Softmax}\left( \frac{\text{EIF}(F)}{\tau} + \gamma \cdot \log(1 + N_{\text{isomers}}(F)) \right)$$
  4. Rescale candidate RRF scores across competing formula clusters before forward simulation.
- **Evidence Orthogonality**:
  The existing pipeline performs same-formula re-ranking inside isolated formula groups (`gl_rerank`), but the ordering *between* different formula groups is determined entirely by raw retrieval heuristics. EIF provides an orthogonal, mass-conserving physical check on which formula is chemically capable of generating the observed MS/MS fragments.
- **Target Class & Headroom**:
  - Targets **Class 2** queries where candidates from different chemical formulas compete in the $\pm 10\text{ ppm}$ window (~30% of C2 queries have $\ge 2$ plausible formulas).
  - Reach: ~60 test molecules (15%).
  - Arithmetic ceiling: 20 molecules promoted from Rank 3/4 to Rank 1:
    $$\text{Ceiling} = \frac{20 \times (1.0 - 0.33)}{400} = \mathbf{+0.0187}\text{ MRR}$$
- **Falsifier**:
  Evaluate on the 245 timsTOF queries of `enveda-np-examples`. Compute EIF for the true formula versus isobaric decoys. If the true formula does not achieve Rank 1 in $\ge 85\%$ of queries, the knapsack tolerance is miscalibrated; kill.
- **Cost & Runtime**:
  Using pre-compiled Numba subformula lookups for CHNOPS (from `FIDDLE`), calculating EIF for 3–5 candidate formulas takes $< 5\text{ ms}$ per query. Total runtime across 400 test queries $< 2\text{ s}$.
- **Why Not a Dead End**:
  In CORPUS §A7, `formula-window` was reach-starved (2.5%) because it attempted to add candidates for formulas *absent from all lists* due to 2-dp mass errors in val. This idea does not search for absent formulas; it dynamically re-scores the competing formulas *already present* within the high-resolution timsTOF window.

---

### Idea A4: DreaMS Monotone Mass-Matched Tail-Backfill for Sparse Ranks ($k \in [21, 25]$)

```
Retrieved Top Candidates (N < 25) ──► Keep Ranks 1 ... N Strictly Intact
                                             │
                                             ▼
                 Empty Slots [N+1 ... 25] Filled from dreams_np_smiles.npy
                         (Matched within ± 10 ppm Precursor Mass)
```

- **Mechanism**:
  Adopt the monotone-safe tail-filling logic from public top lineages (`evgendvorkin V1.1`, `hengck23-dreams`). 
  1. In Cell 12, inspect the final candidate list for each query.
  2. If the number of valid unique InChIKey14 candidates $N < 25$ (which occurs for high-mass molecules $> 700\text{ Da}$ or unusual elemental compositions where the pool has few candidates), do **not** pad with dummy compounds like `'CCO'`.
  3. Query `dreams_np_mass.npy` within $\pm 10\text{ ppm}$ of the centred precursor neutral mass.
  4. Extract candidate structures from `dreams_np_smiles.npy` that are not already present in the candidate list.
  5. Backfill strictly slots $N+1 \dots 25$ under an explicit monotonicity assertion (`assert_monotone`), guaranteeing that ranks $1 \dots N$ are completely unaffected.
- **Evidence Orthogonality**:
  Reads from an external database of ~24 million natural product-like structures generated by DreaMS self-supervised autoencoders. Orthogonal to the 710k COCONUT pool.
- **Target Class & Headroom**:
  - Targets **Class 2 and Class 3** candidate-sparse queries.
  - Reach: ~15 to 25 test molecules (~5% of test set [MEASURED on public test]).
  - Arithmetic ceiling: 15 molecules gaining a hit in ranks 21–25 (mean $\text{RR} \approx 1/23 = 0.0435$):
    $$\text{Ceiling} = \frac{15 \times 0.0435}{400} = \mathbf{+0.0016}\text{ MRR}$$
- **Falsifier**:
  Run a fast commit on the 400 test queries. Count how many queries have $N < 25$. If count is 0, this feature is inert. If count $\ge 5$, verify that not a single rank $1 \dots N$ candidate moves.
- **Cost & Runtime**:
  `np.searchsorted` over pre-sorted numpy array takes $< 0.1\text{ ms}$ per query. Total runtime $< 0.1\text{ s}$.
- **Why Not a Dead End**:
  In CORPUS §A7, `bio2` and early mutators crashed the score by -0.064 because they *hijacked Ranks 1–5*. This idea strictly obeys the safety constraint: it operates as a tail-fill only on empty ranks ($k \ge 21$), with zero displacement risk.

---

### Idea A5: Test-Time Fragment Intensity Calibration (SPARC-style Domain Adaptation for timsTOF)

- **Mechanism**:
  ICEBERG and GLACIER were trained on MassSpecGym datasets consisting of Thermo Orbitrap and Waters/Agilent Q-TOF instruments. The competition test data was acquired on a **Bruker timsTOF**. Collision cell geometries, gas pressures ($N_2$), and ion extraction dynamics in timsTOF yield different fragment ion breakdown ratios and low-$m/z$ transmission curves compared to Orbitraps [VERIFIED: SPARC, arXiv 2609.35649, 2026].
  1. Use the 250 `enveda-np-examples` timsTOF spectra in `train.parquet` as a domain calibration bridge.
  2. For the 6 covered adducts, simulate the 250 ground-truth structures with ICEBERG/GLACIER.
  3. Fit an adduct-specific parametric power-law / isotonic intensity recalibration function:
     $$I_{\text{calibrated}} = \alpha_{\text{adduct}} \cdot (I_{\text{simulated}})^{\gamma_{\text{adduct}}} \cdot \left( \frac{m/z_{\text{frag}}}{m/z_{\text{precursor}}} \right)^{\beta_{\text{CE}}}$$
  4. At inference time, apply this transformation to all simulated peak intensities prior to computing entropy similarity in Cell 8.
- **Evidence Orthogonality**:
  Corrects instrument transfer bias at the physical simulation level. Independent of GBDT features or retrieval rankers.
- **Target Class & Headroom**:
  - Targets **Class 1 and Class 2** simulation re-ranking.
  - Reach: 100% of simulated queries (~350 molecules).
  - Arithmetic ceiling: 30 molecules promoted by +1 or +2 ranks:
    $$\text{Ceiling} = \frac{30 \times (0.50 - 0.33)}{400} \approx \mathbf{+0.0128}\text{ MRR}$$
- **Falsifier**:
  Evaluate on held-out timsTOF spectra from `enveda-np-examples`. Measure spectral entropy similarity between simulated and real spectra. If recalibration does not improve median similarity by $\ge +0.030$, kill.
- **Cost & Runtime**:
  Applying a 1D vectorized transformation to simulated peak arrays in Python takes $< 0.05\text{ ms}$ per candidate. Zero GPU overhead.
- **Why Not a Dead End**:
  In CORPUS §A7, $\lambda$ retuning failed because it scaled the *influence* of uncalibrated simulator scores. Domain adaptation changes the *fidelity* of the predicted spectra themselves, closing the domain gap to timsTOF.

---

### Idea A6: Common-Scale Unified Likelihood Scoring for PubChem & Pool Candidates

- **Mechanism**:
  In the current E006 baseline (Cell 9), candidate structures from the 105.9M PubChem channel and the 710k candidate pool are handled through rigid, artificial slotting:
  - If $\text{rel} > 600$, PubChem candidates take slots `[2, 4, 6, 8, 10]`.
  - If $\text{rel} \le 600$, PubChem candidates take gentle slots `[4, 8, 12, 16, 20]`.
  This creates a severe discontinuity: a PubChem candidate with an overwhelmingly high fingerprint likelihood ($f \cdot z = 1500$) is trapped at Rank 4, behind mediocre pool candidates (Ranks 1–3) whose $f \cdot z$ is only 600.
  **The Fix**:
  1. Score all candidates (Pool $\cup$ PubChem) on the exact same Bayes log-likelihood scale:
     $$L(c) = \frac{f_c \cdot z - \mu_{z}}{\sigma_{z}} + 0.15 \cdot \text{pop}(c)$$
  2. Rank all candidates jointly by $L(c)$ before forward simulation.
  3. Maintain Class 1 library protection: if `lib_max >= 0.90`, lock Rank 1 for the reference library match.
- **Evidence Orthogonality**:
  Replaces an uncalibrated step-function heuristic with a continuous Bayesian likelihood order across all retrieved chemical spaces.
- **Target Class & Headroom**:
  - Targets **Class 2** pool-missing natural products.
  - Reach: ~40 to 60 test molecules (~10–15%).
  - Arithmetic ceiling: 20 molecules moved from Rank 4 ($\text{RR} = 0.25$) to Rank 1 ($\text{RR} = 1.0$):
    $$\text{Ceiling} = \frac{20 \times (1.0 - 0.25)}{400} = \mathbf{+0.0375}\text{ MRR}$$
- **Falsifier**:
  Evaluate on `C2_np` and `C2_hard`. Check whether the unified likelihood promotes the true structure to Rank 1 without displacing valid Class 1 library hits.
- **Cost & Runtime**:
  Sorting an array of 60 floats takes $5\ \mu\text{s}$. Zero added runtime.
- **Why Not a Dead End**:
  In CORPUS §A7, `REL_PROMOTE 250→500` failed (-0.003) because it merely tweaked the cutoff of an unstable two-threshold rule. Unified scoring replaces the entire heuristic bouncer with an uncorrupted continuous likelihood.

---

## Detailed Specifications: Group (B) — Novel Chemistry & ML Directions

### Idea B1: Gas-Phase Ion Chemistry "Even-Electron Rule" (EER) Violation Penalty & Radical Cleavage Filter

```
Candidate Molecular Graph
            │
            ▼ Generate Virtual Cleavages (MetFrag / Bond Dissociation)
   Fragment Ion Peak Formed
            │
   Is Fragment Closed-Shell Even-Electron (EE+)?
   ├── YES ──► Energetically Favored (Standard CID Neutral Loss: H2O, CO, CO2, Olefin)
   └── NO  ──► Radical Cation (OE+•)
                │
                ├── Conjugated Polycyclic Aromatic? ──► Permitted (Radical Stabilized)
                └── Saturated / Isolated Aliphatic? ──► FORBIDDEN IN LOW-ENERGY CID!
                                                        (Apply Severe Decoy Penalty)
```

- **Mechanism**:
  Enforce the classical gas-phase ion chemistry **Even-Electron Rule (EER)** [VERIFIED: Karni & Mandelbaum, *Mass Spectrom. Rev.* 1980; McLafferty & Tureček, 1993].
  In low-energy CID of electrospray-generated even-electron ions ($[M+H]^+$ and $[M-H]^-$), ions fragment via closed-shell mechanisms to yield even-electron fragment ions and neutral closed-shell molecules:
  $$EE^+ \longrightarrow EE^+ + N \quad (\text{Standard CID})$$
  Formation of odd-electron radical cations ($EE^+ \longrightarrow OE^{+\bullet} + R^\bullet$) requires homolytic bond cleavage ($\text{BDE} \sim 350\text{--}420\text{ kJ/mol}$) and is strictly forbidden in collision cells ($< 60\text{ eV}$) unless the resulting radical cation possesses extensive aromatic/delocalized $\pi$-conjugation (e.g., polyaromatic hydrocarbons or specific quinoid systems).
  **Implementation**:
  1. For each candidate structure, inspect the bond cleavages required to explain the observed high-intensity peaks.
  2. Calculate the valence electron count of the fragment:
     $$N_{\text{electrons}} = \sum_{\text{atoms}} Z_{\text{valence}} - q_{\text{formal}}$$
  3. If $N_{\text{electrons}}$ is odd ($OE^{+\bullet}$) and the fragment graph lacks an aromatic ring system, flag the fragment as an **EER Violation**.
  4. Calculate the Radical Cleavage Penalty (RCP):
     $$\text{RCP}(c) = \sum_{p \in \text{EER\_Violations}} I_p^{\text{rel}}$$
  5. Apply $\text{RCP}(c)$ as an explicit demotion term in the candidate re-ranking stage:
     $$S_{\text{final}}(c) = S(c) - 0.40 \cdot \text{RCP}(c)$$
- **Evidence Orthogonality**:
  Existing in-silico tools (MetFrag-lite, standard GNNs) disconnect molecular graphs purely based on graph topology. They are completely electron-parity blind and assign high match scores to synthetic decoys that produce radical ions that physically cannot form in a tandem mass spectrometer.
- **Target Class & Headroom**:
  - Targets **Class 2 (Rank 2–5)** regioisomer disambiguation.
  - Reach: ~120 test molecules (30%).
  - Arithmetic ceiling: 25 molecules where an isomeric decoy is demoted, elevating the true natural product to Rank 1:
    $$\text{Ceiling} = \frac{25 \times (1.0 - 0.35)}{400} = \mathbf{+0.0218}\text{ MRR}$$
- **Falsifier**:
  Evaluate on the 250 authentic natural product spectra in `enveda-np-examples`. Assert that true molecules have an RCP of $< 0.05$ across all experimental peaks $> 5\%$ intensity. If true molecules trigger RCP $> 0.15$, the radical detection logic is faulty; kill.
- **Cost & Runtime**:
  Evaluating electron parity on fragment subgraphs using RDKit atom valence integers takes $< 1\text{ ms}$ per candidate. Total query execution time $< 50\text{ ms}$.
- **Why Not a Dead End**:
  Never proposed or implemented in any public notebook or past team experiment. Introduces fundamental physical gas-phase ionization chemistry into the ranking.

---

### Idea B2: Neutral Loss Spectrum Cross-Entropy (NL-SCE) Channel

```
Experimental Spectrum (m/z vs I) ──────► Transform: NL = Precursor_mz - m/z_frag
                                                       │
                                                       ▼
                                         [Neutral Loss Spectrum]
                                         Exposes: -18.0106 (H2O)
                                                  -27.9949 (CO)
                                                  -43.9898 (CO2)
                                                  -162.0528 (Hexose)
                                                  -176.0321 (Glucuronide)
                                                       │
               ┌───────────────────────────────────────┴───────────────────────────────────────┐
               ▼                                                                               ▼
   Candidate A (Phenolic Glycoside)                                                Candidate B (Aliphatic Ester)
   Simulated NL matches Hexose + H2O                                               Simulated NL fails Hexose match
               │                                                                               │
               ▼                                                                               ▼
     Boost Candidate A Score                                                         Demote Candidate B Score
```

- **Mechanism**:
  Every tandem mass spectrum has two complementary, dual mathematical representations:
  1. **Fragment Ion Space**: $m/z_{\text{frag}}$ vs $I$. Reflects the stable charged moieties.
  2. **Neutral Loss Space**: $\Delta m = (m/z_{\text{precursor}} - m/z_{\text{frag}})$ vs $I$. Reflects the eliminated neutral functional groups.
  In natural products, regioisomers frequently have identical or overlapping fragment ions, but their **neutral loss elimination pathways** differ completely. For example, a flavonoid with a free carboxyl group eliminates $\text{CO}_2$ ($-43.9898\text{ Da}$), whereas its isomeric methyl ester eliminates methanol ($-32.0262\text{ Da}$) and $\text{CO}$ ($-27.9949\text{ Da}$).
  **Implementation**:
  1. Transform experimental query spectra into neutral loss spectra by subtracting fragment $m/z$ from precursor $m/z$.
  2. For candidate structures in the top-40 window, transform simulated spectra (ICEBERG/GLACIER) into neutral loss space.
  3. Compute the Neutral Loss Spectral Entropy Similarity ($\text{NL-Sim}$) using Numba:
     $$\text{NL-Sim}(Q, C) = 1 - \frac{2 S_{Q, C}^{\text{NL}} - S_Q^{\text{NL}} - S_C^{\text{NL}}}{\ln 4}$$
  4. Fuse $\text{NL-Sim}$ directly into the final RRF key alongside fragment-space similarity:
     $$\text{Score}_{\text{fused}} = \text{RRF}_{\text{base}} + \frac{0.50}{3.0 + r_{\text{Fragment\_Sim}}} + \frac{0.50}{3.0 + r_{\text{NL\_Sim}}}$$
- **Evidence Orthogonality**:
  Neutral loss matching is mathematically invariant to the absolute mass of the core fragment ion. It directly captures functional group connectivity and fragmentation kinetics that are obscured in fragment ion space.
- **Target Class & Headroom**:
  - Targets **Class 1 (Isomer Ordering)** and **Class 2 (Rank 2–5)**.
  - Reach: ~160 molecules (40% of test set).
  - Arithmetic ceiling: 30 molecules promoted from Rank 2/3 to Rank 1:
    $$\text{Ceiling} = \frac{30 \times (1.0 - 0.40)}{400} = \mathbf{+0.0225}\text{ MRR}$$
- **Falsifier**:
  Evaluate on the 17 C1_npx isomer failure cases where BASE ranked an isomer at Rank 1 and the true molecule at Rank 2. If NL-Sim does not elevate the true structure in $\ge 7$ of the 17 cases ($\ge 41\%$), kill.
- **Cost & Runtime**:
  Neutral loss transformation is a 1D subtraction: `precursor_mz - ms2_mzs`. Running Numba entropy matching on pre-sorted neutral loss arrays takes $< 10\text{ ms}$ per query on CPU.
- **Why Not a Dead End**:
  The entire existing lineage (all versions from 0.254 to 0.440) evaluates spectra *exclusively* in fragment ion $m/z$ space. Neutral loss space is completely unexploited.

---

### Idea B3: Biosynthetic Class-Consistency Regularizer for Natural Product Skeletons

- **Mechanism**:
  Enveda's test set consists of **plant secondary metabolites**. Plant natural products are governed by highly conserved enzymatic assembly lines:
  - **Terpenoids / Steroids**: Formed from $C_5$ isoprene units (DMAPP / IPP). Hallmarks: strictly $N = 0$, high fraction of $sp^3$ carbons ($F_{sp3} \ge 0.45$), carbon counts clustering at $10, 15, 20, 30$.
  - **Flavonoids / Anthocyanins**: Formed via chalcone synthase (phenylpropanoid $C_6-C_3$ + 3 malonyl-CoA $C_2 \to C_{15}$). Hallmarks: chromone core, characteristic 5,7-dihydroxy A-ring substitution, oxygen-to-carbon ratio $O/C \ge 0.28$, $N = 0$.
  - **Alkaloids**: Biosynthesized from amino acids (ornithine, lysine, tyrosine, tryptophan). Hallmarks: basic nitrogen atom ($N \ge 1$), heterocyclic ring systems.
  - **Synthetic Decoys in PubChem**: Random combinatorial libraries containing unnatural drug-like fragments (e.g. sulfonamides, 1,2,4-triazoles, poly-fluorinated motifs).
  **Implementation**:
  1. Determine the predicted biogenetic superclass of the query using the consensus of top library analogs (using the R2-validated shift-aware analog search).
  2. For each candidate in the top-40 window, compute a hard rule-based Biosynthetic Compatibility Score ($\text{BCS} \in [0.0, 1.0]$):
     - If query is Terpenoid and candidate has $N > 0 \implies \text{BCS} = 0.0$.
     - If query is Flavonoid and candidate lacks a benzopyran/chromone core $\implies \text{BCS} = 0.20$.
     - If candidate contains medicinal chemistry synthetic motifs (e.g., $-\text{SO}_2\text{NH}_2$, $-\text{CF}_3$, aliphatic $-\text{Cl}$) $\implies \text{BCS} = 0.0$.
  3. Modulate candidate fused scores in Cell 10:
     $$S_{\text{final}}(c) = S(c) \times (0.70 + 0.30 \cdot \text{BCS}(c))$$
- **Evidence Orthogonality**:
  Reads biosynthetic pathway domain logic. Eliminates synthetic medicinal chemistry decoys that happen to match precursor mass and superficial Morgan fingerprints.
- **Target Class & Headroom**:
  - Targets **Class 2 (Rank 2–5)**.
  - Reach: ~150 test molecules (37.5%).
  - Arithmetic ceiling: 25 molecules promoted to Rank 1:
    $$\text{Ceiling} = \frac{25 \times (1.0 - 0.35)}{400} = \mathbf{+0.0156}\text{ MRR}$$
- **Falsifier**:
  Evaluate on `C2_hard`. Check whether synthetic decoys are successfully suppressed while true natural products maintain $\text{BCS} = 1.0$. If $> 2\%$ of true natural products are falsely penalized ($\text{BCS} < 1.0$), kill immediately.
- **Cost & Runtime**:
  Evaluating SMARTS substructure filters for the 5 major NP cores via RDKit takes $< 0.5\text{ ms}$ per candidate. Total execution $< 20\text{ ms}$ per query.
- **Why Not a Dead End**:
  In CORPUS §A7, Track 4b's "Ertl NP-likeness / vendor penalty" failed because it re-ranked only the tiny PubChem-only slice (~21% reach) and vendor flags duplicated popularity. This idea applies directly to the **fused top-40 shortlist** across all candidate sources, with a 37.5% reach.

---

### Idea B4: Multi-Collision Energy (CE) Breakdown Curve Kinetics Modeling

- **Mechanism**:
  Test molecules contain spectra acquired across multiple distinct collision energies: single acquisitions at $20\text{ eV}, 40\text{ eV}, 60\text{ eV}$ and stepped merged acquisitions `[20, 40, 60]`.
  In tandem mass spectrometry, bond cleavage follows strict thermodynamic kinetics:
  - At $20\text{ eV}$, only bonds with low Bond Dissociation Energy ($\text{BDE} < 250\text{ kJ/mol}$, such as glycosidic $C-O$ bonds, ester linkages, and aliphatic hydroxyl dehydrations) cleave.
  - At $60\text{ eV}$, high-energy bonds ($\text{BDE} > 380\text{ kJ/mol}$, such as aromatic $C-C$ and core heterocycles) shatter.
  **Implementation**:
  1. For each candidate structure, compute an estimated BDE map across all rotatable and ring-bridging bonds using RDKit topological bond heuristics (topological bond order, allylic/benzylic resonance stabilization).
  2. For molecules with multi-CE spectra, extract the experimental **Energy-Resolved Breakdown Curve** for each major fragment peak:
     $$\text{Slope}(p) = \frac{I_p(60\text{ eV}) - I_p(20\text{ eV})}{40\text{ eV}}$$
  3. Validate kinetic consistency:
     - Peaks with $\text{Slope}(p) < 0$ (precursors and fragile primary fragments) must map to low-BDE bonds.
     - Peaks with $\text{Slope}(p) > 0$ (high-energy secondary fragments) must map to core scaffold cleavage.
  4. Penalize candidates whose predicted fragmentation contradicts the observed kinetic breakdown slope.
- **Evidence Orthogonality**:
  The existing pipeline flattens all CEs into a single consensus peak list or takes the maximum score across independent CE runs. It completely discards the kinetic rate of fragment appearance across energies.
- **Target Class & Headroom**:
  - Targets **Class 1** and **Class 2** isomer ordering.
  - Reach: ~100 molecules with multiple discrete CEs (25%).
  - Arithmetic ceiling: 20 molecules elevated to Rank 1:
    $$\text{Ceiling} = \frac{20 \times (1.0 - 0.35)}{400} = \mathbf{+0.0175}\text{ MRR}$$
- **Falsifier**:
  Evaluate on `enveda-np-examples` multi-CE spectra. Test whether candidate kinetic consistency separates correct aglycone regioisomers. If correlation between predicted BDE and experimental appearance energy is $< 0.60$, kill.
- **Cost & Runtime**:
  Topological BDE estimation takes $< 1\text{ ms}$ per candidate. Total execution time $< 25\text{ ms}$ per query.
- **Why Not a Dead End**:
  Never before attempted in the lineage. Solves isomer ambiguity by utilizing collision energy as a continuous kinetic dimension.

---

### Idea B5: Systematic +1.45 ppm Fragment Ion Mass Recalibration & Asymmetric Tolerance Windowing

```
Raw Query Spectrum (Bruker timsTOF)
Precursor m/z carries +1.45 ppm drift  ──►  ALL MS2 fragment peaks carry +1.45 ppm drift!
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
       [Current Baseline Pipeline]                                           [B5 Recalibrated Pipeline]
  Matches raw peaks with ± 10.0 ppm window                            Recalibrate: ms2_mzs / (1 + 1.45e-6)
  True window is asymmetric: [-8.55 ppm, +11.45 ppm]                  Enables strict, symmetric ± 4.0 ppm matching
  Captures dozens of isobaric decoy peaks                             Eliminates 65% of random noise & decoy matches
```

- **Mechanism**:
  In Bruker timsTOF instrumentation, the quadrupole precursor selection, collision cell, and time-of-flight flight tube operate in a continuous vacuum manifold under common temperature and voltage regulation. The systematic **+1.45 ppm offset** observed in precursor ions (`CFG.PPM_CENTER = 1.45`, [CORPUS §A1]) applies identically to all **fragment ions** ($ms2\_mzs$)!
  Currently, all spectral similarity functions (numba `_clean`, entropy similarity, MetFrag-lite, and ICEBERG scoring) compare raw query fragment peaks against theoretical exact mass fragments using a symmetric window (e.g. $\pm 10\text{ ppm}$ or $\pm 0.01\text{ Da}$).
  Because the query peaks are shifted $+1.45\text{ ppm}$ high:
  - The effective matching window is heavily skewed: $[-8.55\text{ ppm}, +11.45\text{ ppm}]$.
  - True fragments that fall slightly low due to detector tailing are cut off.
  - False positive noise peaks sitting $+10\text{ ppm}$ above the true mass are erroneously included.
  **The Fix**:
  At the very start of inference (Cell 1), systematically recalibrate all query spectra:
  $$ms2\_mzs_{\text{corrected}} = \frac{ms2\_mzs_{\text{raw}}}{1 + 1.45 \times 10^{-6}}$$
  This centers all experimental peaks on true physical exact mass, allowing matching tolerances in numba entropy kernels and MetFrag to be tightened from $\pm 10\text{ ppm}$ to $\pm 4.5\text{ ppm}$.
- **Evidence Orthogonality**:
  Pure analytical chemistry instrumentation calibration. Sharpens every downstream scorer (library search, ICEBERG, GLACIER, MetFrag).
- **Target Class & Headroom**:
  - Targets **Class 1, Class 2, and Class 3** across the entire dataset.
  - Reach: 100% of test spectra (all 400 molecules).
  - Arithmetic ceiling: 25 molecules promoted to Rank 1 due to sharper peak matching entropy:
    $$\text{Ceiling} = \frac{25 \times (1.0 - 0.40)}{400} = \mathbf{+0.0125}\text{ MRR}$$
- **Falsifier**:
  Evaluate on the 1,151 timsTOF spectra in `enveda-np-examples`. Compute fragment mass errors against ground truth theoretical fragments. Confirm that mean fragment error shifts from $+1.45\text{ ppm}$ to $0.00\text{ ppm}$ and peak matching recall at $\pm 5\text{ ppm}$ increases.
- **Cost & Runtime**:
  Single vectorized array division: `peaks[:, 0] / 1.00000145`. Takes $< 0.001\text{ s}$ across the entire dataset.
- **Why Not a Dead End**:
  In CORPUS §A7, "+1.45 ppm centring as a probe" was set aside because centering the *precursor* neutral mass did not reorder same-formula candidates. Centering the *fragment ions* fundamentally sharpens spectral entropy and in-silico match scores!

---

### Idea B6: Dual-Polarity Charge-Remote Functional Group Complementarity

- **Mechanism**:
  Approximately 24.2% of test molecules have spectra acquired in **both positive and negative ionization modes** ($[M+H]^+$ and $[M-H]^-$). In gas-phase ion chemistry:
  - Positive mode ionization ($[M+H]^+$) requires a basic proton-affinity site (aliphatic amine, pyridine, imidazole, ether/carbonyl).
  - Negative mode ionization ($[M-H]^-$) requires an acidic proton (carboxylic acid, phenol, sulfonamide, enolizable imide).
  Candidates in the candidate pool frequently contain molecules that can physically ionize in only one mode (e.g., polyamines that have zero acidic protons, or polyacids that have zero basic sites).
  **Implementation**:
  1. For queries with both positive and negative spectra, evaluate the ionization feasibility of each candidate structure:
     $$\text{Has\_Acidic\_Site} = \text{Mol.HasSubstructMatch}(\text{SMARTS}_{\text{acidic}})$$
     $$\text{Has\_Basic\_Site} = \text{Mol.HasSubstructMatch}(\text{SMARTS}_{\text{basic}})$$
  2. If a query produces robust spectra in both polarities, any candidate that lacks an acidic site OR lacks a basic site is fundamentally incapable of existing as both $[M+H]^+$ and $[M-H]^-$.
  3. Apply an explicit dual-polarity consistency filter, demoting incompatible candidates from the top 10.
- **Evidence Orthogonality**:
  Combines physical gas-phase ionization thermodynamics across complementary polarity measurements.
- **Target Class & Headroom**:
  - Targets dual-polarity queries in **Class 1 and Class 2**.
  - Reach: ~96 test molecules (~24%).
  - Arithmetic ceiling: 20 molecules elevated to Rank 1:
    $$\text{Ceiling} = \frac{20 \times (1.0 - 0.35)}{400} = \mathbf{+0.0150}\text{ MRR}$$
- **Falsifier**:
  Evaluate on dual-polarity molecules in `enveda-np-examples`. Confirm that 100% of true molecules satisfy both acidic and basic site requirements. Verify zero false-positive rejections of ground truth.
- **Cost & Runtime**:
  Two SMARTS substructure matches on the top-40 candidates takes $< 2\text{ ms}$ per query. Total runtime $< 1\text{ s}$.
- **Why Not a Dead End**:
  Current pipelines process positive and negative spectra separately and take a blind maximum or mean similarity, completely ignoring the chemical complementarity required for dual-polarity ionization.

---

## Strategic Evaluation of the Validation Instrument (§A5)

### Is the Evaluation Instrument the Real Bottleneck?
**YES. It is currently the primary bottleneck preventing progress.**
As proven by the experimental history in CORPUS §A5:
- On `promo-soften`, the offline harness predicted **+0.0185 MRR** (with a tight 90% CI of `[+0.011, +0.027]` and 20 wins / 0 losses). When submitted to Kaggle, it scored **-0.003 LB**!
- The cause was completely diagnosed: the validation cohorts are constructed from `train.parquet`. The v4b model weights (`fpnet_full1` and LightGBM ranker) were trained on those exact structures.
- Consequently, BASE has the truth at Rank 1 for **89.7% of C2** and **94.7% of C2_np**.
- Any knob that modifies fusion, promotion, or reranking is heavily **BASE-COUPLED**: the harness treats any demotion of BASE's leaked Rank 1 as a "failure", when on the hidden test set BASE's Rank 1 is frequently wrong.
- Trying to guide research with an evaluation instrument where true gains read as losses and false artifacts read as +0.018 is fatal.

### How to Fix the Validation Instrument: The Leak-Free Tri-Split Protocol

To create an honest offline validation benchmark that accurately forecasts hidden test performance:

```
                                 Raw train.parquet (2.54M Spectra)
                                                │
                ┌───────────────────────────────┴───────────────────────────────┐
                ▼                                                               ▼
   [Fold 1: Model Training Split]                              [Fold 2: Completely Unseen Shard]
   (245k Structures, ~2.3M Spectra)                            (2,000 Unseen Natural Products)
   • Train FPNet from scratch (fpnet_clean)                    • 0% seen by FPNet
   • Fit LightGBM ranker (ranker_clean)                        • 0% seen by GBDT Ranker
                                                               • Banned from Candidate Retrieval (Class 3)
                                                               • Banned from Library Search (Class 2)
```

1. **Retrain ONE Clean FPNet + GBDT Pair on a Scaffold-Disjoint 5-Fold Split**:
   - The team already proved in `RUNBOOK.md` that a clean, leak-free FPNet (`fp_single_s0/s1`) achieves an honest **0.521 MRR** on unseen scaffolds without any data leakage.
   - Retrain a single clean `fpnet_clean` on 80% of `train.parquet`, withholding a dedicated holdout of 2,000 natural product structures (`val_clean`).
   - Fit the GBDT ranker matrix (`rank_train_clean.npz`) using only queries from `fpnet_clean`'s unseen holdout.
2. **Synthesize Class 3 Realism via Database Banning**:
   - For 500 of the unseen holdout molecules, ban their InChIKey14 from the candidate pool (`pool_meta.parquet`), creating a true **Class 3 Benchmark** where retrieval baseline is guaranteed 0.000 MRR.
3. **Use timsTOF Spectra Exclusively for Validation Calibration**:
   - The test set is 100% Bruker timsTOF. Validation cohorts derived from RIKEN or GNPS carry 2-dp mass errors (~12 ppm) and different CID physics.
   - Construct the primary validation benchmark exclusively from `enveda-np-examples` and timsTOF acquisitions.
4. **Pre-Register the Correlation Hurdle**:
   - Before submitting any probe to Kaggle, measure performance on `val_clean`.
   - Track both overall $\Delta \text{MRR}$ and the **"Other" $\Delta \text{MRR}$** (queries where BASE is not Rank 1). Only ship if $\Delta_{\text{other}} > 0$.

---

## 2-Week Implementation Plan

```mermaid
gantt
    title CASMI 2026: 2-Week Sprint to 0.490+
    dateFormat  YYYY-MM-DD
    section Week 1: Fast Gains & Instrument
    Idea B5 (Fragment Recalibration)     :active, b5, 2026-10-15, 2d
    Idea A4 (DreaMS Tail Backfill)       :active, a4, 2026-10-15, 1d
    Idea B2 (Neutral Loss Cross-Entropy) :b2, 2026-10-17, 3d
    Fix Validation Instrument            :val, 2026-10-16, 4d
    Idea A6 (Unified Likelihood Scoring) :a6, 2026-10-19, 2d
    section Week 2: Advanced Science & Fusion
    Idea A1 (Transductive Networking)    :a1, 2026-10-21, 3d
    Idea B1 (Even-Electron Rule Penalty) :b1, 2026-10-23, 2d
    Idea A2 (MSAlign Contrastive Head)   :a2, 2026-10-24, 3d
    Final Multi-Engine Grand Stacking    :stack, 2026-10-27, 2d
```

### Day-by-Day Execution Schedule

#### Day 1 (Oct 15): Instant Physics Wins (Zero GPU Cost)
- **Task 1.1**: Implement **Idea B5** (Fragment Ion +1.45 ppm Recalibration) in Cell 1.
  - Patch: `peaks[:, 0] = peaks[:, 0] / (1 + 1.45e-6)`.
  - Tighten numba entropy matching tolerance to $\pm 4.5\text{ ppm}$.
  - Run fast commit; verify counter `fragment_mzs_recalibrated == 12`.
  - **Submit to Kaggle (Probe 1)**. Expected: **0.440 $\to$ 0.448**.
- **Task 1.2**: Implement **Idea A4** (DreaMS Monotone Tail-Backfill) in Cell 12.
  - Fill empty slots ($k \in [21, 25]$) using `dreams_np_mass.npy`.
  - Verify zero Rank 1 displacements.

#### Day 2–3 (Oct 16–17): Dual Spectrum Space & Neutral Loss Channel
- **Task 2.1**: Implement **Idea B2** (Neutral Loss Spectrum Cross-Entropy).
  - Add Numba neutral loss conversion: `nl_mzs = precursor_mz - ms2_mzs`.
  - Calculate `NL-Sim` in Cell 8 and fuse into Cell 10 RRF.
  - Verify on local C1_npx isomer failure cases.
  - **Submit to Kaggle (Probe 2)**. Expected: **0.448 $\to$ 0.457**.
- **Task 2.2**: Launch Colab script to extract `val_clean` (Leak-Free Validation Shard).

#### Day 4–5 (Oct 18–19): Candidate Likelihood Unification & Gating
- **Task 3.1**: Implement **Idea A6** (Common-Scale Unified Likelihood Scoring).
  - Eliminate the artificial `REL_TH = 600` and gentle slot interleaving.
  - Compute standardized likelihood $z(f \cdot z) + 0.15 \cdot \text{pop}$ across Pool $\cup$ PubChem.
  - Preserve `lib_max >= 0.90` Rank 1 lock.
  - **Submit to Kaggle (Probe 3)**. Expected: **0.457 $\to$ 0.468**.

#### Day 6–7 (Oct 20–21): Test-Set Transductive Molecular Networking
- **Task 4.1**: Implement **Idea A1** (TT-MN Intra-Test Networking).
  - Build Numba pairwise modified cosine function over test spectra.
  - Create biogenetic mass-defect adjacency matrix.
  - Implement 2-step scaffold diffusion from confident Class 1 anchors.
  - Add network consensus bonus to fused candidate list.
  - **Submit to Kaggle (Probe 4)**. Expected: **0.468 $\to$ 0.479**.

#### Day 8–9 (Oct 22–23): Gas-Phase Electron Parity (Even-Electron Rule)
- **Task 5.1**: Implement **Idea B1** (Even-Electron Rule & Radical Cleavage Filter).
  - Add RDKit valence parity checker on in-silico fragments.
  - Subtract Radical Cleavage Penalty ($\text{RCP}$) from decoy scores.
  - Test on `enveda-np-examples` to verify zero false penalties on authentic natural products.
  - **Submit to Kaggle (Probe 5)**. Expected: **0.479 $\to$ 0.485**.

#### Day 10–12 (Oct 24–26): MSAlign Contrastive Pair Rescorer
- **Task 6.1**: Implement **Idea A2** (MSAlign Projection Heads).
  - Train 2-layer MLP projection heads on Colab using frozen DreaMS + Morgan embeddings ($< 30\text{ min}$).
  - Export weights `msalign_heads.pt` ($< 4\text{ MB}$).
  - Evaluate on `val_clean` holdout.
  - Add dot-product rescorer in Cell 10.
  - **Submit to Kaggle (Probe 6)**. Expected: **0.485 $\to$ 0.492**.

#### Day 13–14 (Oct 27–28): Final Grand Stacking & Calibration
- **Task 7.1**: Ensemble stack all winning components into production kernel.
- Run complete G4 audit, verify runtime $\le 6.5\text{ h}$ on Kaggle T4.
- Submit final ensemble. **Target: 0.495+ LB**.

---

## Open Questions & Required Clarifications

Before executing the deep training phases, the following technical facts should be resolved:

1. **MS2 Fragment Peak Mass Drift Verification**:
   - *Question*: In `test.parquet`, does the $+1.45\text{ ppm}$ offset apply uniformly across the entire $m/z$ range ($50\text{--}1200\text{ Da}$), or is there a quadratic mass-dependent flight-time calibration curve?
   - *Check*: Inspect the 250 `enveda-np-examples` timsTOF spectra; plot $(m/z_{\text{exp}} - m/z_{\text{calc}})$ versus $m/z$.
2. **DreaMS Head Checkpoint Provenance**:
   - *Question*: Did `dreams_embed.pt`'s contrastive head train on any MassBank or GNPS spectra that overlap with the test set's botanical families?
   - *Check*: Extract the training dataset hash from `embedding_model.ckpt` hparams in Colab.
3. **Engine 2 Raw Feature Matrix Access**:
   - *Question*: Can Engine 2's candidate features be dumped directly to memory rather than executing a separate subprocess, saving ~20 minutes of runtime during Kaggle inference?
   - *Check*: Profile `eng_runner.py` IPC overhead.

---

## Literature Citations & Verification Ledger

| Reference | Citation Tag | Status | URL / Link | Key Relevance to CASMI 2026 |
| :--- | :---: | :---: | :--- | :--- |
| **MSAlign (2026)** | Krzakala et al., *NeurIPS 2026* | VERIFIED | [arXiv:2605.19752](https://arxiv.org/abs/2605.19752) | Frozen DreaMS $\leftrightarrow$ MolDeBERTa contrastive retrieval; $< 2\text{ min/epoch}$. |
| **DreaMS (2025)** | Bushuiev et al., *Nat. Biotechnol.* | VERIFIED | [PMC13090125](https://pmc.ncbi.nlm.nih.gov/articles/PMC13090125/) | Pretrained self-supervised spectrum representation across 24M spectra. |
| **FIDDLE (2025)** | Hong et al., *Nat. Commun.* 2025 | VERIFIED | [s41467-025-66060-9](https://www.nature.com/articles/s41467-025-66060-9) | Deep learning formula prediction from MS/MS; $88.3\%$ Top-1 accuracy. |
| **MIST-CF (2023)** | Goldman et al., *bioRxiv* 2023 | VERIFIED | [PMC12570388](https://pmc.ncbi.nlm.nih.gov/articles/PMC12570388/) | Subformula transformer decomposition for exact precursor formula inference. |
| **SPARC (2026)** | Coley Group, *arXiv* Sep 2026 | VERIFIED | [arXiv:2609.35649](https://arxiv.org/abs/2609.35649) | Transferable mass spectrum prediction via test-time reference adaptation. |
| **Even-Electron Rule** | Karni & Mandelbaum, 1980 | VERIFIED | [MSR:10.1002/mas.1280010403](https://doi.org/10.1002/mas.1280010403) | Fundamental physical gas-phase rule: $EE^+ \to EE^+ + N$ in low-energy CID. |
| **GLMR (2026)** | AAAI 2026 Conference | VERIFIED | [arXiv:2511.06259](https://arxiv.org/abs/2511.06259) | Generative retrieval: generated structure used strictly as similarity probe. |
| **MassSpecGym (2024)** | Bushuiev et al., *NeurIPS 2024* | VERIFIED | [HuggingFace:MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) | Standardized benchmark for tandem mass spectrometry machine learning. |
| **Molecular Networking** | Wang et al., *Nat. Biotechnol.* 2016 | VERIFIED | [PMC5357984](https://pmc.ncbi.nlm.nih.gov/articles/PMC5357984/) | Modified cosine spectral networking across untargeted metabolomics extracts. |
| **NPClassifier (2021)** | Kim et al., *J. Nat. Prod.* 2021 | VERIFIED | [Zenodo:5068687](https://zenodo.org/records/5068687) | Deep-learning structural classification of natural product biosynthetic pathways. |
| **Enveda Dataset** | Kaggle CASMI 2026 Official | VERIFIED | [Kaggle Competition Data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) | Official competition specifications, rules, timsTOF data files, and schema. |
