# Roadmap & Phased Execution Plan: `specs_novel`

*Specification Version 1.0 — 2026-09-27. Follows strict gatekeeper protocols: each phase has objective criteria and automated verification tests before progressing.*

---

## Phase Overview & Progress Matrix

| Phase | Milestone Name | Primary Artifact / Output | Pass Criteria (Gate) | Status |
|---|---|---|---|---|
| **Phase 0** | **Class 1 Separator & Routing Calibration** | `test_separator.py`, `separator_calibration.json` | Separator precision $\ge 98.5\%$ on C1 holdout, latency $< 0.1\text{s}$/query | **PLANNED** |
| **Phase 1** | **De Novo Spectral Generator Fine-Tuning** | `novel_generator.pt`, `vocab.json` | Valid SMILES rate $\ge 85\%$ on unseen spectra; scaffold-disjoint | **PLANNED** |
| **Phase 2** | **Constrained Beam Search & DFS Engine** | `constrained_search.py`, `test_search.py` | 100% generated molecules within $\pm 8.5\text{ ppm}$; $\ge 20$ valid mols/query in $< 20\text{s}$ | **PLANNED** |
| **Phase 3** | **Unified Forward Simulation Arena (ICEBERG)** | `unified_arena.py`, `arena_benchmark.py` | ICEBERG + MetFrag successfully scores $N_{\text{total}} \approx 100$ candidates in $< 4\text{s}$ | **PLANNED** |
| **Phase 4** | **Kaggle Deployment & Controlled Ablation** | `casmi_novel_submission.ipynb`, `casmi_novel_control.ipynb` | Clean 1-flag ablation submission on Kaggle; runtime $\le 7.5\text{h}$; LB measured | **PLANNED** |

---

## Detailed Phase Breakdown

### Phase 0: Class 1 Separator & Routing Calibration

#### Objective
Establish and validate the decision boundary that segregates Class 1 (direct library search) from Class 2/3 (database retrieval + de novo generation). We must verify that `lv >= 0.85` has near-zero false positives (it must never mistakenly route a novel or Class-2 query to library emission).

#### Steps
1. **Calibration against Validation Cohorts**:
   - Evaluate spectral entropy similarity `lv` across the 1,145 validation molecules (`val_scripts/val_key.pkl`):
     - `C1_npx` (n=245, identical library twins).
     - `C1_xlib` (n=300, cross-library Class 1).
     - `C2` (n=400, no reference spectra).
     - `C2_hard` (n=200, structurally novel / distant decoys).
2. **Precision vs Recall Measurement**:
   - Compute False Discovery Rate at $\tau \in \{0.75, 0.80, 0.85, 0.90\}$.
   - Verify that at `lv >= 0.85`, the fraction of C2/C2_hard queries mistakenly claimed by the separator is $< 1.0\%$.
3. **Execution Latency Check**:
   - Measure the exact search speed of `L.search(query_mz, CFG.PPM_WIN)` with the pre-indexed array. Must execute in $< 50\text{ ms}$ per query.

#### Phase 0 Gate
- **G0.1**: At `lv >= 0.85`, C1 recall $\ge 70\%$ and C2 false-positive rate $\le 1.0\%$.
- **G0.2**: Precursor centering at $+1.45\text{ ppm}$ confirmed functional and tested against timsTOF test set spectra.

---

### Phase 1: De Novo Spectral Generator Adaptation

#### Objective
Prepare and adapt an autoregressive Encoder-Decoder Transformer conditioned on MS/MS spectra to generate SMILES strings.

#### Steps
1. **Vocabulary & Chemical Grammar Definition**:
   - Define a minimal, canonical SMILES token vocabulary:
     - Atoms: `C`, `N`, `O`, `S`, `P`, `F`, `Cl`, `Br`, `I`, `c`, `n`, `o`, `s`, `p`.
     - Bonds & Modifiers: `=`, `#`, `(`, `)`, `[`, `]`, `+`, `-`, `@`, `@@`, `/`, `\`.
     - Ring closures: `1` through `9`, `%10`...`%20`.
     - Special: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`.
   - Freeze vocabulary to `vocab.json` (size $V \approx 64$ tokens).
2. **Model Training / Fine-Tuning**:
   - Input: spectral peaks $(m/z, \text{intensity})$, precursor neutral mass, adduct token, collision energy (eV).
   - Backbone: 6-layer Spectral Transformer Encoder ($d_{\text{model}} = 512, h=8$) + 6-layer Autoregressive SMILES Decoder.
   - Training loss: Cross-entropy token loss on canonical SMILES strings from `train.parquet`.
   - Training split: Scaffold-disjoint split (`_fpnet/pack/splits.npz`) to guarantee the model learns structural decomposition rather than memorizing scaffolds.
3. **Quality Checkpoint**:
   - Check validation loss on unseen scaffolds.
   - Verify greedy decoding validity rate: $\ge 80\%$ of emitted SMILES must parse cleanly in RDKit.

#### Phase 1 Gate
- **G1.1**: The model trains without divergence; validation perplexity $\le 2.5$ on unseen scaffolds.
- **G1.2**: Checkpoint size $\le 250\text{ MB}$; exported as a standalone TorchScript or standard PyTorch state dict.

---

### Phase 2: Constrained Beam Search & Dynamic Prefix Mass Pruning

#### Objective
Implement the constrained tree search engine that guarantees all generated SMILES respect the physical precursor mass of the query within $\pm 8.5\text{ ppm}$.

#### Steps
1. **Dynamic Prefix Mass Calculator**:
   - Build a lookup table of monoisotopic atomic masses for all organic elements.
   - Given any SMILES prefix string $S_{1:t}$, compute the cumulative lower-bound mass $M_{\min}(S_{1:t})$.
   - Rule: An open branch is immediately pruned if $M_{\min}(S_{1:t}) > M_{\text{target}} \times (1 + 8.5 \times 10^{-6}) + 0.005\text{ Da}$.
2. **Constrained Beam Search Execution**:
   - Initialize beam with `<BOS>`.
   - Maintain top-$K$ beams ($K = 30$). At each expansion, mask out tokens that violate mass constraints or syntax rules (e.g. invalid ring closure numbers).
3. **DFS Completion for Bounded Leaves**:
   - When a beam reaches $M_{\min} \in [M_{\text{target}} - 30\text{ Da}, M_{\text{target}}]$, switch to a fast Depth-First Search (max depth 10) to find completing tokens (`<EOS>`, ring closure, terminal hydrogen saturation).
   - Validate completed SMILES with RDKit:
     ```python
     mol = Chem.MolFromSmiles(smiles, sanitize=True)
     if mol is not None:
         exact_mass = Descriptors.ExactMolWt(mol)
         if abs(exact_mass - target_mass) / target_mass <= 8.5e-6:
             ik14 = Chem.MolToInchiKey(mol)[:14]
             valid_candidates[ik14] = smiles
     ```
4. **Throughput Benchmark**:
   - Benchmark the engine on 50 representative test queries. Ensure median search time per query is $\le 15\text{ seconds}$ on a single GPU/CPU worker.

#### Phase 2 Gate
- **G2.1**: 100% of emitted novel molecules pass RDKit sanitization and land within $\pm 8.5\text{ ppm}$ of the target mass.
- **G2.2**: Generates at least 15 unique, valid candidates for $\ge 90\%$ of non-Class-1 queries.

---

### Phase 3: Unified Forward Simulation Arena (ICEBERG)

#### Objective
Validate that generated novel candidates and retrieved database candidates can be scored side-by-side using forward spectral simulation without distribution bias.

#### Steps
1. **Candidate Pool Merging**:
   - For each query:
     - Track A: Retrieve up to 100 database candidates from COCONUT / train pool.
     - Track B: Generate up to 50 novel candidates from Constrained Beam Search.
     - Combine both lists and deduplicate by `InChIKey14`.
2. **Batched Forward Simulation**:
   - Run ICEBERG (`msg_all`) in a batch subprocess (batch size 128) on GPU.
   - Throughput target: $\ge 30\text{ mol/s}$ on T4 GPU (matching measured 36.6 mol/s).
   - Simulate at query's exact CE (in eV) and adduct.
3. **Scoring & Ranking**:
   - Compute spectral entropy similarity $sv$ between predicted and experimental spectra.
   - Compute MetFrag-lite bond-disconnection score $fr$.
   - Compute composite score: $\text{score} = sv + 0.10 \times fr - 0.01 \times |\text{ppm\_error}|$.
4. **Ablation Validation on Synthetic Class 3 Benchmark**:
   - Create a synthetic test set: hide 100 known molecules from the candidate pool (simulating Class 3).
   - Evaluate MRR@25:
     - Baseline (retrieval only): MRR on hidden molecules is strictly **0.000**.
     - Specs Novel (retrieval + de novo): measure the recovered MRR on the hidden molecules.

#### Phase 3 Gate
- **G3.1**: The combined simulation of 150 candidates executes in $\le 5.0\text{ seconds}$ per query.
- **G3.2**: Synthetic Class 3 MRR@25 $> 0.050$ (breaking the 0.000 ceiling).

---

### Phase 4: Kaggle Deployment & Controlled Ablation

#### Objective
Package the entire multi-tier system into an offline, self-contained Kaggle submission notebook and run a clean single-flag ablation against the incumbent baseline.

#### Steps
1. **Generator & Standalone Notebook Construction**:
   - Create `build/mk_novel_submission.py` to compile the modular code into a production notebook.
   - Bundle all dependencies offline:
     - `casmi26-offline-rdkit-2026033` (RDKit wheel).
     - `casmi26-iceberg-offline` (ICEBERG model weights & wheelhouse).
     - `novel_generator.pt` + `vocab.json` (De Novo Generator weights).
     - `coconut_dataset` (Candidate pool arrays).
2. **Watchdog & Fail-Safe Architecture**:
   - Fast-commit wrapper deep-scores 30 molecules during editing, switching to full 400 molecules on `KAGGLE_IS_COMPETITION_RERUN`.
   - Global runtime watchdog: if elapsed time exceeds 7.5 hours, abort generative Track B and fall back to pure database retrieval to guarantee submission completion.
3. **Controlled Single-Flag Submission**:
   - Prepare two twin notebooks differing in exactly one configuration flag:
     - `casmi_novel_submission.ipynb`: `CFG.USE_DE_NOVO = True`
     - `casmi_novel_control.ipynb`: `CFG.USE_DE_NOVO = False`
   - Submit both to Kaggle. The delta $\Delta \text{LB} = \text{LB}_{\text{novel}} - \text{LB}_{\text{control}}$ isolates the exact contribution of the de novo pipeline.

#### Phase 4 Gate
- **G4.1**: Submission notebook completes on Kaggle in $< 8.0\text{ hours}$ without memory leaks or crashes.
- **G4.2**: The generated submission passes all format checks (no nulls, exactly 400 rows, $\le 25$ unique valid SMILES per row).
