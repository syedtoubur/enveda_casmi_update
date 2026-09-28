# CASMI 2026: Novel De Novo Structure Generation & Multi-Tier Verification (Plan & Constitution)

*Specification Version 1.0 — 2026-09-27. Project: `specs_novel`. Grounded in measured benchmarks from `summary.md`, `specs_ms_simulator/`, and `specs_simple_ms_simulator/`.*

---

## 1. Project Mission & Target Scope

The ultimate objective in the Enveda CASMI 2026 competition is to predict the top-25 candidate molecular structures as SMILES strings from tandem mass spectrometry (LC-MS/MS) data, evaluated by **MRR@25 on InChIKey14** after RDKit 2026.03.3 tautomer canonicalization.

### The Problem: The Class 3 Zero-Ceiling Wall
Existing pipelines in the workspace (`enveda-casmi-2026-tier1-patched.ipynb` scoring **0.337 LB** and `casmi_simple_sim_submission.ipynb` scoring **0.289 LB**) operate purely via candidate retrieval against a closed candidate pool (COCONUT 2.0 ∪ train structures = 712,199 molecules):
- **Class 1 (Library Match)**: The query molecule has reference spectra in `train.parquet`. Solved with high accuracy by spectral entropy similarity (`lv`).
- **Class 2 (Known Unmeasured Structure)**: The query molecule exists in COCONUT or PubChem but has no public spectra. Solved by mass-window retrieval (±8.5 ppm) + ranking (FPNet / ICEBERG / MetFrag).
- **Class 3 (Novel Chemical Entity)**: The query molecule is an uncharacterized natural product or novel chemical derivative **not present in any public chemical database**. 

> [!CRITICAL]
> **Any purely retrieval-based pipeline has a mathematical score ceiling of 0.000 on Class 3 queries.** 
> If Class 3 constitutes 15–25% of the hidden test set, a retrieval-only model forfeits up to 0.150–0.250 of potential MRR.

### Core Mission of `specs_novel`
Build an end-to-end, multi-stage hybrid inference system that:
1. **Separates** obvious Class 1 queries (`lv >= 0.85`) from ambiguous queries to preserve GPU runtime.
2. **Generates** de novo SMILES token-by-token using an autoregressive neural decoder conditioned on the query MS/MS spectrum.
3. **Prunes** the search tree dynamically using **Constrained Beam Search with Dynamic Prefix Mass Pruning** and DFS completion, enforcing strict physical precursor mass bounds (pre-centered at +1.45 ppm) and chemical valency validity.
4. **Verifies & Ranks** both retrieved database candidates (Class 2) and novel generated candidates (Class 3) in a unified arena using **ICEBERG neural forward simulation** (`msg_all`) and MetFrag-lite.

---

## 2. High-Level System Architecture

```
                                  Query LC-MS/MS Spectrum
                              (m/z, intensity, precursor m/z, CE, adduct)
                                             |
                                             v
                           [ Precursor Centering: -1.45 ppm ]
                                             |
                                             v
                             +-------------------------------+
                             |    CLASS 1 SPECTRAL GATING    |
                             |   Entropy Sim against train   |
                             +-------------------------------+
                                             |
                      +----------------------+----------------------+
                      | (lv >= 0.85: Definite Class 1)       | (lv < 0.85: Ambiguous / Class 2 / Class 3)
                      v                                      v
          +-----------------------+              +-----------------------+-----------------------+
          | DIRECT LIBRARY EMIT   |              | DUAL-TRACK CANDIDATE GENERATION & RETRIEVAL   |
          | - Rerank top library  |              +-----------------------+-----------------------+
          |   matches by lv       |                          |                                   |
          | - Dedupe InChIKey14   |                          v                                   v
          | - Fill top-25 slots   |              [ TRACK A: Class 2 Retrieval ]      [ TRACK B: Class 3 De Novo ]
          +-----------------------+              - Window: ±8.5 ppm on Pool          - Spectral Encoder
                      |                          - ~40-150 database candidates       - Token-by-token Autoregressive Decoder
                      |                                      |                       - Dynamic Prefix Mass Pruning
                      |                                      |                       - DFS Valid Completion & Sanitization
                      |                                      |                       - ~20-50 valid novel candidates
                      |                                      +-----------------+-----------------+
                      |                                                        |
                      |                                                        v
                      |                                        [ UNIFIED CANDIDATE MERGER ]
                      |                                        - Combine Track A + Track B candidates
                      |                                        - RDKit InChIKey14 deduplication
                      |                                                        |
                      |                                                        v
                      |                                    [ NEURAL FORWARD VERIFICATION ]
                      |                                    - Batched ICEBERG simulation at exact CE
                      |                                    - Spectral entropy similarity (sv)
                      |                                    - MetFrag-lite explain score (fr)
                      |                                                        |
                      |                                                        v
                      |                                          [ UNIFIED ARENA RERANKING ]
                      |                                          - Rank all candidates by (sv, fr, ppm)
                      |                                          - Select top-25 SMILES
                      |                                                        |
                      +----------------------+---------------------------------+
                                             |
                                             v
                                  [ Top-25 SMILES Output ]
                               (MRR@25 on InChIKey14 Evaluated)
```

---

## 3. Component Specifications

### 3.1 Class 1 Separator & Routing Gate
- **Purpose**: Prevent unnecessary generative decoding for queries that have near-identical experimental twins in `train.parquet`, saving GPU compute for hard queries.
- **Threshold**: Conservative cutoff **`lv >= 0.85`** (spectral entropy similarity).
- **Physical Rationale**:
  - The Phase 0 ICEBERG baseline used `lv >= 0.75` for Tier 1. However, setting `lv >= 0.85` ensures that borderline queries (isomers with slightly divergent spectra, or partial scaffold matches) are not prematurely locked into library retrieval and are allowed to explore novel candidate generation.
- **Execution**: Fast Numba entropy kernel against the pre-indexed reference library `train.parquet` in a ±8.5 ppm neutral mass window.
- **Bypass Decision**:
  - If `max(lv) >= 0.85`: emit library matches sorted by `lv`, pad if needed from candidate pool, and skip De Novo Track B completely.
  - If `max(lv) < 0.85`: proceed to parallel candidate acquisition (Track A + Track B).

### 3.2 De Novo Autoregressive Generation Engine (Track B)
- **Model Topology**: Encoder-Decoder Transformer architecture.
  - **Spectral Encoder**: Consumes query peak list `(m/z, intensity)` (up to 128 peaks), precursor m/z, adduct type, and collision energy. Maps spectral features to dense representation vectors $H_{\text{spec}} \in \mathbb{R}^{N_{\text{peaks}} \times d_{\text{model}}}$ ($d_{\text{model}} = 512$).
  - **SMILES Decoder**: Autoregressive Causal Transformer Decoder with cross-attention to $H_{\text{spec}}$. Generates SMILES tokens one by one over a compact, robust molecular token vocabulary (atoms, aromatic symbols, bonds, branches, ring numbers).
- **Training Strategy**:
  - Pretrained / fine-tuned on the 2.54M spectra and 275k unique chemical structures from `train.parquet`.
  - Conditioned on collision energy (eV) and adduct token to learn energy-dependent structural fragmentation signatures.

### 3.3 Constrained Beam Search with Dynamic Prefix Mass Pruning
Unconstrained decoding on SMILES easily falls into hallucinated invalid strings or molecules with wildly incorrect molecular masses. We formulate decoding as a **bounded tree search**:

1. **Prefix Mass Lower-Bound Invariant**:
   - For every SMILES prefix $S_{1:t}$, compute the lower bound of the neutral mass of any valid completion:
     $$M_{\min}(S_{1:t}) = \sum_{a \in \text{atoms}(S_{1:t})} \text{mass}(a)$$
   - Let $M_{\text{target}}$ be the query neutral mass (centered: $m / (1 + 1.45 \times 10^{-6})$).
   - If $M_{\min}(S_{1:t}) > M_{\text{target}} + \Delta_{\text{tol}}$ (where $\Delta_{\text{tol}} = 8.5\text{ ppm} + 0.005\text{ Da}$), the branch is **guaranteed to exceed the precursor mass** and is immediately **pruned**.

2. **Branching & Beam Expansion**:
   - At step $t$, expand each active beam into top-$K$ candidate tokens based on the decoder's log-softmax output $\log P(w_t | w_{<t}, H_{\text{spec}})$.
   - Filter out tokens that violate basic chemical grammar (e.g. consecutive double bonds on incompatible atoms, unmatched closing parentheses without open parentheses).

3. **DFS Completion on High-Scoring Subtrees**:
   - When a beam hits an open ring or incomplete valence near the target mass window ($M_{\text{target}} \pm \Delta_{\text{tol}}$), trigger a local Depth-First Search (DFS) bounded by depth $\le 10$ tokens to find valid ring closures (`1`...`9`) and the `<EOS>` end token.
   - For every completed string $S$:
     - Check exact neutral mass: $|M(S) - M_{\text{target}}| / M_{\text{target}} \le 8.5 \times 10^{-6}$.
     - Check RDKit sanitization: `Chem.MolFromSmiles(S, sanitize=True)` must succeed without valence violations.
     - Hash to `InChIKey14`. If valid, add to the novel candidate buffer.

4. **Output Cap**: Collect top $N_{\text{novel}} \in [20, 50]$ valid, unique InChIKey14 novel structures per query.

### 3.4 Unified Forward Simulation Arena (The Scorer)
Both retrieved database candidates (Track A) and novel generated candidates (Track B) must compete fairly on physical evidence:

- **Forward Simulator**: ICEBERG (`coleygroup/ms-pred`, MIT licence), checkpoint **`msg_all`**.
- **Simulation Invariant**:
  - The simulation precursor is the exact theoretical m/z of the candidate under the query's adduct.
  - The simulation collision energy is the exact query CE in eV.
  - For multi-CE merged spectra, simulate each constituent CE and take $\max(sv)$.
- **Scoring Kernel**:
  - Predicted spectrum $\hat{S}$ vs query spectrum $Q$ scored using high-precision Numba Spectral Entropy Similarity:
    $$sv = \text{entropy\_similarity}(Q, \hat{S})$$
  - Auxiliary explain score $fr$ from MetFrag-lite (explained peak ratio).
- **Ranking Formula**:
  - For all candidates in the unified pool, compute composite score:
    $$\text{score} = sv + 0.10 \times fr - 0.01 \times |\text{ppm\_error}|$$
  - Sort descending by $\text{score}$. Deduplicate by `InChIKey14` (keeping the highest scoring SMILES for that key). Emit top 25.

---

## 4. Anti-Leakage Contracts & Operational Rules

1. **No External Weights with Restrictive / Commercial Licenses**:
   - Forbidden: NIST'20/23/26 weights, CFM-ID commercial versions.
   - Allowed: Pretrained models under MIT, Apache-2.0, or CC-BY (ICEBERG `msg_all`, MassSpecGym, Hugging Face open weights).
2. **Scaffold-Disjoint Evaluation Protocol**:
   - When validating the de novo generator offline, the validation queries MUST belong to held-out Murcko scaffolds that never appeared in the training set of the generator or FPNet.
3. **InChIKey14 Deduplication Invariant**:
   - The submission file allows up to 25 SMILES per molecule.
   - Duplicate InChIKey14 entries in a prediction list waste rank positions and directly degrade MRR. Every candidate list MUST be deduplicated by `InChIKey14` prior to output.
4. **Deterministic Fallback & Fail-Safe Watchdog**:
   - If the generative DFS search hits a timeout (> 45 seconds on a single query), abort generation and fall back exclusively to Track A (database retrieval) sorted by MetFrag-lite / ppm.
   - The notebook MUST finish within 7.5 hours on Kaggle hidden re-runs.

---

## 5. Failure Modes & Mitigations

| Failure Mode | Impact | Mitigation Strategy |
|---|---|---|
| **1. Generative Hallucination** | Generates synthetically impossible or chemically invalid SMILES. | Immediate RDKit sanitization test (`Chem.MolFromSmiles(..., sanitize=True)`) + Dynamic valence check during DFS. |
| **2. Search Explosion in DFS** | High branching factor causes DFS to stall and exceed notebook time limit. | Dynamic Prefix Mass Pruning: prune any path whose minimum constituent mass exceeds precursor mass + 8.5 ppm; cap DFS depth at 10 and timeout per query at 45s. |
| **3. Simulator Mismatch on Novel Structures** | ICEBERG predicts poorly on unusual ring topologies. | Fall back to MetFrag-lite bond-disconnection score ($fr$) and precursor ppm accuracy as regularizers. |
| **4. Redundant Candidate Emission** | Emitting tautomers / stereoisomers that collapse to the same InChIKey14. | Hash canonical RDKit mol to InChIKey14 immediately; store in a set and reject duplicates on insertion. |
| **5. Class 1 False Negative** | Routing a genuine library molecule to the generative engine because `lv < 0.85`. | Track A (database retrieval) is ALWAYS active in parallel when `lv < 0.85`; novel candidates compete in the same fair arena with database candidates. |
