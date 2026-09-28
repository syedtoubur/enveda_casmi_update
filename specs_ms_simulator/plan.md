# CASMI 2026: MS-Simulator & Scaffold-Consensus Architecture (Plan & Constitution)

*Revision 2 — 2026-09-25. Reconciled against the measured state of the codebase
(`summary.md`, `RUNBOOK.md`, `logs.txt`, `build/`). Changes from revision 1 are recorded in
§6 so nothing silently drifts.*

## 1. Project Mission & Scope

The objective is state-of-the-art molecular structure identification (top-25 candidate SMILES,
scored **MRR@25 on InChIKey14**) from LC-MS/MS spectra in the Enveda CASMI 2026 competition.

### Core Targets

1. **Target Novelty Regimes**: maximise retrieval and ranking on **Class 1** (reference spectra
   present) and **Class 2** (known structure in the candidate pool, no public spectra). Class 3
   (de novo generation) is out of scope for this pipeline.
2. **Score Target**: advance from the **0.337 LB** incumbent toward **0.380+** through physical
   and neural feature upgrades. The candidate pool is **not** expanded in this spec; that is a
   separate, later decision (see §5.3).
3. **Execution Topology**:
   - All development, data processing, model training and validation run in **Google Colab**
     (L4 + High-RAM is the reference runtime; A100 where noted).
   - **Kaggle Notebooks are reserved strictly for final inference submissions** — offline,
     ≤ 9 h.
   - Every submission notebook inherits the dynamic fast-commit wrapper
     (`KAGGLE_IS_COMPETITION_RERUN` detection). This already exists in the base notebook; it is
     inherited, not ported.

---

## 2. Base Architecture & The 4 Evidence Channels

The system augments heuristic bond-breaking (MetFrag-lite) with deep neural forward spectrum
simulation, and enhances analog propagation with Murcko scaffold consensus.

```
                                Query Spectrum & Precursor m/z
                                                |
                 +------------------------------+------------------------------+
                 v                                                             v
     [ Candidate Retrieval ]                                      [ Quad-Channel Evidence ]
     Unified Pool (~711,705 mols)                                 -----------------------
     - Centred Neutral Mass                                       Channel 1: Direct Library Match
     - Window: PPM_WIN (see 5.2)                                  Channel 2: Analog + Scaffold Consensus
     - PPM_EXPAND tiers append below                              Channel 3: MetFrag + Neural Simulator
                 |                                               Channel 4: FPNet Fingerprint Model
                 v                                                             |
         ~40-200 Candidates                                                    |
                 |                                                             |
                 +------------------------------+------------------------------+
                                                v
                                    [ 38-Feature Assembly ]
                                                |
                                                v
                                  [ Calibrated GBDT Ensemble ]
                                 (Dual-Prior W1, 4-Seed Bagging)
                                                |
                                                v
                                 [ InChIKey14 Dedup & Emit ]
                                  (Ranked Top-25 SMILES)
```

### Channel 1: Direct Experimental Library Search
- **Input**: query spectrum against 2,539,608 reference spectra in `train.parquet`.
- **Search Scope**: centred target neutral mass
  (`m_target = target / (1 + PPM_CENTER * 1e-6)`, `PPM_CENTER = 1.45`) within `CFG.PPM_WIN`.
- **Metric**: cleaned spectral entropy similarity (power 1.0, linear intensity, entropy weighting).
- **Role**: solves Class 1 when a reference measurement exists.

### Channel 2: Mass-Shifted Analog Search + Scaffold Consensus
- **Input**: reference analog library across a ±200 Da mass window.
- **Representative Set**: `build_rep` with **`CFG.POLARITY_ANALOG = True`** — the richest
  representative per *(scaffold, polarity)* pair (**371,699 spectra across 275,666 scaffolds**,
  measured in `logs.txt`). See §5.1 for why this is the chosen contract.
- **Scaffold Consensus Engine**:
  1. Retrieve top-100 spectral analogs by shifted entropy similarity (`w`).
  2. Map each analog to its Bemis-Murcko core scaffold.
  3. Weighted scaffold frequency `F(S) = sum(w_a^4 for a in S) / (sum(w_a^4) + 1e-9)`.
  4. Score candidates on whether their scaffold matches the consensus family.
- **Role**: identifies the biological family / core ring system of the unknown compound.

### Channel 3: In-Silico Fragmentation — MetFrag-lite + Neural Simulation
- **3a MetFrag-lite (retained, columns 21-24)**: 1-2 bond disconnections, `MAX_BONDS = 60`.
  Measured cost is **5.5% of Stage-C wall clock** and it carried **1.7-2.3% of ranker skill**.
  It is *not* retired on assumption; retirement is a measured decision in Phase 3.
- **3b Neural Virtual Simulation (new, columns 34-37)**: a pretrained forward simulator predicts
  a peak list per candidate SMILES conditioned on adduct and collision energy; the prediction is
  scored against the real query spectrum with the existing numba entropy kernel.
- **Role**: converts Class 2 queries into a virtual library match, capturing real fragmentation
  intensities without rule-based bail-out.

### Channel 4: Deep Molecular Fingerprint Ranker (FPNet)
- **Input**: query peak list, neutral losses, precursor m/z, adduct, collision energy, ionization mode.
- **Model**: 6-layer Transformer with sinusoidal peak/loss embeddings, 6,930 logits (`z`).
- **Scoring**: Bayes log-likelihood dot product (`cf · z`).
- **Role**: orthogonal substructure prediction, agnostic to 2D graph enumeration.

---

## 3. Strict Invariants & Anti-Leakage Contracts

### 3.1 The Separation Contract (extended to two neural models)

Neural models must NEVER have gradient-updated on a molecule used as a Stage-C query or present
in a Stage-D validation cohort. This now binds **two** models with **different** training sets:

| model | its training set | how we enumerate it |
|---|---|---|
| FPNet | essentially all of `train.parquet` | `_fpnet/pack/splits.npz` → `seen_keys` / `es_keys` / `banned_keys` |
| Neural simulator | MassSpecGym train fold (MoNA + MassBank + GNPS + in-house) | **`msg_train_keys.npz`** — InChIKey14 of every MassSpecGym train-fold structure, built in Phase 3 |

**Contract**: a Stage-C query in the *model-unseen* regime must satisfy
`ik14 in es_keys AND ik14 not in msg_train_keys`.
A Stage-D delta attributable to the simulator must be reported **split by** membership in
`msg_train_keys`; only the out-of-set half is evidence.

This is not defensive paperwork. The v1 disaster (LB 0.337 → 0.287) happened because exactly
one model's training set was not excluded. Channel 3b introduces a second model whose training
set demonstrably overlaps the validation cohorts (C2 is 83% gnps/riken; MassSpecGym is built
from GNPS/MassBank/MoNA), so the same failure is available again unless it is measured.

### 3.2 The Checkpoint Provenance Contract

Only **freely and publicly available** pretrained weights may enter the pipeline, per Kaggle's
external-data rule.

- **PERMITTED**: `coleygroup/ms-pred` (MIT licence) **MassSpecGym-trained** checkpoints —
  ICEBERG 2.1 (`msg_all`, `msg_simulation`) and GLACIER. MassSpecGym itself is MIT-licensed and
  publicly hosted on HuggingFace.
- **FORBIDDEN**: any NIST'20/23/26-trained checkpoint from the same repository. Those are
  released only on emailed proof of a commercial NIST licence and therefore fail the rule.
- The Phase-3 pre-flight asserts the loaded checkpoint filename matches the permitted set and
  records its SHA1, exactly as `fp_bits.npy` is guarded today.

### 3.3 The Paired-Regime Contract

Every Stage-C query is processed in two paired regimes:
- `M = 0`: library visible (teaches how Class 1 behaves).
- `M = 1`: library banned (teaches how Class 2 behaves).

The expensive work (candidate window, `model_logits`, MetFrag, neural simulation) is computed
**once**; only the library and analog channels are re-run under both ban settings. Columns that
do not depend on the ban must be byte-identical between the two halves —
`build/test_process_query.py` enforces this and must be extended to cover the new columns.

### 3.4 The Cohort Contract

Stage C draws queries from three cells, not four:

| cell | query structure | library | `M` | mirrors | keep? |
|---|---|---|---|---|---|
| A | model-**seen** | visible | 0 | test Class 1 | **yes** |
| D | model-**unseen** | visible | 0 | cross-library Class 1 | **yes** |
| B | model-**unseen** | banned | 1 | **test Class 2** | **yes** |
| C | model-**seen** | banned | 1 | *no test analogue* | **DROP** |

Cell C is dropped because a Class-2 test molecule has no public spectra and therefore cannot be
in `train.parquet`, so "model-seen + library-banned" cannot occur at test. At MRR 0.9646 it
teaches *"when `lv` is 0, trust the model"* where the model is 91.7% right; at test, `lv = 0`
means the model is 52% right. See `summary.md` §7.2.

### 3.5 The Base Integrity Contract

- The base code is **`enveda-casmi-2026-phase1-enhanced-baseline.ipynb`**, embedded verbatim as
  cells 1-8 by every generator in `build/`, with the single documented substitution
  `/kaggle/input` → the Colab Drive root in cells 2 and 7.
- `CFG.POLARITY_ANALOG = True`, `CFG.EXPAND_BACKFILL = True`, `CFG.PPM_EXPAND` tiers active.
  The matrix is rebuilt under this exact configuration, resolving the train/inference mismatch
  described in §5.1.
- **The feature schema is append-only.** Columns 0-30 keep their present meaning and order
  forever. New channels append at 31+. Rationale: inference performs
  `rank_features(...)[:, :NFEAT]` with `NFEAT = z['X'].shape[1]`, so an older matrix keeps
  working unchanged and a newer matrix enables the new columns automatically. Renumbering would
  break `build/test_gate.py`'s ground-truth validation against the 0.287 and 0.337 matrices.
- **Notebooks are generated, never hand-edited.** `build/mk_*.py` are the source of truth.
  `patch_colab_rdkit.py` is a violation of this rule and is folded into
  `build/mk_stage_c_colab.py` in Phase 1.
- No widening of the candidate window beyond the configured tiers unless candidates are empty
  (`CFG.PPM_FALLBACK`).

---

## 4. The 38-Column Feature Schema

Verified against `rank_features` in phase1 code cell 7. Columns 0-30 are the shipped 31; 31-37
are new.

```
0  - 4    Channel 1  Direct library      lv, rank_norm(lv), lvmax, lv-lvmax, (lv>0)
5  - 13   Channel 2  Analog match        ap, rank_norm(ap), apmax, ap-apmax,
                                         a1, best_tan, top_tan, mean_tan, top_sim
14        Window context                 log(n_candidates)
15 - 20   Channel 4  FPNet               z(raw), rank_norm(raw), raw-max,
                                         z(nrm), rank_norm(nrm), is_top
21 - 24   Channel 3a MetFrag-lite        fr, rank_norm(fr), fr-max, z(fr)
25 - 30   Cross-channel interactions     lv*(1-mr), ap*(1-mr), agree, agree_a,
                                         agree*lvmax, corr(lv, -mr)
--------------------------------------------------------------------------- new
31 - 33   Channel 2b Scaffold consensus  scaffold_vote, is_top_scaffold, scaffold_tanimoto
34 - 37   Channel 3b Neural simulation   sim_entropy, sim_rank_norm, sim_z, sim_is_best
```

The gate's channel index map becomes:

```python
CH = {'library':  [0,1,2,3,4,25],
      'analog':   [5,6,7,8,9,10,11,12,13,26,31,32,33],
      'model':    [15,16,17,18,19,20],
      'frag':     [21,22,23,24],
      'sim':      [34,35,36,37]}
```

---

## 5. Decisions Recorded (with their evidence, and what would reverse them)

### 5.1 `POLARITY_ANALOG = True`

The 0.310 LB observed with polarity-aware representatives was measured at inference **against
the forked matrix, which was not built with them**. That is a train/inference distribution
mismatch, not evidence that polarity representatives are worse. Two further facts support
keeping them on:

- **Mechanism**: `analog_sim` aggregates `agg[k] = max` over a scaffold's representatives. Two
  representatives instead of one means `ap` is monotonically ≥ the single-rep value for every
  candidate — polarity representatives can only *add* analog evidence.
- **Confound**: phase1-enhanced differs from tier1-patched in `PPM_EXPAND`, `EXPAND_BACKFILL`,
  `BACKFILL_PPM` and the fast-commit wrapper as well as `POLARITY_ANALOG`. The −0.027 is the sum
  of all of them; attributing it to polarity alone was an assertion, never a measurement.

**Cost accepted**: the representative set grows 275,810 → 371,699 (+35%) and analog is 91.4% of
Stage-C wall clock, so Stage C rises ~2.8 h → ~3.5-3.8 h.

**What reverses it**: if Stage D still returns HOLD after Phase 4, flipping `POLARITY_ANALOG` to
`False` and rebuilding is the first thing to try. The flag stays live for that reason.

**Known limitation**: Stage D cannot adjudicate this. All four arms share one inference code
path and vary only checkpoints × matrix, so a polarity A/B needs a fifth arm with its own
matrix (~3.5 h Stage C + one Stage D pass). Not budgeted here.

### 5.2 `PPM_WIN` narrowing 8.5 → 5.5

Enabled, because the demotion path already exists and nothing is discarded:
`PPM_EXPAND` becomes `(5.5, 8.5, 20.0, 45.0, 80.0)` and `EXPAND_BACKFILL` appends wider-window
candidates **below** stage A, where they "can never displace a tight-window hit". A truth
between 5.5 and 8.5 ppm is therefore never dropped — only ranked lower — in exchange for ~30%
fewer decoys in every window.

**The riken interaction is in Stage C, not at test.** riken reports m/z to 2 decimals (~12 ppm
at 400 Da). Test spectra are 100% high-precision timsTOF and contain no such rows, so narrowing
does not interact with riken at inference. It does tighten `SC.QUERY_PPM_MAX` (= `CFG.PPM_WIN`),
dropping riken's usable share from the measured 67.2% at 8.5 ppm. VERIFY 1 prints the per-library
usable % under both windows **before** the long loop so the cohort cost is seen, not guessed.

### 5.3 Explicitly deferred

- **Candidate pool expansion** (LOTUS, NPAtlas, PubChem-NP). High expected Class-2 value, but
  `val_scripts` cannot measure the gain — every val truth comes from `train.parquet` and is
  in-pool by construction, so validation can only measure the decoy-dilution *harm*. Needs a
  different evaluation design before it is worth the ingestion work.
- **Class 3 de novo generation.**
- **FPNet retraining** with B1 early stopping gated on `es_MRR` instead of `es_bce`
  (~0.03 es_MRR, ~4.5 h A100). Not worth doing on its own.
- **`CFG.USE_BIO_DB`** — still an assertion, not a measurement. `chebi-lipidmaps-casmi26` is on
  Drive and Stage D can settle it for zero GPU units.

---

## 6. Changes from Revision 1

| # | revision 1 said | corrected to | why |
|---|---|---|---|
| 1 | 35-column schema, renumbered | **38 columns, append-only** | 31 + 3 + 4 = 38; renumbering breaks `test_gate.py`, `merge_shards.py` and the `[:, :NFEAT]` compatibility path |
| 2 | base mirrors `tier1-patched` | base is **`phase1-enhanced-baseline`** | tier1-patched has no fast-commit wrapper, which §1.3 requires; all four generators already embed phase1 |
| 3 | revert `build_rep` to one-rep-per-scaffold | **keep `POLARITY_ANALOG = True`** | §5.1 — the 0.310 datapoint was measured under a mismatched matrix and is confounded |
| 4 | Stage-C queries "strictly `es_keys`" | cells **A, D, B**; drop **C** | strict-`es_keys` deletes cell A, the only test-Class-1 analogue; `summary.md` §7.2 says drop C instead |
| 5 | simulator is "MassSpecGym / MassFormer" | **ICEBERG 2.1 / GLACIER, MassSpecGym checkpoints only** | §3.2 — NIST-trained weights fail Kaggle's external-data rule |
| 6 | MetFrag-lite retired, saving 15-25 s/molecule | **retained; retirement is a Phase-3 measurement** | measured cost is 5.5% of Stage C (~0.4 s of a 7.22 s query), not 15-25 s |
| 7 | rep set ~54,650 structures | **371,699 spectra / 275,666 scaffolds** | measured in `logs.txt`; 54,650 matched nothing |
| 8 | simulator ≤ 20 ms for 50 candidates | **measured in a Phase-3 pre-flight; no number assumed** | the spec figure ignored RDKit graph featurisation and contradicted the project's own 7.22 s/query |
| 9 | Phase 5 re-run is "pass 3 only, ~25 min" | **full pass-1 re-run, ~2-2.5 h** | new channels change what pass 1 caches, invalidating the Drive checkpoint |
| 10 | (absent) | **Phase 0 — close the Stage D HOLD first** | stacking two channels on an unresolved regression makes the next A/B unreadable |
