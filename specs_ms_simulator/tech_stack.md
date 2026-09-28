# Technical Stack, Data Schemas & Execution Constraints

*Revision 2 — 2026-09-25. Figures marked **measured** come from `logs.txt`, `summary.md` §6 or
direct inspection of the artifacts. Figures marked **to measure** are deliberately left open
until the corresponding pre-flight runs; no placeholder numbers.*

## 1. Core Software & Computational Stack

| Component | Technology / Specification | Purpose / Invariant |
|---|---|---|
| **Programming Language** | Python 3.12 | Pinned across Colab and Kaggle submission runtimes |
| **Cheminformatics Core** | RDKit 2026.03.3 (`casmi26-offline-rdkit-2026033.whl`) | Mandatory for tautomer canonicalization and InChIKey14 evaluation |
| **Deep Learning Framework** | PyTorch 2.2+ (CUDA 12.x) | GPU inference for FPNet and the neural simulator |
| **JIT / Fast Kernels** | Numba 0.59+ (`@njit(fastmath=True)`) | Spectral entropy matching, mass-shift alignment, binary search |
| **Tabular GBDT Engine** | Scikit-Learn `HistGradientBoostingClassifier` | Low-latency histogram GBDT with dual-prior bagging |
| **Columnar Storage** | PyArrow 15.0+ | Zero-copy reading of 2.54M `train.parquet` peak arrays |

---

## 2. Neural Model Specifications

### 2.1 Channel 4: FPNet Neural Fingerprint Ensemble

- **Checkpoints**: `fp_single_s0.pt`, `fp_single_s1.pt`, `fp_merged_m0.pt`, `fp_merged_m1.pt`
  (under `casmi26-fpnet-v3`).
- **Input**: ≤ 128 peaks per spectrum (m/z, intensity, neutral loss) + precursor mass + adduct
  embedding (26 classes) + instrument family (5 classes) + collision energy + ionization mode.
- **Backbone**: 6-layer Transformer, scaled dot-product multi-head attention (d=512, h=8,
  dropout 0.10).
- **Output**: 6,930 unnormalized logits (`z`).
- **Bit contract**: `fp_bits.npy` SHA1 `4686f5458acb1259d2dbca42619cda0ca903b1c0`;
  `N_bits = 6,930`, subselected from a 10,407-bit Morgan/RDKit/MACCS concatenation.
- **Measured quality** (scaffold-disjoint holdout, 48 real ±8.5 ppm decoys):

  | checkpoint | es_MRR | es_top1 |
  |---|---|---|
  | `fp_single_s0` | 0.8278 | 0.7425 |
  | `fp_single_s1` | 0.8250 | 0.7366 |
  | `fp_merged_m0` | 0.8222 | 0.7431 |
  | `fp_merged_m1` | 0.8197 | 0.7360 |

### 2.2 Channel 3b: Neural Virtual Spectrum Simulator

| | |
|---|---|
| **Source** | `coleygroup/ms-pred` — **MIT licence** |
| **Permitted checkpoints** | ICEBERG 2.1 (`msg_all`, `msg_simulation`), GLACIER — all **MassSpecGym-trained**, publicly downloadable |
| **Forbidden checkpoints** | any NIST'20/23/26-trained weights (released only on emailed proof of a commercial licence → fails Kaggle's "freely & publicly available" rule) |
| **Training data** | MassSpecGym (**MIT**, HuggingFace) — MoNA + MassBank + GNPS + in-house, structure-disjoint split |
| **Leakage artifact** | `msg_train_keys.npz` — InChIKey14 of every MassSpecGym train-fold structure. Required by `plan.md` §3.1. |
| **Input** | candidate SMILES → molecular graph + collision energy (eV) + adduct token |
| **Output** | predicted peak list `[(mz, intensity), ...]`, scored against the query with the existing numba entropy kernel |
| **Inference mode** | batched PyTorch, fp16 |
| **Latency** | **to measure** (Phase 3 Gate 3) at median 70 / p90 217 / max 676 candidates per window, splitting RDKit featurisation from GPU forward pass |
| **Negative mode** | **to verify** — 4 of the 10 test adducts are negative |

---

## 3. Data Schemas & In-Memory Data Structures

### 3.1 Unified Candidate Pool (`CandidatePool`)

COCONUT 2.0 ∪ unique training scaffolds, pre-sorted by monoisotopic exact mass.
**Measured**: 712,199 structures, built in 264 s at inference (`logs.txt`).

- `pool_mass.npy` — `float64`, shape `(~711705,)` (~5.7 MB)
- `pool_fp.npy` — bitpacked `uint8`, shape `(~711705, 867)` (~617 MB); unpacks to `(N, 6930)`
  in RAM only for the active window
- `pool_scaffolds.pkl` — **`inchikey14` → Bemis-Murcko scaffold SMILES** (~35 MB).
  Keyed by InChIKey14, never by row index: the pool is rebuilt at inference and row order is not
  guaranteed stable. `val_scripts/scaffolds.pkl` already provides **261,890** of these entries
  covering `train.parquet`; only the COCONUT delta needs computing.
- `pool_meta.pkl` — row index → `inchikey14`, `smiles` (~41.5 MB)

### 3.2 Reference Library (`L`)

From `train.parquet`: **2,539,608 spectra / 275,810 structures** (measured).

- `order` — mergesort permutation indexing formula-derived mass `snm` ascending
- `off` — offsets into flattened `allmz` / `allin` (both `float32`)
- `rep` — representative analog set from `build_rep`. With `CFG.POLARITY_ANALOG = True` this is
  the richest spectrum per *(scaffold, polarity)* pair: **371,699 spectra across 275,666 unique
  scaffolds** (measured, `logs.txt`). With the flag off it is one per scaffold (~275,810).
  *Revision 1's figure of ~54,650 was incorrect.*

### 3.3 GBDT Training Matrix

| file | what it is | shape |
|---|---|---|
| `rank_train_old.npz` | the **forked** matrix, LB 0.337 | 142,762 × 31, has `G` |
| `rank_train.npz` (local) | the **v1 leaked** matrix, LB 0.287 | 71,351 × 31, no `G` |
| `rank_train_v3.npz` | target of Phase 4 | N × **38**, with `G` |

Both older matrices must be preserved — `build/test_gate.py` validates the gate against them as
ground truth (FAIL on v1, PASS on forked).

---

## 4. 38-Column GBDT Feature Schema

Verified against `rank_features` in phase1 code cell 7. **Append-only**: columns 0-30 are frozen.

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

**Why append-only is load-bearing**: inference computes
`rank_features(...)[:, :NFEAT]` where `NFEAT = z['X'].shape[1]`. A 31-column matrix therefore
keeps working against 38-column feature code, and a 38-column matrix enables the new channels
automatically. Renumbering would break that path, `merge_shards.py`'s width assertion, and the
gate's ground-truth validation.

---

## 5. Hardware Budgets & Execution Constraints

### 5.1 Colab

**Reference runtime: L4 GPU + High-RAM.** Stage C and Stage D are **CPU-bound** — the GPU runs
`model_logits` (and now the simulator) and nothing else; wall clock goes to numba entropy search
over 2.54M spectra and RDKit bond-breaking, both of which scale with vCPU.

| runtime | vCPU | RAM | units/hr | cores per unit |
|---|---|---|---|---|
| T4 + High-RAM | 4 | 51 GB | ~2.0 | 2.0 |
| **L4 + High-RAM** | **8** | **53 GB** | **~4.8** | **1.7** |
| A100 | 12 | 83 GB | ~13 | 0.9 |

- **RAM**: reference library ~14 GB; pool + scaffold dict ~2 GB. **High-RAM is mandatory** —
  `load_library` holds all 2.54M peak lists in memory and the notebook refuses to start below
  40 GB.
- **VRAM**: < 4 GB with FPNet and the simulator both resident.

**Measured Stage-C profile** (VERIFY 1, 6-query smoke test): 7.22 s/query, split
analog 91.4% | metfrag 5.5% | features 1.4% | fpnet 1.4% | library 0.2%.

| stage | measured / projected |
|---|---|
| Stage C v2 (31 cols, single-rep) | 2.8 h |
| Stage C v3 (38 cols, `POLARITY_ANALOG=True`) | ~3.5-3.8 h from the +35% rep set, **plus** the Phase-3 Gate-3 simulator cost |
| Stage D full run | ~2.5 h on L4 (pass 1 ~2-2.5 h of it) |
| Stage D pass 3 only | ~25 min per matrix variant |

### 5.2 Kaggle Offline Submission

- **Hardware**: T4 or P100, 13 GB RAM, 2 vCPU.
- **Execution limit**: ≤ 9 h (strict).
- **Network**: DISABLED. All models and wheels attached as Kaggle input datasets.
- **Fast-commit profile**: public commit deep-evaluates the first `CFG.FAST_DEV_MOLECULES`
  molecules and mass-fills the rest — **measured 455 s end-to-end** (`logs.txt`), of which 378 s
  is pool construction and library load. Private re-run switches automatically to the deep
  engine on 100% of molecules.
- **Full-run throughput**: **to project** from the Phase-3 Gate-3 measurement. Revision 1's
  "~65 ms per molecule → 400 molecules in ~30 s" is inconsistent with the project's own measured
  7.22 s/query in Stage C and must not be used for planning.
- **Detection contract**:
  ```python
  _env_raw = os.environ.get('KAGGLE_IS_COMPETITION_RERUN')
  looks_public = (n_mol == CFG.PUBLIC_N_MOL and len(te) == CFG.PUBLIC_N_SPEC)
  FAST_MODE = bool(CFG.ALLOW_FAST) and looks_public and not bool(_env_raw)
  ```
  Fail-safe: a full run unless the public test file is positively identified.

### 5.3 Candidate Window Configuration

| knob | value | note |
|---|---|---|
| `PPM_CENTER` | 1.45 | measured timsTOF offset; target is divided by `(1 + PPM_CENTER/1e6)` |
| `PPM_WIN` | **5.5** (was 8.5) | see `plan.md` §5.2 |
| `PPM_EXPAND` | `(5.5, 8.5, 20.0, 45.0, 80.0)` | tiers append **below** stage A; nothing is discarded |
| `PPM_FALLBACK` | 30.0 | fires only when the tight window yields zero candidates |
| `SC.QUERY_PPM_MAX` | `= CFG.PPM_WIN` | Stage-C self-retrievability filter; tightening it reduces the riken cohort (67.2% usable at 8.5 ppm, median \|off\| 3.78 ppm) |
