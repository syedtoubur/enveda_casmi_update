# Tech Stack — Simple MS Simulator Baseline

*Revision 1 — 2026-09-26. **measured** = taken from `logs.txt`, `summary.md` or the phase1
notebook. **to measure** = left open until Phase 0/1 runs. No placeholder numbers.*

## 1. Runtimes

| where | used for | hardware | limits |
|---|---|---|---|
| Google Colab | Phase 0 gate, Phase 1 grid, Phase 2 precompute, Phase 3 analysis | L4 or A100 + High-RAM | a session can drop, so all long jobs are sharded and resumable |
| Kaggle notebook | final inference only | T4/P100 GPU, ~13 GB RAM, few vCPU | **offline**, ≤ 9 h, output `submission.csv` |

## 2. Software

| component | version / source | role |
|---|---|---|
| Python | 3.12 (Kaggle image) | — |
| RDKit | **2026.03.3**, `casmi26-offline-rdkit-2026033` wheel (existing dataset) | parsing, InChIKey14, MetFrag-lite graphs, matches the metric |
| PyTorch | Kaggle image build | ICEBERG forward pass |
| ms-pred (ICEBERG) | `github.com/coleygroup/ms-pred`, MIT | forward simulator |
| DGL, lightning, other ms-pred deps | **offline wheels dataset** (to build in Phase 0 G5) | ms-pred runtime |
| numba | Kaggle image | entropy similarity, MetFrag-lite |
| pyarrow / pandas | Kaggle image | parquet I/O |

## 3. Forward simulator — ICEBERG

| | |
|---|---|
| Source | `coleygroup/ms-pred`, MIT licence |
| **Permitted** checkpoints | MassSpecGym-trained ICEBERG (e.g. `msg_all`, `msg_simulation`), publicly downloadable. MassSpecGym is MIT (`roman-bushuiev/MassSpecGym` on HuggingFace). |
| **Forbidden** checkpoints | any NIST'20/23/26-trained weights. They require emailed proof of a commercial licence, which fails Kaggle's "freely & publicly available" rule. |
| Architecture | two stages: a fragment DAG generator (which bonds break) plus an intensity predictor over the generated fragments. Output: fragment formulae/masses with intensities. |
| Inputs | candidate SMILES, precursor adduct, CE if supported (**to verify**, G3) |
| Output | peak list `[(mz, intensity)]`, normalised to base = 1 and cleaned exactly like the query spectra |
| Adduct / mode coverage | **to verify** (G2). Expected: positive only, `[M+H]+` and possibly `[M+Na]+`. |
| Throughput | **to measure** (G4) on Colab GPU and Kaggle T4, featurisation and forward pass timed separately |
| Guard | checkpoint SHA1 recorded in Phase 0 and asserted at load in Kaggle |

**Coverage of the 10 test adducts** (filled in by Phase 0):

| adduct | mode | ICEBERG? | tier-2 scorer |
|---|---|---|---|
| `[M+H]+` | + | expected yes | `sv` |
| `[M+Na]+` | + | to verify | `sv` or MetFrag |
| `[M+NH4]+`, `[M+K]+`, `[M-H2O+H]+`, `[M-2H2O+H]+` | + | to verify, likely no | MetFrag |
| `[M-H]-`, `[M-H2O-H]-`, `[M+CH2O2-H]-`, `[M+Cl]-` | − | expected no | MetFrag |

## 4. Data artifacts

| artifact | location | content | size |
|---|---|---|---|
| Candidate pool | built at inference by `build_candidate_pool` | COCONUT + train structures, sorted by exact mass; `mass`, `keys` (ik14), `smiles`, packed fp | **measured** 712,199 structures, 264 s to build; dedup by ik14 **to measure** |
| Real library `L` | built at inference by `load_library` from `train.parquet` | flat `mz`/`it` + `off`, `snm` (formula mass, `STRUCT_MASS=True`), ik14 | **measured** 2,539,608 spectra / 275,810 structures, ~30 s to load |
| `_simlib/shards/` | Drive | per-shard simulated spectra + manifest | **to measure** |
| `_simlib/simlib.npz` | Drive → Kaggle dataset | `mz` float32, `it` float16, `off` int64, `key` (ik14), `cond` uint8, `has_real` bool | ≈ structures × conditions × K peaks × 6 B (e.g. 700k × 1 × 64 × 6 B ≈ 0.27 GB per condition); exact figure **to measure** |
| `_simlib/cond_grid.json` | Drive → Kaggle | condition id → (adduct, CE), nearest-grid tolerance | tiny |
| `_simlib/fidelity.parquet` | Drive | per-structure sim-vs-real similarity with adduct, CE, mass, library | analysis only |
| ICEBERG checkpoint + wheels | Kaggle datasets | weights (SHA1-guarded) + offline wheels | **to measure** |

Everything is keyed by **InChIKey14**, never by pool row index.

## 5. Scoring

- **Spectrum cleaning** (phase1 `_clean`): `INT_FLOOR = 0.002`, `MAX_PEAKS = 64`, peaks above
  precursor + 2 Da dropped. The same cleaning is applied to query, real and simulated spectra.
- **Similarity** (phase1 `entropy_sim`): spectral entropy similarity `1 − (2·S_AB − S_A − S_B)/ln 4`,
  `MZ_TOL = 0.01` Da.
- **Mass window:** `PPM_CENTER = 1.45`, `PPM_WIN = 8.5`, `PPM_EXPAND = (8.5, 20, 45, 80)`,
  `PPM_FALLBACK = 30`, `BACKFILL_PPM = 120` (phase1 values, unchanged).
- **Per candidate:**
  - `lv` = max over query spectra × real spectra of that ik14 (phase1 `lib_sim`; 0 if none)
  - `sv` = max over *covered* query spectra of `entropy_sim(query, sim[ik14, cond(query)])`
  - `fr` = MetFrag-lite explain score (phase1 `frag_scores`, `MAX_BONDS = 60`)
- **Ranking (two-tier):**
  1. tier 1: `lv ≥ CFG.TAU` (default **0.75**), ordered by `lv` descending
  2. tier 2: all remaining candidates, ordered by (`sv` desc, `fr` desc, |ppm error| asc). A
     molecule with no covered spectrum uses (`fr`, |ppm|) only.
  3. ik14 dedup → top 25, then PPM_EXPAND backfill if fewer than 25 (phase1 `take_slots`)
- **Multi-CE merged query spectra:** simulate each listed CE and keep the max `sv`. The final
  rule is fixed in Phase 1.

## 6. Runtime budget (Kaggle)

| step | figure |
|---|---|
| pool build | **measured** 264 s |
| library load | **measured** ~30 s |
| simlib load | **to measure** |
| per molecule, library + MetFrag (phase1) | **measured** ~7 s in phase1 incl. analog + FPNet; lower here since analog is dropped |
| per molecule, simlib scoring | **to measure** (dictionary lookups + ~50–200 entropy sims) |
| on-the-fly ICEBERG | option A: every covered candidate × condition. **measured** 36.6 mol/s on 1 T4 at bs 128, 10.4 s start-up per worker call, install 197 s (`t4.json`); estimated ~1.5 h for 400 molecules; the watchdog stops at 7.5 h |
| **total** | must be < 9 h with ≥ 30% margin (Phase 5 check 6) |

The fast-commit wrapper stays: the public commit deep-evaluates the first
`CFG.FAST_DEV_MOLECULES` molecules. The private re-run is always a full run.
