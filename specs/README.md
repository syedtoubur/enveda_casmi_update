# CASMI 2026 — Master Specifications Index

> **Purpose**: Single entry-point for humans and AI agents navigating the research tracks, architectural roadmaps, and code files for the Enveda CASMI 2026 competition.
> **Official Metric**: MRR@25 on InChIKey14 (tautomer-canonical)
> **Best public score**: **0.415 / 0.411 LB** (v29 lineage), but only with the author's **private** popularity arrays. The same code from our account scores **0.393**.
> **Our reproducible base**: **≈ 0.400 LB**: `evenda-casmi-inferencesupernext.ipynb`, own runs 0.399 / 0.401. Judge every new probe against it with a paired control (keep bar ≥ base mean + 0.004).
> **Master Experiment Ledger**: See [EXPERIMENTS.md](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/EXPERIMENTS.md)
> **Last Updated**: 2026-10-05

---

## Quick Start (read in this order)

1. **[EXPERIMENTS.md](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/EXPERIMENTS.md)**: what scored what, and the Graveyard of dead ends.
2. **This file**: find the track, then its *Core Specs*.
3. **The track's `gate_report.md`**: pre-registered thresholds and measured verdicts. Write thresholds there **before** running anything.
4. **[`Dataset_intro/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro)**: what each external dataset contains, before you use it.

Ground rules (from `CLAUDE.md`):
- Never hand-edit generated `.ipynb` files. Change the `build/patch_*.py` / `build/mk_*.py` generator and regenerate.
- Inspect notebooks with `compare.py`, never as raw JSON.
- Kaggle runs offline, in ≤ 9 h, on a T4.
- Never commit `.npz` / `.pt` / `.parquet` / `.zip` files.
- Run `sh build/run_all_tests.sh` before submitting.

---

## Tracks Overview

| Track | Directory | Status (10-05) | Focus |
|---|---|---|---|
| **specs_scratch: own pipeline** | [`specs_scratch/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/specs_scratch) | **Active (10-05)**: S0 done, R0 ready to submit | Our own tested package (`scratch_pkg/`) built rung by rung from the Track 3 base; Tracks 1–4 are its read-only knowledge base. |
| **Track 1: Retrieval & Fusion** | [`track1_retrieval_fusion/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion) | 0.399 LB (v4n fusion); superseded as a base by supernext | Class 1 & 2 retrieval, candidate expansion (PubChem/BIO), dual-engine rank fusion, and scaffold mutation. |
| **Track 2: Novel De Novo** | [`track2_novel_generation/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation) | **Phase 3 (G3.1 PASS)**, next up | Class 3 de novo molecule generation via a SELFIES decoder on frozen DreaMS embeddings. |
| **Track 3: Forward MS Simulator** | [`track3_ms_simulator/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator) | **Validated (+0.035 LB)**, next up | Physical MS/MS forward spectrum prediction (ICEBERG/GLACIER) for candidate re-ranking. |
| **Track 4: Priors & Probes** | [`track4_priors_probes/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes) | Popularity lever used up; our base is supernext ≈ 0.400 | Spectrum-blind priors (PubChem popularity, NP bonus and pool expansion) and the gated Class-3 mutator v2, each as a single-change LB probe. |
| **Track 4b: NP Signals (dataset improvement)** | [`track_dataset_improvement/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track_dataset_improvement) | **CLOSED 10-05, nothing kept** (3 probes at 0.399) | Our own NP signals (vendor-only flag, Ertl NP-likeness, class consistency) re-ranking supernext's PubChem channel. |
| **Datasets** | [`Dataset_intro/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro) | Reference | Descriptions of the external Kaggle datasets the notebooks attach. |
| **Archive** | [`archive/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/archive) | Historical drafts | Superseded early simulator drafts and initial brainstorm notes. |

---

## Track 1: Retrieval & Fusion

**Objective**: Maximize retrieval precision on **Class 1** (reference library present) and **Class 2** (known structure in a database, no public spectra). Combines multiple candidate pools (Train, COCONUT, PubChem, ChEBI, LIPID MAPS), multi-channel GBDT ranking, reciprocal rank fusion (RRF), and scaffold mutation.

### Core Specs
- [`roadmap.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/roadmap.md): Operational roadmap for pushing the baseline from 0.384 toward 0.399+.
- [`plan_v2.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/plan_v2.md): The fusion philosophy and R1–R5 rules (why orthogonality pays and deepening single views fails).
- [`baseline_intro.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/baseline_intro.txt): Comprehensive architecture and cell-by-cell audit of the v4n baseline.
- [`c3_routing.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/c3_routing.md): **Start here for the mutator channel.** Self-contained. Covers:
  - why the two mutator submissions scored 0.399 and 0.397;
  - the Class-3 detector and rank-promotion design;
  - every measurement, with how to reproduce it;
  - the code map and the open gates.
- [`validate_mutator.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/validate_mutator.md): Mutator validation on the `pseudo_c3` Class-3 cohort. **Read §8 alongside §1–§7**: the headline 17.2% is an oracle-scaffold number, and the production-viable ceiling is 4.2%.

### Key Files Produced & Used
- `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb`: **Track 1 best (0.399 LB)**. Weighted RRF fusion (α=0.6) between v4n and a second, orthogonal engine, with `ICE_UNION` scoring engine candidates via ICEBERG/GLACIER.
- `casmi26-v4n-inference (1).ipynb`: single-engine baseline (0.384 LB). A 7-cell modular engine with an isolated background PubChem runner and 10,226-bit FPNet scoring.
- `casmi26-v4h-inference.ipynb`: predecessor baseline (0.373 LB). Introduced the GLACIER transformer alongside GNN scoring.
- `casmi26-v4n-mutator-submit.ipynb`: scaffold mutation inference notebook (0.399 LB, no change). Injects mutated scaffold analogs into candidate pools via a third RRF list.
- `casmi26-v4n-mutator-bio-submit.ipynb`: adds engine-A (BIO + AFIX) as a second scaffold source (0.397 LB, −0.002: one molecule lost to same-formula displacement).
- `casmi26-v4n-mutator-c3-submit.ipynb`: **Class-3 routed mutator**. Fires only on a scale-free Class-3 judgement (`build/c3_gate.py`) and promotes into reserved top ranks via `merge()` instead of competing in the fusion. See `c3_routing.md`.
- `build/c3_gate.py`: the Class-3 detector. It fires on a strong mass-shifted spectral analog, a weak exact-mass match and a shallow formula delta. Similarities are in [0,1] only, so it does not depend on the `f·z` scale.
- `build/eval_mutator_cohort.py` & `build/compare_routers.py`: offline harnesses. The first measures mutator recovery on the `p1_novel` Class-3 cohorts; the second compares how sensitive the two candidate routers are to scale.
- `build/delta_m_mutator.py`: scaffold harvesting and ΔM mutation in **formula space, not mass space**. Exact element-count arithmetic means there is no ppm tolerance anywhere in the generator. 20 biogenetic reaction SMARTS, depth ≤ 6.
- `build/patch_v4n_mutator.py`: **the notebook generator**. Every mutator notebook is produced by this script from the 0.399 fork (`--bio`, `--c3`). Never hand-edit the `.ipynb`; change this and regenerate.
- `build/mut_runner.py`: the mutator subprocess driver. Runs the Class-3 gate, the scaffold sources and the two-pass `f·z` screen, and prints the per-molecule diagnostics.
- The local suite: `build/test_delta_m_mutator.py` (93 checks), `build/test_c3_gate.py` (42), `build/test_mut_runner.py` (45 over 14 subprocess scenarios) and `build/test_mut_cell.py` (29, run against each notebook). `sh build/run_all_tests.sh` runs everything.

---

## Track 2: Novel De Novo Generation

**Objective**: Solve **Class 3** (completely novel structures absent from all chemical databases, where pure retrieval has a 0% ceiling) using an autoregressive SELFIES decoder conditioned on frozen DreaMS spectrum embeddings.

### Core Specs
- [`roadmap.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/roadmap.md): Gate-driven roadmap (Phases P0 to P7). All thresholds are pre-registered before measuring.
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/gate_report.md): Central record of pre-registered thresholds, decision rules, and verified gate measurements.
- [`phase3_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/phase3_plan.md): Architectural design for P3 decoder training and evaluation.
- [`search_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/search_plan.md): Combinatorial chemical search design (prefix-seeded, diverse-beam, masked-beam).
- [`printouts.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/printouts.txt): Latest raw training output and evaluation logs (P3 converged at epoch 19; G3.1 passed).

### Key Files Produced & Used
- `build/novel_decoder.py`: core Python module. SELFIES vocabulary tokenization, tracker logic, legal token masking, and chemical graph validation.
- `build/dreams_infer.py`: vendored, standalone PyTorch DreaMS inference module, plus the exact `dreams_peaks` lexsort logic.
- `build/test_novel_p0.py`: local unit test fixture verifying tracker parity, round-trip correctness, and token masks.
- `casmi_novel_p0_colab.ipynb` & `casmi_novel_p0_kaggle.ipynb`: Phase 0 feasibility runners. Verified byte-identical DreaMS embeddings across Colab and a Kaggle T4.
- `casmi_novel_p1_colab.ipynb`: Phase 1 benchmark runner. Built the evaluation cohorts (`pseudo_c3`, `ts_dec`, `c2_displace`).
- `casmi_novel_p3_train_colab.ipynb`: Phase 3 training notebook. Trained the SELFIES decoder on an A100 GPU (converged at epoch 19, loss 0.41023).
- `casmi_novel_p3_eval_colab.ipynb`: Phase 3 combinatorial search bake-off. `masked_beam` won with 0.3067 maxTan.
- `casmi_novel_p3_null_colab.ipynb`: Phase 3 Gate G3.1 null-arm test. A +0.1885 Tanimoto lift over the `RandomDeNovo` null baseline (8.17% exact match @ 25).

---

## Track 3: Forward MS Simulator

**Objective**: Score and re-rank candidate structures by forward-simulating their tandem mass spectra using the ICEBERG deep-learning fragmentation simulator.

### Core Specs
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/gate_report.md): Results of the paired control ablation test, showing that ICEBERG forward simulation contributes +0.035 LB.
- [`new_gate_results.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/new_gate_results.txt): Empirical adduct coverage gate (G2b). 6 adducts are covered (`[M+H]+`, `[M-H]-`, etc.) and 4 are routed to the MetFrag fallback.
- [`mission.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/mission.md): Scope and architecture of the forward simulator pipeline.

### Key Files Produced & Used
- `casmi_simple_sim_submission.ipynb`: Phase 4 submission notebook (0.289 LB). ICEBERG forward spectrum scoring on the test spectra.
- `casmi_simple_sim_nosim.ipynb`: the one-line ablation control notebook (0.254 LB). Shows ICEBERG provides a clean +0.035 LB gain over pure library matching.
- `casmi_simple_sim_grid_colab.ipynb`: analysis runner. Evaluated collision energy (CE) and adduct distributions across the test sets.
- `casmi_simple_sim_gate_kaggle.ipynb`: gate verification runner. Evaluated per-adduct accuracy and lift on a T4 GPU.

---

## Track 4: Priors & Probes

**Objective**: The author's 0.405 gain came from a spectrum-blind PubChem popularity prior applied to one list. Extend it to every list, enrich it with literature, patent and natural-product flags, then add a gated Class-3 generator. Each LB probe is a single change.

**Where it stands (10-05)**: the popularity lever is used up.
- The v29 0.411/0.415 results depended on private arrays; our copy scores 0.393.
- The NP bonus/expansion probes (0.397) and the c3v2 mutator (0.393) are within noise of their base. See EXPERIMENTS rows 30–35.

### Core Specs
- [`summary.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/summary.md): **Start here.** Handover summary of the popularity-prior probes.
- [`plan_v2.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/plan_v2.md): the t4 plan vs the LB notebooks, the consistent signals in the ledger, and the P1–P6 probe queue on 0.411.
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/gate_report.md): pre-registered keep/kill rules and every probe register (v2, c3v2, and **Track 4b** at the bottom).
- [`probe3_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/probe3_plan.md): P3e / P3, the NP pool expansion and the NP prior bonus on the 0.415 code.
- [`c3v2_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/c3v2_plan.md): Class-3 mutator v2 (gated, anchored, two-stage).
- [`analysis_c3cal_results.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/analysis_c3cal_results.md) and [`analysis_c3v_tau8.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/analysis_c3v_tau8.md): analyses of the c3cal calibration and the c3v2-tau8 notebook. Raw rows are in `c3_cal_v2_results/`.
- [`summary_maccs.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/summary_maccs.md): MACCS keys for core invariance in mutator v2.
- [`track1_summary.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/track1_summary.md): autopsy of Track 1's Class-3 attempts, used as the Track 4 blueprint.
- [`comparision_summary.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/comparision_summary.md): forensic comparison of 0.399 → 0.405 → 0.409 → 0.411.
- [`roadmap.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/roadmap.md): v1 plan on the 0.405 base (superseded): Waves A, B, C.
- [`../../updated_knowledge_base.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/updated_knowledge_base.txt): literature knowledge base and the 0.405 notebook audit.

### Key Files Produced & Used
- `evenda-casmi-inferencesupernext.ipynb` / `.py`: **our reproducible base** (own runs 0.399 / 0.401; the author scored 0.405). It is the 0.399 recipe plus the popularity prior on BASE (`POOLPOP_MU = 0.15`), with GLACIER restricted to `[M+H]+`.
- `build/patch_supernext.py`: notebook generator on supernext. Tests: `build/test_patch_supernext.py`.
  - v1 probe flags → `casmi26-t4-<tag>.ipynb`, including the paired control `casmi26-t4-base.ipynb`.
  - Track 4b flags `--nplike` / `--classcons` → `casmi26-t4b-<tag>.ipynb`.
- `casmi26-v29-glacier-m-h-full-fusion-0-411.ipynb` and `casmi26-v29-fusion-pop-glmh-db1615.ipynb` (0.415): the 0.405 recipe plus the popularity prior inside the PubChem channel (`V17_PATCH`, λ 0.25, `POP_UNION=200`) and the `fz_top` merge gate. **They need the private arrays; our public copy `casmi26-v29-fusion-pop-glmh-0-415.ipynb` scores 0.393.**
- `build/patch_v29.py`: generator for the v2/v3 probes (`--mu`, `--pop-eng`, `--np-bonus`, `--np-expand`, `--pop-precut`, `--lib-gate`) → `casmi26-t4v2-*` / `casmi26-t4v3-*`.
- `build/patch_c3v2.py` and `build/patch_c3cal.py`: generate the c3v2 / c3cal notebooks.

---

## Track 4b: NP Signals (dataset improvement)

**Objective**: Down-weight synthetic same-mass decoys in the 105.9M-row PubChem tier with our own natural-product signals, added as terms on top of the popularity prior.

**Outcome (CLOSED 10-05)**:
- Offline, the vendor flag was killed. NP-likeness and class consistency passed at μ 0.5.
- On the LB, all three probes (NP-likeness, class consistency, stack) scored **0.399** vs base 0.400, so **nothing was kept**.
- **Lesson:** the PubChem-list re-order only reaches the few test molecules whose truth is PubChem-only. Estimate a lever's reach before trusting an offline cohort gain.

### Core Specs
- [`summary.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track_dataset_improvement/summary.md): **Start here.** One page: plan, what was built, results, lessons.
- [`np_signals_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/np_signals_plan.md): the approved plan and status log.
- [`np_class_sig_high_influence.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/np_class_sig_high_influence.md): **Track 4c (10-06)**: spectral class vote (delta-list analog hits) as a signed term on the **zenith fused list**, so it can reach rank 1. Register T4C-* in `gate_report.md`.
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track4_priors_probes/gate_report.md) → "Track 4b register": gates NB.*, amendment A-NB.1, realism check R.*, LB probes T4B-*.

### Key Files Produced & Used
- `build/np_signals_core.py` + `build/test_np_signals_core.py` (37 checks): vendor flags, NP_Score, kNN labels, class consistency, and the same-formula benchmark.
- `build/mk_np_signals.py`: generates `casmi_np_signals_colab.ipynb` (build + benchmark) and `casmi_np_signals_realism_colab.ipynb` (cohort P; optional diagnostic).
- `casmi26-t4b-nplike0.5.ipynb`, `casmi26-t4b-classcons0.5.ipynb`, `casmi26-t4b-nplike0.5-classcons0.5.ipynb`: the LB probes (each 0.399).
- Drive outputs in `MyDrive/casmi_2026/_np_signals/`: vendor arrays, the LOTUS fingerprint bank, train labels, and the benchmark rows.

---

## Datasets Reference (`Dataset_intro/`)

- [`pubchem_related_dataset_analysis.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/pubchem_related_dataset_analysis.md): **Start here.** Master guide to the PubChem tier, the v2 pool and the NP datasets.
- [`popularity_dataset_intro.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/popularity_dataset_intro.txt): Gluzdov popularity prior v2, the `pc_*` / `pool_*` SID, PMID, patent, CID and NP arrays. It has no pool key array; pool keys are in `casmi26-v2-pool/pool_meta.parquet`.
- [`natural_product_knowledge_table.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/natural_product_knowledge_table.md): 152,717 LOTUS/NPAtlas connectivities with scorer keys, NPClassifier classes, taxa, and the `in_casmi_pool` / `in_casmi_train` flags.
- [`casmi-26-iceberg.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/casmi-26-iceberg.md) and [`casmi-26-glacier.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/casmi-26-glacier.md): the ICEBERG / GLACIER offline isomer re-scorers.
- [`Simulated_Ranker_Rows.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/Dataset_intro/Simulated_Ranker_Rows.md): 2,250 simulated class-1 / class-2 molecules for ranker training.

---

## Adding a New Track in the Future

When starting a new direction (e.g. Track 5: Graph Foundation Models):
1. Pick the base by **our own** LB runs, never an author's headline, and plan a paired control.
2. Create a new folder: `specs/track5_<name>/`.
3. Add a `plan.md` and a `gate_report.md` inside it, with thresholds pre-registered before any run. Add a `summary.md` when the track closes.
4. Add a row to the **Tracks Overview** table above.
5. Add a dedicated section with its **Core Specs** and **Key Files Produced & Used**.
6. Log every gate and submission in `EXPERIMENTS.md` with `/update-experiments`.
