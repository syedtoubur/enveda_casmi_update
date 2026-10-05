# CASMI 2026 — Master Specifications Index

> **Purpose**: Single entry-point for humans and AI agents navigating the research tracks, architectural roadmaps, and code files for the Enveda CASMI 2026 competition.  
> **Official Metric**: MRR@25 on InChIKey14 (tautomer-canonical)  
> **Current Best Score**: **0.399 Public LB** (Track 1)  
> **Master Experiment Ledger**: See [EXPERIMENTS.md](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/EXPERIMENTS.md)

---

## Tracks Overview

| Track | Directory | SOTA / Status | Focus |
|---|---|---|---|
| **specs_scratch: own pipeline** | [`specs_scratch/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/specs_scratch) | **Active (10-05)**: S0 done, R0 ready to submit | Our own tested package (`scratch_pkg/`) built rung by rung from the Track 3 base; Tracks 1–4 are its read-only knowledge base. |
| **Track 1: Retrieval & Fusion** | [`track1_retrieval_fusion/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion) | **0.399 LB** (Current Best) | Class 1 & 2 retrieval, candidate expansion (PubChem/BIO), dual-engine rank fusion, and scaffold mutation. |
| **Track 2: Novel De Novo** | [`track2_novel_generation/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation) | **Phase 3 (G3.1 PASS)** | Class 3 de novo molecule generation via SELFIES decoder on frozen DreaMS embeddings. |
| **Track 3: Forward MS Simulator** | [`track3_ms_simulator/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator) | **Validated (+0.035 LB)** | Physical MS/MS forward spectrum prediction (ICEBERG/GLACIER) for candidate re-ranking. |
| **Archive** | [`archive/`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/archive) | Historical Drafts | Superseded early simulator drafts and initial brainstorm notes. |

---

## Track 1: Retrieval & Fusion

**Objective**: Maximize retrieval precision on **Class 1** (reference library present) and **Class 2** (known structure in database, no public spectra). Combines multiple candidate pools (Train, COCONUT, PubChem, ChEBI, LIPID MAPS), multi-channel GBDT ranking, reciprocal rank fusion (RRF), and scaffold mutation.

### Core Specs
- [`roadmap.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/roadmap.md) — Operational roadmap for pushing the baseline from 0.384 toward 0.399+.
- [`plan_v2.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/plan_v2.md) — The fusion philosophy and R1–R5 rules (why orthogonality pays and deepening single views fails).
- [`baseline_intro.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/baseline_intro.txt) — Comprehensive architecture and cell-by-cell audit of the v4n baseline.
- [`c3_routing.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/c3_routing.md) — **Start here for the mutator channel.** Self-contained: why the two mutator submissions scored 0.399 and 0.397, the Class-3 detector and rank-promotion design, every measurement with how to reproduce it, the code map, and the open gates.
- [`validate_mutator.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track1_retrieval_fusion/validate_mutator.md) — Mutator validation on the `pseudo_c3` Class-3 cohort. **Read §8 alongside §1–§7**: the headline 17.2% is an oracle-scaffold number and the production-viable ceiling is 4.2%.

### Key Files Produced & Used
- `casmi26-v4n-engine-fusion-union-lb-0-399.ipynb`: **Current SOTA submission (0.399 LB)** — Weighted RRF fusion (α=0.6) between v4n and a second orthogonal engine, with `ICE_UNION` scoring engine candidates via ICEBERG/GLACIER.
- `casmi26-v4n-inference (1).ipynb`: SOTA single-engine baseline (0.384 LB) — 7-cell modular engine with isolated background PubChem runner and 10,226-bit FPNet scoring.
- `casmi26-v4h-inference.ipynb`: Predecessor baseline (0.373 LB) — Introduced GLACIER transformer alongside GNN scoring.
- `casmi26-v4n-mutator-submit.ipynb`: Scaffold mutation inference notebook (0.399 LB, no change) — Injects mutated scaffold analogs into candidate pools via a third RRF list.
- `casmi26-v4n-mutator-bio-submit.ipynb`: Adds engine-A (BIO + AFIX) as a second scaffold source (0.397 LB, −0.002 — one molecule lost to same-formula displacement).
- `casmi26-v4n-mutator-c3-submit.ipynb`: **Class-3 routed mutator** — fires only on a scale-free Class-3 judgement (`build/c3_gate.py`) and promotes into reserved top ranks via `merge()` instead of competing in the fusion. See `c3_routing.md`.
- `build/c3_gate.py`: The Class-3 detector — strong mass-shifted spectral analog, weak exact-mass match, shallow formula delta. Similarities in [0,1] only, so no dependence on the `f·z` scale.
- `build/eval_mutator_cohort.py` & `build/compare_routers.py`: Offline harnesses — mutator recovery on the `p1_novel` Class-3 cohorts, and the scale-sensitivity comparison of the two candidate routers.
- `build/delta_m_mutator.py`: Scaffold harvesting and ΔM mutation in **formula space, not mass space** — exact element-count arithmetic, so there is no ppm tolerance anywhere in the generator. 20 biogenetic reaction SMARTS, depth ≤ 6.
- `build/patch_v4n_mutator.py`: **The notebook generator** — every mutator notebook is produced by this script from the 0.399 fork (`--bio`, `--c3`). Never hand-edit the `.ipynb`; change this and regenerate.
- `build/mut_runner.py`: The mutator subprocess driver — Class-3 gate, scaffold sources, two-pass `f·z` screen, and the per-molecule diagnostics the notebook prints.
- `build/test_delta_m_mutator.py` (93 checks), `build/test_c3_gate.py` (42), `build/test_mut_runner.py` (45 over 14 subprocess scenarios) and `build/test_mut_cell.py` (29, run against each notebook): the local suite. `sh build/run_all_tests.sh` runs everything.

---

## Track 2: Novel De Novo Generation

**Objective**: Solve **Class 3** (completely novel structures absent from all chemical databases, where pure retrieval has a 0% ceiling) using an autoregressive SELFIES decoder conditioned on frozen DreaMS spectrum embeddings.

### Core Specs
- [`roadmap.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/roadmap.md) — Gate-driven roadmap (Phases P0 to P7). All thresholds are pre-registered before measuring.
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/gate_report.md) — Central record of pre-registered thresholds, decision rules, and verified gate measurements.
- [`phase3_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/phase3_plan.md) — Architectural design for P3 decoder training and evaluation.
- [`search_plan.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/search_plan.md) — Combinatorial chemical search design (prefix-seeded, diverse-beam, masked-beam).
- [`printouts.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track2_novel_generation/printouts.txt) — Latest raw training output and evaluation logs (P3 converged at epoch 19; G3.1 passed).

### Key Files Produced & Used
- `build/novel_decoder.py`: Core Python module — SELFIES vocabulary tokenization, tracker logic, legal token masking, and chemical graph validation.
- `build/dreams_infer.py`: Vendored, standalone PyTorch DreaMS inference module + exact `dreams_peaks` lexsort logic.
- `build/test_novel_p0.py`: Local unit test fixture verifying tracker parity, round-trip correctness, and token masks.
- `casmi_novel_p0_colab.ipynb` & `casmi_novel_p0_kaggle.ipynb`: Phase 0 feasibility runners — verified byte-identical DreaMS embeddings across Colab and Kaggle T4.
- `casmi_novel_p1_colab.ipynb`: Phase 1 benchmark runner — constructed honest evaluation cohorts (`pseudo_c3`, `ts_dec`, `c2_displace`).
- `casmi_novel_p3_train_colab.ipynb`: Phase 3 training notebook — trained SELFIES decoder on A100 GPU (converged at epoch 19, loss 0.41023).
- `casmi_novel_p3_eval_colab.ipynb`: Phase 3 combinatorial search bake-off — evaluated search algorithms (`masked_beam` won with 0.3067 maxTan).
- `casmi_novel_p3_null_colab.ipynb`: Phase 3 Gate G3.1 null arm test — proved +0.1885 Tanimoto lift over `RandomDeNovo` null baseline (8.17% exact match @ 25).

---

## Track 3: Forward MS Simulator

**Objective**: Score and re-rank candidate structures by forward-simulating their tandem mass spectra using the ICEBERG deep-learning fragmentation simulator.

### Core Specs
- [`gate_report.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/gate_report.md) — Results of the paired control ablation test proving ICEBERG forward simulation contributes +0.035 LB.
- [`new_gate_results.txt`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/new_gate_results.txt) — Empirical adduct coverage gate (G2b): 6 covered adducts (`[M+H]+`, `[M-H]-`, etc.) and 4 routed to MetFrag fallback.
- [`mission.md`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/specs/track3_ms_simulator/mission.md) — Scope and architecture of the forward simulator pipeline.

### Key Files Produced & Used
- `casmi_simple_sim_submission.ipynb`: Phase 4 submission notebook (0.289 LB) — Evaluated ICEBERG forward spectrum scoring on test spectra.
- `casmi_simple_sim_nosim.ipynb`: Exact one-line ablation control notebook (0.254 LB) — Proved ICEBERG provides a clean +0.035 LB gain over pure library matching.
- `casmi_simple_sim_grid_colab.ipynb`: Analysis runner — Evaluated collision energy (CE) and adduct distributions across test sets.
- `casmi_simple_sim_gate_kaggle.ipynb`: Gate verification runner — Evaluated per-adduct accuracy and lift on GPU T4.

---

## Adding a New Track in the Future

When starting a new direction (e.g. Track 4: Graph Foundation Models):
1. Create a new folder: `specs/track4_<name>/`.
2. Add a `roadmap.md` and `plan.md` inside that folder.
3. Add a row to the **Tracks Overview** table above.
4. Add a dedicated section with its **Core Specs** and **Key Files Produced & Used**.
