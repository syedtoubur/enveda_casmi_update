# CASMI 2026 — Master Experiment & Submission Ledger

> **Competition**: Enveda CASMI 2026 — Molecule ID From Mass Spectra  
> **Official Metric**: Mean Reciprocal Rank at 25 (`MRR@25`) evaluated on tautomer-canonical `InChIKey14`  
> **Platform & Hardware**: Kaggle Notebooks (≤ 9 hours, GPU T4/P100, completely offline)  
> **Current Workspace State of the Art**: **0.399 Public LB** (`casmi26-v4n-engine-fusion-union-lb-0-399.ipynb`)  
> **Last Updated**: October 2026

---

## 1. Master Public Leaderboard Lineage

This table tracks every submission attempted on the Kaggle public leaderboard, chronological from earliest baselines to the current best.

| # | Submission / Notebook | Date | Core Approach & Changes (Plain English) | Public LB | Delta | Outcome & Verdict |
|---|---|---|---|---|---|---|
| **01** | `casmi_simple_sim_nosim` | 09-27 | • **Control run**: Library Tier 1 (τ 0.75) + MetFrag-lite + precursor ppm<br>• No neural ranker, no forward MS simulator (`CFG.USE_SIM = False`) | 0.254 | Baseline | Pure rule-based retrieval control |
| **02** | `casmi_simple_sim_submission` v2 | 09-27 | • Added **ICEBERG forward mass-spec simulator** to control run<br>• Re-ordered candidate windows by simulated spectrum similarity | 0.289 | **+0.035** | **Huge win**: Proved forward simulation adds independent physical signal |
| **03** | `casmi_submission_notebook` v3 | 09-21 | • Fast-commit wrapper on phase1 baseline<br>• **Bug**: Only the first 30 of 400 molecules were deep-scored | 0.032 | −0.305 | Run-mode logic bug (fast mode leaked into full run) |
| **04** | `new_trained_fp_ranker` (v7) | 09-23 | • New retrained FPNet + **Stage C v1 ranker matrix**<br>• **Bug**: Sampled training queries from FPNet seen data without disjoint holdout | 0.286 | −0.051 | **Leak failure**: Model acted as oracle on 90% of training queries; GBDT discarded other features |
| **05** | `casmi_submission_notebook` v6 | 09-22 | • Exploration on top of early phase1 base | 0.309 | −0.028 | Intermediate tuning |
| **06** | `casmi_submission_notebook` v8 | 09-23 | • Evaluated polarity-aware representatives at inference against forked matrix | 0.310 | +0.024 | Confounded because matrix was not built with them |
| **07** | `casmi_submission_notebook` v11 | 09-25 | • Ranker feature weighting variation | 0.314 | +0.004 | Marginal variation |
| **08** | `enveda-casmi-2026-phase1-enhanced-baseline` (v5) | 09-21 | • Polarity-aware analog channel + `PPM_EXPAND` / backfill<br>• Forked FPNet + forked matrix | 0.319 | +0.005 | Slower than tier1-patched with unexplained −0.018 gap |
| **09** | `casmi_submission_notebook` v9/v10 | 09-25 | • New retrained FPNet + **Stage C v2 ranker matrix** (gate-passed, leak-fixed)<br>• Submitted despite Stage D (`val_scripts`) advising HOLD | 0.323 | **+0.004** | Proved Stage D offline validation was misleading on Class 2 |
| **10** | `v14` (local val tuned) | — | • Hyperparameter-tuned on local GNPS / Enveda validation split | 0.326 | **−0.015** | **Overfitting failure**: Local val lacks blind pool misses |
| **11** | `enveda-casmi-2026-tier1-patched` (v2) | 09-21 | • Forked public baseline: Library + Analog + MetFrag + FPNet → GBDT | 0.337 | **+0.011** | Long-standing early baseline benchmark |
| **12** | `two-ranker engine` (v13 base) | — | • Two-ranker blend + AFIX analog propagation index (`N_ANALOG=200`) | 0.341 | **+0.004** | Structural improvement in analog retrieval |
| **13** | `+ BIO` (candidate expansion) | — | • Expanded candidate pool by injecting **ChEBI + LIPID MAPS** | 0.350 | **+0.009** | **Orthogonal win**: Expanding chemical space directly raises score ceiling |
| **14** | `casmi26-v4g` (v17) | — | • FPNet A + 160 features + 4 LightGBM boosters<br>• Isolated PubChem background subprocess (`pc_runner.py`) + ICEBERG | 0.366 | **+0.016** | Multi-view architecture baseline |
| **15** | `casmi26-v4h-inference` | — | • Added **GLACIER transformer** as 3rd neural scorer (λ = 0.5)<br>• Blended GNN + Transformer predictions | 0.373 | **+0.007** | Model diversity (different neural architectures) pays |
| **16** | `casmi26-v4m` | — | • Swapped engine fingerprint bank `fpnet_0` → `fpnet_full1` | 0.378 | **−0.002** | Replacing one view within same family slightly hurt |
| **17** | `casmi26-v4l` | — | • Boosted physics simulator weights: ICE/GL λ 0.5 → 1.0<br>• Re-weighted neural vs physical evidence | 0.380 | **+0.007** | Giving full voice to independent scorers pays |
| **18** | `casmi26-v4g + fusion` | — | • Weighted Reciprocal Rank Fusion (RRF, α=0.6) with 2nd orthogonal engine | 0.382 | **+0.016** | Fusing two completely independent retrieval systems beats either |
| **19** | `casmi26-v4n-inference (1)` | — | • Combined v4i (PubChem pass-1 depth 5000) + v4l (λ 1.0) + v4m | 0.384 | **+0.004** | SOTA single-engine baseline |
| **20** | `casmi26-v4g + fusion + union` | — | • **ICE_UNION**: Top-40 candidates from 2nd engine scored by ICEBERG/GLACIER | 0.386 | **+0.004** | Allowing physics scorers to evaluate both candidate pools |
| **21** | `casmi26-v4n-engine-fusion-union` | 10-01 | • Strongest v4n base + orthogonal 2nd engine + ICE_UNION + RRF fusion | **0.399** | **+0.015** | **Current State of the Art (0.399 LB)** |
| **22** | `casmi26-v4n-mutator-submit` | 10-02 | • Added the **ΔM mutator** channel: Murcko scaffolds from each molecule's PubChem top-25, decorated to a hit formula by a 20-transform biogenetic palette in formula space<br>• Third RRF list at `ALPHA_MUT = 0.25`, fired on the weakest 30% by `best_pool_fz` | 0.399 | **0.000** | **Arithmetic ceiling, not a bug.** At `ALPHA_MUT=0.25, KRR=3` the mutator lands exactly **one** candidate in the top 25, at rank 20, so its best case was ≈+0.0007 — below 3-dp resolution. `roadmap.md` §8 predicted ≈+0.001 before shipping |
| **23** | `casmi26-v4n-mutator-bio-submit` | 10-02 | • Same, plus engine-A top-40 (BIO = ChEBI + LIPID MAPS, AFIX analog propagation) as a **second scaffold source**<br>• `MUT_K` 25 → 40; ICE union capped at 25 | 0.397 | **−0.002** | **One molecule, lost to friendly fire.** The mutator emits same-formula regioisomers, so it always lands in the truth's own formula group where the `z(RRF)+z(ICE)+z(GL)` rerank can swap the two: `(1−1/20)/400 = 0.0024`. `MUT_K` 25→40 was a provable no-op (`fuse2` cuts at `n=40`) |

---

## 2. Component & Offline Validations (Colab / Holdout / Gates)

Offline benchmarks that validated individual modules before or alongside Kaggle submission.

### A. Forward MS Simulator (ICEBERG) Adduct Coverage Gate (`new_gate_results.txt`)
- **Benchmark Cohort**: 380 queries with valid mass, 7,551 candidate predictions.
- **Speed**: 67.6 mol/s on NVIDIA A100 (111.6s total for 7,551 molecules).
- **Adduct Coverage Rule (G2b)**: Requires $N \ge 15$, top-1 $\ge 2.0\times$ chance, and $p < 0.01$.

| Adduct | Samples ($N$) | Candidates | Top-1 Accuracy | Chance Top-1 | Lift vs Chance | MRR | Covered by Simulator? |
|---|---|---|---|---|---|---|---|
| `[M+H]+` | 40 | 21.0 | **90.0%** | 4.9% | **18.55x** | **0.943** | **COVERED (Pass)** |
| `[M-H]-` | 40 | 21.0 | **57.5%** | 5.2% | **10.99x** | **0.712** | **COVERED (Pass)** |
| `[M+Na]+` | 40 | 21.0 | **35.0%** | 4.8% | **7.26x** | **0.499** | **COVERED (Pass)** |
| `[M+CH2O2-H]-` | 40 | 21.0 | **22.5%** | 5.0% | **4.48x** | **0.405** | **COVERED (Pass)** |
| `[M+Cl]-` | 40 | 21.0 | **22.5%** | 4.9% | **4.63x** | **0.384** | **COVERED (Pass)** |
| `[M+NH4]+` | 40 | 21.0 | **20.0%** | 5.0% | **4.00x** | **0.371** | **COVERED (Pass)** |
| `[M-H2O+H]+` | 40 | 21.0 | 20.0% | 8.6% | 2.33x | 0.429 | **UNCOVERED** (routed to MetFrag-lite fallback) |
| `[M-2H2O+H]+` | 40 | 21.0 | 12.5% | 5.4% | 2.33x | 0.274 | **UNCOVERED** (routed to MetFrag-lite fallback) |
| `[M+K]+` | 40 | 21.0 | 15.0% | 5.3% | 2.84x | 0.279 | **UNCOVERED** (routed to MetFrag-lite fallback) |
| `[M-H2O-H]-` | 19 | 21.0 | 21.1% | 8.6% | 2.45x | 0.384 | **UNCOVERED** (routed to MetFrag-lite fallback) |

---

### B. Neural FPNet Checkpoints on Scaffold-Disjoint Holdout (`RUNBOOK.md`)
- **Evaluation Set**: Scaffold-disjoint holdout evaluated against 48 real ±8.5 ppm decoys.

| Checkpoint | Architecture | es_MRR | es_top1 | Verdict |
|---|---|---|---|---|
| `fp_single_s0` | Single-CE model, seed 0 | **0.8278** | **0.7425** | Generalizes strongly across unseen chemical scaffolds |
| `fp_single_s1` | Single-CE model, seed 1 | 0.8250 | 0.7366 | Consistent with seed 0 |
| `fp_merged_m0` | Merged multi-CE model, seed 0 | **0.8222** | **0.7431** | Crucial for the 24.6% merged spectra in test set |
| `fp_merged_m1` | Merged multi-CE model, seed 1 | 0.8197 | 0.7360 | Consistent with seed 0 |

---

### B2. ΔM Mutator on a Leak-Free Class-3 Cohort (`build/eval_mutator_cohort.py`)
- **Cohort**: `p1_novel/pseudo_c3.parquet` — 600 natural products, truth banned from candidate retrieval (`pool_ban_keys`), every MassSpecGym/ICEBERG-seen structure excluded. `val_scripts/val_key.pkl` carries `cls ∈ {1,2}` only, so this is the project's first honest Class-3 benchmark.
- **Why the headline number is not the production number**: the original run seeds from each truth's *own* Murcko scaffold and *own* formula (GM.0a/GM.0b oracle conditions). In production the scaffold must be harvested from a retrieved hit.

| Condition | n | Exact ik14 hits | Rate | % of cohort | Verdict |
|---|---|---|---|---|---|
| All formula-reachable cyclic (**oracle** scaffold + formula) | 418 | 72 | 17.2% | 12.0% | corroborates GM.0b (21.6%) on independent data |
| **+ scaffold harvestable** (`group_size > 1`) | 240 | 25 | 10.4% | **4.2%** | **GC.0 PASS** — 53.2% of the cohort is alone on its Murcko scaffold, taking 65% of the hits out of reach |
| **+ delta within 2 transforms** | 71 | 20 | 28.2% | 3.3% | the band worth gating on |

- **`q₃` (truth in the mutator's top 3, random-screen floor)**: depth ≤ 1 → **31.4%** on 2.5% of the cohort; depth ≤ 2 → 7.6% on 11.8%. Same expected gain, a quarter of the exposure, so `c3_gate.MAX_STEPS = 1`. **GC.1 PASS**.
- **Acyclic wall**: 129 of 600 (21.5%) have no Murcko scaffold at all and account for 70.9% of misses — materially worse than the 9.4% previously assumed.
- **Still open**: GC.2 (`a`, the pipeline's top-1 accuracy on the fired set) and GC.3 (detector precision) need the engine channels, i.e. a Colab/Kaggle run.

---

### C. Novel De Novo Molecule Generation (`specs_novel`)
Generates novel Class-3 structures using a SELFIES decoder on frozen DreaMS embeddings.

| Milestone / Gate | Experiment & File | Metric & Pass Condition | Measured Result | Verdict |
|---|---|---|---|---|
| **G0 Feasibility (P0)** | `casmi_novel_p0` (Colab + Kaggle) | Vendored DreaMS inference vs Colab official $\ge 0.9999$; SELFIES round-trip $\ge 99\%$ | Vendored agreement: 1e-11 on Kaggle T4; SELFIES round-trip: 100.0% over 194,589 structures; Tracker parity: 100% | **PASS** (G0.1-R2 verified) |
| **G1 Cohorts (P1)** | `casmi_novel_p1_colab` | Balanced benchmark cohorts for novel molecules | `pseudo_c3`: 600, `ts_dec`: 155, `c2_displace`: 400; Training set: 94.06% | **PASS** |
| **G3.2 Convergence (P3)** | `casmi_novel_p3_train_colab` | Early stopping convergence (patience 5) | Early stopped at epoch 25; best dec_val loss **0.41023** at epoch 19 (A100) | **PASS** |
| **Search Bake-Off** | `casmi_novel_p3_eval_colab` | Highest median top-25 max Tanimoto on `dec_val` | • `prefix_seeded`: 0.2910 (2.17s/mol)<br>• **`masked_beam`**: **0.3067** (15.85s/mol)<br>• `diverse_beam`: 0.3067 (21.49s/mol) | **`masked_beam` won** |
| **G3.1 Null Arm Gate (P3)** | `casmi_novel_p3_null_colab` | Median top-25 max Tanimoto on `pseudo_c3` vs RandomDeNovo null $\ge +0.10$ | • `masked_beam`: **0.3092**<br>• RandomDeNovo: **0.1206**<br>• **Lift**: **+0.1885** (well above +0.10 threshold)<br>• Exact match @ 25: **8.17%** (null: 0.0%) | **PASS** |

---

## 3. The Graveyard: Negative Results & Pitfalls

Understanding what failed and why saves weeks of wasted compute.

| Attempted Change | Observed Result | Why It Failed / Root Cause | Rule to Remember |
|---|---|---|---|
| **Forward λ = 2.0** | LB dropped **−0.034** (0.386 → 0.352) | Over-weighting the simulator destroys confident spectral library matches in Class 1. | Keep forward physics scoring balanced ($0.5 \le \lambda \le 1.0$). |
| **Local GNPS / Val Split Tuning** | LB dropped **−0.015** (v13 0.341 → v14 0.326) | Local validation is biased and blind to real pool misses in Class 2. | Do not overfit hyperparameters to local validation cohorts. |
| **Stage C v1 Training Leak** | LB dropped **−0.051** (0.337 → 0.286) | Sampled queries from the FPNet training set. Model acted as 90% oracle, so GBDT learned to rely 60% on model and discarded library/analog. | Always strictly separate FPNet-seen from FPNet-unseen queries in Stage C. |
| **Swapping FPNet Backbone** | LB dropped **−0.002** (v4l 0.380 → v4m 0.378) | Swapped `fpnet_0` with `fpnet_full1`. Both models make the same fundamental errors. | Replacing a model within the same family yields zero gain. |
| **Deepening Window `TOPN 60 → 100`** | LB changed by **0.000** (stayed 0.386) | Non-orthogonal depth captures more decoys at the same rate as truths. | Deepening non-orthogonal views burns time without improving MRR. |
| **Generated candidates inside the RRF (`ALPHA_MUT = 0.25`)** | LB **0.000** then **−0.002** (0.399 → 0.399 → 0.397) | Two separate faults. (1) A weight below the base list's own tail caps the entry rank: at `KRR=3` the mutator's best candidate lands at fused rank 20, paying `1/20`, so the channel could not move a 3-dp score whatever its chemistry. (2) Generated structures are same-formula regioisomers of the target, so they enter the truth's own formula group and the λ=1+1 forward rerank can promote a decoy over a correct rank-1 answer — the same failure as forward λ=2.0, which cost −0.034. | **A source with no retrieval score must not be fused; it must be routed.** Decide separately whether it applies, then splice it into reserved ranks. Check the entry rank *before* shipping: the payoff is `1/rank`, and a weight that caps it below ~1/12 cannot pay. |
| **Calibrating an absolute `f·z` threshold on a commit run** | not submitted — caught offline | `pc_lists_bio.json` holds exactly 12 molecules (`SMOKE_N`), and a commit run scores a slice of `train.parquet`, so every molecule in it is a *training* structure. The median `best_pool_fz` moves +172.3 between it and the real 400-molecule list; shifting the distribution ±300 swings a `< 300` gate between **16 and 147** molecules. | Gate on quantities that are scale-free — similarities in [0,1], counts — or on a **quantile of the current run's own distribution**. Never on an absolute `f·z` cut fitted to placeholder output. |
| **Fast-Commit Logic Bug** | LB dropped to **0.032** (v3) | Only first 30 of 400 test molecules were scored; remaining received dummy guesses. | Always assert full-test output shape and non-trivial candidate counts. |
