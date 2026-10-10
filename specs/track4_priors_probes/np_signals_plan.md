# Track 4b — Our own NP signals for the PubChem channel

## Context
- The only spectrum-blind lever that has paid is the borrowed popularity prior `pop = log1p(SID) + log1p(PMID)`. It is added as `z(spectral) + μ·pop`: μ 0.15 in the pool list (C7) and λ 0.25 in the PubChem channel (`V17_PATCH`).
- No ranker has an NP feature, and none of the scored notebooks uses an NP term.
- Synthetic decoys enter through the **105.9M PubChem tier**. The 710k pool is COCONUT plus the training structures, so it is already almost all NP and an NP signal barely separates candidates there.
- Decision from the grilling session: **keep the borrowed popularity arrays unchanged.** Add three new NP signals of our own, built from our own datasets, as extra additive terms.
- **Base = `evenda-casmi-inferencesupernext.ipynb`**. From our own account it scored **0.399 and 0.401** (mean 0.400). The 0.405 in EXPERIMENTS row 24 is the *author's* score, not ours.
  - The v29 lineage scores 0.393–0.399 from our account: its 0.411/0.415 depended on the author's private arrays.
  - In supernext, the PubChem channel is ordered by `f·z` only: `CASMI_POP_LAM` is set but never read. That gives the NP signals a clean slot.
  - Same-code noise is ±0.004. np1/npx at 0.397 against a single 0.393 control is **not** a proven +0.004, and their counters were never read.
- Goal: finish fast, using 2–4 LB submissions in total. Signals are filtered offline first so the LB budget is not wasted.

## Decisions (settled in grilling)
| # | Decision |
|---|---|
| Objective | New NP signals only. The popularity arrays are not changed. |
| Signals | (1) vendor-only decoy flag, (2) Ertl NP-likeness, (3) class consistency |
| Build venue | Colab + Drive → our private Kaggle dataset `casmi26-np-signals` |
| Integration | additive `+ μ_s·signal`, one signal per probe |
| Channel | PubChem channel first; the fused list (after `fuse2`) only after a PubChem-channel win |
| Order | vendor → NP-likeness → class consistency; kill a signal as soon as it fails its gate |
| Vendor form | binary `vendor_only = (all SIDs from depositors in PubChem's "Chemical Vendors" category) AND PMID == 0`. Applied as a penalty: `− μ·vendor_only` |
| NP-likeness | RDKit `Contrib/NP_Score`, computed on the fly on the post-pass-2 shortlist only (no 105M precompute) |
| Class labels | NP knowledge table labels (`np_superclass`/`np_class`), plus kNN propagation from the nearest LOTUS neighbour (Tanimoto ≥ 0.7). Unlabelled = **neutral 0**, never a penalty. **Fallback (later phase): NPClassifier run on all candidates and analogue hits**, if coverage is the bottleneck. |
| Offline gate | same-formula PubChem-tier window. Rank by `pop + μ·s` vs `pop` alone, μ ∈ {0.25, 0.5, 1.0}. KEEP if MRR@25 ≥ +0.010 on the no-spectrum LOTUS cohort AND the loss on the training-structure cohort is ≤ 0.005 |
| LB control | **paired**: resubmit unchanged supernext in the same batch as the first probe. Base = mean of all our own unchanged supernext runs (currently 0.399 and 0.401, plus the paired run) |
| LB keep bar | ≥ base mean + 0.004 **and** the signal's counters are non-zero (`top-1 changed`, signal hits) |
| Stacking | if ≥ 2 signals pass, one stacked probe; it must beat the best single signal |

## Steps

### Step 0 — Pre-register (before any measurement)
- Append a "Track 4b NP signals" register to `specs/track4_priors_probes/gate_report.md`. It holds the offline gate and the LB bar above, plus the coverage gate for class consistency:
  - ≥ 50% of queries have at least one labelled analogue hit, and
  - ≥ 20% of same-formula windows hold ≥ 2 distinct labelled superclasses.
  - Otherwise drop the signal (or move to NPClassifier).
- Plan doc: `specs/track4_priors_probes/np_signals_plan.md` (this plan). Add a row to `specs/README.md`.

### Step 1 — Colab build: `build/np_signals_colab.ipynb`
Inputs, pulled with the Kaggle API and the PubChem FTP:
- the `casmi26-pubchem-tier` arrays (`pc_mass`, `pc_off`, `pc_smiles`)
- Gluzdov popularity v2 (`pc_cid`, `pc_lsid`, `pc_lpmid`, `pc_np`, `pool_*`)
- `natural_products.parquet`
- PubChem `Substance/Extras/SID-Map`, `CID-SID` and the PubChem data-source table (source → category)

Outputs, aligned row for row with `pc_mass.npy` (assert len = 105,875,489) and `pool_meta` (assert len = 710,701):
- `pc_vendor_only.npy` / `pool_vendor_only.npy` (uint8, packable to bits). The CID mapping uses Gluzdov's `pc_cid` (the most-deposited CID per row).
- `np_labels.parquet`: `scorer_key → superclass, class, label_source ∈ {table, knn}`, covering the table plus kNN-propagated training structures and tier rows. The kNN runs only over the rows the benchmark and inference can reach: training structures plus the shortlist-reachable tier rows. If that is too expensive, label lazily at inference from a packed LOTUS fingerprint bank (decide by measuring the time on Colab).
- `MANIFEST.json` with row counts, source dates and the RDKit version (2026.03.3, scorer-key parity check).

Sanity checks, printed in the notebook:
- the share of NP-table compounds with `vendor_only = 1` must be ≈ 0 (it should be < 1%; otherwise the vendor categorisation is wrong)
- the overall `vendor_only` rate in the tier (expect most of the 105M)

### Step 2 — Colab offline benchmark: `build/np_signals_bench.ipynb`
- Cohort T: training structures (unique scorer keys). Cohort L: `natural_products.parquet` with `in_casmi_train == False`.
  - Sample about 2k each.
  - Use the same-formula window from the tier, deduplicated by scorer key, with the truth guaranteed to be in the window.
- For each signal and μ, report MRR@25 and hit@1/5/25 of the truth for `pop` vs `pop + μ·s`.
- NP-likeness is scored on the window members. Class consistency needs a query-side class: use the class of the truth's nearest *other* training structure (leave-one-out), which stands in for the spectral-analogue hit.
- Write the results to `specs/track4_priors_probes/np_signals_bench.md`, then apply the pre-registered gates. **A signal that fails stops here.**

### Step 3 — Notebook patch: extend `build/patch_supernext.py` with NP-signal flags
- Source: `evenda-casmi-inferencesupernext.ipynb`/`.py`.
- First locate where supernext's PubChem channel sorts its candidates by `f·z` (the PubChem runner source embedded in or attached to C4). Put a `must_replace` anchor there, and add `+ μ_s·signal` to that sort key.
- Leave the unused `CASMI_POP_LAM` alone: turning the PubChem popularity prior on would be a second change.
- Reuse the existing flag and test pattern of `patch_supernext.py`/`test_patch_supernext.py`. The output notebooks are `casmi26-t4b-<tag>.ipynb`.
- Flags:
  - `--vendor μ`: loads `pc_vendor_only.npy` via `mmap`
  - `--nplike μ`: RDKit `npscorer` on the shortlist, z-scored within each molecule's window
  - `--classcons μ`: superclass agreement with the classes of the molecule's library or analogue hits, weighted by similarity, at superclass level, k = 5
  - `--fused`: applies the winning term again after `fuse2` (only in a later probe)
- Every flag sits inside `try/except` with a printed no-op notice, and honours `KAGGLE_IS_COMPETITION_RERUN`.
- Counters printed: signal hit-rate, `top-1 changed in N`, and the min/median/p99 of the term.
- The `fz_top` merge gate stays on pure spectral evidence. Do not touch it.
- Tests: `build/test_patch_np_signals.py`, wired into `build/run_all_tests.sh`.

### Step 4 — LB probes (2–4 submissions)
1. Vendor flag at the best offline μ, if it passed.
2. NP-likeness at the best offline μ, if it passed.
3. Class consistency, if it passed the coverage and offline gates.
4. A stack of the winners, or `--fused` on the single winner.

Submit unchanged supernext as the paired control in the same batch as probe 1. Check each probe against the base-mean + 0.004 bar, and log each result with `/update-experiments`.

### Step 0b — Ledger corrections (with `/update-experiments`)
- Row 24: 0.405 is the author's score. Our own unchanged runs are 0.399 and 0.401.
- Row 29 (A1 precut, 0.401): its "−0.004 → REVERT" was measured against the author's 0.405. Against our own 0.399/0.401 it is ±0, so re-mark it as **neutral (noise)**, not a loss.
- Rows 32–33: note that +0.004 against a single control is within noise, and that the counters were never read.

### Deferred / later
- NPClassifier labels (the class-consistency fallback).
- NP signals in the pool list (C7) and engine-A list (low expected value: the pool is already NP).

## Critical files
- `evenda-casmi-inferencesupernext.py` (base; popularity config lines 58–63, pool prior lines ~197–203)
- `build/patch_supernext.py` + `build/test_patch_supernext.py` (to extend); `build/patch_v29.py` (reference only, for the `V17_PATCH` style of hooking into the PubChem channel)
- `specs/track4_priors_probes/gate_report.md`, `EXPERIMENTS.md`, `specs/README.md`
- `specs/Dataset_intro/natural_product_knowledge_table.md` (label source, cohort L)

## Verification
- Colab build: row-count asserts, the scorer-key parity check against the notebook's `score_key`, and the vendor sanity check on NP-table compounds.
- Benchmark: the truth is in its own window for 100% of the cohort, and the baseline `pop`-only MRR is reported.
- Patch: `compare.py base.ipynb variant.ipynb` shows only the intended cell delta, and `sh build/run_all_tests.sh` passes.
- Kaggle commit run (12 molecules): counters print, the term scale is sane, and runtime fits the 9 h budget (NP_Score on the shortlist is about 400 × 100 molecules, which takes seconds).
- LB: `submission.csv` is (400, 2) with ≤ 25 unique IK14 values per row; apply the keep bar.

## Status (2026-10-04)
- Steps 0 and 0b are done: the Track 4b register is in `gate_report.md`, the ledger corrections are in EXPERIMENTS rows 24, 29, 32 and 33 and the header, and the README is updated.
- Steps 1 and 2 are built as **one** notebook: `casmi_np_signals_colab.ipynb`, generated by `build/mk_np_signals.py` from `build/np_signals_colab.py`. Its core is `build/np_signals_core.py`, covered by 32 checks in `build/test_np_signals_core.py` (wired into `run_all_tests.sh`).
- **Next (user):** run it on Colab with CPU High-RAM. The inputs are read from Drive: `MyDrive/casmi_2026/casmi26-pubchem-tier`, `casmi26-pubchem-popularity-prior` (v2) and `casmi26-natural-product-knowledge-table`. The tier and popularity arrays are copied to local disk first. No Kaggle API is needed. Paste the printout back, then upload `_np_signals/` as the private dataset `casmi26-np-signals`.
- Step 3 (the patch flags) waits for the gate results: only signals that pass get a flag.
- Open risk: PubChem's source-table CSV could not be fetched locally (server busy), so its category column is unverified. If it is missing, the notebook marks the vendor signal VOID and still benchmarks the other two.

## Status (2026-10-04, after the benchmark)
- Benchmark (gate_report NB.*): vendor **KILLED**, NP-likeness and class consistency **PASS at μ 0.5** under amendment A-NB.1 (rule 5 caps μ at 0.5 on first appearance).
- LB notebooks come from `build/patch_supernext.py`: `--nplike 0.5`, `--classcons 0.5`, and the stack (both). The NP block re-orders the PubChem list after its ICE/GL rerank and keeps `pc_fz[0]`, so the `rel` gate is unchanged. Class consistency needs only the NP knowledge table attached: the kNN bank is built from its labelled SMILES.
- Paired control: `casmi26-t4-base.ipynb` (instrumentation only, the same code path as the probes).
- Realism check `casmi_np_signals_realism_colab.ipynb` (R.1–R.4, pre-registered): cohort P (pool-missing NP-table compounds, pool members excluded from each window), plus the `member@μ` and `stack@μ` variants.
