# R5 — Spectral-analogue channel (PSSM + Tanimoto kernel): plan, build, status

**Read this first if you are picking up R5.** It records:
- what was decided and why (grilling, 2026-10-08);
- what was built and where it lives;
- what has run so far;
- exactly what is left.

Related docs:
- The original proposal is `pssm.md`. Its §0 summarises the settled design, and §0 wins wherever the proposal text disagrees.
- The pre-registered gates are in `gate_report.md`, section "R5". The roadmap row is S7 in `roadmap.md`.

---

## 1. Context: why this lever

- **Base:** `casmi_scratch_r3_full1_r2gl` is the best scratch run, **LB 0.346**. It is two-tier:
  - Tier 1 is the real-library match, lv ≥ 0.75, sorted by lv.
  - Tier 2 is ranked by RRF (k = 3) of two lists: GLACIER spectral similarity sv (plus the R2 class-vote bonus) and the FPNet full1 score f·z.
- **R4 failed:** adding a second FPNet family scored 0.345, which is neutral. A second model of the same spectrum→fingerprint task is a near-copy. New evidence must come from **different data**.
- **The idea:** library spectra within ±200 Da that match the query directly or with a mass shift are *analogues*. Their known structures share substructures with the unknown. Candidates that look like the analogues should rank higher. No current tier-2 list reads this.
  - Before R5, analogue hits fed only the coarse R2 superclass vote, worth about +0.005, which is within noise.
  - The old-lineage AFIX ("Analog Propagation", N_ANALOG = 200) was a ranker feature with 15% of GBDT gain, but it was never tested as an RRF list here.
- **The lesson that applies** (memory "fusion beats tuning"): only adding orthogonal sources fused by RRF has ever paid off. Tuning has not.

## 2. Settled decisions and their reasons

| # | Decision | Why |
|---|---|---|
| D1 | **Score-only.** The list re-ranks the existing R1b pool and injects no analogue structures. | Pool injection via mutator or scaffold was a measured dead end (0.000 / −0.002), and score-only keeps one variable per probe. |
| D2 | **Open ±200 Da search** with `shift_entropy_sim`. A peak matches directly or shifted by the precursor difference. | This is the user's design, and it catches arbitrary modifications. The "delta" list mode (±10 ppm + 13 NP deltas) is faster but only sees known deltas. |
| D3 | **Analogue library = full train.parquet, enveda-180 included.** | User's call: enveda-180 (synthetic, drug-like, timsTOF, ~66% of library structures) still shares substructures with NPs, which is signal. |
| D4 | **Train structures are excluded from the analogue list only**, using pool flag bit `F_TRAIN`. They get a **neutral term** in that list. The pool, tier 1, sv and f·z lists still see them. | Synthetic analogues give drug-only bits a high weight, which could push drug-like decoys up. After this exclusion the list can only promote NP-source structures (COCONUT, BIO, LIPID MAPS, NP table). It also stops a train candidate from retrieving its own library spectrum (T = 1) and echoing lv. Excluding train structures from the pool globally would delete every Class-1 answer and break tier 1. |
| D5 | **Neutral term** = the mean RRF term of the scored candidates. | If excluded candidates simply ranked last, every train structure in tier 2 would lose a whole RRF term. That is a hidden "demote train" prior, it hurts truths that missed tier 1 (lv < 0.75), and it confounds the LB delta. |
| D6 | **Two scorers built, both sent to the LB:** PSSM and Tanimoto kernel. | The PSSM collapses the analogues into one profile and loses multiple families. The kernel keeps them. Which one wins is unknown. |
| D7 | **PSSM math (fixes the proposal).** w = S⁴. p̃ = (Σw·x + α·q)/(Σw + α), α = 1. W = clip(logit p̃ − logit q, ±3). Score s = C·W. | The proposal's `log((p+ε)/(q+ε))` with ε = 1e-4 over present bits gives about −9 for every bit that no analogue carries. The true molecule usually carries such bits (its modification), so it would be buried. The pseudocount prior plus the full Bernoulli log-odds bound that penalty. The result has the same form as f·z. |
| D8 | **PSSM fingerprint = 16,520 raw Family-A bits.** The layout is Morgan r2 4096, Morgan r3 4096, FeatMorgan r2 2048, RDKitFP 2048, AtomPair 2048, TopologicalTorsion 2048, MACCS 167. The 10,226-bit subset is *not* used. | User's call: the 10,226 subset was selected on the older v2 pool, and R1 is larger. |
| D9 | **Tanimoto kernel.** s = Σ S⁴·T(c, aᵢ) / Σ S⁴ over the top 20, with T computed on the **Morgan r2 4096 slice** (the first 4096 raw bits). | Standard structural similarity. In the full concatenation, the dense RDKitFP and AtomPair blocks would dominate T. |
| D10 | **Background q** = bit frequency over the non-train R1b pool, i.e. the structures the list scores. It is a seeded runtime sample of 50k rows, or an attached `analog_q.npy` if present. | A runtime sample needs no new Kaggle dataset upload. The audit notebook writes `analog_q.npy` in case it is wanted. |
| D11 | **Clip W to ±3.** | COCONUT still holds halogenated and aminated marine NPs that share bits with drugs, so no single rare bit may decide a ranking. |
| D12 | **Multi-spectrum:** search each spectrum within its own polarity, union the hits, keep the max S per analogue structure (ik14), and deduplicate by structure. | The metric scores molecules, so evidence is pooled per molecule. Deduplication stops a heavily sampled structure (enveda-180 has many spectra per structure) from dominating. |
| D13 | **S ≥ 0.5, PSSM top M = 100, kernel top K = 20.** Fewer than 3 analogues with a parsable structure means **no list** for that molecule. | Defaults. A near-empty profile would only add rank noise. |
| D14 | **Fusion:** tier-2 key `1/(3+r_sv+bonus) + 1/(3+r_fz) + 1.0/(3+r_an)` in stages A and B. Tier 1 is untouched, stage C is unchanged (off), and the weight is **1.0**, never tuned. | User's call (w = 1.0 over the proposal's 0.4–0.8). Tuning on slice I is unreliable because it is optimistic. |
| D15 | **Class vote:** on in `r5_pssm` and `r5_tani`. **Twins** `r5_pssm_novote` and `r5_tani_novote` are identical except `class_vote=False`. | User request: the twins show whether the analogue list makes the R2 vote redundant. |
| D16 | **No runtime guard or fallback** for the open search. | User's call: public AFIX notebooks run the same kind of search, and the fast commit showed about 3 h total (§6). |
| D17 | **LB decision:** no same-day control, **KEEP iff LB ≥ 0.354** (0.346 + 0.008, the no-control rule). If several pass, keep the highest. Compare vote on vs off within a scorer at the ±0.004 noise floor. | LB run-to-run noise is about ±0.004 (0.411 vs 0.415 for identical code). |
| D18 | **Offline audit runs alongside the LB and is informational.** It uses slice I with the own structure banned, sv = 0, and **the held-out truth counted as non-train**, while every other train structure stays excluded. | Every slice I truth is a train structure. Without that exemption the channel could never score the truth, so slice I would measure nothing. The exemption reproduces the hidden test: an NP truth among train-structure decoys. Slice I is analogue-dense and full1 saw slice I, so the audit explains the LB result and does not decide it. |

## 3. What was built (all in `scratch_pkg/`)

### Package `casmi_scratch/`

- **`config.py`**: new R5 knobs, all off by default:
  - `analog_channel` ("" | "pssm" | "tani");
  - `analog_open_da` 200, `analog_smin` 0.5, `analog_m` 100, `analog_k` 20;
  - `analog_alpha` 1.0, `analog_clip` 3.0, `analog_w` 1.0;
  - `analog_min_n` 3, `analog_q_n` 50,000.
- **`analog.py`**, section "R5 analogue channel":
  - `molecule_analogs(lib, specs, pols, target, cfg, exclude)`: open-window hits per polarity, max S per key.
  - `analog_set(hits, s_min, m)`: top m with S ≥ s_min.
  - `pssm_weights(X, S, q, alpha, clip)`: the D7 math.
  - `tani_kernel(C, X, S)`: the D9 kernel.
  - `train_smiles(train_path, keymap)`: library structure key (mapped to the pool's scorer_key) → `normalized_smiles`.
  - `estimate_q(pool, feat, n, seed, mask)`: background q from a seeded pool sample.
  - `AnalogBank(smiles_of, q, cfg)`:
    - it wraps a raw Family-A `fpnet.Featuriser` and keeps a packed per-key cache;
    - `.profile(hits, mode)` returns `{mode, n, W}` for "pssm" or `{mode, n, X, S}` for "tani", or None below `analog_min_n`;
    - `.score(prof, keys, smiles)` returns one score per candidate, NaN where there is no fingerprint.
- **`rank.py`**:
  - `analog_rrf(score, k, w)`: RRF term over the scored candidates; unscored (NaN) candidates get the mean term.
  - `rank_two_tier(..., an=None, an_w=1.0)` adds that term to the tier-2 key. It works with or without the FP list. With `an=None` the output is identical to r3.
- **`pipeline.py`**:
  - `plan_molecule` runs the open search when the channel is on, then stores `an_hits` (the top 4·max(M, K)) and `an_s`.
  - `run(..., analog_bank=None, analog_free=None)` builds `plans[mid]["an_prof"]` in pass 1. `analog_free` is {mid: keys} exempt from the train exclusion; validation uses it for the truth. It also accumulates `times["analog_s"]`.
  - `analog_scores(plan, idx, pool, cfg, bank)` returns candidate scores, with NaN for `F_TRAIN` structures not in `an_free`. It is wired into stage A and stage B.
  - Diagnostics per molecule: `an_ok`, `an_scored`, `an_reordered`.
  - `analog_counters` returns `n_analog_mol`, `n_analog_dropped`, `n_analog_ok`, `n_analog_reorders` and `analog_n_median`, all in `run_report.json`.
- **`audit.py`**:
  - `analog_discrimination(mols, lib, pool, truth_of, fp_bank, an_bank, cfg, vote_label)` gives the truth's rank under 12 arms: `fp{0,1}_{none,pssm,tani}_v{0,1}`. It also records analogue-only ranks, analogue counts and search time.
  - `summarize_analog(rows)` returns MRR@25 per arm, up/down against the base `fp1_none_v1`, coverage, analogue-only MRR, and a verdict per scorer: Δ ≥ +0.010 and up > down.

### Drivers and notebooks

- **`drivers/r0_kaggle.py`**:
  - An R5 setup block builds the `AnalogBank` from `train_smiles`. For "pssm" it loads q from `analog_q.npy` if attached, otherwise from `estimate_q` over the non-train pool.
  - It prints `[analog] <mode> | {...}`, passes `analog_bank` into `pipeline.run`, adds `analog_info` to the report, and prints the R5.0 check lines.
  - In fast mode it asserts `n_analog_mol > 0`.
- **`drivers/r5_audit_colab.py`**: the Colab audit. It needs on Drive `train.parquet`, `casmi26-scratch-pool` v2, `casmi26-fpnet-full1` and `casmi26-v2-pool`. Results are cached in `MyDrive/casmi_2026/_r5_audit/`, and the slice I rows go to `slice_i.json`.
- **`tools/mk_notebook.py`**, `NOTEBOOKS`: four R5 probes plus the audit, each differing from `r3_full1_r2gl` only in its override line and `RUN_TAG`:
  - `casmi_scratch_r5_pssm.ipynb`, `casmi_scratch_r5_tani.ipynb`
  - `casmi_scratch_r5_pssm_novote.ipynb`, `casmi_scratch_r5_tani_novote.ipynb`
  - `casmi_scratch_r5_audit_colab.ipynb`
  - Regenerate with `python scratch_pkg/tools/mk_notebook.py --all`. Never hand-edit the `.ipynb` files.

### Tests

- **`tests/test_r5.py`**, 10 tests:
  - PSSM weights: a conserved bit is rewarded, an absent bit gets a bounded penalty, the clip holds, and weak analogues shrink towards the prior.
  - `analog_set`, and the neutral term of `analog_rrf`.
  - `an=None` gives the r3 order, and tier 1 is untouched.
  - The profile is dropped below 3 analogues.
  - The kernel matches RDKit `BulkTanimotoSimilarity`.
  - Train exclusion applies only in the analogue list, and the `an_free` exemption works.
  - `estimate_q`.
  - An audit row on a toy library.
  - A **notebook dry run for pssm and tani** on a fake R1 pool built from the local `test.parquet`.
- **Full suite:** `python scratch_pkg/tests/run.py` → **OK** (R1–R5 and the notebook-current check, 2026-10-08).

### Docs

- `pssm.md` §0 (settled design), `gate_report.md` R5 (gates, written before any run), and the S7 row in `roadmap.md`.

## 4. Gates (copied from `gate_report.md` R5)

| Gate | Threshold |
|---|---|
| R5.0 | Fast commit, per probe: the analogue list is built for ≥ 50% of deep molecules, `n_analog_ok` > 0 and `n_analog_reorders` > 0; `analog_s` and `total_s` recorded. Full run ≤ 7.5 h. |
| R5.audit | Slice I, informational. Arm `fp1_<mode>_v1` vs `fp1_none_v1`: ΔMRR@25 ≥ +0.010 and up > down. |
| R5.LB | Per probe, no control: KEEP iff LB ≥ **0.354**; if several pass, keep the highest. |

## 5. How to run

1. **Kaggle probes.** Attach the same inputs as `r3_full1_r2gl`:
   - the competition data, casmi26-iceberg, rdkit-wheels, casmi26-glacier;
   - casmi26-scratch-pool v2;
   - casmi26-fpnet-full1 and casmi26-v2-pool.

   R5 needs **no new dataset**. Use GPU T4, Internet off. Commit (the fast commit covers 20 molecules), check that the `R5.0` lines print PASS, then submit (the hidden rerun scores all molecules).
2. **Colab audit.** Open `casmi_scratch_r5_audit_colab.ipynb` on a GPU T4 with High-RAM and run all cells. It takes roughly 1.5–2 h, mostly the open search over about 245 slice I molecules. Paste the printout back.
3. **Record results** with the `/update-experiments` skill (EXPERIMENTS.md), and write the verdicts into `gate_report.md` R5.

## 6. Status (as of 2026-10-09)

| Item | Status |
|---|---|
| Code, tests, notebooks, docs | done 10-08, tests OK. **Not committed to git yet.** |
| R5.0 `r5_tani` fast commit | **PASS** (Kaggle 10-08). `n_analog_mol` 20/20, dropped 0, `n_analog_ok` 19, `n_analog_reorders` 19, `analog_n_median` 20. `analog_s` 441 s for 20 molecules (~22 s each). `total_s` 622. FP parity 50/50, self-rank 5/5. sim_ok 1.0. |
| R5.0 `r5_pssm` fast commit | **PASS** (Kaggle 10-08). `q_source` sample, `q_mean` 0.069, q setup 214 s. `n_analog_mol` 20/20, dropped 0, ok 19, reorders 19, `analog_n_median` 100. `analog_s` 394 s for 20 (~20 s each). `total_s` 771. |
| R5.0 `_novote` twins | fast commits not yet reported. Same code with `class_vote=False`; expect identical analogue counters. |
| R5.LB (all 4) | **DROP ×4** (10-09): pssm 0.323, tani 0.322, pssm_novote 0.320, tani_novote 0.317 vs base 0.346. See §9. |
| R5.audit (Colab) | **not run** |
| R5 verdicts in `gate_report.md` / EXPERIMENTS.md | not written yet (R5.0 PASS above not yet recorded there) |

Notes on the fast-commit logs:
- The `CHECK tier-1 rate` line and `tier1_rate` 0.0 are expected. The public test file is enveda-180 rows, which R1b removed from the pool. R3 and R4 showed the same.
- **Projected hidden-run time:** about 400 molecules × 20–28 s gives a 2.2–3 h analogue search, plus about 4 min for q, about 10 min of simulation and about 10 min of setup and pass 2. That is **about 3 h**, well under the 7.5 h budget and the 9 h limit. The search runs in pass 1, before simulation, and only uses up the simulation's slack (about 13×).

## 7. What is left, and how to read results

1. Run the `_novote` fast commits and check R5.0.
2. Submit all four and record the LB scores. Then decide:
   - Any probe ≥ 0.354 → KEEP, and the highest becomes the new scratch base.
   - None ≥ 0.354 → record R5 as negative, and `r3_full1_r2gl` stays the base.
   - Read pssm vs tani, and vote vs novote, at the ±0.004 noise floor. A gap smaller than that is no difference.
3. Run the Colab audit. Use it to *explain* the LB result: which scorer moves truths up, the analogue-only MRR, coverage, and whether the vote and the analogue list overlap. It does not override the LB.
4. Commit the R5 work. It is on branch `specs-scratch-r0`, together with uncommitted earlier work. Another agent edits this repo too, so stage only R5 files:
   - `scratch_pkg/casmi_scratch/{analog,rank,pipeline,config,audit}.py`
   - `scratch_pkg/drivers/{r0_kaggle,r5_audit_colab}.py`, `scratch_pkg/tools/mk_notebook.py`, `scratch_pkg/tests/test_r5.py`
   - the regenerated notebooks
   - `specs/specs_scratch/{pssm,r5_pssm_tani,gate_report,roadmap}.md`

## 8. Gotchas for whoever continues

- **Never read raw `.ipynb` JSON.** Edit the drivers and package, then regenerate with `mk_notebook.py --all`. `test_notebook` fails if any notebook is stale.
- **Keys:** the library is remapped to the pool's `scorer_key` (`remap_keys` with `train_keymap.parquet`), and `train_smiles` uses the same keymap. Keep the two in one key space.
- **`F_TRAIN`** is pool provenance bit 1 (`pool.py`). The R0 runtime pool has `flags=None`, so no exclusion happens there.
- The analogue search stores only the top 4·max(M, K) hits per molecule. Storing the full ±200 Da hit dicts would blow memory over 400 molecules.
- `rank_two_tier` with `an` all-NaN equals the order without `an` (tested). `analog_rrf` returns zeros when nothing is scored.
- Any later tuning of M, K, α, the clip or w is "tuning", which this project has repeatedly found not to pay (memory: fusion beats tuning). Prefer a new orthogonal source over sweeping R5 knobs.
- If R5 passes, a natural next probe is replacing the R2 class vote with R5 outright. The `_novote` twins already measure that.

## 9. R5b: the train exclusion as a switch (2026-10-09)

**Diagnosis.** No code bug was found: keys, flags, q and tier-2 indexing all check out. pssm ≈ tani, so the loss comes
from something the two scorers share. The main suspect is D4/D5:
- `F_TRAIN` marks every train.parquet structure, the GNPS/MoNA NPs included. So a Class-1 truth whose lv < 0.75
  (instrument, CE or adduct mismatch) sits in tier 2 with only the mean RRF term (about analogue rank 50–300 at k = 3).
  The top scored candidates get 0.25, 0.20, 0.17 and so on.
- The open search still finds the truth's own library spectrum (Δ ≈ 0, highest S), which goes into the profile. The list
  therefore promotes the non-train structures most like the truth, which are its closest decoys.
- The audit exemption (`an_free` = the truth) hides this, so slice I could not have shown it.
- Secondary suspect: weight 1.0 at k = 3 for a list whose own quality was never measured.

**Switch.** `config.analog_exclude_train` (default **True** = R5 as submitted). False makes `pipeline.analog_scores`
score every candidate, and the driver samples the PSSM q over the whole pool instead of the non-train rows. An attached
`analog_q.npy` would still override that sample, and none is attached. `analog_info.exclude_train` goes into the log and
`run_report.json`.

**Notebooks.** `mk_notebook.py` now writes 8 R5 probes. The 4 R5 ones set `analog_exclude_train=True` explicitly and
behave as before. The 4 new `_inclt` twins (`r5_pssm_inclt`, `r5_tani_inclt`, `r5_pssm_novote_inclt`,
`r5_tani_novote_inclt`) differ from their R5 twin only in that override and `RUN_TAG`. Gates are in `gate_report.md`,
R5b. Later probes: `--set analog_exclude_train=False` combined with e.g. `analog_w=0.3`.

