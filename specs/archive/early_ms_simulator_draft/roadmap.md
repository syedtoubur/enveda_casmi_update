# Phased Implementation Roadmap & Verification Milestones

*Revision 2 — 2026-09-25. Every phase is developed and validated in **Google Colab**
(L4 + High-RAM reference runtime) before graduating to the Kaggle submission notebook.*

```
   +-------------------------------------------------------------+
   | Phase 0: Close the Stage D HOLD                             |
   | (diag -> drop cell C -> Stage D pass 3. ~1 h, no new code)  |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 1: Base Consolidation & Generator Hygiene             |
   | (single base, schema widened to 38, tests green)            |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 2: Scaffold Precomputation & Consensus Pipeline       |
   | (COCONUT scaffold delta + 3 consensus features)             |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 3: Neural Simulator - licence, leakage & latency gate |
   | (ICEBERG 2.1 MassSpecGym; GO/NO-GO before any long run)     |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 4: Stage C v3 - leak-free 3-cell matrix, 38 columns   |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 5: Stage D 4-way validation (+ PPM arm)               |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   | Phase 6: Final Kaggle Offline Submission Notebook           |
   +-------------------------------------------------------------+
```

---

## Phase 0: Close the Stage D HOLD

**Why first.** Stage D currently scores `both` at −0.044 on C2 and −0.040 on C2_hard against
paired SEs of 0.013 and 0.018. Adding two channels on top of an unresolved regression makes the
next A/B unreadable — you would not know whether a delta came from the new channels or from the
thing that was already wrong.

**Actions**
1. Run `stage_d_diag_cell.py` in a live Colab kernel.
   - **Part A** splits every delta by `ce_known`. C2 is 77% CE-null and C2_hard 85%, while the
     test set is 100% CE-known; those molecules score through FPNet's 25.0 eV fallback.
   - **Part B** measures each FPNet's in-window truth top-1 per cohort against the new model's
     known honest rate (74.2% vs 48 decoys; 52.1% in a ~70-candidate window on unseen structures).
2. Drop `(regime == seen) & (M == 1)` rows — cell C — from `rank_raw.npz` and refit.
3. Re-run **Stage D pass 3 only** (~25 min per matrix variant; pass 1 is checkpointed on Drive
   and pass 2 takes minutes).

**Prerequisite**: `rank_raw.npz` must still exist on Drive. If it does not, Phase 0 folds into
Phase 4 and the cell-C drop happens at build time instead.

**Exit criterion**: either the C2 regression is explained (CE-null composition + val-banned
training asymmetry) or it is shown to be real. Both outcomes are acceptable and both unblock
Phase 1 — what is not acceptable is proceeding without knowing.

### Result of step 1 (run 2026-09-25)

**The C2 regression is an artefact.** Of the two suspected confounds, one is dead and one is
confirmed.

- **Collision energy — FALSIFIED.** CE-known C2/C2_hard and CE-known C1_xlib draw on the same
  libraries with the same NCE-heavy units mix and the same ~30 eV median, yet the new FPNet
  scores 36-38% in-window top-1 on the first and 79.8% on the second. Identical inputs cannot
  produce a 43-point gap.
- **Training-exclusion asymmetry — CONFIRMED.** The new FPNet's exclusion set is the union of
  the val `ban_keys`. C1_npx and C1_xlib have **empty** `ban_keys`, so their truths were never
  excluded and the new model scores 80.4% / 83.4% on them. C2 and C2_hard truths *were* denied
  it: 59.9% / 60.0%, against its independently measured honest rate of **52.1%** in
  ~70-candidate windows. The new model is behaving exactly as an honest model should. The
  forked model, which excluded nothing, scores 75.5% / 73.3% there against its own 42.0%
  reference.

**Consequences carried forward:**
1. `base` holds a memorisation advantage on C2/C2_hard that does not exist on the leaderboard.
   Its C2 numbers are inflated, so the −0.044 is not a verdict on the new stack.
2. **Both C1 cohorts are memorised by both models** (empty `ban_keys`), so `val_scripts`
   overstates every arm on Class 1 as well. New entry alongside the pool-miss blindness.
3. Stage D as composed **cannot** deliver a clean Class-2 verdict. This is a cohort-design
   limit, not a compute limit — more runs will not fix it. Judge a new FPNet against its own
   honest in-window rate, not against `base`.

Steps 2-3 remain outstanding. Paste-in cells: `stage_c_dropC_cell.py` (Stage-C kernel) and
`stage_d_pass3_noC_cell.py` (Stage-D kernel).

---

## Phase 1: Base Consolidation & Generator Hygiene

**Primary objective**: one base, one schema, one set of green tests — before any new feature
code is written.

**Actions**
1. Confirm `enveda-casmi-2026-phase1-enhanced-baseline.ipynb` is the sole `SRC` in all four
   generators (it already is) and record `CFG.POLARITY_ANALOG = True` as a deliberate decision
   in the notebook header, citing `plan.md` §5.1.
2. Fold `patch_colab_rdkit.py`'s RDKit-bootstrap replacement into `build/mk_stage_c_colab.py`,
   then delete the patch script. Notebooks are generated, never hand-edited.
3. Widen `rank_features` from 31 to 38 columns, **append-only**, with the new columns emitting
   zeros when their inputs are absent — exactly how `mfeat` / `ffeat` already degrade. This keeps
   the function callable before Phases 2 and 3 land.
4. Update `merge_shards.py`'s hardcoded `assert X.shape[1] == 31` to accept 31 or 38 and record
   which it saw.
5. Extend the gate's `CH` map with the `analog` additions (31-33) and the new `sim` channel
   (34-37), keeping a 31-column compatibility path so `build/test_gate.py` still validates
   against the 0.287 and 0.337 matrices.
6. Regenerate all notebooks: `python build/mk_stage_c_colab.py` (and the three siblings).

**Verification**: `sh build/run_all_tests.sh` green, including `test_gate.py` still returning
FAIL for the v1 matrix and PASS for the forked one.

---

## Phase 2: Scaffold Precomputation & Consensus Feature Pipeline

**Primary objective**: extract Bemis-Murcko scaffolds for the candidate pool and inject soft
consensus features into Channel 2. No learned weights, so no new leakage surface.

**Actions**
1. **Reuse, do not rebuild.** `val_scripts/scaffolds.pkl` is already a Murcko scaffold dict
   keyed by `inchikey14` with **261,890 entries** covering `train.parquet`. Only the COCONUT
   delta (~436k structures) needs computing. Extend that dict; do not start from scratch.
2. Save to Drive as `_scaffolds/pool_scaffolds.pkl`.
   **Invariant**: keyed by `inchikey14`, never by pool row index — the pool is rebuilt at
   inference (`logs.txt`: 712,199 structures in 264 s) and row order is not guaranteed stable.
3. In `analog_sim()`, map the top-100 retrieved analogs to their Murcko scaffolds.
4. Emit 3 features per candidate (columns **31-33**):
   - `scaffold_vote` — weighted scaffold frequency among the top-100 analogs,
     `F(S) = sum(w_a^4 for a in S) / (sum(w_a^4) + 1e-9)`
   - `is_top_scaffold` — 1.0/0.0 if the candidate shares the most frequent analog scaffold
   - `scaffold_tanimoto` — topological Tanimoto between the candidate scaffold and the top
     analog scaffold

**Validation check**: pre-flight on 50 sample queries confirming added latency < 5 ms/query
(dict lookups over an existing top-100 list) and a non-trivial spread in `scaffold_vote` — a
column that is constant carries nothing.

---

## Phase 3: Neural Simulator — licence, leakage and latency gate

**Primary objective**: add Channel 3b **only if** it passes three independent gates. Each gate is
cheap and runs before the expensive work. This phase is explicitly allowed to end in NO-GO.

### Gate 1 — Licence (blocking, minutes)

- Use `coleygroup/ms-pred` (**MIT**) with a **MassSpecGym-trained** checkpoint: ICEBERG 2.1
  (`msg_all` or `msg_simulation`) or GLACIER. MassSpecGym is MIT-licensed and publicly hosted.
- **Never** a NIST'20/23/26 checkpoint — those require emailed proof of a commercial licence and
  fail Kaggle's "freely & publicly available" rule.
- Record the checkpoint SHA1 and assert it at load, mirroring the existing `fp_bits.npy` guard.
- Confirm the checkpoint covers **negative ionization mode** and eV collision energy. The test
  set uses 10 adducts, 4 of them negative. If the checkpoint is positive-only, the channel can
  still ship but must emit zeros in negative mode, and that must be stated, not discovered.

### Gate 2 — Leakage (blocking, ~30 min)

1. Pull MassSpecGym from HuggingFace in Colab; compute InChIKey14 for every **train-fold**
   structure → `msg_train_keys.npz`.
2. Report `|es_keys \ msg_train_keys|` — the structures unseen by **both** models.
3. **GO/NO-GO**: if fewer than ~2,000 structures survive, there is no honest Stage-C cohort for
   this channel. Abandon the pretrained route and train an in-house simulator on
   `train.parquet` under the FPNet split instead (the Option-1 fallback, ~6-12 h GPU).
4. If GO, `msg_train_keys.npz` becomes a first-class artifact: Stage C filters cell D/B queries
   through it, and Stage D reports every delta split by membership in it.

### Gate 3 — Latency (blocking, ~15 min)

Measure, do not assume. Benchmark on the real distribution: a query window is median ~70
candidates, p90 217, max 676 — not a flat 50.

- Report ms/query at the median, p90 and max window sizes, separating RDKit featurisation from
  GPU forward pass, with and without a per-run SMILES→graph cache.
- **Budget**: Stage C must stay under ~5 h wall clock and Kaggle inference under 9 h for 400
  molecules. Extrapolate from the measured number and state the projection before committing.

### Then, and only then

Implement `simulate_virtual_spectra(cand_smiles, ce_ev, adduct)` and
`sim_virtual_scores(query_spectrum, virtual_spectra)` (reusing the existing fast numba entropy
matcher), emitting columns **34-37**: `sim_entropy`, `sim_rank_norm`, `sim_z`, `sim_is_best`.

**MetFrag-lite is retained.** Its retirement is decided by the Phase-4 gate's channel-share
readout, not in advance — measured cost is 5.5% of Stage C (~0.4 s of a 7.22 s query) and it
carried 1.7-2.3% of ranker skill.

---

## Phase 4: Stage C v3 — leak-free 3-cell matrix, 38 columns

**Primary objective**: a GBDT training matrix that teaches balance across five evidence sources.

**Actions**
1. Load `fpnet_split_keys.npz` and (if Phase 3 went GO) `msg_train_keys.npz`.
2. Sample queries into **three** cells — A (seen/visible), D (unseen/visible), B (unseen/banned).
   **Cell C is not built.** "Unseen" means unseen by *both* neural models.
3. Run the paired dual-regime extraction (expensive channels once, library+analog twice).
4. Emit `X` (38 columns), `Y`, `M`, `G`, plus the regime mask needed by the gate.

**VERIFY 1 (pre-flight, before the long loop)** must print:
- per-library self-retrievability at both 8.5 and 5.5 ppm, so the riken cohort cost of the
  narrower window is seen before it is paid
- `|es_keys \ msg_train_keys|`
- a 6-query smoke test with a per-channel timing breakdown and a **measured ETA**

**VERIFY 2 (the gate)** — channel destruction on held-out queries. Target shares:

| channel | target share |
|---|---|
| library | 10-30% |
| analog + scaffold consensus | 15-25% |
| neural simulator | 15-25% |
| FPNet | 25-40% |
| MetFrag | whatever remains (informs retirement) |

Retained hard rules from v2, which test the leak mechanism rather than a proxy for it:
- model channel > 50% → FAIL
- library or analog < 5% → FAIL
- FPNet top-1 on > 75% of model-unseen queries → FAIL
- seen − unseen top-1 gap < 0.10 → FAIL
- MRR > 0.88 on the model-unseen + library-banned cell → FAIL

**New rule**: `sim_is_best` accuracy on the model-unseen + library-banned cell must not exceed
the simulator's published retrieval accuracy on MassSpecGym's *test* fold. Above that, the
holdout is not a holdout.

The matrix is **not written** if the gate fails.

---

## Phase 5: Stage D validation

**Primary objective**: prove statistically that the new pipeline beats 0.337 before submitting.

**Cost correction**: pass 1 caches `cand / lv / fsc / ids / sims / lib_hits`. The new channels
change what pass 1 must produce, so the Drive checkpoint is invalid — this is a **full pass-1
re-run, ~2-2.5 h**, not the ~25 min that applies to Phase 0's cell-C drop.

**Actions**
1. Run `casmi_validate_ab_colab.ipynb` over the 1,145 validation molecules and four cohorts:
   `C1_npx` (245), `C1_xlib` (300), `C2` (400), `C2_hard` (200).
2. Report every delta **split by `msg_train_keys` membership** (per plan.md §3.1) and by
   `ce_known` (the Phase-0 diagnostic's finding carried forward).
3. Optional 5th arm: `PPM_WIN = 5.5` at inference against the same matrix. This is a cheap first
   read that mismatches only column 14 (`log(n_candidates)`); if it wins, rebuild the matrix at
   5.5 and re-measure properly.

**Pass / Fail**
```
IF delta(C2) >= +0.020 AND delta(C1_npx) >= -0.005
   AND the C2 gain survives restriction to structures OUTSIDE msg_train_keys:
    PASS -> proceed to Kaggle packaging
ELSE:
    HOLD -> diagnose channel interference; first lever is CFG.POLARITY_ANALOG = False
```

**Read Stage D as relative deltas only.** Every val truth comes from `train.parquet` and is
in-pool by construction, so val MRR always reads above the leaderboard; the gap is the pool miss
rate.

---

## Phase 6: Final Kaggle Offline Submission Notebook

**Primary objective**: package code, checkpoints and offline wheels into a production notebook.

**Actions**
1. Package artifacts:
   - the permitted simulator checkpoint (licence + SHA1 recorded)
   - `rank_train_v3.npz` — the 38-column gate-passed matrix
   - `pool_scaffolds.pkl` — Murcko scaffolds keyed by `inchikey14`
   - the FPNet checkpoints the matrix was built from — **never ship a matrix without them**
2. Verify the offline RDKit wheel loads cleanly (`casmi26-offline-rdkit-2026033`).
3. Fast-commit verification on Kaggle: commit ≤ 7 min; `submission.csv` has exactly one row per
   `molecule_id` and ≤ 25 semicolon-separated SMILES each.
4. Confirm the private-rerun path switches to the deep 4-channel engine, and that the projected
   full-run time from Phase 3 Gate 3 fits inside 9 h.
5. Submit and record the leaderboard MRR.
