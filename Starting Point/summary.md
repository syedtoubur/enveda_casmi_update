# CASMI 2026 — every approach so far, what worked, what did not

Consolidated record for the Enveda CASMI 2026 competition (predict a 2D structure from MS/MS;
submit 25 SMILES per molecule; scored **MRR@25 on InChIKey14**). Written so that an agent with no
prior context can learn what was tried, what was measured and which conclusions are safe.

Two approaches have been worked on:

- **Approach A — the FPNet + GBDT stack** (§4). Four evidence channels fused by a gradient-boosted
  ranker. Retraining FPNet and the ranker ourselves. Specs: `RUNBOOK.md`, `specs_ms_simulator/`.
- **Approach B — the simple ICEBERG baseline** (§5). No learned ranker: real-library matches
  first, then a pretrained forward simulator (ICEBERG) scores the rest. Specs:
  `specs_simple_ms_simulator/`.

Every number below is **measured**, with its source. Anything not measured says so.

---

## 0. TL;DR

| question | answer | evidence |
|---|---|---|
| Best leaderboard score so far | **0.337**, `enveda-casmi-2026-tier1-patched.ipynb` (a forked public notebook). Nothing built since has beaten it. | §2 |
| Does a forward MS simulator help? | **Yes.** ICEBERG adds **+0.035 LB** in a clean one-flag ablation (0.254 → 0.289). | §5.6 |
| Does retraining FPNet + ranker ourselves help? | Slightly on the phase1 base: 0.319 → 0.323. The retrained ranker **must** be built on FPNet-unseen queries; the first attempt leaked and scored 0.286. | §4.2, §4.6 |
| Is `val_scripts` a reliable guide? | **No, for absolute numbers and for Class 2.** It cannot see pool misses and both of its Class-1 cohorts are memorised by the models. Stage D said HOLD on a stack that then scored *higher* on the LB. | §4.6, §6 |
| Largest unexplained gap | tier1-patched (0.337) vs phase1-enhanced (0.319): **−0.018**, never attributed. All later work was built on phase1-enhanced. | §2, §7 |

---

## 1. The task, and facts that constrain every design

- **Scoring:** MRR@25 on the first 14 characters of the InChIKey, after RDKit 2026.03.3
  canonicalisation. Duplicate ik14s in one row waste slots, so dedupe before emitting.
- **Runtime:** Kaggle notebook, internet **off**, ≤ 9 h, GPU T4 ×2 (4 vCPU). Only freely and
  publicly available external data and models are allowed.
- **Visible `test.parquet`** (what a commit sees): 400 molecules, 1,213 spectra; 24.6 % of spectra
  are merged multi-CE spectra. The LB is computed on a **hidden re-run** of the notebook
  (`KAGGLE_IS_COMPETITION_RERUN` set) whose logs we never see.
- **Test spectra are 100 % Bruker timsTOF** with a systematic **+1.45 ppm** precursor offset
  (`CFG.PPM_CENTER`). CE is always known, in eV.
- **Adducts in the visible test** (spectra): `[M+H]+` 959, `[M-H]-` 193, `[M+CH2O2-H]-` 31,
  `[M+Na]+` 22, `[M+NH4]+` 4, `[M+Cl]-` 2, `[M+K]+` 2 (source: Colab grid run,
  `specs_simple_ms_simulator/colab_grid_partial_results.txt`).
- **Molecule classes:** Class 1 = the structure has public reference spectra; Class 2 = the
  structure is in PubChem/COCONUT but has no public spectra; Class 3 = novel structure.
- **Candidate pool:** COCONUT 2.0 ∪ train structures = 712,199 rows, 711,690 unique ik14. Any
  molecule outside it (all of Class 3, and Class 2 molecules missing from COCONUT) scores 0 whatever
  the ranking. The pool has never been expanded.
- **Reference library:** `train.parquet`, 2,539,608 spectra / 275,810 structures.

---

## 2. Leaderboard history

Source: `specs_simple_ms_simulator/scores_performed.txt`, the user's notes, and past session
records. Dates are approximate (from "n days ago").

| date | submission | what it is | LB |
|---|---|---|---|
| 09-21 | `casmi_submission_notebook` v2 | `enveda-casmi-2026-tier1-patched.ipynb`, the forked public notebook with forked FPNet + forked ranker matrix | **0.337** |
| 09-21 | v3 | **run-mode bug**: only the first 30 of 400 molecules were deep-scored (§6, lesson 7) | 0.032 |
| 09-21 | v5 | `enveda-casmi-2026-phase1-enhanced-baseline.ipynb` (polarity-aware analog channel + PPM_EXPAND / backfill + fast-commit wrapper), forked FPNet + forked matrix | 0.319 |
| 09-22 | v6 | **not documented** in the workspace | 0.309 |
| 09-23 | `new_trained_fp_ranker` | new FPNet + **Stage C v1 (leaked)** ranker matrix (§4.2). Recorded elsewhere as 0.287 | 0.286 |
| 09-23 | v8 | per project memory: polarity-aware representatives at inference against the forked matrix, which was not built with them, so the result is confounded. **Not fully documented** | 0.310 |
| 09-25 | v9, v10 | new FPNet + **Stage C v2** ranker matrix (gate-passed), phase1 base. Submitted despite Stage D's HOLD | **0.323** |
| 09-25 | v11 | **not documented** in the workspace | 0.314 |
| 09-27 | `casmi_simple_sim_submission` v2 | Approach B: library tier 1 → ICEBERG → MetFrag-lite → ppm | **0.289** |
| 09-27 | `casmi_simple_sim_nosim` | Approach B control, `CFG.USE_SIM = False`: library tier 1 → MetFrag-lite → ppm | **0.254** |

**Owner to fill in:** what v6 (0.309), v8 (0.310) and v11 (0.314) changed, and whether v9 and v10
differ at all.

---

## 3. Verdict table — what worked and what did not

| idea | verdict | measured by |
|---|---|---|
| ICEBERG forward simulation as a tier-2 scorer | **works: +0.035 LB** | clean ablation, §5.6 |
| Retrained FPNet (listwise fine-tune) | works offline: es_MRR 0.556 → 0.828 on unseen scaffolds | §4.1 |
| Retrained ranker on **FPNet-seen** queries (Stage C v1) | **failed: −0.05 LB** (0.337 → 0.286) | §4.2 |
| Retrained ranker on seen + unseen queries (Stage C v2) + new FPNet | small gain on the phase1 base: 0.319 → 0.323 | LB, §4.6 |
| phase1-enhanced changes over tier1-patched | **−0.018 LB**, not attributed | LB |
| Stage D (`val_scripts`) as a ship/hold gate | **misleading for Class 2**: said HOLD, LB went up | §4.6 |
| Precomputing a simulated library on a CE grid | **not viable**: ICEBERG's output changes too sharply with CE (tolerance 0 eV) | §5.3 |
| On-the-fly ICEBERG on Kaggle | viable: 36.6 mol/s on one T4; the notebook finished inside the limit | §5.3, §5.6 |
| Covering water-loss adducts / `[M+K]+` with ICEBERG | **no**: no isomer separation at p < 0.01 | §5.2 |

---

## 4. Approach A — the FPNet + GBDT stack

### 4.1 Pipeline

Four evidence channels, fused by a gradient-boosted ranker:

| channel | what it is | columns in `rank_features` |
|---|---|---|
| 1 — library | entropy similarity against 2.54M reference spectra in a ±8.5 ppm window | 0–4, 25 |
| 2 — analog | mass-shifted entropy propagation across a ±200 Da window | 5–13, 26 |
| 3 — MetFrag | in-silico 1–2 bond fragmentation, peaks explained | 21–24 |
| 4 — model | FPNet spectrum→6,930-bit fingerprint, ranked by `f·z` (Bayes log-likelihood dot product) | 15–20 |

`rank_features()` emits **31 columns**, of which **12 are functions of FPNet logits**. The ranker
is 8 `HistGradientBoostingClassifier`s: `W1_PRIORS=(0.30,0.60)` × `SEEDS=(0,1,2,3)`, weighted
`W = where(M==0, w1, 1-w1)` where `M=0` means library evidence is available.

**The coupling that governs everything:** those 12 columns tie `rank_train.npz` to the exact FPNet
checkpoints that produced it. Swapping one without the other is a distribution mismatch; Stage D
measured the cost in both directions (§4.5).

Stages: **A** = data packing, **B** = FPNet training (B1 BCE pretrain, B2 listwise fine-tune),
**C** = build `rank_train.npz`, **D** = 4-way A/B on 1,145 validation molecules.

### 4.2 Stage A+B — FPNet (worked, keep it)

On the scaffold-disjoint early-stop split, against 48 real ±8.5 ppm decoys:

| checkpoint | es_MRR | es_top1 |
|---|---|---|
| `fp_single_s0` | 0.8278 | 0.7425 |
| `fp_single_s1` | 0.8250 | 0.7366 |
| `fp_merged_m0` | 0.8222 | 0.7431 |
| `fp_merged_m1` | 0.8197 | 0.7360 |

The listwise stage earned its keep: 0.556 → 0.828 (single), 0.689 → 0.822 (merged), on unseen
scaffolds. Known weakness, not fixed: B1 early-stopped on `es_bce` (bottomed at ep02) while
`es_MRR` was still climbing, leaving roughly 0.03 es_MRR (~4.5 h A100 to fix).

### 4.3 Stage C v1 — the regression (LB 0.337 → 0.286)

v1 excluded the *validation* structures from ranker-training queries but not the structures the
*FPNet had trained on*. The FPNet trained on essentially all of `train.parquet`, so every query was
memorised and `f·z` was a near-oracle.

| | forked matrix | v1 matrix |
|---|---|---|
| truth is the model's in-window top-1 | 42.0% | **90.2%** |
| `_z(raw)` for the truth | 1.64σ | **2.64σ** |
| library-channel separability (Cohen *d*, M=0) | 1.63 | **4.48** |
| row-weighted mean window | 194.7 | **106.8** |

Channel destruction on held-out queries (permute one channel within each query, re-measure MRR):

| channel destroyed | forked ranker | v1 ranker |
|---|---|---|
| library | −0.104 (14.5% of skill) | −0.026 (**2.7%**) |
| analog | −0.107 (15.0%) | −0.002 (**0.2%**) |
| model (`f·z`) | −0.242 (33.9%) | −0.576 (**60.2%**) |
| MetFrag | −0.016 (2.3%) | −0.000 (**0.0%**) |
| held-out MRR | 0.713 | 0.958 |

A GBDT handed a feature that alone solves 90% of its training queries uses that feature and
discards the rest. At test time, where Class-2/3 structures are unseen, nothing was left.

### 4.4 Stage C v2 — the rebuild (gate passed)

**(a) Queries from both model-strength regimes.** `_fpnet/pack/splits.npz` marks the scaffolds the
FPNet never trained on. Four cells of roughly equal size:

| cell | query structure | library | `M` | mirrors |
|---|---|---|---|---|
| A | FPNet-seen | visible | 0 | test Class 1 |
| C | FPNet-seen | banned | 1 | a Class-2 molecule the model happens to nail (**no test analogue**, §7) |
| D | FPNet-unseen | visible | 0 | cross-library Class 1 |
| B | FPNet-unseen | banned | 1 | **test Class 2** |

**(b) Paired two-regime scoring:** expensive work once per query; only library and analog re-run
under both ban settings. **(c) Full windows** (cap 400, fired on 1.1%). **(d) A `G` column** for
group reconstruction.

**Self-retrieval bug (caught by VERIFY 1).** The library is indexed on the label's exact mass and
the pool on `ExactMolWt`, but a query's window uses its reported `precursor_mz`. riken reports m/z to
2 dp (~12 ppm at 400 Da), so riken queries fell outside their own window: `lv` = 0 and the group had
no positive row. Measured usable share within ±8.5 ppm: riken 67.2%, mona 92.5%, msdial 98.8%,
massbank 86.8%, gnps 91.9%, enveda-np-examples 99.7%, masaryk 99.8%. Fix: keep only queries whose
corrected target lands within `CFG.PPM_WIN` of their label mass.

**Matrix and gate (VERIFY 2):**

```
2,788 groups | 196,174 rows | size mean 70.4 median 39 p90 184 max 400
truth is the model's in-window top-1: FPNet-seen 91.7% | FPNet-unseen 52.1%   (forked 42.0%, v1 90.2%)
held-out MRR 0.9108   M=0 0.9415 | M=1 0.8796
  library destroyed ->  8.4% of skill   (forked 14.5, v1  2.7)
  analog  destroyed ->  6.3%            (forked 15.0, v1  0.2)
  model   destroyed -> 36.4%            (forked 33.9, v1 60.2)
  frag    destroyed ->  1.7%            (forked  2.3, v1  0.0)
cells: A 0.9642 | C 0.9646 | D 0.9201 | B (test Class 2) 0.7971    (forked M=1: 0.6318)
```

The first gate rejected this matrix on an absolute rule (held-out MRR > 0.88) that was calibrated
on a weaker model. It was replaced with mechanism rules: FPNet-unseen top-1 ≤ 0.75, a seen−unseen
top-1 gap ≥ 0.10, MRR ≤ 0.88 on the unseen + banned cell, plus channel-share rules.
`build/test_gate.py` proves it rejects the 0.286 matrix and accepts the 0.337 one.

### 4.5 Stage D — the 4-way A/B (1,145 val molecules)

| config | C1_npx | C1_xlib | C2 | C2_hard | ALL |
|---|---|---|---|---|---|
| `base` (old FPNet / old matrix) | 0.8401 | 0.8941 | 0.8107 | 0.7783 | 0.8332 |
| `model_only` (new/old) | 0.8683 | 0.8987 | 0.7443 | 0.7143 | 0.8060 |
| `matrix_only` (old/new) | 0.5994 | 0.9084 | 0.8445 | 0.7939 | 0.8000 |
| `both` (new/new) | 0.8435 | 0.9117 | 0.7669 | 0.7382 | 0.8162 |

`both` vs `base`: C2 −0.044 ± 0.013, C2_hard −0.040 ± 0.018 → **HOLD**. The mirror-image control
arms (`model_only` −0.066 on C2; `matrix_only` −0.241 on C1_npx) demonstrate the FPNet↔matrix
coupling in both directions.

### 4.6 Why Stage D said HOLD and the LB said +0.004

`stage_d_diag_cell.py` (2026-09-25) resolved it:

- **Collision energy — falsified.** CE-known C2 and CE-known C1_xlib share libraries and CE mix,
  yet the new FPNet's in-window top-1 is 36–38% vs 79.8%.
- **Training-exclusion asymmetry — confirmed, and it is the whole story.** The new FPNet was denied
  the val `ban_keys`. C1 cohorts have **empty** `ban_keys`, so both models memorised them (new model
  80.4% / 83.4%). C2 truths *were* denied to the new model: 59.9% / 60.0%, close to its honest
  52.1%. The forked model excluded nothing and scores 75.5% / 73.3% there. `base` therefore had a
  memorisation advantage on C2 that does not exist on the LB.

The stack was submitted anyway and scored 0.323 vs 0.319 for the same base with forked artifacts.
Stage D as composed cannot deliver a clean Class-2 verdict.

Validation cohorts (from `val_scripts/val_key.pkl`):

| cohort | n | source library | `ce_known` | ban_keys |
|---|---|---|---|---|
| C1_npx | 245 | 100% enveda-np-examples | 100% | 0 |
| C1_xlib | 300 | gnps 165, riken 46, mona 40 | 32% | 0 |
| C2 | 400 | gnps 281, riken 51 | 23% | 1 |
| C2_hard | 200 | gnps 157, riken 17 | 15% | median 3, max 7,608 |

### 4.7 Planned but not executed: ICEBERG inside the stack

`specs_ms_simulator/` plans a 38-column matrix: scaffold consensus at columns 31–33 and neural
simulation at 34–37, append-only (inference truncates with `[:, :NFEAT]`). Its Phase 0 step 1 (the
diag) is done; the cell-C drop (`stage_c_dropC_cell.py`, `stage_d_pass3_noC_cell.py`) and
everything after it are **not run**.

---

## 5. Approach B — the simple ICEBERG baseline

### 5.1 Design

Goal: a lean, standalone notebook whose score shows directly what a forward simulator is worth.
No FPNet, no analog channel, no learned ranker.

```
query molecule → neutral mass (median over spectra, −1.45 ppm centring)
 → candidate window: ±8.5 ppm (fallback 30), PPM_EXPAND tiers, 120 ppm mass-nearest backfill (phase1)
 → per candidate:  lv = best real-library entropy similarity (phase1 lib_sim)
                   sv = best entropy similarity(query, ICEBERG spectrum at the query's adduct + exact CE)
                   fr = MetFrag-lite explain score;   ppm = |mass error|
 → tier 1: lv ≥ τ (0.75), by lv        tier 2: everything else, by (sv, fr, ppm)
 → ik14 dedup → top 25
```

- **Simulator:** ICEBERG from `coleygroup/ms-pred` (MIT), checkpoint **`msg_all`** (trained on
  MassSpecGym), CE in eV, instrument QTOF. NIST-trained weights are not allowed (not freely
  available). SHA1s: gen `c626bae1…`, inten `46583684…`; ms-pred commit `708148c2`.
- Tier-1 candidates are not simulated. The simulation precursor is the candidate's theoretical m/z.
  Merged multi-CE spectra: simulate each listed CE and keep the best `sv` (`MERGE_RULE = "max"`).
- Watchdog: requests run in priority order and simulation stops before 7.5 h; unsimulated
  candidates fall back to `fr`/ppm order.

### 5.2 Phase 0 — ICEBERG gate (Colab A100 + Kaggle T4)

Source: `specs_simple_ms_simulator/gate_report.md`.

- **G1 licence** pass; **G3 CE** conditions the output (median sim between 10 eV and 60 eV
  predictions 0.350; `[M+H]+` MRR 0.971 in eV vs 0.927 in NCE); **G5 offline** install on Kaggle in
  177 s, 20/20 predicted.
- **G6 sanity:** ms-pred emits every fragment both with and without the adduct shift, so water-loss
  adducts produce peaks 17/35/19 Da *above* the precursor. The worker now drops peaks above
  precursor + 2 Da (≤ 2% of intensity).
- **G2b isomer test** (truth + 20 decoys from its ±8.5 ppm window) — the decisive test:

| adduct | true fidelity | top1 | chance | covered? |
|---|---|---|---|---|
| [M+H]+ | 0.425 | 0.900 | 0.049 | **yes** |
| [M-H]- | 0.187 | 0.575 | 0.052 | **yes** |
| [M+Na]+ | 0.055 | 0.350 | 0.048 | **yes** |
| [M+Cl]- | 0.021 | 0.225 | 0.049 | **yes** |
| [M+CH2O2-H]- | 0.027 | 0.225 | 0.050 | **yes** |
| [M+NH4]+ | 0.106 | 0.200 | 0.050 | **yes** |
| [M-H2O+H]+ | 0.257 | 0.200 | 0.086 | no |
| [M+K]+ | 0.028 | 0.150 | 0.053 | no |
| [M-2H2O+H]+ | 0.132 | 0.125 | 0.054 | no |
| [M-H2O-H]- | 0.091 | 0.211 | 0.086 | no |

**Fidelity is not discrimination.** Water-loss adducts reproduce the spectrum well but match
fragments the decoys share; `[M+Cl]-` / `[M+Na]+` have near-zero fidelity yet win on a single
correct peak. The first G2 rule (decoys of different mass) overstated coverage. These top-1 values
are **upper bounds**: the truths are train structures, many likely in MassSpecGym, and real
windows hold 20–115 candidates, not 21.

### 5.3 Phase 1 — CE tolerance and throughput

- **CE tolerance is 0 eV.** Median similarity between predictions at CE and CE ± 2.5 eV is 0.846
  (rule: ≥ 0.9). A precomputed CE grid would be wrong for any off-grid query, so **all simulation
  runs on the fly** on Kaggle at the query's exact CE. roadmap Phases 2–3 (precompute, fidelity
  atlas) were skipped.
- **Throughput:** Kaggle T4 36.6 mol/s at batch 128 on one GPU (`t4.json`); a second T4 adds
  nothing (4 vCPU, CPU-bound). Worker start-up 10.4 s per call; offline install 197 s. A100:
  53–68 mol/s.
- **Merge-rule study not done.** The Colab grid's step 3 (max vs sum-merge vs mean CE on merged
  spectra) ran out of RAM. The OOM was fixed in `read_response`, but the step was never re-run, so
  `max` is untested against the alternatives.

### 5.4 Phase 4 — the Kaggle notebook

`build/simple_sim_submit.py` → `build/mk_simple_sim.py` → `casmi_simple_sim_submission.ipynb` and
`casmi_simple_sim_nosim.ipynb`. The two differ in exactly one line (`CFG.USE_SIM`), which a test
enforces. Phase1's library, pool, evidence and kernel cells are embedded verbatim. The notebook
installs ICEBERG into its own venv from the offline wheelhouse and runs it as a subprocess on GPU 0.
A fast commit deep-scores 30 molecules and writes `run_report.json` with PASS/CHECK flags.

Kaggle inputs: competition data, `casmi26-iceberg-offline`, the COCONUT dataset
(`coco_meta.pkl`, `coco_mass.npy`, `coco_fp.npy`), the dataset with `fp_bits.npy`, and the RDKit
2026.03.3 wheel. GPU T4, internet off.

### 5.5 Implementation bug worth remembering

The first watchdog compared the chunk-limited row count against the budget, so it would have stopped
simulating whenever more requests remained than one chunk (20,000) — i.e. almost always on the
hidden run. Caught by the offline test with a fake worker, before any submission.

### 5.6 Result — the ablation

| notebook | tier 2 order | LB |
|---|---|---|
| `casmi_simple_sim_nosim` | MetFrag-lite, ppm | 0.254 |
| `casmi_simple_sim_submission` | **ICEBERG `sv`**, MetFrag-lite, ppm | **0.289** |

**ICEBERG's contribution = +0.035 LB.** The read-out thresholds were written down before the control
was scored (`gate_report.md`: ≥ +0.010 = real signal). Both runs score the same molecules and differ
only in tier-2 order, so the difference is much less noisy than either absolute score.

For scale: the phase1 GBDT stack (library + analog + MetFrag + FPNet → ranker) scores 0.319, i.e.
+0.065 over the no-simulation control. ICEBERG alone, with no learning, recovers about half of that.
This comparison is **confounded** (the stack also uses weak library matches as a continuous feature,
the two-tier rule does not), so read it as an order of magnitude only.

Caveats:
- The hidden-run logs are not visible, so whether the watchdog capped any simulation there is
  unknown. The notebook did finish inside 9 h.
- ICEBERG covers 6 of 10 adducts. In the visible test the covered ones are 1,211 of 1,213 spectra.
- Weak real-library matches (`lv` < 0.75) are ignored once below tier 1. That is a limitation of
  this baseline, not of the simulator; the control shares it.
- τ = 0.75 and `MERGE_RULE = "max"` are untuned.

---

## 6. Lessons that hold across both approaches

1. **One change per LB submission, with a control.** 0.289 alone was uninterpretable (it added
   ICEBERG *and* removed three channels). A one-line control made it a clean +0.035. The unexplained
   0.337 → 0.319 step is what happens without one.
2. **A fused ranker must train where every channel is realistically weak.** A channel that is an
   oracle on training queries (memorised FPNet) gets all the weight and the rest are discarded
   (−0.05 LB). Ranker-training queries must include structures the model never saw.
3. **Pretrained models bring their own training set.** ICEBERG `msg_all` was trained on
   MassSpecGym (MoNA + MassBank + GNPS), which overlaps `train.parquet` and the val cohorts. Any
   ranker trained on ICEBERG features needs a second exclusion set — MassSpecGym train structures —
   or it repeats lesson 2 with a new actor.
4. **`val_scripts` cannot see pool misses and its C1 cohorts are memorised.** Val MRR always reads
   above the LB; use it for relative deltas only, and not for Class-2 verdicts against a model with
   a different exclusion set.
5. **Fidelity is not discrimination.** Test a scorer on its real task: isobaric candidates inside
   the actual window.
6. **FPNet checkpoints and `rank_train.npz` are a matched pair.** Never ship one without the other.
7. **Run-mode detection must fail safe.** v3 scored 0.032 because the fast-commit switch
   (`len(samp) > 400`, and an exact `== 'true'` string compare on `KAGGLE_IS_COMPETITION_RERUN`)
   evaluated False on the hidden run, so only 30/400 molecules were deep-scored, and the fast path
   took the 25 *lowest-mass* candidates instead of the nearest. Current code: full run unless the
   public file is proven, env var tested by truthiness.
8. **Absolute held-out MRR is not a leak test.** Gate on mechanisms (seen/unseen gap, channel
   shares) and validate the gate against matrices whose LB is known.
9. **Measure throughput and sensitivity before designing storage.** The precompute plan (~70 A100-h)
   died on a single measurement (CE tolerance 0 eV); on-the-fly simulation was cheaper and correct.
10. **Test the orchestration, not just the maths.** The watchdog bug (§5.5) would have silently
    removed most of ICEBERG's effect on the hidden run.

---

## 7. Open questions and candidate next steps

These are options, not decisions.

1. **ICEBERG inside the GBDT stack** (`specs_ms_simulator/`, columns 34–37). Both halves measured
   positive, but whether ICEBERG adds on top of FPNet — both target Class 2 — is unknown. Cost:
   ICEBERG spectra for every Stage-C training window on Colab, plus a MassSpecGym exclusion set
   (lesson 3). The licence memo sets a go/no-go: if `|es_keys \ msg_train_keys|` < ~2,000 there is
   no honest training cohort.
2. **Cheap one-flag LB probes on Approach B:** τ (0.6 / 0.85), `MERGE_RULE` (`sum` / `mid`), using
   `lv` < τ inside tier 2. Each costs one submission and no retraining.
3. **Attribute 0.337 → 0.319.** phase1-enhanced differs from tier1-patched in POLARITY_ANALOG,
   PPM_EXPAND / EXPAND_BACKFILL / BACKFILL_PPM and the fast-commit wrapper. The best-scoring notebook
   is not the one everything is built on.
4. **Cell C has no test analogue** (FPNet-seen + library banned). It is half the `M=1` training
   groups and teaches "when `lv` is 0, trust the model" at 91.7% model accuracy, where test Class 2
   gets ~52%. Dropping it needs only Stage D pass 3 (~25 min) — prepared, not run.
5. **Pool recall.** `CFG.USE_BIO_DB = False` (ChEBI / LIPID MAPS) is an assertion, never measured.
   Class-2 molecules outside COCONUT are unreachable by any ranker.
6. **FPNet early stopping** on `es_bce` instead of `es_MRR`: ~0.03 es_MRR, ~4.5 h A100.

---

## 8. Files

### Notebooks (all generated from `build/`, never hand-edited)

| file | approach | what it does | status |
|---|---|---|---|
| `enveda-casmi-2026-tier1-patched.ipynb` | A | forked public notebook, LB 0.337 | reference |
| `enveda-casmi-2026-phase1-enhanced-baseline.ipynb` | A/B | the base whose cells every generator embeds verbatim | base |
| `casmi_fpnet_train.ipynb` | A | Stage A+B, FPNet training | done |
| `casmi_export_fpnet_split.ipynb` | A | seen / es / banned keys from `_fpnet/pack/` | done |
| `casmi_rank_matrix_colab.ipynb` / `casmi_rank_matrix.ipynb` | A | Stage C v2 + VERIFY + gate (Colab / Kaggle sharded) | done / unused |
| `casmi_validate_ab_colab.ipynb` / `casmi_validate_ab.ipynb` | A | Stage D 4-way A/B | done |
| `casmi_simple_sim_gate_colab.ipynb` / `casmi_simple_sim_gate_kaggle.ipynb` | B | Phase 0 ICEBERG gate | done |
| `casmi_simple_sim_grid_colab.ipynb` | B | Phase 1 CE-tolerance / merge-rule grid | step 3 not re-run |
| `casmi_simple_sim_t4_kaggle.ipynb` | B | T4 throughput | done |
| `casmi_simple_sim_submission.ipynb` | B | the submission, LB 0.289 | done |
| `casmi_simple_sim_nosim.ipynb` | B | the control, LB 0.254 | done |

### Generators and shared code — `build/`

| file | purpose |
|---|---|
| `mk_stage_c_colab.py`, `mk_stage_c.py` | Stage C (Colab / Kaggle) |
| `mk_stage_d_colab.py`, `mk_stage_d.py` | Stage D; the Colab build derives its cells from the Kaggle one |
| `fpnet_train.py`, `export_split.py` | Stage A+B and split-export sources |
| `mk_simple_sim_gate.py`, `mk_simple_sim_grid.py` | Approach B Phase 0 / Phase 1 notebooks |
| `simple_sim_submit.py`, `mk_simple_sim.py` | Approach B submission source and generator (both notebooks) |
| `simsim_common.py` | Approach B helpers: adduct physics, windows, spectrum prep, request building, two-tier ranking |
| `iceberg_worker.py` | ICEBERG subprocess worker, embedded into the notebooks |
| `py2nb.py` | cell-marked `.py` → `.ipynb` |
| `mk_stage_c_v1_broken.py.bak` | the generator that produced the leaked 0.286 matrix, kept for reference |

### Tests — `build/`, run by `run_all_tests.sh`

| file | what it proves |
|---|---|
| `selftest.py` | Stage-B machinery on synthetic data |
| `test_process_query.py` | paired two-regime invariant |
| `test_cohort.py` | four-cell cohort, holdout respected, coarse-m/z libraries filtered |
| `test_gate.py` | the gate rejects the 0.286 matrix and accepts the 0.337 one |
| `test_stage_d_colab.py` | Colab Stage D differs from Kaggle only where intended — **currently failing** ("kaggle 19, colab 22" cells), pre-existing |
| `test_simple_sim_gate.py`, `test_simple_sim_grid.py` | Approach B gate / grid notebooks |
| `test_simple_sim.py` | submission notebook: verbatim embedding, helpers, pass 1 / simulate / pass 2 on a toy pool with a fake worker, watchdog, crash paths, the no-sim control |

`run_all_tests.sh` pipes each test through `tail -1`, which hides exit codes: it reports success
even when a test fails. Read its output, not its exit status.

### Paste-in cells (live Colab kernel)

`stage_c_hotfix_cell.py`, `stage_c_gate2_cell.py`, `stage_d_diag_cell.py` (run),
`stage_c_dropC_cell.py`, `stage_d_pass3_noC_cell.py` (prepared, not run).

### Documentation

| file | purpose |
|---|---|
| `summary.md` | this file |
| `RUNBOOK.md` | Approach A operations and decision rules |
| `knowledge_base.txt` | background: competition, models, external resources |
| `specs_ms_simulator/` | plan for simulation + scaffold consensus inside the GBDT stack (not executed past Phase 0 step 1) |
| `specs_simple_ms_simulator/` | Approach B: mission, roadmap, tech stack, `gate_report.md`, raw gate/grid/T4 results, `scores_performed.txt` |

---

## 9. Standing rules

- Never ship a `rank_train.npz` without the FPNet checkpoints it was built from.
- Ranker-training queries must include model-unseen structures, for every learned channel
  (FPNet, and ICEBERG / MassSpecGym if it becomes a feature).
- `rank_features` columns 0–30 are frozen; widen append-only.
- Only freely and publicly available checkpoints; record SHA1s and assert them at load.
- Notebooks are generated from `build/mk_*.py`; regenerate after editing a generator and run
  `sh build/run_all_tests.sh` before shipping.
- Key everything by InChIKey14; dedupe before emitting.
- Read `val_scripts` as relative deltas only; the LB is the arbiter for Class 2.
- Every LB experiment gets a control that differs in one setting.
