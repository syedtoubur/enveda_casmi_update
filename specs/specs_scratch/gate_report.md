# Gate report

Thresholds are written **before** the run. Verdicts are appended after, never edited.

## R0 — port of the Track 3 base (pre-registered 2026-10-05)
Reference: `casmi_simple_sim_submission.ipynb` with `CFG.MAX_PEAKS=100` → LB 0.296 (own run).

| Gate | Threshold | Why |
|---|---|---|
| R0.1 package tests | `python scratch_pkg/tests/run.py` green | |
| R0.2 fast commit | finishes; `sim_ok_share ≥ 0.99`; tier-1 rate ≥ 0.9 on the public file | the public file is train rows, so a real library match must win |
| R0.3 full run | `capped_rows == 0`, `failed_rows == 0`, total ≤ 7.5 h | |
| R0.4 LB parity | LB ≥ 0.292 (0.296 − noise 0.004) | R0 also extends `MAX_PEAKS=100` to `sv`, so a move up is possible; a drop below 0.292 means a port bug → diff against the base with `compare.py` before anything else |

### Verdicts
- **R0.4 FAIL** (LB 2026-10-07): `casmi_scratch_r0` 0.290 < 0.292. The base itself scored 0.296, so R0 is −0.006, just
  outside the ±0.004 noise floor. Pre-registered next step: diff R0 against the base with `compare.py` for a port bug.
  Mitigating evidence, which does not override the gate: every R1 notebook on the same port scores 0.301–0.303, above the
  base. R1 was submitted before this prerequisite was met.
- **R0.4 diff (2026-10-07, done in parallel with R2 as agreed):** `build/simple_sim_submit.py` vs `casmi_scratch` by hand
  (`compare.py` on the notebooks is useless here: R0 embeds the package as one string). The target centring, pass 1
  (lv, fr, covered spectra, sim set, priority), the watchdog, pass 2 stages A/B/C and the slot dedup are the same. The one
  material difference is the planned one: R0 cleans query spectra for `sv` with `max_peaks` 100, while the base used
  `simsim_common.TOPK` = 64 for `sv` (`MAX_PEAKS=100` reached only lv/fr). No port bug found. The −0.006 is either
  that change or noise (1.5× the floor). An `sv_max_peaks=64` probe on the current base would separate the two; not built.

## R1 — unified candidate pool (pre-registered 2026-10-05, before any measurement)
Plan: `r1_pool_plan.md`. R1a = union pool (v2-pool ∪ R0 coco+train ∪ BIO ∪ NP table, scorer_key dedup).
R1b = R1a minus enveda-180-only structures (in no NP source). Sim priority unchanged from R0.
Prerequisite: R0.4 passed.

| Gate | Threshold | Why |
|---|---|---|
| R1.1 no loss | every R0 pool scorer_key ∈ R1a (0 lost) | R1 must be a pure superset of R0 |
| R1.2 recall | truth-in-window recall on slice I (245 enveda-np-examples mols, 1.45 ppm centring, R0 `ppm_win`): R1a ≥ R0; report the gain and R1b's recall | coverage is the lever |
| R1.3 cost | median / p95 window growth reported; R1a run ≤ 7.5 h on T4; truth-in-simulated rate logged | bigger windows dilute the sim budget |
| R1.LB | keep R1a / R1b iff ≥ paired-control mean + 0.004 and counters > 0 (pool size, windows grown, e180 rows dropped) | standard protocol |
| CV class vote (definitions below) | slice I, own-ik14 library spectra banned, same polarity, shift-aware entropy; top 10 hits sim ≥ 0.5, similarity-weighted superclass vote. Two variants reported separately: (a) direct ±10 ppm + NP delta list, (b) open ±200 Da. PASS per variant iff purity@superclass ≥ majority-class baseline + 0.10 **and** coverage ≥ 50% (purity@pathway also reported) | R2 uses (a) if both pass; idea dropped before R2 if neither does |

Definitions fixed in code on 2026-10-06, still before any measurement (`casmi_scratch/audit.py`, `analog.py`):
- **R1.1** An R0 row is lost if RDKit parses it but gives no scorer key, or if its scorer key is not in R1
  (`pool_build.superset_check`, recorded in `MANIFEST.json`).
- **R1.2** Each pool in its own key space: R0 = plain inchikey14, R1 = scorer key (truth via `train_keymap`).
  Window = ±`ppm_win` (8.5 ppm) around the 1.45 ppm-centred target; the scored range (with expansion) is also reported.
- **R1.3** Truth-in-simulated: the request budget of a real run (`rows_that_fit` at `Config.rate`, 400 test molecules)
  split evenly per molecule and per (adduct, CE) condition. The truth counts as simulated iff its R0 sim-priority rank
  is below that share. Own library spectra are banned.
- **CV** NP delta list (each ±10 ppm, both signs): CH2 14.01565, O 15.99491, H2O 18.01056, CH2O 30.01056,
  C2H2O 42.01056, CO2 43.98983, C5H8 68.06260, C3H2O3 86.00039, C5H8O4 132.04226, C9H6O2 146.03678,
  C6H10O4 146.05791, C6H10O5 162.05282, C6H8O6 176.03209. Truth label = the R1 pool label of the truth's scorer key
  (table or kNN). Coverage = voted / molecules with a truth label. Purity = matching votes / voted. Baseline = share of
  voted molecules whose truth carries the most frequent truth superclass. A multi-valued truth label (`A|B`) matches
  any of its values.

### Verdicts
- **R1.1 PASS** (2026-10-06, `casmi26-scratch-pool` v1 build, `MANIFEST.json`): 712,199 R0 rows, all parsed, 706,289
  R0 scorer keys; lost_unkeyable 0, missing_keys 0. R1a = 778,095 structures (+71,806 vs R0, +10.2%).
- **R1.2 PASS, but uninformative** (audit 2026-10-06): window, scored-range and in-pool recall were all 1.000 for R0,
  R1a and R1b, with 0 recovered and 0 lost. Every slice-I truth is a train structure, and train is in every pool, so
  this slice cannot measure a coverage gain. The gate was badly designed (cf. memory "val blind to pool misses"),
  so the LB is the only coverage judge for R1.
- **R1.3 reported:** median ±8.5 ppm window R0 53.5 → R1a 57 → R1b 47 candidates. R1a/R0 ratio: median 1.03,
  p95 1.20 (+2 median, +8 p95). R1b/R0: median 1.00 (p95 1.19). Scored-range median is 60 / 60 / 53.5, held at the
  `topn` floor by expansion. Truth-in-sim was 1.000 for all three because the budget (2,405 requests per molecule at
  rate 36.6/s) is about 13× the roughly 60 candidates × conditions requested. The sim budget does not bind; the
  runtime part (≤ 7.5 h) is still to be checked on the Kaggle run.
- **CV PASS, both variants.** Superclass: (a) delta coverage 0.929 (196/211), purity 0.878 vs baseline 0.168
  (Flavonoids), pathway purity 0.905. (b) open coverage 0.910, purity 0.776 vs baseline 0.167, pathway purity 0.867.
  Median hits with sim ≥ 0.5: delta 18, open 974.5. CPU time: delta about 30 s, open about 41 min.
  → Per the pre-registered rule, R2 uses (a). Caveats before building R2 (see kb_index):
  - vote purity does not show discrimination among the window's candidates;
  - slice I is 230/250 in MassSpecGym, so its analog families are dense in the library;
  - kNN labels can be inherited from the truth's own structure.
- **R1.LB PASS** (LB 2026-10-07; control = R0 0.290, bar 0.294): r1b 0.301 (+0.011), r1a_glacier 0.301 (+0.011),
  r1b_glacier 0.303 (+0.013). vs the Track 3 base 0.296: +0.005 / +0.005 / +0.007. Caveats: single runs, R0 failed R0.4,
  and r1a (ICEBERG) is not yet submitted. R1a vs R1b (within GLACIER) is 0.301 vs 0.303, noise: R1b (597k, 15% smaller)
  is the cheaper default.

## R1-GL — GLACIER replaces ICEBERG (pre-registered 2026-10-06, before any measurement)
Twins `casmi_scratch_r1a_glacier` / `r1b_glacier` = R1a / R1b with `sim_model="glacier"`. That is the only change: pool,
lv/fr, sim priority, scheduler, `spectrum_sv` (max over CEs), tau and two-tier ranking are untouched. GLACIER
(casmi26-glacier, MassSpecGym checkpoint) predicts `[M+H]+` / `[M+Na]+` only. Spectra with the other four covered
adducts get no sim requests, so `sv` = 0 and their candidates rank on fr/ppm. Motivation:
`specs/track4_priors_probes/iceberg_bug_comparison.md` (GLACIER alone beat ICE+GLACIER, but *in fusion* and in single
author runs). In our pipeline ICEBERG is the only simulator, so this is a new hypothesis.

| Gate | Threshold | Why |
|---|---|---|
| R1-GL.LB | keep GLACIER iff the twin's LB ≥ its ICEBERG twin (r1a / r1b, same day) mean + 0.004 | LB noise floor ±0.004 |
| R1-GL.run | fast commit: `[GLACIER]` setup line with MANIFEST sha OK, smoke `n_peaks_40` > 0, `sim_ok_share` ≥ 0.99, `n_predicted` > 0; full run ≤ 7.5 h, `capped_rows` and `failed_rows` reported | the swap must actually run |
| R1-GL.coverage | `n_covered_spec` reported for both twins (ICEBERG vs GLACIER) | reads how much sim coverage the 2-adduct restriction costs |

Reading a drop: the coverage loss on non-`[M+H]+`/`[M+Na]+` spectra outweighs any better isomer scores. The
follow-up would be a hybrid (GLACIER on `[M+H]+`/`[M+Na]+`, ICEBERG on the rest), which is not built.

### Verdicts
- **R1-GL.run PASS** (fast commit of `casmi_scratch_r1a_glacier`): MANIFEST sha OK, smoke 100 peaks at 40 and 120 eV,
  `sim_ok_share` 1.0, 0 capped / failed rows, 103 requests/s (ICEBERG 36.6), 43 covered spectra on 20 molecules.
  The full-run counters (`n_covered_spec`, `total_s`) have not been read.
- **R1-GL.LB, r1b pair: NOT KEPT** (LB 2026-10-07): r1b_glacier 0.303 vs r1b 0.301 = +0.002 < +0.004. Neutral.
- **R1-GL.LB, r1a pair: PENDING**: r1a_glacier 0.301; `casmi_scratch_r1a` not yet submitted.
- **R1-GL.coverage**: not read for the full runs.
- Reading: GLACIER simulates 2 of the 6 covered adducts and still ties ICEBERG, so restricting the simulator costs
  nothing measurable. The hybrid (GLACIER on `[M+H]+`/`[M+Na]+`, ICEBERG on the rest) is the next probe; pre-register it first.

## R2 — class vote in tier-2 ranking (pre-registered 2026-10-07, before any measurement)
Plan: grilling 10-07 (summary in `roadmap.md` S3). Probes **R2-ICE** = r1b + class vote, **R2-GL** = r1b_glacier + class
vote. One change each: `class_vote=True`. Labels come from **NPClassifier** (mwang87/NP-Classifier, MIT, Zenodo
5068687 weights). It is run on Colab for all 778,095 R1a structures and published as `pool_npc.parquet` in
`casmi26-scratch-pool` v2. The same labeller is used on the vote side (library analogs) and the candidate side.
Mechanism: the vote is `analog.class_vote` (delta list ±10 ppm, shift-aware entropy, top 10 hits with sim ≥ 0.5, same
polarity), applied to each hit's top-1 NPClassifier superclass; `conf` = the winner's share of the vote weight.
Tier-2 sort key = `sv + β·conf·P_cand(voted superclass)`, with P from the candidate's top-3 superclass probabilities
(0 if the voted superclass is not among them) and **β = 0.10 fixed**. Tier 1 (lv ≥ τ) and stage C are untouched. The
bonus also applies when sv = 0. No NP_Score and no provenance prior.

| Gate | Threshold | Why |
|---|---|---|
| R2.0 spike | NPClassifier weights load on Colab; on 50 random NP-table rows with a superclass, the model's top-1 superclass ∈ the table label for ≥ 45/50; fingerprint parity with the repo's `calculate_fingerprint` on those 50; 10k SMILES timed | the labeller must reproduce its own published labels. Fail → fallback LR on the 152k table labels (report holdout accuracy) |
| R2.1 labels | all R1a rows that RDKit parses get top-3 superclasses; unparsable count logged; top-1 agreement with table labels reported | full coverage is the reason for relabelling |
| R2.2 CV re-run | slice I, own-ik14 ban, delta variant, NPClassifier top-1 labels on both sides: purity@superclass ≥ majority baseline + 0.10 **and** coverage ≥ 50%; table-only purity also reported | the R1 CV PASS used table/kNN labels |
| R2.3 discrimination (kill) | slice I, own-ik14 library spectra banned, scored range, **sv = 0** (no simulator on Colab: fr is the tier-2 base, as in GLACIER's no-sim spectra), β = 0.10. PASS iff (1) the truth's P(voted) > the window median P(voted) in ≥ 60% of voted molecules **and** (2) the bonus moves the truth up in more molecules than it moves it down. Reported: reach (% top 25 changed), MRR@25 before/after, median top-1/top-2 fr gap | purity ≠ discrimination; FAIL → no LB build, recorded as a dead end |
| R2.run | fast commit: `n_vote_mol` > 0, `n_bonus_reorders` > 0, `sim_ok_share` ≥ 0.99, 0 capped/failed rows; full run ≤ 7.5 h | the lever must fire |
| R2.LB | same day: R2-ICE vs control r1b, R2-GL vs control r1b_glacier. Keep each iff ≥ its control + 0.004 and its counters > 0 | standard protocol |

### Verdicts
- **R2.0 FAIL** (Colab 2026-10-07): weights loaded (heads 7 / 77 / 687), fingerprint parity 50/50, but top-1 ∈ table
  label only **1/50**; 3.5 s per 10k. Diagnosis (local, numpy forward pass of the Zenodo superclass hdf5, so TensorFlow
  is not involved): RDKit changed the Morgan bit assignment between **2021.03.5** and **2021.09.5** (2020.09.5 =
  2021.03.5; 2021.09.5 = 2022.03.5 = 2026.03.3 on 14 test structures). The net was trained on the old bits. With
  RDKit 2021.03.5 fingerprints it labels all 14 correctly (quercetin/naringenin Flavonoids, cholesterol/testosterone
  Steroids, palmitic/oleic Fatty Acids and Conjugates, nicotine Nicotinic acid alkaloids, strychnine Tryptophan
  alkaloids, limonene/menthol Monoterpenoids, artemisinin Sesquiterpenoids, glucose Saccharides, emodin Polycyclic
  aromatic polyketides, caffeine Pseudoalkaloids), mostly at p ≈ 1.0. With 2026.03.3 fingerprints its outputs are
  noise. The parity check could not see this because it ran the repo's code on the same new RDKit. Not a model failure,
  so the LR fallback is not triggered yet.
- **R2.0b re-run (pre-registered 2026-10-07, before the run):** fingerprints computed in a side env with
  `rdkit-pypi==2021.3.5` (Python 3.9, via `uv`; `resources/npc_fp_worker.py`); TensorFlow stays in the notebook env.
  Same thresholds as R2.0: parity of our worker vs the repo's `calculate_fingerprint` 50/50 (both in the old env),
  top-1 ∈ table label ≥ 45/50 on 50 random table-labelled rows (seed 0), 10k timed. Also reported, not gated: the 14
  structures above, expected labels. FAIL → LR fallback as pre-registered.
- **R2.0b PASS** (Colab 2026-10-07, `spike_npc_r20b.json`): fingerprint env RDKit 2021.03.5 (built in 13 s),
  parity **50/50** with the repo's `calculate_fingerprint` in that env, top-1 ∈ table label **49/50** (≥ 45; the
  one miss: model Diterpenoids vs table Pseudoalkaloids), known structures **14/14** (p 0.988-1.0), 4.2 s per 10k.
  Weights load under tf_keras (heads 7 / 77 / 687). R2.0 → R2.0b: 1/50 → 49/50 from the RDKit change alone.
- **R2.1 PASS** (Colab 2026-10-07, NPClassifier, fingerprints from RDKit 2021.03.5): 778,095 / 778,095 labelled,
  0 unparsable; top-1 agreement with the 131,644 table-labelled rows **0.970**; sp1 median 0.710; 5.0 min.
  `pool_npc.parquet` written, MANIFEST v2. Top-1 counts are led by Lysine alkaloids 68,271, Tryptophan alkaloids
  66,094 and Nicotinic acid alkaloids 56,019, then Glycerolipids 37,365. Likely out-of-domain argmax on non-NP
  N-heterocycles (unverified). The bonus is weighted by P, but the vote uses each hit's top-1 label unweighted, so
  R2.2 / R2.3 will show whether it matters.
- **R2.2 PASS** (Colab 2026-10-07, `_r2_audit/cv.json`): delta variant, NPClassifier top-1 labels on both sides,
  coverage **0.948** (237/250), purity@superclass **0.730** vs baseline 0.139 (Flavonoids), +0.59 (bar +0.10).
  Pathway purity 0.848 vs 0.316 (Alkaloids). Table-only purity (NPClassifier vote vs table truth label) 0.816 (n 190).
  Down from R1's 0.878, which used table/kNN labels: multi-valued table labels (`A|B` matches either) and
  kNN labels inherited from the truth inflated R1. 0.730 is the more honest figure.
- **R2.3 PASS** (Colab 2026-10-07, `_r2_audit/disc_rows.json`, sv = 0, β = 0.10): voted 237/250, all in range. Truth
  P(voted) above the window median in **0.802** of voted molecules (bar 0.60). Truth moved up in **122**, down in **91**
  (bar up > down). Reach 0.928. MRR@25 0.2691 → 0.2987 (+0.030). Median tier-2 fr gap 0.017.
  Reading: with sv = 0 the bonus (up to 0.10) dwarfs the fr gaps, so it acts as the main tier-2 key. Reach 0.93 and
  +0.030 are an upper bound on the effect, not a forecast. 91 molecules got worse, so the gain is a net of large
  moves both ways. Slice I is analog-dense (230/250 in MassSpecGym), so test-set vote coverage and purity will be lower.
  → LB build allowed.
- **R2-GL.LB PASS** (Kaggle 2026-10-07): `casmi_scratch_r2_gl` **0.311** vs control `casmi_scratch_r1b_glacier` 0.303
  (same day, 10-07): +0.008 ≥ +0.004. Kept. Fast-commit counters (`n_vote_mol`, `n_bonus_reorders`, `n_vote_first`)
  not yet copied here. R2-ICE (control r1b 0.301) not yet submitted. Best specs_scratch run (R0 0.290 → R1 0.303 → R2 0.311).

## R3 — FPNet channel, RRF in tier 2 (pre-registered 2026-10-08, before any measurement)
Plan: `r3_fpnet_plan.md` (grilling 10-07/08). Bases: **r1b_glacier** (0.303) and **r2_gl** (0.311, R2-GL LB).
One change per probe: `fp_channel`. Candidate score `f·z` (binary candidate bits · predicted logits; z averaged over
models, then `0.5·(mean per-spectrum + mean per-polarity merged)`). Tier 1 (lv ≥ τ) and stage C untouched. Tier-2 key
in stages A and B = `1/(3+r_sv) + 1/(3+r_fp)` (r_sv = rank of sv + vote bonus, r_fp = rank of f·z, descending, average
ranks for ties), tie-breaks fr then |ppm|. k = 3 and equal weights fixed, never tuned.
Checkpoints: **own** = `casmi-fpnet` s0,s1,m0,m1 (family P, 6,930 bits, C2/C2_hard banned from training);
**full1** = `casmi26-fpnet-full1` (family A, Ahmed engine FPNet, 10,226 bits, trained on all folds).
Probes: `r3_own_gl`, `r3_full1_gl`, `r3_own_r2gl`, `r3_full1_r2gl`. No same-day control (user's choice).

| Gate | Threshold | Why |
|---|---|---|
| R3.run | fast commit: (a) featuriser bit count = ckpt nbits and own fp_bits sha1 = ckpt sha1; featuriser parity vs the precomputed rows (`pool_fp.npy`) ≥ 45/50 for ahmed (own: reported only, see amendment); (b) self-rank smoke: 5 train molecules, truth's f·z in the top 5 of its ±8.5 ppm window for ≥ 4/5; (c) `n_fp_mol` > 0 and `n_fp_reorders` > 0; full run ≤ 7.5 h | catches wrong bit order, adduct vocab or peak prep; the lever must fire |
| R3.LB | keep iff ≥ its base's earlier score **+ 0.008** (r1b_glacier probes ≥ 0.311, r2_gl probes ≥ 0.319) and counters > 0. A result in [+0.004, +0.008) gets a paired re-run with its control the next day | no same-day control, so the noise bar (±0.004) is doubled |
| R3.audit | val_key **C2 (400) ∪ C2_hard (200)**, own library spectra banned (`ban_rows`), scored range, sv = 0. Own bank: PASS iff MRR@25(with FP) − MRR@25(without FP) ≥ **+0.02** **and** the truth moves up in more molecules than down. full1 and C1_npx (slice I) reported as leaked upper bounds only | does not block the LB; decides whether to keep iterating on FP |

Amendment (2026-10-08, build time, before any measurement): gate (a) asserts featuriser parity only for **ahmed**,
whose model was trained on targets read from `pool_fp.npy`. For **own**, the training targets were computed by
`fpnet_train.py` with RDKit 2026.03.3 (the runtime RDKit), while `coco_fp.npy` may come from another RDKit, so parity
there is reported but not asserted. Gate (b), self-rank, is asserted for both.

### Verdicts
- **R3.run fast commit PASS, `r3_own_r2gl`** (Kaggle 2026-10-08): own bank = fp_merged_m0/m1 + fp_single_s0/s1 from
  `casmi-fpnet`. It picked `fp_bits.npy` from `prvsiyan/coconut-casmi26-candidates` by sha1 `4686f545…`, which is the
  known bit contract. (a) featuriser parity 50/50 against `coco_fp.npy`; (b) self-rank 5/5, all at rank 1
  (slice I was in training, so this proves the wiring, not the quality); (c) `n_fp_z` 20, `n_fp_mol` 20,
  `n_fp_reorders` 20/20. `fp_z_s` 0.7 s for 20 molecules; sim_ok 1.0; 0 capped / failed rows. The tier-1 rate on the
  placeholder file is 0.0 ("CHECK"). That check is R0.2's, not R3's.
- **R3.LB PASS ×4** (Kaggle 2026-10-08, single runs, no same-day control):

  | probe | base (old score) | bar | LB | Δ vs base | verdict |
  |---|---|---|---|---|---|
  | `r3_own_gl` | r1b_glacier 0.303 | 0.311 | 0.322 | +0.019 | KEEP |
  | `r3_own_r2gl` | r2_gl 0.311 | 0.319 | 0.320 | +0.009 | KEEP (by 0.001) |
  | `r3_full1_gl` | r1b_glacier 0.303 | 0.311 | 0.341 | +0.038 | KEEP |
  | `r3_full1_r2gl` | r2_gl 0.311 | 0.319 | **0.346** | +0.035 | KEEP, best scratch run |

  Readings:
  - **Family:** full1 beats own on both bases (+0.019 gl, +0.026 r2gl), 5–6× the ±0.004 noise floor. A larger bit set
    (10,226 vs 6,930) and training on every fold both count. On the LB that is legitimate, because the hidden test
    structures are NP molecules that are not in train. It also means the C2 audit cannot judge full1.
  - **Class vote on top of FP:** +0.005 with full1, −0.002 with own. Both are within noise, so the R2 vote is no longer
    shown to add anything once FP is in tier 2. Keep it in the best line (it does not hurt), but do not credit it.
  - **Lever size:** FP is the largest specs_scratch lever so far (+0.035…0.038, against R1 +0.011 and R2 +0.008).
    The gap to zenith (0.425) shrinks from 0.114 to 0.079.
  - Full-run times and the full1 fast-commit counters were not recorded.
- **R3.audit:** pending (`casmi_scratch_r3_audit_colab`).


## R4 — FPNet family fusion + FP-ordered stage C (pre-registered 2026-10-08, before any measurement)
Plan: `r4_famfuse_plan.md` (grilling 10-08). Base and same-day control: **`r3_full1_r2gl`** (0.346).
Probe: **`r4_famfuse`**. It differs from the base only in `fp_banks=("ahmed","prv")`, `fp_weights=(1.0,0.6)`.
Families:
- **A** = `fpnet_full1` (as R3);
- **P** = prvsiyan `casmi26-fp-models-v2` (`fp_single_s2`, `fp_merged_m1`), 6,930 bits = the `fp_bits.npy` next to `coco_fp.npy`.

Each family gives its own f·z over the same candidates.

Ranking:
- **Tier-2 key** in stages A and B: `1/(3+r_sv) + 1/(3+r_A) + 0.6/(3+r_P)`. Average ranks, descending, NaN last within its own family. Tie-breaks fr, then |ppm|.
- **Stage C** (the mass-nearest top-up) is ordered by `1/(3+r_A) + 0.6/(3+r_P)`, then |Δmass|.
- Tier 1 is untouched. k = 3 and the weights (1.0, 0.6) are fixed and never tuned. 0.6 is zenith's engine-2 α.

| Gate | Threshold | Why |
|---|---|---|
| R4.run | fast commit, per bank: (a) featuriser parity vs the precomputed rows ≥ 45/50 (A: `pool_fp.npy`, P: `coco_fp.npy`); (b) self-rank smoke: truth's f·z in the top 5 of its ±8.5 ppm window for ≥ 4/5; (c) `n_fp_mol` > 0 and `n_fp_reorders` > 0 for both families. `n_stagec_slots` reported. Full run ≤ 7.5 h with `total_s` recorded | catches a wrong bit basis or loader for prvsiyan; both channels must fire |
| R4.LB | same-day pair, control `r3_full1_r2gl` re-run. KEEP iff probe − control ≥ **+0.006**. [+0.004, +0.006) → repeat both the next day and keep on a mean Δ ≥ +0.004. Otherwise DROP | ±0.004 run-to-run noise |
| R4.audit | none: prvsiyan's training split is unknown and full1 saw every fold, so no offline cohort is honest | the LB pair decides |

### Verdicts
- **R4.run fast commit PASS, `r4_famfuse`** (Kaggle 2026-10-08):
  - A = `fpnet_full1`, with the `casmi26-v2-pool` fp_bits (sha1 `3d87ee9a…`, 10,226 bits).
  - P = prvsiyan `fp_merged_m1` + `fp_single_s2`, with the fp_bits from `prvsiyan/coconut-casmi26-candidates` (sha1 `4686f545…`, the same contract as the own family).
  - (a) Parity 50/50 for both families.
  - (b) Self-rank 5/5 for both, all at rank 1. Slice I was in both families' training data, so this proves the wiring only.
  - (c) `n_fp_mol_by` [20, 20] and `n_fp_reorders_by` [20, 19]. Dropping P changes the top 25 in 19 of 20 molecules.
  - `n_stagec_mol` 0, so stage C did not fire.
  - `fp_z_s` 0.8 s for 20 molecules (both families). sim_ok 1.0; 0 capped / failed rows. Fast-commit `total_s` 184.
  - The tier-1 "CHECK" is the placeholder-file artefact (as in R3).
- **R4.LB DROP (neutral)** (Kaggle 2026-10-08): `r4_famfuse` **0.345** vs `r3_full1_r2gl` 0.346 (its earlier score; the control was not re-run the same day). Δ −0.001, far below the +0.006 bar and inside the ±0.004 noise, so no repeat pair is needed. Reading: P is live (it reorders the top 25 in 19/20 molecules) but brings no new information next to full1. Both families predict the same fingerprint task from the same spectra. Zenith's two-engine gain came from fusing whole systems (pool, rankers, analog features), not from a second FPNet. Stage C never fired (0/20), so it neither helped nor hurt. `r3_full1_r2gl` stays the best scratch run.

## R5 — Spectral-analogue channel: PSSM / Tanimoto kernel (pre-registered 2026-10-08, before any measurement)
Plan: `pssm.md` (v2, grilling 10-08). Base: **`r3_full1_r2gl`** (0.346). **No same-day control.**

Four probes. Each differs from the base only in `analog_channel` (and `class_vote` for the novote twins):
- **`r5_pssm`** and **`r5_tani`**, with the class vote on as in the base.
- **`r5_pssm_novote`** and **`r5_tani_novote`**, identical but with `class_vote=False`.

Channel definition:
- **Analogue search:** open ±200 Da, shift-aware entropy similarity over the full train library, enveda-180 included. Each spectrum is searched within its own polarity. Keep the max S per structure and S ≥ 0.5. Fewer than 3 analogues with a structure means no list for that molecule.
- **PSSM:** top 100 analogues, 16,520 raw Family-A bits.
  - w = S⁴.
  - p̃ = (Σw·x + 1·q) / (Σw + 1).
  - W = clip(logit p̃ − logit q, ±3).
  - Score s = C·W.
  - q is the bit frequency over a seeded sample of 50k non-train R1b pool rows, unless an `analog_q.npy` is attached.
- **Tanimoto kernel:** top 20 analogues; s = Σ S⁴·T / Σ S⁴, with T on Morgan r2 4096.
- **Train structures** (pool flag `train`) are not scored by this list. They get the neutral term, the mean of the scored candidates' terms. The pool, tier 1, the sv list and the f·z list are untouched.
- **Tier-2 key:** `1/(3+r_sv+bonus) + 1/(3+r_fz) + 1.0/(3+r_an)` in stages A and B. Stage C is unchanged (off). k, weights and every analogue parameter are fixed, never tuned.

| Gate | Threshold | Why |
|---|---|---|
| R5.0 | fast commit, per probe: the analogue list is built for ≥ 50% of deep molecules (`n_analog_mol`), and it fires (`n_analog_ok` > 0, `n_analog_reorders` > 0); `analog_s` and `total_s` recorded. Full run ≤ 7.5 h | the channel must be live. No runtime fallback: the open search runs as in the public AFIX notebooks |
| R5.audit | slice I, Colab `casmi_scratch_r5_audit_colab`, run alongside the LB. Own structure banned from the library, sv = 0, and the truth counted as non-train. Per scorer, arm `fp1_<mode>_v1` vs base `fp1_none_v1`: ΔMRR@25 ≥ **+0.010** and truth up > down | **informational only**: it explains the LB result and does not decide it. Slice I is analogue-dense (optimistic), and full1 saw slice I (the FP arms are leaked) |
| R5.LB | per probe, with no control: KEEP iff LB ≥ **0.354** (0.346 + 0.008, the no-control rule as in R3). If several pass, keep the highest. Vote on vs off within a scorer is read at the ±0.004 noise floor | ±0.004 run-to-run noise; there is no same-day control |

### Verdicts
- **R5.0 PASS** (pssm, tani fast commits 10-08; counters in `r5_pssm_tani.md` §6).
- **R5.LB DROP ×4** (10-09): r5_pssm 0.323, r5_tani 0.322, r5_pssm_novote 0.320, r5_tani_novote 0.317. All are below
  the 0.354 bar and 0.023–0.029 below the base 0.346. pssm ≈ tani, so the cause is something both scorers share.
- Suspected cause (diagnosis 10-09, not yet measured): the D4/D5 train exclusion. Every `F_TRAIN` candidate, including
  Class-1 truths with lv < 0.75, gets the mean RRF term (about analogue rank 50–300 at k = 3), while the truth's own
  library spectrum enters the profile and promotes its closest non-train decoys. The R5.audit `an_free` exemption hides
  this. Weight 1.0 is a secondary suspect.

## R5b — train structures scored in the analogue list (pre-registered 2026-10-09, before any measurement)

One variable: `analog_exclude_train=False` (default True = R5). Every candidate gets its own analogue score, and the
PSSM q is sampled over the whole pool. Twins: `r5_{pssm,tani}{,_novote}_inclt`, each paired with its R5 notebook.

| Gate | Threshold | Why |
|---|---|---|
| R5b.0 | fast commit: `[analog]` line shows `exclude_train: False`; the R5.0 counters pass | the flag must reach the run |
| R5b.cause | per pair, `_inclt` − R5 twin ≥ **+0.008** → the exclusion caused the R5 loss. Under +0.004 on all pairs → the exclusion is not the cause, and the next probe is `analog_w` | ±0.004 noise; paired twins share all other code |
| R5b.LB | KEEP iff LB ≥ **0.354** (the R5 bar). If several pass, keep the highest | R5 has to beat the base, not just recover |

### Verdicts
- pending
