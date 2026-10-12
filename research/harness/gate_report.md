# Harness gate report (pre-registered 2026-10-10, before any measurement)

Instrument: `build/patch_harness_dump.py` + `build/casmi_post.py` (exploit worktree), runs in
`research/exploit/20261010-harness-{draft,c2,c1npx}`. Replay and metrics: `research/harness/harness.py`.
Plan: P1 of `~/.claude/plans/here-i-am-trying-wiggly-pine.md`.

## Scope and known limits (read before trusting any number)
- Cohorts: val_key **C2** (400, all truth library rows banned: Class 2 simulation) and **C1_npx** (245, timsTOF,
  only the query's own rows banned: Class 1 simulation). C2_hard and C1_xlib are not dumped yet.
- **Levels are not LB forecasts** (L07): truths stay in the pools (train structures and COCONUT), and fpnet_full1 plus
  Engine 2's models saw these structures. Only **paired deltas** between configs on the same dump count.
- The dump starts **after** ICEBERG/GLACIER scoring, so only post-scoring changes replay (C8 rerank loop, C9 gate,
  C10 fusion, C12 tail). Ranker, feature, pool and simulator-input ideas need a new val-mode run (the heavy path).
- **Leak check is a proxy until a feature-level dump exists.** Every compare also reports the delta on the molecules
  where no FPNet-driven list (PubChem f·z list, Engine 2 list) already has the truth at rank 1. If the sign flips there,
  the idea is `leak-sensitive` and only the LB decides.

## Gates
| gate | what | PASS iff |
|---|---|---|
| H0 | replay equivalence, in-run | `POST EQUIV ... mismatches 0` in the draft run and in both val runs |
| H1 | dump sanity | `VAL ban-row check ok a of b` with a/b ≥ 0.95, and every val molecule present in state.te_mids |
| H2 | back-test on C2 (primary) | the replay delta has the LB's sign in ≥ 3 of the 4 flips below |

H2 flips (each vs the unchanged base config; the expected sign comes from our LB history):
| flip | LB evidence | expected harness Δ |
|---|---|---|
| `CELL19_ICE=True` (ICE on the fused list) | b420 0.402 vs b421 0.417 | < 0 |
| `USE_ENG=False` (no two-engine RRF) | two-engine fusion +0.016 (E#39) | < 0 |
| `FUSED_GL=False` (no GLACIER on the fused list) | GLACIER +0.006–0.007 (E#15, v27) | < 0 |
| `PROMOTE=False` (no PubChem promotion) | b417 promotion ≥ +0.024 | < 0 |

A flip whose sign is wrong marks that **kind of change** as harness-advisory (only the LB decides it). If H2 fails
overall (< 3 of 4), the harness is advisory for every idea, and the next step is a feature-level dump, not ideas.

## LB-slot bar for ideas (from the grill)
Harness Δ MRR@25 ≥ +0.005 on the idea's target slice, paired-bootstrap 90% CI lower bound > 0 (2000 resamples,
seed 0), and the same sign on the leak-proxy subset.

## Results
(filled in after the runs; never edit the thresholds above)

### 2026-10-11: harness-c2 v1 (crashed) and H1
- **H0 draft**: `casmi-0433-harness-draft` completed (human report, 10-11). `POST EQUIV` line not yet read from its log: confirm on fetch.
- **H1 (C2, as worded): FAIL.** v1 log: `VAL ban-row check ok 344 of 400` = 0.86 < 0.95. The run then died at cell 3
  (`NameError: np`, the VAL cell sat above the base's numpy import; fixed, rule R07).
- **H1b (POST-HOC, added 10-11 after seeing H1; not a pre-registered gate).** Suspected cause: C2 bans the truth's rows
  for every adduct, and the mass check only passes rows at the query's own adduct. v1 also ignored `ban_keys`.
  The v2 cell bans `ban_rows` ∪ every train row with the truth IK14 (plain RDKit IK14, which matches val_key 400/400).
  PASS iff, in the v2 C2 log: `VAL key-ban leftover truth rows 0` AND `VAL ban-row misses N: other-adduct explained N`
  (every miss explained). If H1b fails, the C2 dump is not used.

## P1.5 gates: own val generator (pre-registered 2026-10-11, before any generator run; never edit these thresholds)
Shards: C1-timsTOF, C2-NP, C2-syn (~400 each), from `build/gen_val.py`; the inherited val_key C2/C1_npx is the frozen anchor.
| gate | what | PASS iff |
|---|---|---|
| G-gen1 | ban integrity | `VAL key-ban leftover truth rows 0` on every C2 shard; on C1 shards only the query's own rows are banned |
| G-gen2 | replay equivalence | `POST EQUIV ok n of n \| mismatches 0` on every shard |
| G-gen3 | covariate match | every weighted SMD vs the reference covariates < 0.1 (precursor mass, adduct, CE, peak count, entropy, NP class) |
| G-gen4 | leak removed | fpnet_0 top-1 rate on our shards < full1's top-1 rate on the same molecules |
| Combiner LB bar | learned combiner | grouped-CV Δ MRR@25 ≥ +0.005 over tuned RRF, 90% CI lower bound > 0 (grouped bootstrap, 2000, seed 0), same sign in every stratum and on the anchor shard; then one LB probe with a paired control via `decide()` |
- Combiner rank constant (pre-registered 10-11, before any real dump): k is chosen INSIDE each CV training fold from {2, 10, 60}; the held-out fold never sees the choice. Tuned RRF gets the same per-fold tuning, so the comparison is fair.

### 2026-10-11: H1b, evaluated offline (probe `syedtoubur/casmi-gen-val-probe` train_meta: ik14 per train row)
- Clause 1, key-ban leftover truth rows: **C2 0** (400 molecules); C1_npx 0. Every val_key ban row IS a truth row (0 non-truth rows in either cohort).
- Clause 2, "every mass-check miss explained by another adduct": **FAIL as worded**, 0 of 56 (C2) and 0 of 68 (C1_npx).
- **H1b = FAIL as worded** (clause 2).
- Diagnosis (post-hoc, recorded, not a gate): the H1 mass check compares each ban row to the molecule's MEDIAN precursor
  over all its spectra. 112/400 C2 and 119/245 C1_npx molecules have spectra at more than one adduct, so the median
  falls between adducts (e.g. VAL00547 [M+H]+ 337.143 / [M-H]- 335.129 → median 336.136).
- Per-spectrum version: same-adduct ban row within 0.02 Da for **810 of 810** C2 spectra and **604 of 604** C1_npx
  spectra. The ban is exact; the H1 metric was mis-specified.
- Using the C2 / C1_npx dumps despite the H1/H1b FAIL-as-worded is a human decision (logged below when made).
- **Human decision 2026-10-11:** USE the C2 / C1_npx dumps, flagged. H1/H1b stay FAIL as worded. The override rests on
  the per-spectrum evidence above (bans exact). Future val runs print the per-spectrum check.

### 2026-10-11: val runs ended in ERROR (no H0 yet)
- harness-c2 v2: base assert `Engine 2 did not cover all active molecules` (C10) after ICE/GL; early state dump written (400 molecules); no submission.csv, no POST EQUIV.
- harness-c1npx v1: base completed; final harness cell failed (`_pk` rebound to a list by a base cell; harness name collision); early state dump written (245 molecules); no POST EQUIV.
- H0 (val) remains OPEN. Plan: offline H0 for C1_npx from its early dump + submission.csv; fix and re-push C2 (handoff §4, §6).

### 2026-10-11: H0 (val), offline from the ERROR runs' early dumps
Correction to the entry above: both ERROR runs kept their /kaggle/working outputs (`kaggle kernels files` lists
nothing for an ERROR run, `kaggle kernels output --file-pattern` still fetches them). c2 HAS submission.csv: its assert
`Engine 2 did not cover all active molecules` sits in the base's C12 final-pass cell (cell 13), not C10, so c2's
submission.csv = baseline_submission.csv = the C10 output. The early state.pkl is intact in both (the c1npx crash is on
the `_pk.dump` lookup, before `open(..., 'wb')`). Missing early-dump constants (S_TAU, POP_TAU, ALPHA, KRR, CELL19_ICE)
equal casmi_post.CFG (base cells 17/19: 6.0, 5.0, 0.6, 3.0, False). Replay = worktree build/casmi_post.py, local
RDKit 2026.03.3, dreams=None.
- **H0 C1_npx: PASS.** Full replay == submission.csv **245/245**; FINAL_PASS=False == baseline_submission.csv 245/245
  (C12 changed 2 molecules, as the run's own `changed_molecules: 2`).
- **H0 C2: PASS to C10.** FINAL_PASS=False == submission.csv **400/400**. C12 has no in-run reference on C2 (it is the
  cell that asserted); the same _final_pass code is exact on C1_npx. Full replay changes 12 C2 molecules in C12.
- C2 flag: Engine 2 has no list for **VAL00724, VAL00770** (398/400); the replay leaves them unfused, as the notebook
  did. Harness deltas on C2 cover all 400; these two are reported, not dropped.
- Not re-run: both dumps serve the harness as they are. Replay speed: identity cache (dumps/<shard>/ident_cache.pkl,
  `harness.py warm`, one-time 11 min C1_npx / ~15 min C2 on 6-7 cores) takes a full replay from 18 min to 4-5 s.

### H2 back-test on C2 (FAIL: 2 of 4 signs match)

| flip | Δ MRR@25 | 90% CI | up/down | Δ without FP-oracle | sign as LB |
|---|---|---|---|---|---|
| `CELL19_ICE=True` | -0.0219 | [-0.0379, -0.0066] | 27/66 | -0.0224 | yes |
| `USE_ENG=False` | +0.2568 | [+0.2330, +0.2803] | 203/1 | +0.2627 | NO |
| `FUSED_GL=False` | -0.0674 | [-0.0868, -0.0484] | 37/90 | -0.0689 | yes |
| `PROMOTE=False` | +0.0252 | [+0.0168, +0.0348] | 26/0 | +0.0258 | NO |
- H0 draft (in-run): `POST EQUIV ok 400 of 400 | mismatches 0` (draft log, smoke run, 12 active molecules). PASS.

### 2026-10-11: H2 diagnosis (POST-HOC, not a gate; H2 stays FAIL 2 of 4)
Per-channel truth rank on the dumps (replay, unchanged config):

| dump | BASE top-1 / in list | Engine 2 top-1 / in list | final top-1 / MRR@25 |
|---|---|---|---|
| C2 (n 400) | **0.897** / 0.970 | 0.022 / **0.037** | 0.415 / 0.631 |
| C1_npx (n 245) | 0.894 / 0.996 | 0.788 / 0.992 | 0.898 / 0.933 |

- On C2 the BASE channel (full1 FPNet, trained on every train structure; fold audit) ranks the banned truth first 90%
  of the time, while Engine 2 almost never has it. Any step that mixes Engine 2 or PubChem into BASE can only demote
  the leaked answer, so `USE_ENG=False` (+0.257, 203/1) and `PROMOTE=False` (+0.025, 26/0) read as gains: both wrong
  signs are the leak. The two reranker flips (CELL19_ICE, FUSED_GL) keep the LB sign.
- The FP-oracle proxy missed it: it looks at the PubChem and Engine 2 top-1 only, never BASE (Δ without oracle ≈ Δ).
- Loss-map consequence: C2's 51% "rank 2-5" bucket is this demotion (93 of 118 rank-2 cases have a same-formula
  isomer at rank 1), not headroom we can trust.
- Per the H2 rule: the harness is **advisory for every idea** until a leak-free channel exists.

### 2026-10-11: E006 on the harness (advisory; H2 FAIL)
| dump | knobs | Δ MRR@25 | 90% CI | up/down |
|---|---|---|---|---|
| C2 | ICE_FALLBACK + METRIC_FUSION | +0.0005 | [-0.0041, +0.0054] | 4/6 |
| C2 | METRIC_FUSION | -0.0004 | [-0.0012, +0.0000] | 1/2 |
| C2 | ICE_FALLBACK | +0.0009 | [-0.0036, +0.0058] | 3/4 |
| C1_npx | ICE_FALLBACK + METRIC_FUSION | +0.0027 | [+0.0000, +0.0068] | 2/0 |
| C1_npx | METRIC_FUSION | +0.0020 | [+0.0000, +0.0061] | 1/0 |
| C1_npx | ICE_FALLBACK | +0.0007 | [+0.0000, +0.0020] | 1/0 |

LB (own account): E006 0.440 vs control-0433 0.433 (+0.007, one run each). The harness sees E006 as neutral on C2 and
small-positive on C1_npx: same sign as the LB on C1_npx, magnitude below the +0.005 LB-slot bar on both.

### 2026-10-11: H2 answer key was mis-cited (LB-history audit; recorded before the back-test suite runs)
- `CELL19_ICE`: the 0.417 is b416, not b421 (b421 = zenith 0.425), and b420 → b421 also turned BASE_ICE off: a bundle.
- `USE_ENG`: "E#39" is the zenith row; the real evidence is row 18 (v4g, old pipeline).
- `FUSED_GL`: "v27 +0.006" is the upstream GLACIER [M+H]+ input filter, not a fused-list switch.
- `PROMOTE`: "b417 ≥ +0.024" is inferred; no b417 LB score exists.
- So H2 (FAIL 2/4) tested against an invalid key. It stays logged as run; the suite below replaces it.

### 2026-10-11: back-test suite + capability map, PRE-REGISTERED (grill 2026-10-11; written before any suite run)
Spec: `research/harness/backtests.yaml` (append-only). Command: `harness.py suite --dump C2 --dump C1_npx`.
Evidence tiers: **S** = sure, own account, paired (E006 vs control-0433). **A** = same-author single runs isolating
the change (sign only). **B** = bundle replayed whole (advisory). Old-pipeline (v4g/v4n) and inferred numbers excluded.

| flip | family | tier | LB key |
|---|---|---|---|
| `ICE_FALLBACK=True,METRIC_FUSION=True` | e006_overlay | S | +0.007 |
| `ICE_FALLBACK=True` | ice_fallback | A | +0.004 |
| `FINAL_PASS=False` | tail | A | −0.008 (V1.1 → V1; also pinned RDKit) |
| `CELL19_ICE=True,BASE_ICE=True` | reranker | B | −0.015 |
| `USE_ENG=False`, `PROMOTE=False`, `FUSED_GL=False`, `CELL19_ICE=True` | fusion / promotion / reranker | none | behaviour rows |

Sweeps: ICE_LAM, GL_LAM ∈ {0.25, 0.5, 1.0, 1.5, 2.0} are evidence (LB-tuned at 1.0; PASS iff the harness argmax is
within one grid step of 1.0 on every dump). ALPHA, KRR, LIB_TAU, REL_TH, SLOTS_GENTLE, S_TAU, POP_TAU, REL_PROMOTE:
sensitivity only (max |Δ| reported, no verdict). Mechanical sanity (PASS iff all three): identity Δ == 0 exactly;
oracle (truth inserted at rank 1) Δ == 1 − MRR within 1e-9; seeded shuffle of each final list Δ < 0.

Per-flip sign status: **match** = every dump has the LB sign; **mismatch** = every dump has the opposite sign;
**regime-dependent** = dumps disagree; **zero** = Δ == 0 on some dump. Until C2_np exists the dumps are C2 + C1_npx
(agreement rule); then C2_np is primary and a flip's status is read on C2_np alone (C2/C1_npx reported).

Family verdict (keyed flips = tiers S/A/B; S/A = evidence flips):
1. **BLIND** if keyed flips with status mismatch outnumber those with status match.
2. **TRUSTED** if ≥ 3 S/A flips, all match, and every LB-null flip's 90% CI covers 0.
3. **TRUSTED-PROVISIONAL** if ≥ 1 S/A flip matches, no S/A flip is mismatch or regime-dependent, the λ sweep PASSES and
   mechanical sanity PASSES.
4. Otherwise **ADVISORY** (the harness ranks ideas, the LB decides). Families with no keyed flip are ADVISORY.
Diagnostic (no verdict): every flip's Δ on C2 split by BASE-top-1-is-truth; channel table (BASE / Engine 2 / PubChem
truth top-1 and in-list rates). Use: `compare --json` carries the family + verdict; BLIND → `lb_slot` false, and the
campaign passes such an idea only with a written prior (LB decides). Preflight R08 enforces it.

### 2026-10-11: Option A0, PRE-REGISTERED (fpnet_0 holdout level on enveda-np-examples; written before the A0 run)
Instrument: Kaggle CPU kernel `syedtoubur/casmi-a0-np-holdout` = worktree `build/fold_audit.py` `run_a0` (tests
`build/test_fold_audit.py`). Query sets: **np** = every enveda-np-examples train spectrum; **other** = the same 250
structures' spectra in OTHER libraries (≤ 4 per structure, distinct libraries first, [M+H]+ preferred); **control** =
150 library-unique non-np structures drawn from the other set's libraries in proportion. Each spectrum's truth ranked
by f·z in the ±10 ppm pool window (fold-audit method; truth appended if absent). Models: fpnet_0, full1 (reference),
fe_A/fe_B/fe_D (MANIFEST fe_models), v3 fpnet_0/1, prvsiyan fp_merged_m1 / fp_single_s2 (scored on their own 6,930
bits with candidates re-fingerprinted by Engine 2's fp_and_mass recipe; a model that cannot be scored is reported as
such, not forced).
- **Rule** on r = MRR@25(fpnet_0) / MRR@25(full1) over **other**: r ≤ 0.8 → structures held out → build **C2_np**;
  r ≥ 0.9 → only spectra held out → build **C1_np_fp0**; 0.8 < r < 0.9 → **STOP**, the human decides.
- **Residual leak** (reported, not blocking): any model other than fpnet_0 with MRR ratio ≥ 0.9 × full1 on **np**.
- Controls are a sanity row: fpnet_0/full1 on control expected ≈ 1 (fold audit 0.96–1.09); reported, no verdict.

### 2026-10-11 16:04: back-test suite result (C2, C1_npx)
| family | verdict | flips | knobs |
|---|---|---|---|
| e006_overlay | **ADVISORY** | e006 | METRIC_FUSION, FUSION_WINDOW |
| ice_fallback | **ADVISORY** | ice_fallback | ICE_FALLBACK |
| tail | **ADVISORY** | final_pass_off | FINAL_PASS |
| reranker | **ADVISORY** | cell19_base_ice, fused_gl_off, cell19_ice | CELL19_ICE, BASE_ICE, FUSED_GL, PC_SIM_RERANK, ICE_LAM, GL_LAM |
| fusion | **ADVISORY** | use_eng_off | USE_ENG, ALPHA, KRR, FUSE_N |
| promotion | **ADVISORY** | promote_off | PROMOTE, S_TAU, POP_TAU, REL_PROMOTE |
| merge_gate | **ADVISORY** | - | LIB_TAU, REL_TH, SLOTS_AGG, SLOTS_GENTLE, ICE_PC, TOPN |

Global: λ sweep FAIL, mechanical sanity PASS.

Full tables: research/harness/capability_map.md

### 2026-10-11: Option A0 result: fpnet_0 held out the np STRUCTURES → build C2_np
Kernel `syedtoubur/casmi-a0-np-holdout` v2 (COMPLETE, 1,493 s; log + csv in `.scratch/valgen/a0_v2/`). MRR@25 ratio
vs full1 per query set (np n 1,184 spectra; other n 996 = 250 structures × ≤ 4 other-library spectra; control n 150):

| model | np | other | control |
|---|---|---|---|
| full1 (ref; MRR np/other/control) | 0.916 | 0.896 | 0.904 |
| **fpnet_0** | 0.631 | **0.634** | 0.953 |
| fe_A | 0.631 | 0.634 | 0.953 |
| fe_B | 0.627 | 0.646 | 0.944 |
| fe_D | 0.652 | 0.647 | 0.810 |
| v3 fpnet_0 | 0.631 | 0.634 | 0.953 |
| v3 fpnet_1 | 0.627 | 0.646 | 0.944 |
| prvsiyan fp_merged_m1 | 0.453 | 0.582 | 0.824 |
| prvsiyan fp_single_s2 | 0.451 | 0.893 | 1.016 |

- **Rule: r(other) = 0.634 ≤ 0.8 → structures held out → C2_np.** fpnet_0 top-1 on np spectra = 0.427.
- Residual leak (≥ 0.9 × full1 on np): **none** (max fe_D 0.652). fe_A and v3 fpnet_0 score identically to fpnet_0
  on every row (same weights, it seems); fe_B ≡ v3 fpnet_1. So every A-family channel except full1 is np-clean.
- prvsiyan (6,930 bits, scored on its own bits; f·z proxy, not Engine 2's GBM): fp_single_s2 is np-clean on np spectra
  (0.451) but near full1 on the same structures' other-library spectra (0.893): it held out np **spectra only**, so
  Engine 2's FP features may know these structures. Reported, not blocking (prvsiyan ranker unchanged per decision).
- Engine 2 ranker rows (`sim_rank_rows_nofp.npz`, 399,790 rows): query-set field **`Q`** (int8) indexes **`sets`** =
  `['np', 'gnps', 'riken', 'mona', 'msdial', 'massbank', 'pluskal_ms2']`; np = code 0 = **43,864 rows**.
  `rank_train.npz` (prvsiyan) has no query-set field (X, Y, M, G only).

### 2026-10-11: Option A C2_np GPU run, PRE-REGISTERED (written before the push)
Kernel `syedtoubur/casmi-0433-harness-c2np` (idea `research/exploit/20261011-harness-c2np`, kind instrument). C2_np =
the 245 C1_npx molecules, `ban_keys=[inchikey14]`, cls 2; BASE bank = fpnet_0; Engine 2 "our ranker" fitted without
the Q == np rows. Base cells byte-identical (G1); everything else in harness-only cells.
- **H0**: log prints `POST EQUIV ok 245 of 245 | mismatches 0`, and offline replay == submission.csv 245/245.
- **Ban counters**: `VAL ban extra rows by key` = 54,660 ± 2% (the other-library rows of the 245 truths, plus any
  np rows of them outside ban_rows); `VAL key-ban leftover truth rows 0` exactly.
- **Swap / filter counters**: `HARNESS fpnet_0 swaps 1`; `HARNESS ranker rows dropped 43864 of 399790`.
- **Coverage guard**: `HARNESS ENG fill n` with n ≤ 12 (5% of 245); the run asserts below 95%.
- **BASE top-1 on C2_np ≤ 0.527** (= fpnet_0 np top-1 0.427 + 0.10; C2 was 0.897). Above that bar = the BASE channel
  still leaks on C2_np → the shard is not a leak-free primary; stop and report.
- Reported without a bar: Engine 2 rank-given-reachable (truth rank among molecules where Engine 2's list holds it)
  on C2_np vs C2; BASE / Engine 2 / PubChem truth top-1 and in-list rates; every back-test flip with USE_ENG on and off.

### 2026-10-11: Option A C2_np result: FAIL on the BASE top-1 bar → the shard is NOT a leak-free primary (stopped)
Kernel `syedtoubur/casmi-0433-harness-c2np` COMPLETE; outputs in `research/harness/dumps/C2_np/` (cache warmed 44,612
entries); offline checks `.scratch/valgen/c2np_analysis.py C2_np --c2` → `dumps/C2_np/option_a_checks.json`.

| gate (pre-registered) | measured | verdict |
|---|---|---|
| POST EQUIV (log) | ok 245 of 245, mismatches 0 | PASS |
| offline replay == submission.csv | 245/245, 0 mismatches | PASS |
| VAL ban extra rows by key = 54,660 ± 2% | 54,660 | PASS |
| VAL key-ban leftover truth rows = 0 | 0 | PASS |
| HARNESS fpnet_0 swaps 1 | 1 | PASS |
| HARNESS ranker rows dropped 43864 of 399790 | 43,864 of 399,790 | PASS |
| HARNESS ENG fill n ≤ 12 | 0 | PASS |
| **BASE top-1 on C2_np ≤ 0.527** | **0.947** (in-list 0.996; C2 0.897 / 0.970) | **FAIL** |

Per the pre-registered rule the BASE channel still leaks on C2_np, so C2_np is not used as the primary dump and the
suite is NOT rerun with it. Reported without a bar (C2_np vs C2): Engine 2 top-1 0.041 / 0.023, in-list 0.073 /
0.038; Engine 2 rank-given-reachable n 18 / 15, top-1 0.56 / 0.60, MRR 0.70 / 0.70; PubChem 0 / 0; final MRR 0.596 /
0.631. lib_max median 0.55 (C1_npx 0.84): the library ban works.

**What this falsifies:** the H2 diagnosis "BASE leaks because full1 saw every train structure". BASE with fpnet_0
(np-structure-clean per A0, f·z top-1 0.427 on these spectra) is *higher* (0.947), so the leak is not the FPNet.
**Leading hypothesis (POST-HOC, not verified):** the v1 engine's pool artefacts are symlinked prebuilt files
(`.scratch/nb1_433.py:192`: `pool_meta.parquet`, `pool_fp.npy`, `train_structs.parquet`, `train_fp_sel.npy`, …) built
from the FULL train set; only the spectrum cache is rebuilt from the banned train.parquet (:196). So the engine / its
4-booster ranker can still see that the truth is a train structure (a train-membership or nearest-train-structure
feature, or pool popularity), on C2 and C2_np alike. Next diagnosis: list `RK['features']` and the v1 engine features
that read `train_structs` / `tfp`, and rerun with those artefacts rebuilt from the banned train set.

Flips on C2_np (Δ MRR@25, USE_ENG on | off), reported only: e006 −0.0003 | 0; ice_fallback −0.0003 | 0;
final_pass_off 0 | 0; cell19_base_ice −0.0044 | −0.0267; use_eng_off +0.308; promote_off +0.029 | +0.061;
fused_gl_off −0.090 | 0; cell19_ice +0.033 | 0.

### 2026-10-12: C2_np BASE-leak diagnosis (CPU, no new run; diagnostic only, no gate)
Probes `.scratch/valgen/c2np_leak_probe{1,3,4}.py` on the existing dumps; pool files `pool_meta.parquet`
(ahmedberatozer/casmi26-v2-pool) + `pool_lsid/lpmid.npy` (dmitriigluzdov/casmi26-pubchem-popularity-prior) in
`C:/casmi/dl_pool`. Outputs `dumps/<shard>/leak_probe{1,3}.csv`.

| hypothesis | test | result | verdict |
|---|---|---|---|
| prebuilt-artefact misalignment (previous block's lead) | read `casmi/build.py`, `library.py`, `engine.py` | sids of the rebuilt spectrum cache come from the prebuilt `train_structs` via ik14, so lib / analog / train_fp stay aligned; truth spectra gone → `lib_*` can't hit the truth; fe families document "no pool src / train_sid" | falsified as a carrier |
| H-taut: truth variants survive the native-ik14 ban | P1: train rows with RDKit ik14 == truth but native ik14 ≠ truth; P2: BASE top-1 SMILES plain ik14 | 0 rows; 99.1% of hits are the exact truth structure (2 variants) | falsified |
| H-small: tiny pool window | P2/P3: list and window size | window median 58 (C2 42); hit rate 0.94 even with 30–60 pool entries | falsified |
| FP banks (model_views) | A0 table | fe_A/B/D, fpnet_0 top-1 0.42–0.44 on np vs full1 0.88 | not a carrier |
| H-pop: v19 popularity re-rank | P3/P4: pop of truth, rerank with the 0.15·POOL_POP term removed (exact) | truth pop median 13.0 vs window 2.4; pop alone top-1 0.689 (C2 0.216); BASE top-1 MU 0 = **0.886** (as run 0.947) | partial carrier (+0.061) |
| H-rk: the v4b ranker (and fe tables) were fit on sims whose queries include the `np` fold | circumstantial: `casmi/traindata.py` holdout folds (np, nplib, syn, plusk, twin) are the FPNet-unseen folds = the ranker's query pool; fragnet decision log reports ranker-CV per fold incl. "np ALL" over 18,465 groups | ranker-only top-1 C2_np 0.886 > C1_npx 0.747: hiding the truth's library spectra *raises* accuracy, i.e. C2_np reproduces the ranker's training condition for these exact structures | **leading, unverified** (needs per-candidate features or a ranker not fit on np) |

Ranker-only (MU 0) / as run / popularity only, BASE top-1: C2_np 0.886 / 0.947 / 0.727; C2 0.905 / 0.897 / 0.250;
C1_npx 0.747 / 0.894 / 0.731. The pre-registered C2_np FAIL stands. The 0.527 bar assumed BASE ≈ fpnet_0 + margin; it
did not model the popularity prior or the ranker's query set.

### 2026-10-12: Option 2 BASE-coupling rule (PRE-REGISTERED, before the run)
Decision (user, after the C2_np diagnosis): BASE can't be measured on any train-derived val shard (each structure was
seen by the FPNet or, most likely, by the v4b ranker). The harness keeps judging only families whose effect does not
run through BASE's leaked rank 1.
- **coupling(flip or sweep point, dump)** = share of leak-hit molecules (BASE rank 1 == truth) whose final truth rank
  differs from the base config. **coupling(family)** = max over the family's flips, the sweep points of its knobs, and
  the dumps.
- **coupling(family) > 0.05 → BASE-COUPLED** (harness evidence not citable; the LB decides). Handled exactly like BLIND:
  `compare()` lb_slot False; R08 and campaign `harness_gate` pass only with prior.md; capmap_verdict orders it next to
  BLIND. Otherwise the family keeps the verdict from the 2026-10-11 suite rules (unchanged).
- Dumps for verdicts: C2 + C1_npx (as pre-registered 2026-10-11). The `--eng-off` run is reported alongside. C2_np is
  not used (it failed its BASE bar).

### 2026-10-12 10:57: back-test suite result (C2, C1_npx, Engine 2 off)
| family | verdict | coupling | flips | knobs |
|---|---|---|---|---|
| e006_overlay | **ADVISORY** | 0.000 | e006 | METRIC_FUSION, FUSION_WINDOW |
| ice_fallback | **ADVISORY** | 0.000 | ice_fallback | ICE_FALLBACK |
| tail | **ADVISORY** | 0.000 | final_pass_off | FINAL_PASS |
| reranker | **BASE-COUPLED** | 0.064 | cell19_base_ice, fused_gl_off, cell19_ice | CELL19_ICE, BASE_ICE, FUSED_GL, PC_SIM_RERANK, ICE_LAM, GL_LAM |
| fusion | **ADVISORY** | 0.000 | - | USE_ENG, ALPHA, KRR, FUSE_N |
| promotion | **BASE-COUPLED** | 0.196 | promote_off | PROMOTE, S_TAU, POP_TAU, REL_PROMOTE |
| merge_gate | **BASE-COUPLED** | 0.073 | - | LIB_TAU, REL_TH, SLOTS_AGG, SLOTS_GENTLE, ICE_PC, TOPN |

Global: λ sweep FAIL, mechanical sanity PASS.

Full tables: research/harness/capability_map.md

### 2026-10-12 11:06: back-test suite result (C2, C1_npx)
| family | verdict | coupling | flips | knobs |
|---|---|---|---|---|
| e006_overlay | **ADVISORY** | 0.019 | e006 | METRIC_FUSION, FUSION_WINDOW |
| ice_fallback | **ADVISORY** | 0.014 | ice_fallback | ICE_FALLBACK |
| tail | **ADVISORY** | 0.003 | final_pass_off | FINAL_PASS |
| reranker | **BASE-COUPLED** | 0.323 | cell19_base_ice, fused_gl_off, cell19_ice | CELL19_ICE, BASE_ICE, FUSED_GL, PC_SIM_RERANK, ICE_LAM, GL_LAM |
| fusion | **BASE-COUPLED** | 0.563 | use_eng_off | USE_ENG, ALPHA, KRR, FUSE_N |
| promotion | **BASE-COUPLED** | 0.156 | promote_off | PROMOTE, S_TAU, POP_TAU, REL_PROMOTE |
| merge_gate | **BASE-COUPLED** | 0.267 | - | LIB_TAU, REL_TH, SLOTS_AGG, SLOTS_GENTLE, ICE_PC, TOPN |

Global: λ sweep FAIL, mechanical sanity PASS.

Full tables: research/harness/capability_map.md
