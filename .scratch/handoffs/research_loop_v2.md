# Handoff: research loop v2 + replay harness + own val generator (as of 2026-10-11, end of session 2)

Self-contained: a fresh session can start from this file alone. Everything is **uncommitted** (main checkout branch
`specs-scratch-r0`, exploit worktree `E:/casmi/lines/exploit` branch `line/exploit`). Read CLAUDE.md first (repo
rules: never read raw .ipynb, use `/analyze-notebook`; no .parquet/.npz/.pt in git; agents push Kaggle kernels, only
the human submits).

Plans of record: grill 1 `~/.claude/plans/here-i-am-trying-wiggly-pine.md`, grill 2/3 `~/.claude/plans/i-am-working-with-reflective-moonbeam.md`.

---

## 0. START HERE: state in one screen

- **SOTA / paired control: 0.433** (own-account `casmi-control-0433`, unchanged `casmi-notebook-0-433.ipynb`). The
  adduct-iceberg probe (0.432) is **reverted** by `poll_lb.decide()`.
- **Unverified external claim: E006 = 0.440** (`casmi26-e006-window-40-ice-fallback-true.ipynb`; analysis in
  `.scratch/handoffs/comparison_sotas.md`). Its knobs are already in `casmi_post.py` (off by default). A concurrent agent
  built a separate "smart gate" on top of E006 in `.scratch/smart_gate/` (memory `casmi-smart-gate-scratch`, outside
  this workflow; it needs an E006 paired control).
- **P0 done. P2 built. P1 is blocked**: both val-mode harness kernels ended in **ERROR** on 10-11, for two different
  bugs (§4). Each wrote the early state dump, but neither printed `POST EQUIV`.
- **P1.5 (own val generator) is DEFERRED** behind campaign 1 (grill 3). The train probe and fold audit are done, and
  their findings voided two grill-2 decisions (§2.3).
- **The very next actions** are in §6.

---

## 1. Goal and context (why this exists)

The `research/` loop (10-10 version) decided wrongly: adduct-iceberg-rerank was declared "SOTA 0.432" with no scored
control and a G4 log saying `molecules reranked 0`. Spikes assumed their gains, and nobody knew where the base loses
MRR. The goal is a loop where:
- Opus reasons over measured headroom
- ideas die cheaply on an **offline replay harness** before they cost code or LB slots
- KEEP/REVERT lives in code
- every new mistake becomes an enforced rule with a fixture

It must stay lean on a **Claude Pro** quota.

Competition: MRR@25 on InChIKey14 (first 14 characters); the test is 100% timsTOF with a +1.45 ppm offset. The LB
noise floor is ±0.004.

---

## 2. Settled decisions

### 2.1 Grill 1 (10-10): loop design
| Topic | Decision |
|---|---|
| Roles | Opus thinks (grill, replan, spec, attack). Sonnet builds (stress tests, code, G1–G4). Gemini via `agy -p` (C:/Users/h/AppData/Local/agy/bin/agy, `--output-format json`) fetches: scouting, running scripts, digests. Gemini critic dropped. |
| Lines | Explore frozen. **Exploit only** (0.433 base). |
| Instrument | Replay harness first: a Kaggle val-mode kernel dumps post-scoring state; CPU replay through `casmi_post.py`, which is shared with the notebook. |
| Cohorts | val_key C2 (all truth library rows banned; Class 2 sim) + C1_npx (only the query's own rows banned; Class 1 sim). Reported per slice. |
| Leakage | Accepted with guards: every harness delta is reported with the FPNet channel on and off (FP-oracle proxy); a sign flip = `leak-sensitive`, and only the LB decides. One-time back-test H2. |
| LB gate | An idea gets an LB slot only if harness Δ ≥ +0.005, the paired-bootstrap 90% CI lower bound > 0, and the sign survives the FPNet ablation. |
| Upstream ideas | Heavy path: one val-mode Kaggle run, max 2 per campaign. |
| Cadence | One grill → a campaign of 3–5 autonomous cycles with an adaptive Opus replan. Stops: 5 cycles / 4 LB probes / 2 harness deaths in a row on one branch / quota pause > 12 h. |
| Critic | `preflight.py` (static rules) + Opus `attack.md` (3–6 attempts to break the idea, each with a pre-registered death condition) → Sonnet `stress_test.py`. |
| Knowledge | Each checkable lesson becomes a rule `research/rules/R*.py` + fixture; `--selftest` proves it. LESSONS keeps judgment lessons only. |
| Decisions | `decide()`: a scored paired control, LB ≥ control + 0.004, counter **values** > 0 → `keep-pending` → one replicate → `keep`. |
| Context | campaign.py writes `brief_<step>.md` ≤ ~1.5k tokens; steps never read README / LESSONS / EXPERIMENTS in full. |
| Driver | `research/bin/campaign.py`: a deterministic state machine calling `claude -p` / `agy -p`, one fresh context per step; resumable. |

### 2.2 Grill 2 (10-11): own val generator. Rows marked ✗ were voided later (§2.3)
| Topic | Decision |
|---|---|
| H1 gate | H1 FAIL logged as worded; **H1b** added post-hoc. Both later FAIL as worded; the **human decided to USE the dumps, flagged** (§5). |
| Goal | Primary: a fast offline attack (replay). A learned combiner is a second use of the same dumps. |
| N sets | Confidence comes from the **number of molecules**, not of batches: a few **regime shards**, with folds and bootstrap cut offline. |
| Instrument vs chemistry | Draw from all libraries; **importance-weight to `enveda-np-examples`** (the user confirmed it as the reference distribution: the hidden test is never observable, and the visible test.parquet is a placeholder train slice); +1.45 ppm shift on non-timsTOF precursors; the sign must hold per stratum. |
| Dropout | Random per-feature corruption / −∞: dropped. **Channel dropout** (a whole channel masked) is a regulariser only; it does NOT fix the leak (a leaked channel is overconfident, not absent). |
| ✗ Leak fix | ~~fpnet_0 as an unseen FPNet for np/nplib/syn/plusk~~ → voided by the fold audit. |
| Combiner order | Tune RRF/channel weights → small learned combiner (≤ 20 params) → full 31-feature GBDT only with unseen features. |
| Compute | Dumps on Kaggle (~10 GPU-h/week); Colab (~170 CU) for training. **The Colab CLI Drive mount fails (Google 400)**, so Kaggle CPU kernels are used for data jobs. |
| Regimes | C1 + C2 now; C3 (pool ban) later. |
| ✗ Eligibility | ~~Strict: only structures in fpnet_0 held-out libraries~~ → voided. |
| Class mix | Estimated from `enveda-np-examples`; deltas are always reported per class. |
| Anchor | The inherited val_key C2 / C1_npx dumps = a **frozen anchor shard**. |
| Campaign 1 | Runs on the anchor dumps, RRF/weight ideas only. First candidate: E006 (metric fusion + ICE fallback). |

### 2.3 Grill 3 (10-11, after the fold audit): replaces the voided rows
| Topic | Decision |
|---|---|
| Leak fix v2 | **No learned combiner on leaked data.** Ideas stay A/B with the FP-oracle proxy, PLUS an **exact leak check on the 250 np-example molecules (= C1_npx anchor)**: swap full1's f·z for fpnet_0's (fpnet_0 never saw these 250 structures, same feature format) and require the idea's sign to hold. RRF weight tuning (few params) is allowed if it passes there. K-fold FPNet retrain is parked (no training loop exists in `.scratch/casmi_v4`). |
| gen_val.py | **Deferred behind campaign 1.** Build regime shards only if campaign 1's CIs are too wide to decide. |

---

## 3. Phase status

### P0: decision integrity — DONE (10-10)
`poll_lb.decide()` + `counters_fired()`; watcher counter-**value** rule (`?regex` = may be 0); research/bin/test_bin.py 49/49.
control-0433 scored 0.433 → STATUS `scored`; adduct → `revert` (10-11, `poll_lb.py --write`).

### P1: harness — BLOCKED (val runs failed; §4)
Built and working:
- Worktree `build/casmi_post.py`: replay of C8 loop / C9 gated PubChem merge / C10 two-engine RRF + GL / C12 final pass.
  - Knobs: USE_ENG, FUSED_GL, PROMOTE, PC_SIM_RERANK, FINAL_PASS, CELL19_ICE, BASE_ICE, and E006's ICE_FALLBACK,
    METRIC_FUSION, FUSION_WINDOW (all default to the base). Since 10-11 the C12 loop is `_final_pass()`.
- Worktree `build/patch_harness_dump.py`: generates `casmi-0433-harness{,-c2,-c1npx}.ipynb` from the base by adding three things:
  - a VAL cell after `# SMOKE switch`
  - a state snapshot before the C8 loop (early `post_dump/state.pkl`)
  - a final cell: `POST EQUIV ok n of n | mismatches 0` + the full dump
  - Never hand-edit the generated notebooks; change the script and regenerate (`python build/patch_harness_dump.py --shard C2|C1_npx` / `--shard=`).
- Worktree `build/test_patch_harness_dump.py`: 12/12 (incl. `test_inserted_cells_bind_their_names` and 5 E006 knob tests).
- `research/harness/harness.py`: `fetch <idea>`, `compare --dump C2 --a "" --b "K=V,..."`, `lossmap --dump C2 --dump C1_npx`, `backtest`.
  - The replay uses the WORKTREE casmi_post.py by default (`--post` overrides).
- `research/harness/metrics.py`: mrr_at25, grouped paired_bootstrap, stratum_signs, grouped_kfold, importance_weights, smd.
- `research/harness/combiner.py`: rrf, tuned_rrf, Combiner with channel dropout, cv_report.
- Tests: `research/harness/test_metrics.py` + `test_harness.py` green.
  - Synthetic result: the combiner beats uniform RRF (+0.075) but not tuned RRF (−0.016, CI crosses 0).
- Gates H0/H1/H2 (+ H1b, P1.5 gates, combiner k rule) pre-registered in `research/harness/gate_report.md`.

Open: H0 on a val run, the loss map, H2 back-test, the first E006 compare.

### P1.5: own val generator — DEFERRED (grill 3). Done so far:
- **Train probe** (Kaggle CPU kernel `syedtoubur/casmi-gen-val-probe`, code `build/gen_val_probe.py`) → `.scratch/valgen/train_meta.parquet`:
  - one row per train spectrum (2,539,608 rows)
  - columns row, lib, instrument_type, adduct, ionization_mode, precursor_mz, n_peaks, ce_n, ik14, ik14_rdkit, scaffold
- **Fold audit** (CPU kernel `syedtoubur/casmi-fold-audit`, `build/fold_audit.py` + `test_fold_audit.py` 48/48) →
  `.scratch/valgen/fold_audit.csv` and `fold_audit_queries.csv` (§5).
- Pre-registered P1.5 gates G-gen1..4 + combiner LB bar are in gate_report.md. G-gen3/G-gen4 assumed things the audit
  disproved: re-read them before any generator run.
- Not built: `build/gen_val.py`, fpnet_0 f·z in the dump, the shard upload.

### P2: loop — BUILT (10-10)
- `research/bin/{campaign.py, brief.py, preflight.py}` + `test_campaign.py` (dry run green).
- Rules R01–R07 + fixtures: `python research/bin/preflight.py --selftest` 7/7.
  - R07 is new on 10-11: the `research/rules/_nbnames.py` static check that a notebook cell reads no name before some
    cell binds it.
- Skills: `.claude/skills/{campaign-grill, replan, spec, attack, build}`, `.agents/skills/scout`; retired ones in
  `.agents/skills/_retired/`.
- README v2, CLAUDE.md §4, config.yaml (`campaign`, `harness`, `claude.models`, `kaggle.watch_timeout_s: 3600`).

Not done:
- Gemini digest from a code-generated status JSON
- BASE facts generator with hash check
- feature-level wide dump

### P3: first campaign — NOT STARTED
Needs P1 (loss_map.md, H2) first. Run `/campaign-grill` with loss_map.md (+ scout.md), then
`python research/bin/campaign.py research/campaigns/<id>`. Scope: anchor dumps, RRF/weight ideas, E006 first, with the
np-examples fpnet_0 leak check.

---

## 4. Why the two val kernels failed (10-11) and how to fix them

Logs: `.scratch/valgen/logs/harness-c2-v2.log`, `harness-c1npx-v1.log`. Both completed ICE + GLACIER scoring (c2
~2.9 h, c1npx ~2.1 h). Both printed the early `POST DUMP /kaggle/working/post_dump molecules 400|245`, so
`post_dump/state.pkl` exists in each kernel's output. Neither reached `POST EQUIV`.

1. **c1npx (`casmi-0433-harness-c1npx` v1): harness bug, name collision.**
   - The final cell's `_harness_dump` calls `_pk.dump(...)`, but the base notebook rebinds `_pk` to a list after the
     snapshot (its merge self-test does `_b, _bk, _p, _pk = ...`).
   - Error: `AttributeError: 'list' object has no attribute 'dump'`.
   - The base itself finished, so `submission.csv` should be in the output.
   - Fix in `patch_harness_dump.py` SNAP: import pickle/copy/shutil under collision-proof names (e.g. `_hx_pickle`), or
     re-import inside `_harness_dump`.
   - Extend R07 / `_nbnames` with a test that SNAP/FINAL helper names are never re-bound by base cells (a "rebound
     after definition" check).
2. **c2 (`casmi-0433-harness-c2` v2): the base's own assert.**
   - `AssertionError: Engine 2 did not cover all active molecules` (C10, before submission.csv).
   - Engine 2 (the BIO+AFIX two-ranker subprocess) skipped some of the 400 C2 molecules in val mode.
   - C1_npx (245) passed the same assert. Diagnose from the c2 log: which molecules Engine 2 missed and why. Hypotheses:
     - an adduct or polarity Engine 2 doesn't handle
     - molecules with no candidates after the library ban
     - Engine 2 reading its own smoke/rerun switch rather than the harness's `IS_RERUN = True`
   - The fix belongs in the VAL cell or a harness-only guard; never change base cells (G1 = base cells byte-identical).
3. **Key-ban line.** Train has no `smiles` column, so c2 v2 printed `VAL key-ban skipped`. That's already fixed:
   the cell now uses train's native `inchikey14` column (smoke `.scratch/valgen/smoke_val_cell.py` green, notebooks
   regenerated, NOT pushed). H1b was settled offline anyway (§5).

Recovery options, cheapest first:
- **c1npx offline H0 (no GPU).**
  - `harness.py fetch research/exploit/20261010-harness-c1npx` (downloads `post_dump/` only; also fetch submission.csv).
  - Replay the early state.pkl with the final cell's constants. The early dump lacks S_TAU, POP_TAU, ALPHA, KRR,
    CELL19_ICE; take their values from the base notebook via `/analyze-notebook`.
  - Compare against that run's submission.csv. Exact equality = H0 for C1_npx.
- **c2:** after the Engine 2 fix and the `_pk` fix, regenerate and re-push (one T4 session, ~3 h).

Push command pattern (two GPU sessions max; watchers give up after 60 min with `g4-timeout`, so finalize later with
`python research/bin/kaggle_push.py --check`):

```
python research/bin/kaggle_push.py research/exploit/20261010-harness-c2 "E:\casmi\lines\exploit\casmi-0433-harness-c2.ipynb" --preset harness_0433_preset --slug casmi-0433-harness-c2 --counter "POST EQUIV ok" --counter "VAL ban-row check ok" --counter "?VAL key-ban leftover truth rows"
```

---

## 5. Facts established this session (trust these over older docs)

- **Train schema**: 2,539,608 rows, 277,566 structures.
  - Columns: ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type,
    adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks,
    base_peak_intensity, collision_energy_ev / _orig / _orig_units.
  - `inchikey14` == plain RDKit IK14 on 100% of rows; val_key uses plain IK14 too (400/400).
- **Libraries** (rows / structures):

  | library | rows | structures | notes |
  |---|---|---|---|
  | enveda-180 | 1.15M | 183k | all timsTOF, drug-like synthetic |
  | pluskal_ms2 | 528k | 47k | Orbitrap |
  | riken | 347k | 16k | |
  | gnps | 221k | 46k | |
  | massbank | 102k | 9k | |
  | mona | 92k | 12k | |
  | spectraverse | 51k | 10k | |
  | msdial | 41k | 9k | |
  | drug_plus | 2.5k | 2.5k | |
  | masaryk | 0.7k | 0.4k | |
  | enveda-np-examples | 1,184 | 250 | timsTOF; all 250 structures also in other libraries; = the C1_npx anchor cohort |

  28,113 structures sit in more than one library.
- **Fold audit**: fpnet_0 vs full1 MRR@25, 150 library-unique structures per library, rule pre-registered (yes if
  ≥ 0.8×, no if ≤ 0.5×).
  - Every library has a ratio of 0.96–1.09 → trained_on = yes.
  - **enveda-np-examples: 0.622 vs 0.933 (ratio 0.67)**. fpnet_0 held out ONLY the 250 np-example structures.
  - `specs/Dataset_intro/fpnet_v4_model_code_intro.md` §4.4–4.5 (np/nplib/syn/plusk holdout) is **wrong**.
  - full1 saw everything, so every train-derived val molecule is FPNet-leaked except those 250 (for fpnet_0).
- **H1 / H1b**:
  - The bans are exact: 0 leftover truth rows, and 0 banned rows that aren't truth, in C2 and C1_npx.
  - Per spectrum, a same-adduct banned truth row within 0.02 Da exists for 810/810 C2 and 604/604 C1_npx spectra.
  - The H1 metric was mis-specified: it used the median precursor of molecules with spectra at two or more adducts
    (112/400 C2).
  - H1/H1b stay **FAIL as worded**. Human decision 10-11: **use the dumps, flagged**. All of it is logged in
    gate_report.md.
- **E006** (agent-verified from the notebook):
  - `FUSION_WINDOW=40` is a no-op (the base already caps at FUSE_N=40).
  - Metric fusion is a post-C12 overlay that rebuilds rows from the C9 lists (one vote per IK14 per engine).
  - The ICE fallback (ICEBERG rerank when GLACIER is empty) only runs inside that overlay.
  - Its lift must come from those two. Our reverted adduct probe was a similar ICE-on-non-[M+H]+ idea whose counter
    read 0.
- **Infra**:
  - The Colab CLI (WSL, `colab new|exec|drivemount|stop`) works, but `drivemount` fails with a Google 400. The user's
    own attempt failed too, despite reporting success. Use Kaggle CPU kernels (script type, no GPU quota) for data
    jobs. Staging examples: `.scratch/valgen/kernels/`. Write kernel-metadata.json **without a BOM**.
  - `colab exec -f` runs inside a kernel: use `parse_known_args`.
- **LB**: an undescribed submission from 2026-10-10 20:21 was still PENDING on 10-11 (probably the harness draft, which
  would just re-score the base). Check with `kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra`.

---

## 6. Next steps (in order)

1. **Fix `_pk` collision (TDD).**
   - Red test: a base cell re-binding a SNAP helper name must fail (extend `_nbnames` / R07 + fixture, and the worktree test).
   - Then rename the aliases in `patch_harness_dump.py` SNAP/FINAL and regenerate the notebooks.
2. **c1npx offline H0.**
   - Fetch c1npx `post_dump` + submission.csv.
   - Replay with the base's constants.
   - Exact match → record H0 (C1_npx) in gate_report.md. Mismatch → diagnose `casmi_post` (remember the C12 →
     `_final_pass()` refactor came AFTER these kernels were generated: first prove the default knobs reproduce).
3. **Diagnose the c2 Engine 2 coverage assert** from its log. Fix in the VAL cell / harness only, then re-push c2
   (§4 command).
4. **Confirm the draft's H0.**
   - `kaggle kernels status syedtoubur/casmi-0433-harness-draft` = COMPLETE.
   - Fetch its log, read `POST EQUIV ok 12 of 12 | mismatches 0`.
   - Set its STATUS from `g4-pushed` (finalize with `kaggle_push.py --check`).
   - The human previously interrupted a manual output download; confirm before large downloads.
5. With C2 + C1_npx dumps: `harness.py lossmap --dump C2 --dump C1_npx` → `research/harness/loss_map.md`;
   `harness.py backtest` → H2 (≥ 3 of 4 signs; appends to gate_report.md).
6. **E006 offline test:**
   - `harness.py compare --dump C2 --b "ICE_FALLBACK=True,METRIC_FUSION=True"` (+ each knob alone, + `--dump C1_npx`).
   - Implement the np-examples leak check: on C1_npx, swap full1 f·z for fpnet_0 f·z.
     - This needs fpnet_0 logits for the 245 query molecules and fingerprints of the dumped candidates.
     - Reuse `build/fold_audit.py`'s scoring code in a CPU kernel.
     - Pre-register the check in gate_report.md first.
7. Record results with `/update-experiments`, then **P3**: `/campaign-grill` with loss_map.md → charter →
   `campaign.py`.
8. Only if campaign 1's CIs are too wide: P1.5 `gen_val.py` from `.scratch/valgen/train_meta.parquet` (re-read the
   P1.5 gates first: G-gen3/4 need re-specifying after the fold audit).

---

## 7. File map

**Main checkout** (`E:\ENVEDA_casmi`):
- `research/bin/`:
  - poll_lb.py (`decide`, `--write`)
  - kaggle_push.py (push + detached watcher + `--check`)
  - campaign.py, brief.py, preflight.py, call_claude.py
  - test_bin.py, test_campaign.py
- `research/rules/`: R01–R07, `_util.py`, `_nbnames.py`, fixtures/.
- `research/harness/`: harness.py, metrics.py, combiner.py, test_harness.py, test_metrics.py, gate_report.md (all
  gates + results + human decisions; never edit thresholds).
- `research/exploit/`: control-0433 `scored`, adduct-iceberg-rerank `revert`, fusion-depth-expansion `dead`,
  harness-draft `g4-pushed` (stale; COMPLETE on Kaggle), harness-c2 / harness-c1npx `g4-timeout` (stale; ERROR on
  Kaggle).
- `.scratch/valgen/`: train_meta.parquet (git-ignored), fold_audit*.csv, logs/ (probe, fold audit, both val runs),
  kernels/ (staging dirs + metadata), smoke_val_cell.py.
- `.scratch/handoffs/comparison_sotas.md` (E006 vs 0.433; its "window" point is wrong).
- `.scratch/smart_gate/` (other agent).
- `.scratch/casmi_v4/` (Ahmed's engine code: FPNet model/data, pool, chem; no train loop).
- EXPERIMENTS.md: header, row 52 REVERT, row 53 control 0.433, Section 2 harness H0/H1/H1b rows, two graveyard rows (10-11).
- Memory: `casmi-research-loop-v2`, `casmi-own-val-generator` (fold-audit corrected).

**Exploit worktree** (`E:/casmi/lines/exploit`):
- `build/`: casmi_post.py, patch_harness_dump.py, test_patch_harness_dump.py, gen_val_probe.py, fold_audit.py,
  test_fold_audit.py.
- Generated notebooks: casmi-0433-harness.ipynb (draft), -c2.ipynb, -c1npx.ipynb.
- Base: `casmi-notebook-0-433.ipynb` (read only).

**Kaggle**:
- Dataset `syedtoubur/validation-data-casmi-2026` (val.parquet, val_key.pkl; already exists, don't re-upload).
- Preset `harness_0433_preset` (17 sources) in `specs/Dataset_intro/dataset_catalog.json`.
- Kernels: casmi-control-0433, casmi-0433-harness-draft / -c2 / -c1npx, casmi-gen-val-probe, casmi-fold-audit.

## 8. Gotchas (each cost time this session)

- **Inserted notebook cells:**
  - They must bind their own names: the draft (`VAL_SHARD=''`) never executes the val branch (R07).
  - Their helper names must not collide with base variables (`_pk`).
- **Pre-registered gates:** never edit thresholds. Post-hoc checks are appended and labelled, and the human decides on
  overrides.
- **Watcher timeout:** the watcher gives up at 60 min, but val runs take 2–3 h. `g4-timeout` doesn't mean failure;
  check `kaggle kernels status`.
- **Agent-written docs are unverified:** the fold-map doc was wrong. Verify claims about models empirically before
  building on them.
- **Placeholder test data:** the visible test.parquet is a train slice. Never treat it as test covariates.
- **Concurrency:** other agents edit this repo (memory `casmi-concurrent-agents`). Re-read files before editing.
