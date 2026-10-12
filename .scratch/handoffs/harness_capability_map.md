# Handoff: replay harness, back-test suite, leak-free C2_np shard
**Written:** 2026-10-11 ~16:45 · **Branch:** `specs-scratch-r0` (repo) / `line/exploit` (worktree `C:/casmi/lines/exploit`)
**Everything below is UNCOMMITTED.** No commit was requested; do not commit without asking.

Read first: `CLAUDE.md`, `research/README.md`, `research/harness/gate_report.md` (from "### 2026-10-11: H0" down),
`research/harness/capability_map.md`, `research/harness/loss_map.md`, `EXPERIMENTS.md` header + rows 54–55.
Plan of record: `C:\Users\h\.claude\plans\here-i-am-trying-memoized-badger.md` (approved after a 5-round grill).

---

## 1. Goal
Every idea is measured on the offline replay harness (`research/harness/`) before it gets an LB slot. The harness
replays `build/casmi_post.py` (post-scoring knobs) on a state dump of the 0.433 kernel run on a val shard. The question
of this workstream: **which idea families can the harness judge?** Output = a capability map with one verdict per
family (TRUSTED / TRUSTED-PROVISIONAL / ADVISORY / BLIND), plus a leak-free primary shard (C2_np) to make it useful.

## 2. Hard rules (from CLAUDE.md + user)
- Never read raw `.ipynb` JSON. Use `/analyze-notebook`, or the py exports `.scratch/nb1_433.py` (0.433 base) /
  `.scratch/nb2_e006.py` (E006).
- Never commit `.npz/.pt/.parquet/.zip`. Dumps live in `research/harness/dumps/` (git-ignored).
- Agents may push Kaggle kernels (`research/bin/kaggle_push.py`); **only the human submits**.
- Pre-register gate thresholds in `gate_report.md` BEFORE a run; never edit them, append only.
  `backtests.yaml` is append-only too.
- Kaggle inference: offline, ≤ 9 h on one T4.
- Log results with `/update-experiments`.
- Grill questions go in AskUserQuestion. Work step by step and show the reasoning.
- Stay in scope. The libmax / smart-gate work in `.scratch/smart_gate` and `.scratch/kaggle_libmax_eval` is a
  separate track, not part of this workflow. Don't touch it.
- Base notebook cells stay byte-identical (G1). Harness changes go only in harness-only cells
  (`build/patch_harness_dump.py`).

## 3. State of the instrument
**Dumps** (`research/harness/dumps/<shard>/`: `post_dump/state.pkl`, `submission.csv`, `baseline_submission.csv`, log,
`ident_cache.pkl`):
| shard | n | MRR | H0 | note |
|---|---|---|---|---|
| C2 | 400 | 0.631 | PASS 400/400 up to C10 | kernel crashed in base C12 assert (Engine 2 covered 398/400) |
| C1_npx | 245 | 0.933 | PASS 245/245 full + FINAL_PASS=False | crash was in harness FINAL cell, after the dump |
| C2_np | 245 | ? | pending | **GPU run pushed 16:37** (see §5) |

**The leak (why H2 failed 2/4):** the BASE channel (full1 FPNet, trained on every train structure) puts the banned
truth at rank 1 for 0.897 of C2 (in-list 0.970). Engine 2 manages top-1 0.022. So on C2 any fusion with Engine 2 or
PubChem reads as a loss (USE_ENG=False +0.257). Any C2-only conclusion about fusion, promotion or reranking is suspect.

**Speed:** an identity cache (`score_key` / `canonicalize_smiles_to_ik14`) cuts a replay from 18 min to 4–5 s.
A new dump needs `python research/harness/harness.py warm --dump <shard> --procs 7` once (~11 min). The cache is
tagged with the RDKit version.

**Harness CLI** (`python research/harness/harness.py ...`): `fetch <idea>`, `warm --dump X`,
`compare --dump X --b '{json cfg}' [--a ...] [--json out]`, `lossmap --dump X`, `backtest` (old H2),
`suite --dump X [--dump Y] [--eng-off] [--no-report]`.
**Kaggle quirk:** an ERROR kernel's `kaggle kernels files` lists nothing, but
`kaggle kernels output <k> -p DIR --file-pattern '...'` still downloads `state.pkl` etc.

## 4. Workstream 1 (done): back-test suite → capability map
- `research/harness/backtests.yaml`: families (every CFG knob mapped, test-enforced), flips with tiers, sweeps.
  Tiers: S = sure own-account paired; A = same-author single runs isolating the change (sign only); B = bundle;
  none = behaviour row without a valid LB key. Old-pipeline (v4g/v4n) and inferred LB numbers are excluded.
  YAML gotcha: write `1.0e+9`, not `1.0e9` (PyYAML parses the latter as a string; a test asserts this).
- Verdict rules (pre-registered in gate_report "back-test suite ... PRE-REGISTERED"):
  - BLIND: mismatches > matches.
  - TRUSTED: ≥ 3 S/A flips, all match, and the null CIs cover 0.
  - TRUSTED-PROVISIONAL: ≥ 1 S/A match, no S/A mismatch or regime-dependent flip, λ sweep PASS and sanity PASS.
  - Otherwise ADVISORY.
  - Before C2_np, a sign counts only if C2 and C1_npx agree (disagreement = regime-dependent). After C2_np, C2_np is
    primary.
- Code:
  - `harness.py`: `load_spec`, `family_of`, `flip_status`, `family_verdict`, `sweep_ok`, `suite`, `_write_capmap`.
    `compare()` now returns `family` + `verdict`, and a BLIND verdict forces `lb_slot=False`.
  - `research/rules/R08_harness_family_verdict.py` + 2 fixtures.
  - `campaign.py` `harness_gate`: BLIND passes only with `prior.md`.
  - `brief.py`: replan and spec read `capability_map.md`.
  - Tests all green: `test_suite.py`, `test_harness.py`, `test_metrics.py`, `research/bin/test_campaign.py`,
    `test_bin.py` 51/51, `preflight.py --selftest` 8/8.
- **Result (C2 + C1_npx): every family is ADVISORY.**
  - Sanity PASS. The λ sweep FAILs: the ICE_LAM argmax is 0.25 on both dumps, while the LB-tuned value is 1.0.
  - e006 and ice_fallback match on sign but are tiny: C2 +0.0005 / C1_npx +0.0027.
  - final_pass_off reads Δ 0 because the replay runs with `dreams=None`, so the harness is blind to the DreaMS fill.
  - cell19_base_ice matches but is 3× too large.
  - Consistent lead with no LB key: less PubChem promotion is better on both dumps.
  - Full tables are in `capability_map.md` / `.json`.

## 5. Workstream 2 (Option A, in flight): C2_np
- **A0 done** (Kaggle CPU `syedtoubur/casmi-a0-np-holdout` v2, log in `.scratch/valgen/a0_v2/`):
  - fpnet_0/full1 MRR ratio on the np structures' other-library spectra = **0.634 ≤ 0.8**, so the structures are held
    out and the branch is **C2_np**.
  - No residual leak in the A family: fe_A ≡ v3 fpnet_0 ≡ fpnet_0, and fe_B ≡ v3 fpnet_1.
  - prvsiyan fp_single_s2 held out np **spectra only** (other-library ratio 0.893). This is reported, not blocking.
  - The ranker npz field `Q` indexes `sets` (np = code 0, 43,864 of 399,790 rows).
- **C2_np kernel** `syedtoubur/casmi-0433-harness-c2np`, idea folder `research/exploit/20261011-harness-c2np`
  (STATUS `g4-pushed`, kind instrument, pushed 16:37):
  - Shard: 245 C1_npx molecules, `ban_keys=[ik14]`, cls 2.
  - BASE bank swapped to fpnet_0.
  - Engine 2 "our ranker" is fitted without the np rows (site .pth hook).
  - Coverage guard: Engine 2 must cover ≥ 95%.
  - `_pk` fix (`_hx_*` helpers + R07 rebound check).
  - Regenerate with `python build/patch_harness_dump.py --shard C2_np --ranker-field Q --ranker-drop np --ranker-names sets`
    in the worktree. `test_patch_harness_dump.py` passes 18/18.
- **Pre-registered gates** (gate_report "Option A C2_np GPU run"):
  - `POST EQUIV ok 245 of 245`, and the offline replay matches 245/245.
  - `VAL ban extra rows by key` = 54,660 ± 2%, and the leftover truth rows = 0.
  - `HARNESS fpnet_0 swaps 1`, and `HARNESS ranker rows dropped 43864 of 399790`.
  - `HARNESS ENG fill n` with n ≤ 12.
  - **BASE top-1 on C2_np ≤ 0.527.** Above that, the shard still leaks: stop and report.
- **RESULT (2026-10-11 late): STOPPED on a gate miss.** Cache warmed (`--procs 2`; RAM is tight, ~450 MB free).
  Offline H0 PASS 245/245, all counters PASS, but **BASE top-1 on C2_np = 0.947 > 0.527 bar → FAIL**: BASE still
  leaks, so C2_np is NOT a leak-free primary and the suite was not rerun with it. STATUS `needs-human`; full block in
  gate_report "Option A C2_np result"; raw numbers `dumps/C2_np/option_a_checks.json`.
  - This falsifies "BASE leaks because full1 saw every structure": with the np-clean fpnet_0, BASE is *higher*.
  - Leading hypothesis (unverified): the v1 engine's pool artefacts are symlinked prebuilt files built from the FULL
    train set (`.scratch/nb1_433.py:192`: `pool_meta.parquet`, `pool_fp.npy`, `train_structs.parquet`,
    `train_fp_sel.npy`, …); only the spectrum cache is rebuilt from the banned train.parquet (:196). So the engine or
    its 4-booster ranker (`RK['features']`) can still see that the truth is a train structure.
  - The human decides the next step. Options: (a) diagnose which feature carries it (list `RK['features']` and the
    v1-engine features that read `train_structs`/`tfp`); (b) rebuild those artefacts from the banned train set in a
    harness-only cell and rerun C2_np; (c) accept that BASE is unmeasurable on val and judge only post-BASE knobs.
  - **(a) DONE 2026-10-12 on CPU** (gate_report "C2_np BASE-leak diagnosis"; probes `.scratch/valgen/c2np_leak_probe{1,3,4}.py`):
    the prebuilt-artefact hypothesis is FALSIFIED (sids aligned via train_structs ik14), as are tautomer survivors and
    small windows. Carriers: (1) popularity prior, +0.061 top-1 (np truths are famous: pop 13 vs 2.4); (2) leading,
    unverified: **the v4b ranker was fit on sims whose queries include the `np` fold** (FPNet-holdout folds = ranker
    query pool), so it memorizes these structures. Signature: ranker-only top-1 C2_np 0.886 > C1_npx 0.747. So option
    (b) as written (rebuild pool artefacts) would NOT fix it; rebuilding needs a ranker not fit on np, which we don't
    have.
  - **User chose option 2 (2026-10-12), DONE.** Pre-registered gate_report "Option 2 BASE-coupling rule":
    coupling = share of leak-hit molecules whose truth rank moves; > 0.05 → **BASE-COUPLED** (handled like BLIND:
    `compare()` lb_slot False, R08 + campaign `harness_gate` need prior.md, brief wording). Code: `harness.py`
    (`coupling_rate`, `coupled_verdict`, `COUPLING_TH`, `NOT_CITABLE`), `R08` + fixture `R08_base_coupled_no_prior`,
    `campaign.py`, `brief.py`; tests green. Map (C2 + C1_npx): e006_overlay 0.019 / ice_fallback 0.014 / tail 0.003
    **ADVISORY**; reranker 0.32 / fusion 0.56 / promotion 0.16 / merge_gate 0.27 **BASE-COUPLED**. The `--eng-off`
    run is in gate_report (there fusion reads 0 only because its knobs are inert with Engine 2 off).
    `capability_map.md` = primary run. C2_np idea STATUS `dead`. Logged in EXPERIMENTS.md §2 + Graveyard.

## 6. Next steps (in order)
1. DONE (option 2): capability map with BASE-COUPLED verdicts; C2_np closed (no clean train-derived shard exists).
2. DONE: C2_np gates, diagnosis and the new map logged in EXPERIMENTS.md.
3. Raise these points with the user (not decided yet):
   - The harness is blind to the DreaMS fill (`dreams=None`). Should DreaMS be replayed?
   - The λ evidence is old-pipeline. It was a pre-registered choice, so the rule is not changed retroactively.
   - The promotion lead (less PubChem promotion) has no LB key, and calibration probes were declined.
   - An **E006 replicate is owed** (one 0.440 run, so it is keep-pending).
   - The **smart-gate LB result** (submitted 10-11 04:09) is pending, with E006 0.440 as its paired control.
4. Later: P3 `/campaign-grill` that feeds `loss_map.md` + `capability_map.md` into the first campaign charter.

## 7. Settled decisions (don't re-litigate)
Deliverable = capability map; strict per-family trust; tiered evidence S/A/B; specs_scratch R0–R5 results are family
priors only (not replayable); no calibration LB probes; skip the offline Engine-1 proxy (the dump holds only ≤ 60
candidates + final scores); Engine 2 leakage handled by **dropping np ranker rows**, no FPNet retrain now (FPNet ↔
`rank_train.npz` coupling: never swap one without the other); old H2 rows kept as behaviour rows (tier none);
agreement rule across dumps; ICE_LAM/GL_LAM sweeps = evidence, other sweeps = sensitivity.
