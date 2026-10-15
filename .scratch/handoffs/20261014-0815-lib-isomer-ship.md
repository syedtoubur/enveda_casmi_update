# Handoff 2026-10-14 08:15 (Claude): ship lib-isomer-margin (falsifier + build notebook)

## Task for the new session (the human approved all three actions)
1. Create the private Kaggle dataset `syedtoubur/casmi-lib-isomer-inputs`.
2. Push the CPU falsifier kernel `syedtoubur/casmi-lib-isomer-falsifier`.
3. Push the build notebook `casmi26-e006-lib-isomer.ipynb` with `kaggle_push.py`. This is G4, a draft run on the
   pinned E006 image.

**Nothing is submitted.** Only the human submits. Do not step `campaign.py` until the falsifier verdict is in (see
"Campaign state").

## Context in five lines
- Campaign `research/campaigns/20261014-qoder-structural`, cycle 3. The current idea is
  `research/exploit/20261014-lib-isomer-margin`, at STATUS `attacked`, branch `scorer`.
- **Idea**: on each molecule's final top 3, re-sort the candidates that have their own library spectra by their own
  library similarity, if the best beats the first such slot by more than 0.05. Candidates without references never
  move.
  - Spec: `research/exploit/20261014-lib-isomer-margin/spec.md`. Attack (kill tests A1–A4): `attack.md`.
  - Origin: `research/ideas/2026-10-14-libgate-diagnosis.md` (H1). C1_npx loses 0.052 MRR where BASE ranks a
    library-backed isomer above the truth, 17 of those at rank 2.
- **Pre-registered kills**:
  - F1: on C1_npx eligible rank-2/3 cases, the truth's own libsim beats the rank-1 isomer's by > 0.05 in < 50% → dead;
  - F2: fire rate on C2_np > 2% → dead;
  - ceiling: oracle ΔMRR(C1_npx) ≤ 0 → dead;
  - A3: the gain vanishes without ≥ 0.97 self-copies → dead.
- `lib_sim` is the engine's own per-candidate library similarity, `X[:, FEATURES.index('lib_max')]`
  (`research/harness/dumps/C1_npx/post_dump/src/casmi/engine.py:262-301`). No new similarity code.

## Files (all built and reviewed; tests green)
**Falsifier** (main repo, `research/exploit/20261014-lib-isomer-margin/falsifier/`):
- `export_inputs.py` has already run (21 min), producing `inputs/cands.json` (1.6 MB) and `inputs/ban_rows.json`
  (0.6 MB).
  - Bans per cohort: C1_npx 1170, C1_xlib 3958, C2 10079, C2_np 55830 rows. C2_np = the C1_npx bans plus every row of
    the 245 truths, as in VAL_CELL.
- `lib_isomer_falsifier.py` is the kernel body; `build_nb.py` builds `lib_isomer_falsifier.ipynb`.
  `test_falsifier.py` passes 10/10. `kernel-metadata.json` is CPU, internet off, private.
- It builds the library from the full `train.parquet` with `casmi.build.build_spec_cache`. Library index i equals
  train row i (verified in `build.py:48-100`: every row group is kept in order). Each cohort's ban rows are excluded
  with a boolean mask.
- It prints `FALSIFIER {json}` with `verdicts.overall` = `SURVIVES` or `DIES: ...`, and writes
  `/kaggle/working/ceiling.json`.
- Review fix applied: the swap only uses libsim for candidates in the BASE top-10, for parity with the shipped knob.
  F1 still reads the truth's own libsim.

**Build** (worktree `E:/casmi/lines/exploit`, branch `idea/lib-isomer-margin`, nothing committed):
- `build/casmi_post.py`:
  - `CFG['LIB_ISOMER_MARGIN']=None`;
  - pure `lib_isomer_rows()`;
  - `_lib_isomer()` as the last step of `run()`, which no-ops without `state['lib_sim']`. Verified: the old C1_npx dump
    with the knob on gives byte-identical output.
- `build/patch_harness_dump.py`:
  - the C7 graft `LIB_SIM[mid] = {smiles: X[i, lib_max]}` for BASE `order[:10]`, pickled as `state['lib_sim']`;
  - **the RDKit wheel fix**, see below.
- `build/patch_lib_isomer.py` builds `casmi26-e006-lib-isomer.ipynb` from E006 v1 with `--margin 0.05`.
  - `compare.py` shows only C18 (the C7 graft) MODIFIED and C31 ADDED.
  - C31 is the ship cell after `FACTORIAL_REPORT`. It re-sorts `submission.csv` and prints `LIB_ISOMER {json}`. On any
    error it prints `LIB_ISOMER_FAILED` and keeps the base submission.
- Tests: `python build/test_patch_lib_isomer.py` 12/12; `python build/test_patch_harness_dump.py` 22/22.
- The ship cell uses plain RDKit InChIKey14 for matching, while the replay uses tautomer-canonical. A mismatch can
  only cause a miss (no swap), never a wrong swap.

## RDKit wheel fix (done; on the idea branch)
- **Problem**: the harness dump notebooks are built from `casmi-notebook-0-433.ipynb`, whose `_claw_preseed_site`
  hard-codes the cp313 rdkit-2025.3.6 wheel. `kaggle_push.py` pins dump runs to E006's image (Python 3.12), so ICE
  died and `state["ice"]` was empty.
- **Fix**: `PRESEED_OLD`/`PRESEED_NEW` in `build/patch_harness_dump.py`, applied in `build()` with
  `must_replace(...)`. It takes the first rdkit-2025.3.6 wheel (bundled `.whl.ice` or mounted) whose tags
  `pip._vendor.packaging.tags.sys_tags()` accepts, which is E006's `simulator_rdkit_wheel` rule. It logs
  `[harness] simulator rdkit wheel: ... python X.Y`.
- Tests: `test_preseed_picks_wheel_by_interpreter_tag` and `test_patch_only_adds`.
- **Unproven on Kaggle** until the next harness re-dump.

## Commands (run from `E:\ENVEDA_casmi`)
1. **Dataset**:
   - Create `research/exploit/20261014-lib-isomer-margin/falsifier/inputs/dataset-metadata.json` with
     `{"title": "casmi-lib-isomer-inputs", "id": "syedtoubur/casmi-lib-isomer-inputs", "licenses": [{"name": "CC0-1.0"}]}`.
   - Then run `kaggle datasets create -p research/exploit/20261014-lib-isomer-margin/falsifier/inputs` (private by
     default; never pass `--public`).
   - Wait until `kaggle datasets status syedtoubur/casmi-lib-isomer-inputs` says ready.
2. **Falsifier push**:
   - Make a clean push dir in the scratchpad that holds only `lib_isomer_falsifier.ipynb` and `kernel-metadata.json`.
   - Run `kaggle kernels push -p <dir>`. Check status once later with
     `kaggle kernels status syedtoubur/casmi-lib-isomer-falsifier`; don't poll in a loop.
   - When it is COMPLETE, fetch the output with
     `kaggle kernels output syedtoubur/casmi-lib-isomer-falsifier -p research/exploit/20261014-lib-isomer-margin/falsifier/kaggle_output`.
     A failed kernel still keeps its outputs (memory).
   - Then copy `ceiling.json` to `research/exploit/20261014-lib-isomer-margin/ceiling.json` and fill `spike_result.md`
     (template in the file) from the `FALSIFIER` line.
3. **Build push (G4)**:
   ```
   python research/bin/kaggle_push.py research/exploit/20261014-lib-isomer-margin E:/casmi/lines/exploit/casmi26-e006-lib-isomer.ipynb --counter "\"lib_sim_mols\": " --counter "?\"lib_isomer_swapped\": "
   ```
   - The defaults are line `exploit`, preset `sota_0433_inference_preset` (the same 16 sources as E006), slug
     `casmi-lib-isomer-margin`, and the pinned control image.
   - It starts a detached G4 watcher; check once later with
     `python research/bin/kaggle_push.py --check research/exploit/20261014-lib-isomer-margin`.
   - **Watch out**: the commit run is the 12-molecule placeholder (a train slice). G4 also needs the draft to differ
     from the control draft in ≥ 1 molecule (`g4_min_changed: 1`). If no swap fires on those 12, G4 FAILs.
   - A G4 FAIL auto-starts a `/run-idea` fix agent (`gates.g4_fix_rounds: 2`), which spends Claude tokens.
   - If it fails only on "no diff" or "swapped 0", tell the human rather than letting the fix agent loop. The swap
     legitimately can be 0 on 12 placeholder molecules.

## Campaign state (don't break it)
- `kaggle_push.py` moves STATUS to `g4-pushed`, which is TERMINAL for `campaign.py`. A `--once` step would then close
  cycle 3 and count an LB probe before the falsifier and the scorecard exist. **Do not step `campaign.py`** until:
  - **the falsifier DIES**: write `spike_result.md`, set STATUS `needs-human` (a pre-slot kill, not `dead`), then let
    `--once` close the cycle;
  - **the falsifier SURVIVES**: the next measurement is the scorecard. Regenerate the dump notebooks with the graft
    (`python build/patch_harness_dump.py --shard <C1_npx|C1_xlib|C2|C2_hard|C2_np> --out ...`; C2_np needs
    `--ranker-field/--ranker-drop`, see `research/exploit/2026101*-harness-*` STATUS/gates for the values used
    before). Re-dump on Kaggle (GPU; ask the human), then run the spec's scorecard command and the gate one-liner
    (`.scratch/qoder_prompt.md` §4.2). Ask the human before these GPU runs.
- **Open loop proposals for the human**:
  - `research/campaigns/20261014-qoder-structural/questions.md`: split C1 leak-hit into library-driven vs FPNet, or
    the "other Δ" reading zeroes every library idea;
  - `research/exploit/20261014-lib-isomer-margin/questions.md`: a new harness family `lib_isomer`.
- Submit autonomy: OFF.

## Verified vs hypothesis
- **Verified locally**:
  - all the tests above;
  - the compare.py diff;
  - the old-dump no-op;
  - the library row = train row mapping (code read);
  - `train_structs.parquet` comes from `ahmedberatozer/casmi26-v2-pool` (`POOL_DIR`, `scratch/casmi26-e006.py:292`);
  - `chem.score_key` matches the base keys on 100% of the BASE top-10 entries.
- **Hypothesis**: the falsifier's CPU runtime (est. 1–3 h, budget 2.5 h, truncates gracefully); the RDKit preseed fix
  on Kaggle; that swaps fire on the 12 placeholder molecules.
