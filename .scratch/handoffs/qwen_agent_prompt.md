# Autonomous Research Loop: Complete Handoff & Execution Prompt for Qwen Agent

> **Target Agent**: Qwen AI Assistant / Autonomous Coding Agent  
> **Repository Root**: `e:\ENVEDA_casmi`  
> **Active Worktree**: `E:\casmi\lines\antigravity-exploit` (Branch `idea/20261017-e180-drop-dreamsnp-pclite`)  
> **Campaign**: `research/campaigns/20261017-antigravity-sota-exploit/`  
> **Generated At**: 2026-10-17 13:55:00 +06:00  

---

## 1. Mission Overview & Problem Context

You are taking over the **Antigravity Exploit Loop** for the **Enveda CASMI 2026** competition on Kaggle (`enveda-CASMI26-molecule-id-mass-spectra`).

- **Objective**: Push the SOTA MRR@25 score from the current baseline of **~0.447–0.451** up towards **0.470+** through disciplined, leakage-free post-scoring, candidate filtering, and reranking patches.
- **Evaluation Metric**: Strict Mean Reciprocal Rank at top 25 (`MRR@25`) evaluated on **tautomer-canonical InChIKey14** (RDKit 2026.03.3). Stereocenters and tautomers are ignored. Duplicates in top 25 waste slots.
- **Data Reality**: 100% of test spectra are Bruker timsTOF with a systematic **+1.45 ppm offset** (`CFG.PPM_CENTER = 1.45`).
- **Hard Constraints**:
  - Kaggle notebooks must run **completely offline** within **<= 9 hours** on a single T4 GPU. All dependencies (RDKit wheels, etc.) are loaded from offline datasets.
  - Fast-commit protocol: 10–20 molecules on visible draft commit (`KAGGLE_IS_COMPETITION_RERUN` false), full scoring on hidden re-run.
  - Strict data boundaries: never commit `.npz`, `.pt`, `.parquet`, or `.zip` files to Git.
  - **Never hand-edit `.ipynb` JSON directly**. Always write and run deterministic Python patch scripts (`build/patch_<idea>.py`) that transform the base notebook.

---

## 2. Settled Empirical Decisions (Do NOT Re-Litigate)

The following results have been rigorously measured and settled across all 6 validation cohorts:

1. **Baseline Model**:
   - `easonyanyan/casmi26-v22-rel150-e2r2-clean` (`REL_PROMOTE=150.0`, `FRAG_GAP=True`, `E2R2=True`).
   - Calibrated parity is **0.0000** across all cohorts (`C1_npx`, `C1_xlib`, `C2`, `C2_hard`, `C2_np_other`, `C2_np_lotus`).
2. **Wave 1 Outcomes**:
   - **`e180-drop-dreamsnp-pclite` (Idea B8)**: **PASS -> `status: build`**!
     - **C2 Delta**: **+0.0026** (90% CI `[+0.0007, +0.0053]`, 8 up, 0 down, 0 displaced).
     - **C2_hard Delta**: **+0.0079** (90% CI `[+0.0016, +0.0156]`, 6 up, 0 down, 0 displaced).
     - **Displacements**: **0 across all 1,390 molecules**.
     - **Mechanism**: Prunes enveda-180-only synthetic combinatorial decoys after simulator rerank, backfilling to 25 slots from overflow/extras and mass-matched DreamsNP natural products. Includes E2R2 Decoy Guard preventing synthetic decoys from taking Rank 2.
   - **`cross-formula-cluster` (Idea B1)**: **DEAD** (K3 degeneracy: 0 wrong-formula decoys in top-10 on honest molecules; lowering thresholds caused 10 displacements and -0.0265 C2).
   - **`nl-sim-rda` (Idea B7)**: **DEAD** (7 displacements on C2, -0.0161 C1_npx).
   - **`transductive-net-stoich` (Idea B3)**: **DEAD** (no rank 2-10 congener pairs, +0.0000 C2).
3. **Wave 2 Outcomes So Far**:
   - **`tautomer-canon-unify` (Idea A6)**: **DEAD** ($\Delta = +0.0000$ across all 1,745 validation molecules).
     - *Root cause*: `metric_identity` in the V22 baseline already applies RDKit 2026.03.3 `TautomerEnumerator().Canonicalize` + `RemoveStereochemistry` during `metric_fusion` and `safe_extend`. Candidate representations are already fully consolidated at post-scoring; remaining mismatches are upstream in raw database `.npy`/`.parquet` files.
   - **`v4b-union-xenc` (Idea A1)**: **IN FLIGHT** in `research/antigravity-exploit/v4b-union-xenc/` (cross-encoder reranker conditioning on DreamsFP embeddings and MetFrag fragment evidence over the 40-candidate fused union).
4. **Infrastructure State Just Completed**:
   - Post-scoring engine updated in worktree: `E:/casmi/lines/antigravity-exploit/build/casmi_post.py`.
   - Unit test suite: `E:/casmi/lines/antigravity-exploit/build/test_e180_drop.py` (**5/5 PASS**).
   - Patch script: `E:/casmi/lines/antigravity-exploit/build/patch_e180_drop.py`.
   - Patch test: `E:/casmi/lines/antigravity-exploit/build/test_patch_e180_drop.py` (**100% PASS**).
   - Generated probe notebook: `E:/casmi/lines/antigravity-exploit/casmi26-e180-drop-dreamsnp-pclite.ipynb`.
   - Worktree Git commit: `69c3f7ebf77d9f491b7c7bffd627d65bdbe14e8c` on branch `idea/20261017-e180-drop-dreamsnp-pclite`.
   - **Kaggle Dataset Published**: `syedtoubur/casmi26-e180-keys` containing `e180_only_ik14.txt` (181,542 keys) is live on Kaggle.

---

## 3. Immediate Step-by-Step Instructions (Execute Next)

As the incoming Qwen agent, execute the following steps in sequence:

### Step 1: Push Kaggle G4 Kernel for `e180-drop-dreamsnp-pclite`
Push the probe notebook as a private Kaggle kernel version using the standard pipeline:
```powershell
python research/bin/kaggle_push.py research/antigravity-exploit/e180-drop-dreamsnp-pclite E:/casmi/lines/antigravity-exploit/casmi26-e180-drop-dreamsnp-pclite.ipynb --preset sota_0433_inference_preset --source syedtoubur/casmi26-e180-keys
```
- The push command launches a detached watcher that polls `kaggle kernels status`.
- Check status anytime with:
  ```powershell
  python research/bin/kaggle_push.py --check research/antigravity-exploit/e180-drop-dreamsnp-pclite
  ```
- Verify the fast-commit draft diff has >= 10 molecules and **0 displacements**.

### Step 2: Check & Finalize `v4b-union-xenc` Scorecard
Inspect the results of `v4b-union-xenc` once its harness run finishes:
```powershell
python -c "
import os, json
p = os.path.join('research', 'antigravity-exploit', 'v4b-union-xenc', 'harness_result.json')
if os.path.exists(p):
    with open(p) as f:
        data = json.load(f)
    print('=== v4b-union-xenc Scorecard ===')
    for cohort, vals in data.get('cohorts', {}).items():
        print(f'  {cohort}: delta={vals.get(\"delta\")} (up={vals.get(\"up\")}, down={vals.get(\"down\")}, disp={vals.get(\"displacements\")})')
else:
    print('v4b-union-xenc: harness_result.json not found yet')
"
```
- If `v4b-union-xenc` passes gate criteria (`C2 delta > 0`, `disp == 0`): mark `status: build` and package into a stacked kernel with B8.
- If it fails: mark `status: dead` in its `STATUS` file with root-cause analysis in `gates.md`.

### Step 3: Final Submission Packaging
- Once G4 passes for `e180-drop-dreamsnp-pclite` (single-factor) or the stacked kernel (if `v4b-union-xenc` passes):
  - Record results in `research/campaigns/20261017-antigravity-sota-exploit/log.md`.
  - Update `EXPERIMENTS.md` with the new verified SOTA score.
  - Present the final submission recommendation to the human partner.

---

## 4. Key Operational Guardrails

- **Worktree Separation**: Always operate in `E:/casmi/lines/antigravity-exploit` (branch `idea/20261017-e180-drop-dreamsnp-pclite` or new idea branch). Never touch `E:/casmi/lines/c3-commit` (which is reserved for Claude Code's Track 1 run).
- **Execution Working Directory**: Always run shell commands with `Cwd: e:\ENVEDA_casmi` (the repo root).
- **Offline Parity**: Ensure that any new asset or file added to notebooks is registered in `specs/Dataset_intro/dataset_catalog.json` and passed via `--source <owner/slug>`.
- **Zero Displacements Rule**: Any candidate change that causes even 1 Rank-1 displacement on the honest validation cohorts (`C2` or `C1_npx`) is immediately disqualifying unless compensated by >= 3 honest wins with positive net delta and clean stress survival.

You now have 100% of the project context. Proceed directly with **Step 1**!
