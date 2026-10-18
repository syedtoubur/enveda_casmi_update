# Brief: iso-rank STAGE A (CPU validation), Claude Code Sonnet builder

Read first: `spec.md`, `a1a2_result.md` (the API), the gate block `### 2026-10-18: c3-iso iso-rank, PRE-REGISTERED`
(end of `research/harness/gate_report.md`; its Split and Stage A bullets are the contract), CLAUDE.md §3 (heavy compute only
on Kaggle CPU kernels), and `research/c3/20261017-isomer-mut-ice/{i1b0_result.md,i1b1_result.md}`.

## Where
Code in worktree `E:/casmi/lines/c3-iso-rank-a` (branch `idea/20261018-iso-rank-stage-a`, off c83b7e6), in a sub-dir `stage_a/` (separate from the ship agent's files: do not
touch `build/patch_iso_rank.py`, `scratch/`, or the notebook). You may add small helpers under `build/` only if new files.
Existing inputs:
- pseudo_c3: frozen children/top-150 (`E:/casmi/c3_iso/i1b_pseudo_c3_full_chunk*.pkl`, `frozen_pseudo_c3_full.pkl`), I1b.1
  ICE outputs on Kaggle datasets `syedtoubur/casmi26-c3iso-i1b1-ice-out` and `syedtoubur/casmi26-c3iso-i1b-measure`; the
  scripts `build/run_i1b1_measure.py`, `build/prep_ice_inputs.py`, `build/isomer_reach.py` show the formats.
- C1/C2 dumps: harness kit dataset `syedtoubur/casmi26-harness-kit` (see `E:/casmi/kaggle_kit/build_kit.sh` and
  `E:/casmi/kaggle_kit/kernel/run_harness_kit.py` for how a kernel mounts it).

## Steps
1. **A3 (Kaggle CPU kernels):** for every molecule of C1_npx, C1_xlib, C2 and C2_hard (~1,390), generate children from
   the V22 fused top 10 with `isomer_mut.mutate_many` (cap 600, 120 s), frag-cut to the top 150, and compute frag + NL +
   multiplicity + parent rank, plus the parents' frag. Split across 2 kernels (4 workers each). Truth is never loaded in
   this step. Output a compact per-molecule table (pkl/parquet as a kernel output, NOT committed to git). For pseudo_c3,
   compute NL/multiplicity/parent rank on the frozen children in a third (small) kernel, joined with the I1b.1 ICE.
2. **A4 (analysis, can be a Kaggle CPU kernel or local if it is < 2 min on small fetched tables):** the half split
   exactly as pre-registered; on half A of pseudo_c3: forward-select lists (keep iff within-pool MRR@20 rises), pick the
   provisional m_ice from {0, 0.01, 0.02, 0.05, 0.1}, freeze both in `stage_a_frozen.json` BEFORE scoring half B.
3. **A5:** on half B, report everything in the Stage A bullet: Δ_pseudo_c3 (bootstrap CI by scaffold; acyclic/ring,
   timsTOF, truth ∈ MSG if `msg_ik14.npy` is reachable, else say so), the C1/C2 frag-only fire rate (fire with the ICE
   conjunct dropped) and its worst-case parent-anchored cost (child treated as sure-wrong; Δ from moving the V22 row),
   and the composite bracket at π 0.15/0.25/0.35. Also report the fast-ship config's own numbers (all six lists minus
   GLACIER, m_ice 0) on half B, for the LB-vs-offline calibration. Replays use `iso_rank.apply` on the V22 rows
   from the dumps (V22 cfg `METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40,REL_PROMOTE=150.0,FRAG_GAP=true,E2R2=true`).
4. Verdict per the pre-registration (**DEAD iff Δ_pseudo_c3 half B < 0.012**). Write
   `research/c3/20261018-iso-rank/stage_a_result.{json,md}`. STATUS stays `needs-human` (checkpoint 1).

## Rules
Truth only scores; never selects children; never touches the frozen choices after half B is read. No FPNet. Fetch only
small result files (`kaggle kernels output --file-pattern`). Never submit. If Kaggle auth fails, stop and report.
Report at the end: kernels + versions, the Stage A numbers, the verdict, deviations.
## Kaggle auth (2026-10-18)
The Kaggle CLI token expired; the human re-authenticates. Do all local work first (patch script, tests, kernel dir,
runtime.md skeleton). Before the first Kaggle call run `kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra`;
if it says "Authentication required", stop there and report exactly what is ready and what remains.
