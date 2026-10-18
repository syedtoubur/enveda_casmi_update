# Brief: iso-rank FAST-SHIP (Claude Code Sonnet builder)

Read first: `spec.md`, `a1a2_result.md` (the API A1/A2 built), the gate block `### 2026-10-18: c3-iso iso-rank,
PRE-REGISTERED` (end of `research/harness/gate_report.md`, its "Fast-ship config" bullet is the contract, do not change it),
and CLAUDE.md §3. The ship does not wait for Stage A/B.

## Goal
A Kaggle kernel = the V22 notebook + one iso stage at the end, `ISO_MUT=True` with the fast-ship config, pushed (not
submitted). The human submits.

## Where
- Code: worktree `E:/casmi/lines/c3-iso-rank` (branch `idea/20261018-iso-rank`). Base notebook (read-only source):
  `E:/casmi/lines/c3/scratch/baseline/casmi26-v22-rel150-e2r2-clean.ipynb` with its export `casmi26_baseline.py`; copy
  both into the worktree's `scratch/baseline/`. Inspect notebooks only through `.py` exports or
  `python .agents/skills/analyze-notebook/scripts/compare.py` (never raw .ipynb JSON).
- Pattern to copy: `E:/casmi/lines/c3/build/patch_gen1_keep.py` (+ its tests). That patch snapshots what it needs from the
  V22 state and rewrites the final rows as a last step. The pushed example is in
  `research/c3/20261017-gen-rank1-keep/kaggle_push/` (kernel-metadata, push_args, counters.txt).
- The control's docker_image (pin it) and sources: `research/c3/20261017-control-v22/` and the gen-rank1-keep STATUS.

## Stage to add (`build/patch_iso_rank.py` → `casmi26-v22-iso-rank.ipynb`)
Per molecule, as the last step before `submission.csv` is written:
1. Parents = the V22 fused top 10 (the same list `build/isomer_reach.py` uses; find where V22 binds it); children by
   `isomer_mut.mutate_many` (cap 600, 120 s per molecule) in a 4-worker pool; plain-IK14 dedupe; drop retrieved IK14s.
2. CPU frag (`frag_rescore.py`) on the measured spectra → top K children; `iso_nl.nl_scores`.
3. ICEBERG (the base's worker, the base's adducts/CEs, `msg_all` weights) on the K children + parents; GLACIER
   (`gl_runner.py` as V22 runs it) on `[M+H]+`/`[M+Na]+` spectra. Reuse V22's ICE scores for the parents if they are
   already in the state; otherwise score them too.
4. `iso_rank.apply` with the fast-ship cfg (m_ice 0, k 10, all six lists, tail on).
5. Wall-clock guard: record the notebook start time; skip remaining molecules (rows unchanged) once elapsed > 8.3 h;
   `iso_skipped_time` counter. Print one line `ISO_COUNTERS iso_children=… iso_fired=… iso_tail=… iso_skipped_time=… n_mol=…`.
- Fast-commit protocol: on the visible draft (no `KAGGLE_IS_COMPETITION_RERUN`) run the stage on the 12 draft molecules
  only; on the hidden re-run, all of them.
- R07: inserted cells must not read names that a later base cell binds; helper names must not collide with base names.

## Choosing K (runtime only)
V22 hidden ≈ 4.5 h (human). Measure GLACIER and ICE throughput on children in a short GPU smoke (one kernel, ≤ 20 min,
~20 molecules from the draft; record cand/s). Project hidden time for 400 molecules =
4.5 h + mutation (≈ 20 s/mol CPU / 4 workers) + ICE + GLACIER. Use K = 150 if ≤ 8.5 h, else K = 50. Write the
arithmetic in `research/c3/20261018-iso-rank/runtime.md` before the push.

## Push and check
- `research/bin/kaggle_push.py` (GPU T4, internet off, the base's datasets + `ahmedberatozer/casmi26-glacier` +
  any dataset carrying `build/` code). Kernel slug `syedtoubur/casmi26-iso-rank`.
- G4 on the draft output: rows may differ from the V22 control draft only on molecules where the stage changed something;
  `iso_children` > 0 on the draft (a zero `iso_fired` on 12 molecules is acceptable); no exception in the log.
- Write `research/c3/20261018-iso-rank/card.md` (submit card: kernel/version, control = V22 0.447 ref 57306053
  unpaired-flagged, KEEP rule and counter band from the gate block, runtime projection) and add a row to
  `research/submit_queue.md`. STATUS → `ready-to-submit`.
- Never `kaggle competitions submit`. If Kaggle auth fails, stop and report (the human re-authenticates).

## Rules
No FPNet in the child path; no truth anywhere; heavy compute only on Kaggle; local runs < 2 min (unit tests only).
Report at the end: kernel slug/version, K chosen + runtime numbers, draft counters, G4 verdict, anything unresolved.

## Kaggle auth (2026-10-18)
The Kaggle CLI token expired; the human re-authenticates. Do all local work first (patch script, tests, kernel dir,
runtime.md skeleton). Before the first Kaggle call run `kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra`;
if it says "Authentication required", stop there and report exactly what is ready and what remains.
