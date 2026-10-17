# Spec: c3-slot-curve (S1, instrument)

Campaign `20261017-c3-iso`. Gate block: gate_report `### 2026-10-17: c3-iso campaign, PRE-REGISTERED`, bullet S1.

## Question
Does inserting the generator's top-1 at one fixed slot s, for EVERY molecule and without any detector, raise the expected
MRR@25 on a test mix with C3 share π? If so, which s?

## Inputs (all exist)
- V22 replays: `harness.load_dump(shard, POST)` with POST = `E:/casmi/lines/c3/build/casmi_post.py` and the V22 cfg
  `METRIC_FUSION=True, ICE_FALLBACK=True, FUSION_WINDOW=40, REL_PROMOTE=150.0, FRAG_GAP=True, E2R2=True`;
  cohorts pseudo_c3, C1_npx, C1_xlib, C2, C2_hard (C2_np advisory).
- Generator top-1 for pseudo_c3: `E:/casmi/c3_gen/p4_raw.pkl`, mut-first rule exactly as `build/c3_detector.py:gen_top1`
  (mutator top-1 if any mutant, else decoder top-1). Reuse that function.
- Truth: `harness.truths()`; IK14 via `casmi_post.canonicalize_smiles_to_ik14`.

## Method (`build/slot_curve.py` + `build/test_slot_curve.py` in the idea worktree)
1. Per cohort, get the V22 final rows (`casmi_post.run` with the V22 cfg; reuse `harness.ranks_for`-style code).
2. For s in 2..25 and each molecule: new row = row[:s-1] + [x] + row[s-1:] with x's IK14 removed from the rest
   (no duplicate IK14), cut at 25. pseudo_c3: x = mut-first top-1 (skip insert if none). C1/C2: x = a sure-wrong
   placeholder token (cost only; never equal to the truth).
3. Δ(s) per cohort = MRR(new) − MRR(V22), with the harness scaffold-bootstrap 90% CI (2000 resamples, seed 0;
   reuse the harness bootstrap function) and the leak-hit / other split.
4. Expected Δ(s, π) = π·Δ_pseudo_c3(s) + (1−π)·mean over C1_npx, C1_xlib, C2, C2_hard of Δ_other(s), for π 0.15/0.25/0.35.
   Also a second variant using all-molecule Δ (not only "other") for C1/C2, reported.
5. s* = argmax_s Expected Δ(s, 0.25) (other-variant). Report s*, its value and CI.

## Output
`research/c3/20261017-c3-slot-curve/result.json` + `result.md` (table s × cohort, the π curves, s*). STATUS →
`harness-done` with `kind: instrument` (no gate kill; gate.py is not run on an instrument). Runtime: ≤ 4 procs.

## Tests
- Insertion keeps length ≤ 25, no duplicate IK14, rank order of the others preserved.
- At s = 26 (no-op) Δ = 0 on every cohort (parity check of the code path).
- Non-vacuous: on real pseudo_c3 rows, the count of molecules with a generator top-1 is > 500.
