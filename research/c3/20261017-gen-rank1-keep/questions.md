- **G4 push blocked by preflight R08** (2026-10-17 20:03): `harness_result.json` has `family: "unknown:GEN1_KEEP"`; the
  knobs `E2R2, FRAG_GAP, REL_PROMOTE (promotion), GEN1_KEEP, ...` of the c3 base cfg are not all in
  `research/harness/backtests.yaml` `families:` (that file is append-only and pre-registered, so I did not edit it, and
  I did not hand-edit the gate's `harness_result.json`).
  Needed from the human: add `GEN1_KEEP` (and the missing V22 knobs `FRAG_GAP`, `E2R2`) to a family in `backtests.yaml`
  (suggestion: new family `gen1_keep: [GEN1_KEEP]`, `rank_overlay: [FRAG_GAP, E2R2]`), re-run the scorecard so
  `harness_result.json` carries a mapped family + verdict, and, if the family is BLIND / BASE-COUPLED (pseudo_c3 is
  FPNet-leaky, the gain sits on leak-hit molecules), write `prior.md` (LB decides). Then rerun:
  `python research/bin/kaggle_push.py research/c3/20261017-gen-rank1-keep E:/casmi/lines/c3/casmi26-v22-gen1-keep.ipynb --slug casmi26-gen-rank1-keep --counter "GEN1_KEEP gen1_checked" --counter "?GEN1_KEEP gen1_kept" --counter "?GEN1_KEEP gen1_moved"`
  Everything else is ready: probe notebook, G1/G2 in gates.md, card.md, STATUS `g4_min_changed: 1`.
- **Control replicate already pushed** (this was not blocked): `syedtoubur/casmi26-control-v22-replicate` v1,
  folder `research/c3/20261017-control-v22`. `lines.c3.control` in config.yaml still points at the E006 placeholder, so
  the probe's draft diff will be against the E006 draft until the human repoints it.
- **G4 fix round 1 -> needs-human** (2026-10-18 02:47): the draft vs the V22 control is identical (0/400), because
  GEN1_KEEP legitimately fired 0 times on the 12 visible molecules (gen1_checked 12, kept 0; generated #1 is ~1-2% on
  C1/C2, so 0/12 is expected). No code bug found (g4_fix_1.md). Decide one of:
  (a) accept 0 draft fires: set STATUS `g4_min_changed: 0` (L38 waiver; the hidden-rerun `gen1_kept` counter is the
      evidence) and re-run the brief's push command, or mark ready-to-submit from the existing completed run;
  (b) keep the gate and drop the idea from LB testing (it cannot change a 12-molecule draft by design).
  Re-pushing unchanged would fail the same way, so I did not push.
