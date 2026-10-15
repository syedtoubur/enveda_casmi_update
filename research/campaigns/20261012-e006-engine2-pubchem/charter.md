# Campaign charter: 20261012-e006-engine2-pubchem

Grill 2026-10-12 (interactive, Opus). **Supervised run**: the human steps the driver with `campaign.py --once` and
decides at three checkpoints: after /replan, at harness-done, and before /build ship. The purpose is both to move the
LB and to collect paired harness-vs-LB data points, which tell us whether the harness can be trusted.

## Scope
- Line: exploit. Base and paired control: **E006 v1** = `syedtoubur/casmi26-e006-window-40-ice-fallback-true`
  version 1, LB 0.440 (`research/exploit/20261012-control-e006`). Version 2 of that kernel is the smart gate (0.429,
  REVERT) and is never the base. Not replicated, by the human's choice.
- Probe base notebook: `casmi26-e006-window-40-ice-fallback-true.ipynb` in the worktree. Cite base lines from
  `scratch/casmi26-e006.py` (worktree).
- Every harness compare uses the control's config as `--a`: `{"METRIC_FUSION": true, "ICE_FALLBACK": true, "FUSION_WINDOW": 40}`.
  `--b` = that config plus the knob. Report C2 and C1_npx, the 90% CI, coupling, and the leak-hit / other split.
- Cycle order: **IDEA1 (engine2-fppair) first**, then promo-soften (A), then slots-gentle (B) only if A is not dead.
- Heavy-path ideas (IDEA1) skip the replay: stress_test.py checks the inputs offline, and harness_result.json carries
  verdict BLIND, heavy_path true. prior.md is written from the spec, and G4 plus the LB decide.
- BASE-COUPLED families (A, B): the harness number is reported but cannot be cited. The human decides at the
  checkpoint whether to write prior.md and spend an LB slot.

## Portfolio
| slug | branch | slice | claim | knob | heavy-path | prior | EV |
|---|---|---|---|---|---|---|---|
| engine2-fppair | engine2-model | C2 rank 2–5 (51%, +0.298 headroom) | Replacing Engine 2's no-fingerprint ranker rows `sim_rank_rows_nofp.npz` with `sim_rank_rows_fppair.npz` plus the leak-free FPNets (`ours_fpnet_single_16k.pt`, `ours_fpnet_merged_26k.pt`) raises Engine 2's own ranking, and RRF carries that into the fused list | the Engine 2 cell (`E.main(... find('sim_rank_rows_nofp.npz') ..., w_pv=0.88)`, `.scratch/nb2_e006.py:235`); the spec must establish which blend side (pv 0.88 / ours) reads the npz and how the fp feature blocks are computed at inference (L02, L25) | yes | Dataset card `specs/Dataset_intro/Simulated_Ranker_Rows.md:11-14`: Class-2 MRR 0.655 → 0.689 (fp16k) → 0.710 (fppair) on np-examples 5-fold CV. L10: orthogonal/stronger views pay; L01: FPNets hold out all 3,150 sim structures | high |
| promo-soften | pubchem-voice | C2 + C1_npx rank 2–5 | Less PubChem promotion: raising `REL_PROMOTE` 250 → 500 stops PubChem candidates from displacing better fused candidates | `REL_PROMOTE` (C9 promotion, `build/casmi_post.py:38`) | no | Harness on the 0.433 config: REL_PROMOTE 500 +0.019 C2 / +0.009 C1_npx; with promotion fully off, C1_npx "other" +0.021. Family BASE-COUPLED (0.156). L12 popularity overshoot | med |
| slots-gentle | pubchem-voice | C2 rank 2–5 | Fewer gentle merge slots for PubChem: `SLOTS_GENTLE` [4,8,12,16,20] → [2,4,6,8,10] | `SLOTS_GENTLE` (`build/casmi_post.py:38`) | no | Harness on the 0.433 config: +0.016 C2 / −0.003 C1_npx; family merge_gate BASE-COUPLED (0.267) | low–med |

## Bars
- KEEP (poll_lb.decide): LB ≥ E006 0.440 + 0.004 = **0.444**, and the counters fired (values > 0). A replicate is
  required before the final KEEP (L05).
- Harness (A, B): pre-registered LB-slot bar from gate_report.md (Δ ≥ +0.005 on the target slice, 90% CI lower bound
  > 0, same sign on the leak proxy). Because both families are BASE-COUPLED, the bar is reported, not enforced. The
  human decides, and a prior.md records the reasoning.
- IDEA1 G4: complete with no `g4_fail_markers`, the line's `g4_required` regexes present, and value-checked counters
  proving (a) the fppair rows were loaded (row count > 0, feature count printed), (b) the leak-free FPNet features
  are non-zero for the candidates, (c) Engine 2 lists exist for every molecule (`engine lists 400` on the rerun). The
  runtime stays ≤ 9 h on one T4 (L24).

## Stop rules
max_cycles: 3
max_lb_probes: 3
dead_streak: 2
quota_pause_h: 12

## Out of scope
- New FPNet training, ranker retraining beyond the shipped fppair rows, and any FPNet swap without its matching rows
  (L02).
- GL_LAM / ICE_LAM retuning (L08), the smart library gate (REVERT, separate track), and the DreaMS-fill harness
  instrument.
- Class-3 / de novo generation (L18), and new val-mode harness dumps.
- Any `kaggle competitions submit`: the human submits.

## Amendment 2026-10-12 (human, after the cycle 1–2 LBs)
- LBs vs E006 0.440: engine2-fppair v2 **0.429** (−0.011, revert; bundled rows + w_pv 0.88→0.50); promo-soften
  **0.437** (−0.003, revert; the harness +0.0185 C2 did not transfer).
- **Cycle 3 = engine2-pv88** (`research/exploit/20261012-engine2-pv88`), replacing slots-gentle. It is the same
  generator (`build/patch_engine2_fppair.py --w-pv 0.88`), so only the fppair rows + leak-free FPNets differ from
  E006. Purpose: attribute the −0.011. It runs without replan/spec/attack (one constant in a G4-proven generator).
  slots-gentle is dropped because it is the same BASE-COUPLED pubchem-voice family as the flat promo-soften.
- Pre-registered reading bands (LB vs E006 0.440), fixed before the push:
  - ≥ 0.444: the rows help and w_pv 0.50 caused the loss → keep-pending + a replicate.
  - 0.436–0.443: the rows are neutral at 0.88, so the −0.011 is attributed to w_pv 0.50 → close both the weight
    change and the rows (no LB lift).
  - ≤ 0.435: the fppair rows hurt by themselves → close fppair rows on the engine2-model branch.
- It uses the last LB probe (3/3); the campaign closes after it is scored.
