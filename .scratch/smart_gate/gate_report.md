# Smart gate: pre-registered bars (written 2026-10-11, before any Kaggle run; never edit the thresholds)

Arm: `casmi26-e006-smartgate.ipynb` (built by `build_nb.py` from `casmi26-e006-window-40-ice-fallback-true.ipynb`).
Design and audit: `PLAN.md`.

## Paired control (required)
E006's 0.440 is an **external claim** that has not been run on our account (EXPERIMENTS.md header, 10-11).
Submit the unchanged E006 notebook as the paired control with the same data versions and T4 GPU. Every delta
below is measured against **that** score, never against 0.440 and never against 0.433.

## Gates
| gate | when | PASS iff |
|---|---|---|
| S0 local | before push | `test_smart_gate.py` failures 0; `test_overlay_integration.py` prints `overlay integration OK`; `build_nb.py` writes 34 cells |
| S1 smoke | visible commit (12 molecules, placeholder test) | the run completes; the log has `SMART_GATE {"health": "ok"`, `FACTORIAL_VERIFICATION` with `"experiment": "E006_SG"` and `same_run_control_reconstructed: true`, and `E004_VERIFICATION`. Tier counts are placeholder-only and mean nothing |
| S2 runtime | hidden rerun | `SMART_GATE ... "secs"` ≤ 900 and `timed_out` 0; the notebook finishes inside 9 h |
| S3 alive | hidden rerun | t1+t2+t3 ≥ 10 in `merge stats` AND pinned_moved + inserted + top2_moved ≥ 1 in `sg_events`. If not, the LB delta is noise by construction |
| S4 keep | LB | arm ≥ paired control + 0.005 → KEEP. Between −0.004 and +0.005 → NEUTRAL (do not stack). Below −0.004 → REGRESSION |

## If S4 regresses
T3 is the first suspect: it has the highest uncalibrated false-lock rate (multi-spectrum max ×k, NP isomer families).
The next arm sets T3 to audit-only: `T3_ACTIVE = False` in `smart_gate.py` (T3 is still logged but acts like T4), then rebuild. One LB slot.

## Results
(fill after runs)
