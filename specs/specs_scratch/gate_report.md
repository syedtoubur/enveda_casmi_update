# Gate report

Thresholds are written **before** the run. Verdicts are appended after, never edited.

## R0 — port of the Track 3 base (pre-registered 2026-10-05)
Reference: `casmi_simple_sim_submission.ipynb` with `CFG.MAX_PEAKS=100` → LB 0.296 (own run).

| Gate | Threshold | Why |
|---|---|---|
| R0.1 package tests | `python scratch_pkg/tests/run.py` green | |
| R0.2 fast commit | finishes; `sim_ok_share ≥ 0.99`; tier-1 rate ≥ 0.9 on the public file | the public file is train rows, so a real library match must win |
| R0.3 full run | `capped_rows == 0`, `failed_rows == 0`, total ≤ 7.5 h | |
| R0.4 LB parity | LB ≥ 0.292 (0.296 − noise 0.004) | R0 also extends `MAX_PEAKS=100` to `sv`, so a move up is possible; a drop below 0.292 means a port bug → diff against the base with `compare.py` before anything else |

### Verdicts
_(none yet)_
