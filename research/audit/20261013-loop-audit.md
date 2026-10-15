# Loop audit and loop v3 (2026-10-12 → 2026-10-13)

Input: `.scratch/handoffs/loop_audit.md` and `loop_audit_evidence.md` (W1–W19). Three code audits plus spot checks
verified each row. The grill ran 2026-10-13 (rounds Q1–Q34). Plan: `~/.claude/plans/read-scratch-handoffs-loop-audit-md-the-ancient-cerf.md`.

## Verified ledger (V = verified, P = partly; N/new = found in this audit) and what was done

**A. A probe could pass G4, or spend a slot, wrongly**

| id | verdict | finding | fix (2026-10-13) |
|---|---|---|---|
| N2 | V new | `counter_value` read the number after the match, so `engine lists \d+` read the runtime | kaggle_push.counter_value + test |
| N1/W6 | V | promo-soften passed G4 with print-only counters (`promoted: 0`); there was no comparison against a control | G4 draft-diff vs research/exploit/20261013-control-draft (`gates.g4_draft_diff`) |
| N3 | V new | `E2_ROWS FALLBACK` was not a fail marker | added to `gates.g4_fail_markers` |
| GATE | V new | the harness gate passed any BLIND/BASE-COUPLED idea with a prior.md, which the spec agent wrote itself | research/bin/gate.py (rank-only) + gate_report 2026-10-13 block |
| W3/W11/N4 | V | overrides and deviations existed only in prose; a "died" attack did not block | `--override --reason`; `deviation` and `blocking` fields in stress_result.json; gate.py |
| W17 | V | there was no effect-size gate; config bars were unread | superseded by the rank-only gate (human decision) |
| W13 | V+ | submissions matched by substring, one submission could match several ideas, and controls got verdicts | exact-token match, `poll_lb --claim` refs, controls never decided |
| W7 | V | a control without an image pin let a probe run silently on Kaggle's latest image | push_preconditions; the image is recorded at push |
| N5/N6 | V new | a push needed no clean preflight, and a G2 FAIL did not block | push_preconditions |
| W19 | V | KEEP rests on a single control run | kept as is (human decision) |

**B. Orchestration**

| id | verdict | finding | fix |
|---|---|---|---|
| W14 | V+ | no campaign lock, and campaign calls skipped the global Claude lock | campaign.lock + the global lock around every model call |
| LOOP | V new | the exit code was ignored, so a step that made no progress re-ran forever | `progressed()`, step_retries 2 → needs-human |
| W15 | V+ | `--dry-run` wrote files and could stop a campaign | `describe()`, read-only |
| W4 | V | no checkpoints | autonomy up to ready-to-submit; the human gate is the submit (+ submit_queue.md) |
| CRASH | V new | the gate crashed on bad JSON or a missing delta | gate.py never raises |
| HIST | V new | history froze at g4-pushed, and LB results never reached /replan | refresh_history (now / lb / decision), `submitted` count |
| W12 | V | `family_of` used all of b; the charter's JSON broke parse_set | `knob_diff(a, b)`; skills and briefs use KEY=value |
| W9 | P | relative Edit rules were already fixed; `Bash(python:*)` bypasses the Edit denies; `kaggle datasets download` missing | settings proposal for the human (handoff) |
| W8 | V | G2 stopped at the first failure | run_tests.py keep-going, SKIP-INPUT, harness tests included |
| W16 | V (minor) | briefs ran 26–754 chars over 6 k | briefs are now about 3 k (line-generic reads) |
| W10 | P | R04 bare-filename false positives, no fixture | R04 code-dir lookup + 2 fixtures |
| W18 | V | campaigns never learned | /report → research/proposals/ → promote.py (human approves) |
| W2 | V | LB keys had no pipeline tag | `lb_pipeline`; attack skill rule |
| W5 | P | heavy-path checks could not run | heavy path ranks by a measured ceiling (gate.py) |

**C. Instrument**

| id | verdict | finding | status |
|---|---|---|---|
| W1 | P, cause corrected | the cheap families' coupling was 0.003–0.019, under the 0.05 limit; TRUSTED was unreachable as coded (≥ 3 S/A flips per family, a global `lam_ok`, an uncoded CI clause) | replaced by the scorecard + calibration ledger (gate_report 2026-10-13) |
| STAT | V new | no grouped bootstrap; "match" ignored the CI; coupling = max over extreme sweeps | scorecard: Bemis-Murcko-grouped bootstrap, leak split, displacements |
| CAL | V | 4 sign-only LB points | research/harness/calibration.md (2 paired rows so far) |

**D. Research capability**

| id | finding | fix |
|---|---|---|
| I1 | no idea generation; a fixed 3-row portfolio | /scout (L, web) → /diagnose (M) → /ideate (H) when the pool runs dry; /fetch answers research/requests/ |
| I2 | LB results never reached replan | history refresh + calibration.md in the replan reads |
| I4 | no Reporter | /report each cycle |
| I5 | 3 slots per campaign, ±0.004 noise | rank-only queue: the human picks what to submit, with numbers |

## Ranking used (cycles or slots lost × frequency ÷ cost)
1 silent G4 pass · 2 orchestration safety · 3 self-certified gate · 4 no ideation or learning · 5 instrument ·
6 LB bookkeeping · 7 the rest. Build order followed this ranking.

## First harness-vs-LB pairs
- promo-soften: harness C2 +0.0185, of which +0.0204 on leak-hit and +0.0020 on other molecules → LB −0.003.
- engine2-fppair: heavy path, ceiling +0.021 C1_npx → LB −0.011.

## Grill decisions
See the memory file `casmi-loop-v3-grill` and the plan. In short: autonomy up to the submit; rank-only gate;
corrected multi-cohort battery (equal class weights); ideation engine (scout L → diagnose M → ideate H); lesson and
rule changes go through approved proposals; Kaggle GPU pool 30 h/week, ≤ 4 h per training run; Colab training
only with approval; the loop is line-generic, and explore = modules grafted onto E006.

## Open after this session
- Battery dumps for C2_hard, C1_xlib (VAL_SHARD only), pseudo_c3 and c2_displace (port the patch_c3cal pool-ban into
  VAL_CELL). Each costs 2–3 GPU-h from the pool.
- Settings: the web settings file and narrowing `Bash(python:*)` (human paste).
- The human approves or rejects research/proposals/20261013-loop-audit.md (P1–P7).
- The `20261013-e006-structural` charter cites the 0.008 bar, which the rank-only block supersedes.
