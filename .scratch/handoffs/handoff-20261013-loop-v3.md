# Handoff: loop v3 built (2026-10-13); next is the battery dumps, then the first v3 campaign

**For:** a fresh session, or any model via `.scratch/prompt.md`. **Audit + fixes:** `research/audit/20261013-loop-audit.md`.
**Decisions:** memory `casmi-loop-v3-grill`, plan `~/.claude/plans/read-scratch-handoffs-loop-audit-md-the-ancient-cerf.md`.

## What exists now (verified by tests: test_bin 62/62, test_campaign, harness tests)
- `research/bin/gate.py`: the rank-only gate (pre-registered in gate_report.md, 2026-10-13 block). `campaign.py` uses it.
- `research/harness/harness.py scorecard`:
  - covers every dumped cohort;
  - CI from a scaffold-grouped bootstrap;
  - leak-hit/other split and displacements;
  - composite score;
  - `knob_diff`.
- `research/harness/calibration.md`: harness-vs-LB pairs (2 rows).
- `campaign.py` adds:
  - a lock, plus the global Claude lock around every model call;
  - a retry cap with a progress check;
  - per-model call caps;
  - a read-only `--dry-run`, `--resume` and `--override`;
  - live history (now/lb);
  - the ideation chain (scout → diagnose → ideate) when `/replan` sets `ideate`;
  - `/fetch` for `research/requests/`;
  - `/report` after every cycle;
  - the dossier at cycle close.
- `kaggle_push.py` adds:
  - the `counter_value` fix;
  - the draft-diff vs the control draft (`firing_check`);
  - `E2_ROWS FALLBACK` as a fail marker;
  - push preconditions (pinned image, G2, preflight);
  - the GPU pool;
  - kernel version and image recorded in STATUS;
  - the queue refreshed on PASS.
- `poll_lb.py`: exact-token matching, `--claim` refs, controls never decided, `lb_pipeline`.
- `run_tests.py`: G2 keep-going with SKIP-INPUT, plus the harness tests.
- `dossier.py`: writes `research/submit_queue.md`, per-idea `card.md` and `digest_cycle<N>.md` (the 4 loop metrics).
- `gpu.py` keeps the ledger `research/gpu_ledger.json`. `promote.py` applies approved proposals, with selftest-diff
  rollback.
- `colab_run.py` adds `--check-mount` (live), fetch-on-failure, `CASMI_RESUME_DIR`, and `--kind train` gated by
  `colab_train: approved`.
- Skills:
  - new: scout, fetch, diagnose, ideate, report;
  - rewritten: replan, spec, attack, build, campaign-grill;
  - all line-generic (exploit/explore via config `lines.<line>`).
- `brief.py` is line-generic and covers the new steps; briefs are about 3k chars.
- Config: explore = modules grafted onto E006 (same control); `harness.cohorts` plus `class_weights`; `call_caps`;
  `web_settings`; the GPU pool; `control_draft`.

## In flight
- Kaggle kernel `syedtoubur/casmi-control-e006-draft` v1: the E006 v1 base notebook on its pinned image, draft
  only, never submitted. Its watcher writes `research/exploit/20261013-control-draft/kaggle_output/submission.csv`,
  which is the control draft G4 needs. Until it lands, every probe's G4 ends `needs-human` ("no control draft").
  Check with `python research/bin/kaggle_push.py --check research/exploit/20261013-control-draft`.

## Next steps (in order)
1. **Human, 5 min**:
   - review `research/proposals/20261013-loop-audit.md` (P1–P7), then run `python research/bin/promote.py research/proposals/20261013-loop-audit.md`;
   - paste the settings below.
2. **Battery dumps**, in the exploit worktree (`build/patch_harness_dump.py`):
   - C2_hard and C1_xlib: set `VAL_SHARD` only, push as instrument ideas like `research/exploit/20261010-harness-c2`,
     then `harness.py fetch`;
   - pseudo_c3 and c2_displace: port the pool-ban from `build/patch_c3cal.py` into VAL_CELL first (the P1 parquets
     carry `train_rows`, `lib_ban_keys`, `pool_ban_keys`; `p1_novel/*.parquet`);
   - each dump costs 2–3 GPU-h from the pool (`gpu.py`);
   - after each dump, re-run the promo-soften scorecard (`REL_PROMOTE=500`) and add its rows to `calibration.md`.
3. **First v3 campaign**: `/campaign-grill`; the human names the line.
4. Charter `research/campaigns/20261013-e006-structural` (another session) cites the 0.008 bar that the rank-only
   block supersedes. Re-grill it, or let `gate.py` apply (it does regardless).

## Settings the human pastes (Claude may not edit .claude/)
**a) New file `.claude/settings.headless.web.json`** for the scout and fetch steps. Copy
`settings.headless.json` and change only these parts:
- Remove `"WebFetch"` and `"WebSearch"` from `deny`.
- Add `"WebFetch"` and `"WebSearch"` to `allow`.
- Keep every Edit deny. Also add these to `deny`, because a web step should write only its outputs:
  `"Edit(//E:/casmi/lines/**)"`, `"Edit(E:/casmi/lines/**)"`, `"Bash(kaggle kernels push:*)"`.

**b) In `.claude/settings.headless.json` `allow`** (W9):
- Replace `"Bash(python:*)"`, which lets `python -c` write to LESSONS.md and config.yaml past the Edit denies, with:
  `"Bash(python research/*)"`, `"Bash(python E:/casmi/lines/*)"`, `"Bash(python //e/casmi/lines/*)"`,
  `"Bash(python build/*)"`, `"Bash(python scratch/*)"`.
- Trade-off: agents lose ad-hoc `python -c` checks. Keep the old rule if stalls come back.
- Add `"Bash(kaggle datasets download:*)"`. Keep downloads under `E:/casmi/data/`; the skills say so.

**c) In `deny`**: `"Edit(research/rules/**)"`, `"Edit(//E:/ENVEDA_casmi/research/rules/**)"`,
`"Edit(research/harness/gate_report.md)"`, `"Edit(research/harness/calibration.md)"`. Rules, pre-registrations and
calibration change only through promote.py or an interactive session.

## Verified vs hypothesis
- **Verified (tests + real runs)**:
  - the scorecard reproduces promo-soften's +0.0185 C2 / +0.0070 C1_npx;
  - the dossier renders the real queue;
  - control-draft push v1 accepted.
- **Hypotheses**:
  - Kaggle drafts are deterministic run to run, so `g4_draft_noise` is 0. Raise it if two control drafts differ.
  - The "other" (leak-free) Δ tracks the LB better than the overall Δ (1 data point).
