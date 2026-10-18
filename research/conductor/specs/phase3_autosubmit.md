# Phase 3 spec: auto-submit within budget

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decision 12).
Builder: Sonnet subagent working DIRECTLY in E:/ENVEDA_casmi (no worktree; other agents write `research/bin/conductor.py`
and `research/bin/desk.py` concurrently; touch only the files listed).

## Goal
The conductor may spend every daily Kaggle submission slot except `autosubmit.reserve_for_human` (1). It submits only
ideas that passed every code gate, binds the submission to the idea, and never touches the final-submission selection.

## Read first
`research/bin/poll_lb.py` (fetch, claim, decide, KEEP_MARGIN, OPEN states, `--claim` flow), `research/bin/dossier.py`
(`ideas()`, `score_of()`, `recommend()`, `write_queue()`, `kernel_ref()`), `research/bin/gate.py` (`evaluate`),
`research/bin/kaggle_push.py` (G4 PASS markers, draft diff vs control: grep `g4_draft_diff`), `research/bin/_common.py`,
`research/config.yaml` (`competition`, `lines.<line>.control`, `autosubmit:`), `research/README.md` (STATUS machine).

## Files
- NEW `research/bin/autosubmit.py`
- NEW `research/bin/test_autosubmit.py` (fakes only; never call the real `kaggle competitions submit`)
- EDIT `research/bin/run_tests.py`: add test_autosubmit.py to the repo test list.

## `python research/bin/autosubmit.py --once [--dry-run]`
1. Refuse (print one line `autosubmit: off (<why>)`, exit 0) when `autosubmit.enabled` is false, or
   `daily_submissions` is null, or `research/conductor/conductor.yaml` has `paused: true`.
2. Today's count: `kaggle competitions submissions -c <competition> --csv` (reuse `poll_lb.fetch(cfg)`), count rows whose
   date is today in UTC (Kaggle's day boundary is UTC midnight; document that). `budget = min(daily - reserve,
   max_auto_per_day or inf) - auto_submitted_today` where `auto_submitted_today` comes from `research/conductor/autosubmit_log.jsonl`;
   also require `daily - count_today > reserve` (the human's slot is never consumed, even if the human already used some).
3. Candidates, in priority order:
   a. **Replicates**: ideas in STATUS `keep-pending` (poll_lb says "submit one replicate") with no `lb_rep` → submit the
      same kernel version again.
   b. **Line controls not yet scored** that probes are waiting on (STATUS `scored-unpaired` probes whose line control has
      no `lb`): submit the control (paired control is required before any KEEP).
   c. **Queue items**: STATUS `ready-to-submit` ideas ranked like `dossier.write_queue()`, eligible only if ALL hold:
      `recommend(score_of(idea), meta) == "submit"`; `gate.evaluate(...)["ok"]` is True (or STATUS has `gate:` line from
      campaign.py's harness gate); G4 PASS recorded in STATUS/gates.md AND draft differs from control (check how
      kaggle_push records it; require the evidence string, else ineligible); the idea's line control (config
      `lines.<line>.control`) exists and has `lb:` in its STATUS (paired control scored); `kernel:` and
      `kernel_version:` present; not `kind: control|instrument`; no `override:` in STATUS (human overrides are human-submitted).
4. For each candidate while budget > 0: `kaggle competitions submit -c <competition> -k <kernel> -v <version>
   -f submission.csv -m "<idea folder name> | <kernel>@v<version> | auto"` (verify flag names with `kaggle competitions
   submit --help`, which is safe to run). On success: append a JSON line to `autosubmit_log.jsonl` (time, idea, kernel,
   version, kind a/b/c), then run `poll_lb.claim` the same way `poll_lb.py --claim <idea>` does (call `poll_lb.main(["--claim", rel])`),
   which moves STATUS to `submitted`. On failure: log, mark the idea STATUS line `autosubmit_error: <msg>` via
   `write_status(idea, None, autosubmit_error=...)`, continue.
5. `--dry-run` prints the plan (budget, candidates with eligibility reasons) and submits nothing.
6. `final_pick_note() -> str` (importable): a markdown paragraph recommending the two final submissions = best `keep`
   idea by mean LB (lb, lb_rep) + the most different high-LB idea (different line or different family), with numbers. It
   NEVER selects anything on Kaggle; it is text for the digest.

## Tests
- off states (disabled / null daily / paused) → no submit call.
- budget math: daily 5, reserve 1, 2 human submissions today, 1 auto today → 1 slot; daily 5, 4 used → 0.
- eligibility: each failing condition makes an idea ineligible with a reason; replicate and unscored-control priorities.
- success path: fake submit runner → log line + claim called with the idea path.
- `--dry-run` on the real repo runs without crashing and submits nothing (fake runner injected, or `enabled` forced
  false → prints off). Paste its output.

## Done checklist
- `python research/bin/test_autosubmit.py` green; `python research/bin/test_bin.py` no new failures (67/68 baseline).
- Do not commit; do not edit config.yaml (the human enables it). Report: files, test tails, dry-run output, deviations,
  and the exact `kaggle competitions submit --help` flags you used.
