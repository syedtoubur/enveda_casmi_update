# Brief: replan

**Goal**: Pick the next idea of this campaign (or stop) from the portfolio and the idea pool, the loss map and every finished cycle's result (state.json history carries each idea's current STATUS and LB).

## Campaign scope (from charter.md; decisions already made, do not reopen)
- Line: necropsy
- Mode: necropsy
- Base: V22 = `casmi26-v22-rel150-e2r2-clean` (keep_bar 0.451, paired control `research/c3/20261017-control-v22`);
  worktree `E:/casmi/lines/necropsy`, branch `line/necropsy`.
- Idea pool: `research/ideas/*-revivals.md`, written by `/necropsy-synth` from the `necropsy.md` of every dead idea
  (all lines). When the pool is dry the driver autopsies up to 5 more dead ideas and re-synthesises.
- Each revival is a NEW idea folder under `research/necropsy/`, whose hypothesis.md carries `revives:`, `cause:`, `fix:`
  (rule R12). Combinations are allowed only when the parts act on different cohorts or leak-hit/other strata.
- Ideas from exploit (E006 base) that need porting to V22 say so in spec.md; the port is part of the build.
- Heavy compute runs on Kaggle (CPU default; GPU within the 30 h/week pool, autonomous since 2026-10-18).
- Isolation: never write in another line's worktree or another campaign's idea folders; necropsy.md files in dead
  idea folders are the only exception (side files, STATUS untouched).

### Bars
- Gate: the rank-only gate (`research/harness/gate_report.md`, 2026-10-13 block); judge by the "other" Δ, not leak-hit.
- KEEP at submit: `poll_lb.decide` (paired V22 control + 0.004, counters > 0, one replicate).

### Out of scope
- Re-running a measurement that already killed an idea without a mechanism change (R12).
- Signals in `research/necropsy/dead_signals.yaml` unless re-MEASURED (R10, R11).
- Edits to the loop code, skills, CLAUDE.md, or other campaigns' charters.

## Rules preflight.py enforces on your output (do not restate them; satisfy them)
- R01: Counts from local test.parquet or a draft run are placeholder data (a train slice), never the hidden test (L04).
- R02: Every MEASURED number cites what produced it: a harness result json, a script with line, or a log file.
- R03: A spike must read data (a dump, cached arrays, a log) in its measuring loop; a loop over literal numbers or `pass` is a formula, not a measurement.
- R05: A pushed probe has at least one counter whose printed value must be > 0; presence alone proves nothing (L05).
- R06: A submit card names the paired own-account control kernel; KEEP is decided only by poll_lb.decide() (L05, L06).
- R08: A harness result counts only for families the capability map does not call BLIND or BASE-COUPLED; it must carry its family + verdict, and such a family's idea goes to the LB on a written prior.md (LB decides).
- R09: A spec that cites a spike log must have spike_result.md filled (question, script, numbers, verdict), not the blank template.
- R10: A candidate scorer's reference must be the query's measured spectrum; a spectrum simulated from the truth structure ranks the truth first by construction.

## Judgment lessons cited by this campaign or idea
- none cited

## You may open only
- E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/charter.md
- E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/state.json
- E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/log.md
- research/harness/loss_map.md
- research/ideas/ (newest candidates file, if any)
- research/harness/gate_report.md (Results section only)
- result files named in log.md

## Output contract
Either `research/bin/new_idea.py necropsy <slug>` + hypothesis.md (claim, branch, target cohort, mechanism with code cites, prior with source: MEASURED or ASSUMED, `novelty: new|variant`, source) and state.json `current` = that folder; or, when neither the portfolio nor the newest candidates file has a live idea left, state.json `ideate` = <why> (the driver runs scout/diagnose/ideate and calls you again); or state.json `stop` = <reason> only for a scope problem a human must settle. Append one line to log.md.

Never read README, LESSONS or EXPERIMENTS in full, never read a raw .ipynb, never ask a question (write it to questions.md), never submit. Need one specific fact that is not in the repo (a paper detail, a model's licence, whether pretrained weights exist)? Write `research/requests/<yyyymmdd>-<slug>.md` (`status: open`, the question, why it matters) and continue; a web-enabled fetch step answers it before your next step.
