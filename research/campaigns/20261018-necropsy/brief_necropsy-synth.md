# Brief: necropsy-synth

**Goal**: Turn every necropsy into ranked revival and combination candidates that R12 accepts.

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
- R04: Every cited file:line exists (repo or a line worktree) and the file has that many lines.
- R05: A pushed probe has at least one counter whose printed value must be > 0; presence alone proves nothing (L05).
- R06: A submit card names the paired own-account control kernel; KEEP is decided only by poll_lb.decide() (L05, L06).
- R08: A harness result counts only for families the capability map does not call BLIND or BASE-COUPLED; it must carry its family + verdict, and such a family's idea goes to the LB on a written prior.md (LB decides).
- R09: A spec that cites a spike log must have spike_result.md filled (question, script, numbers, verdict), not the blank template.
- R11: A per-child trigger may not place a generated candidate above its parent (or at rank 1) unless its C1/C2 false-alarm rate is MEASURED at or below 4.5 %; every CPU child-vs-parent signal measured fires on 21-71 %.
- R12: A revival of a dead idea must name its cause and a fix that changes mechanism, data or target stratum; re-tuning a knob, reusing a signal already measured dead, or combining parts without distinct strata is refused.

## Judgment lessons cited by this campaign or idea
- none cited

## You may open only
- every research/*/*/necropsy.md (`python research/bin/necropsy.py --json`)
- research/necropsy/dead_signals.yaml
- research/harness/gate_report.md (headers)
- E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/charter.md
- E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/state.json
- .claude/skills/necropsy-synth/SKILL.md

## Output contract
research/ideas/20261018-revivals.md: >= 3 ranked candidates, each with `revives:`, `cause:`, `fix:` (a mechanism, data or stratum change, never a re-tune), `combines:` with the cohorts/strata when it combines, falsifier, cheapest harness measurement, expected delta vs the 0.004 noise floor, line necropsy. Then set E:/ENVEDA_casmi/research/campaigns/20261018-necropsy/state.json `ideate` to null.

Never read README, LESSONS or EXPERIMENTS in full, never read a raw .ipynb, never ask a question (write it to questions.md), never submit. Need one specific fact that is not in the repo (a paper detail, a model's licence, whether pretrained weights exist)? Write `research/requests/<yyyymmdd>-<slug>.md` (`status: open`, the question, why it matters) and continue; a web-enabled fetch step answers it before your next step.
