---
name: replan
description: Campaign step (tier H, headless): pick the next CASMI idea, ask for new ideas when the pool is dry, or stop the campaign. Invoked by research/bin/campaign.py as `/replan research/campaigns/<id> --brief <path>`.
---

# Replan

You are the campaign's strategist between cycles. Read the brief first; it lists the only files you may open.

1. **Read the evidence**:
   - the charter portfolio;
   - `state.json`: history (each idea's status now and its LB), closed branches;
   - `log.md`;
   - the loss map;
   - `research/harness/calibration.md`;
   - the result files of every closed cycle (`harness_result.json`, `stress_result.md`, the STATUS reason).

   Done when you can state, for each closed cycle, one line on what it taught. An LB result outranks any harness number.
2. **Rank the open pool**, which is the charter portfolio plus the newest `research/ideas/*-candidates.md`.
   - EV = reach × expected gain (MEASURED where a scorecard exists, else ASSUMED with its source) × P(survives attack) ÷ cost (knob = 1, pipeline = 2, heavy path or training = 5).
   - Prefer the "other" (leak-free) Δ over the overall Δ where calibration.md says it tracks the LB better.
   - Drop dominated ideas and ideas on a closed branch.
   - You may reshape an idea inside the charter scope.
3. **Decide**:
   - **Next idea**:
     - run `python research/bin/new_idea.py <line> <slug>`;
     - write `hypothesis.md`: claim, branch, target cohort, mechanism with code cites, prior with source, `novelty: new|variant`, source (scout link, diagnosis row, or mutation of <idea>), and the EV line;
     - set STATUS to `proposed` with a `branch: <name>` line;
     - set `state.json` `current` to the folder path (repo-relative).
   - **Pool dry**: if no live idea is left, set `state.json` `ideate` to a one-line reason. The driver runs scout, diagnose and ideate, then calls you again.
   - **Stop**: set `state.json` `stop` only for a problem a human must settle (scope, rules, data access).
4. Append one line to `log.md`: the pick, the ideate request or the stop, and its EV line.

Done when `state.json` holds exactly one of `current` (with a `proposed` folder behind it), `ideate`, or `stop`.
