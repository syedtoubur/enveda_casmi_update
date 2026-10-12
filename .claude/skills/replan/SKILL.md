---
name: replan
description: Campaign step (Opus, headless): pick the next CASMI idea or stop the campaign. Invoked by research/bin/campaign.py as `/replan research/campaigns/<id> --brief <path>`.
---

# Replan

You are the campaign's strategist between cycles. Read the brief first; it lists the only files you may open.

1. **Read the evidence**: the charter portfolio, `state.json` (history, closed branches), `log.md`, the loss map, and the result files of every closed cycle (harness_result.json, stress_result.md, the STATUS reason). Done when you can state, for each closed cycle, one line on what it taught about the loss map.
2. **Rank the open portfolio**: EV = reach (molecules in the loss-map bucket it targets) × expected rank gain (MEASURED where a harness result exists, else ASSUMED with its source) × P(survives attack) ÷ cost (post-scoring knob = 1, heavy path = 5). Drop dominated ideas and ideas on a closed branch. You may reshape an idea inside the charter scope; anything outside the scope goes to `questions.md` for the next grill.
3. **Decide**:
   - Next idea: run `python research/bin/new_idea.py exploit <slug>`, then write `hypothesis.md` (claim, branch, target slice, mechanism with code cites, prior with source, the EV line) and set STATUS to `proposed` with a `branch: <name>` line. Set `state.json` `current` to the folder path (repo-relative).
   - Or stop: set `state.json` `stop` to a one-line reason (portfolio exhausted, every branch closed, scope needs a human).
4. Append one line to `log.md`: the pick (or the stop) and its EV line.

Done when `state.json` holds exactly one of `current` (with a `proposed` folder behind it) or `stop`.
