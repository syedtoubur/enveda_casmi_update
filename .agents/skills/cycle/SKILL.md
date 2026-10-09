---
name: cycle
description: Start one CASMI research cycle on a line (exploit or explore). Grill the human on the next hypothesis, then on your own research, spike, spec, get it critiqued and hand it to Claude. Use when the user types /cycle <line>.
---

# Cycle (Ideator)

Argument: `exploit` or `explore`. You are the **Ideator**: Gemini Flash, cheap, high-volume. Claude is scarce: call it only with a spec that has survived a critique.

Read first, and only these: `research/README.md`, `research/config.yaml` (your line), `research/LESSONS.md`, the **last 15 rows** of the EXPERIMENTS.md table, the newest `research/daily/*.md`, and the STATUS of the open ideas in `research/<line>/*/`.

## 1. Grill the human (the only interactive step)
Use the grill-me skill. Before asking, prepare **3–5 candidate hypotheses** drawn from:
ledger gaps (untested or confounded rows, VOIDs never re-run), lessons that point at an untried lever (L10: orthogonal sources), the open ideas' results, and new public Kaggle notebooks or discussions.
For each candidate: claim, class targeted, arithmetic ceiling (L13), cost, and dead-end overlap.
Ask the human to pick one or reshape one, plus their priorities and constraints for this cycle. Re-read free-text answers and clarify anything ambiguous.
Then run `python research/bin/new_idea.py <line> <slug>` and fill `hypothesis.md`. **From here on, do not ask the human anything**; write questions to `questions.md`.

## 2. Research
Use the research skill and the science plugin (literature_search_arxiv / europepmc / openalex, pubchem_database) for prior art, Kaggle notebooks and discussions for this competition, and grep over EXPERIMENTS.md and specs/ for related rows. Every claim gets a source. To compare notebooks, use `python .agents/skills/analyze-notebook/scripts/compare.py`, never the raw .ipynb. Write `research.md` with 3 ranked concrete ideas. STATUS `spiking`.

## 3. Spike
Use the `spike` skill on the top idea; if it shows NO SIGNAL, spike the second. If both fail, set STATUS `dead` with the reason on line 2, propose a one-line lesson in `spike_result.md`, and stop.

## 4. Spec
Use the to-spec skill in autonomous mode (see `docs/agents/issue-tracker.md`): write `spec.md` from the template, including the **full CASMI section** with numeric gate thresholds pre-registered. Exploit line: the change is one patch on `casmi-notebook-0-433.ipynb`. Explore line: a `scratch_pkg` module change. One change per idea (L15). STATUS `specced`.

## 5. Critique (separate agent)
Start a **new agent** with fresh context and give it only: "Use the critic skill on research/<line>/<idea>". If it returns BACK-TO-IDEATOR, fix spec.md and repeat, at most 2 rounds; then hand to Claude anyway with the open points listed at the top of critique_gemini.md.

## 6. Hand to Claude
Set STATUS `ready-for-judge`, then run:
`python research/bin/call_claude.py research/<line>/<idea> --detach`
If Claude is busy, the idea is queued. When STATUS later reads `revise`, read `verdict.md`, revise spec.md, set `ready-for-judge` and call again. Bounces are counted in STATUS; at 3 the judge escalates to the human.

Token rules: pass file paths, not contents. Never read EXPERIMENTS.md in full. Every role starts in a fresh agent.
