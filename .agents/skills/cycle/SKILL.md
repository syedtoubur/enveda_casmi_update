---
name: cycle
description: Start one CASMI research cycle on a line (exploit or explore). Grill the human on the next hypothesis, then on your own research, spike, spec, get it critiqued and hand it to Claude. Use when the user types /cycle <line>.
---

# Cycle (Ideator)

Argument: `exploit` or `explore`. You are the **Ideator**: Gemini Flash, cheap, high-volume. Claude is scarce: call it only with a spec that has survived a critique.

Read first, and only these: `research/README.md`, `research/config.yaml` (your line), `research/LESSONS.md`, `research/<line>/BASE.md` (what the base really does), the **last 15 rows** of the EXPERIMENTS.md table, the newest `research/daily/*.md`, and the STATUS of the open ideas in `research/<line>/*/`.

**Control first (L31)**: if the line's `control` idea (config.yaml) has no `G4 | … | PASS` row in its gates.md, run `python research/bin/kaggle_push.py --check` once. If the control is still not PASS, tell the human in the grill: probes on this line can be specced, but their push will be refused until it is.

## 1. Grill the human (the only interactive step)
Use the grill-me skill. Before asking, prepare **3–5 candidate hypotheses** drawn from:
ledger gaps (untested or confounded rows, VOIDs never re-run), lessons that point at an untried lever (L10: orthogonal sources), the open ideas' results, and new public Kaggle notebooks or discussions.
For each candidate: claim, class targeted, arithmetic ceiling (L13) with each factor marked MEASURED or ASSUMED, cost, and dead-end overlap. Any statement about what the base does now cites BASE.md or a code line; if BASE.md does not answer it, grep the base code before claiming it.
Ask the human to pick one or reshape one, plus their priorities and constraints for this cycle.
**Colab login (ask now, while the human is here)**: run `python research/bin/colab_run.py --check-login`. It is free (it reads the local token; no session). If it prints `LOGIN NEEDED` (rc 7), add one grill question: "Colab needs a login for this cycle's spike and smoke runs. Log in now, or run this cycle without Colab (Kaggle draft run instead)?" If they choose to log in, show the exact command it printed (to type as `! <command>`), wait, and re-run `--check-login` until it prints `login OK`. If they decline, write `colab: off` into STATUS and spike on local CPU only. Re-read free-text answers and clarify anything ambiguous.
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
If Claude is busy, the idea is queued. When STATUS later reads `revise`, read `verdict.md`, revise spec.md, set `ready-for-judge` and call again. Read its `## Lesson candidates` too: they are the mistakes the judge found in your work. Bounces are counted in STATUS; at 3 the judge escalates to the human.

Token rules: pass file paths, not contents. Never read EXPERIMENTS.md in full. Every role starts in a fresh agent.
