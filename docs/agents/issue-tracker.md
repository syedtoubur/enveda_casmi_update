# Issue tracker: local markdown (research/)

Specs and tickets for this repo are local files inside an **idea folder**, `research/<line>/<YYYYMMDD-slug>/`
(see `research/README.md`). Do not use `gh issue`: it costs tokens and Antigravity agents share these files.
Create an idea folder with `python research/bin/new_idea.py <line> <slug>`.

## Conventions

- **The spec** is `<idea>/spec.md`. Its template includes a required CASMI section.
- **Tickets** are `<idea>/tickets/NN-<slug>.md`, one per vertical slice. The first lines of each ticket file are:
  ```
  # NN <title>
  Status: todo | doing | done
  Blocked by: NN, NN   (or: none)
  ```
  The rest of the file holds the description and acceptance criteria. Implementers set `Status: done` when the ticket's tests pass, so a cut-off run can resume from the first ticket that isn't done.
- **Labels and triage state** live in the first line of `<idea>/STATUS` (state machine in `research/README.md`). `ready-for-agent` corresponds to STATUS `approved`.
- **Comments** are the role files: `critique_gemini.md`, `verdict.md`, `gates.md`.

## Autonomous mode

When `<idea>/STATUS` contains `mode: autonomous`, no human is watching:
- `/to-spec`: skip "check with the user that these seams match". Pick the highest existing seam and record it under Testing Decisions as `Assumed seam: ... (autonomous)`.
- `/to-tickets`: skip "Quiz the user". Write the tickets directly and list any assumptions at the top of `tickets/00-index.md`.
- Any other question for the user: write it to `<idea>/questions.md` and carry on with the most conservative choice. Never block.

## When a skill says "publish to the issue tracker"

Write the file into the idea folder: the spec goes to `spec.md`, tickets go to `tickets/NN-*.md`.

## When a skill says "fetch the relevant ticket"

Read `<idea>/tickets/NN-*.md`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Wayfinding operations

Not used. One idea folder is one map; tickets are its children, and their `Blocked by:` lines are the edges.
