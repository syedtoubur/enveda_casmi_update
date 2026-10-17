---
name: run-idea
description: Headless pipeline for one CASMI idea folder. Judge → tickets → implement in the line worktree → gates G1–G4 → submit card. Resumes from STATUS. Invoked by research/bin/call_claude.py as `/run-idea research/<line>/<idea>`.
---

# Run idea

Argument: `research/<line>/<idea>`. Nobody is watching: **never ask questions**. Write anything you would ask to `<idea>/questions.md`, take the most conservative option, and continue.

Context: `research/README.md` (roles, STATUS machine), `research/config.yaml`, `research/LESSONS.md`. Read only what a step needs. Never open a raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py`.

**Resume rule**: read `<idea>/STATUS` and start at the step for that state. After each step, update STATUS (first line) before starting the next, so a cut-off run resumes cleanly.

**Worktree**: all code changes go in `lines.<line>.worktree` from config.yaml, on branch `lines.<line>.branch`. Commit there after each finished ticket (`git -C <worktree> add <paths> && git -C <worktree> commit -m ...`). Never commit in the main checkout. Never `git push`. Never stage `.npz/.pt/.parquet/.zip/.ipynb` data (L28).

**Shell**: permissions are an allowlist (`.claude/settings.headless.json`). Run **one command per Bash call**: no `cd … &&` chains and no pipes into other tools. Use `git -C <dir>`, absolute paths, and the Grep/Glob/Read tools instead of shell grep/cat. A denied command is a signal to use another allowed route, not to retry.

**No background work**: never use `run_in_background`, and never start anything that outlives this session (L33). The session ends when you stop, and a background job dies with it. Long waits (Kaggle runs) belong to the detached watcher that `kaggle_push.py` starts by itself (L34); `kaggle_push.py --check` is its backup.

## Steps by STATUS

**ready-for-judge | judging** → run the `judge` skill on the folder. If STATUS is then `revise` or `escalate`, **stop here**.

**approved** → tickets. Read the to-tickets skill file (`~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills/*/skills/engineering/to-tickets/SKILL.md`) and follow it with the local tracker in `docs/agents/issue-tracker.md`, in autonomous mode (skip the quiz). Write `tickets/NN-*.md` with `Status: todo` and `Blocked by:`. Keep 2–6 tickets, each fitting one context. Exploit-line ideas follow the pattern `build/patch_<idea>.py` + `build/test_patch_<idea>.py` (copy an existing `build/patch_*.py` / `test_patch_*.py` pair for style). Set STATUS `implementing`.

**implementing** → read the implement skill file (`.../skills/engineering/implement/SKILL.md`) and follow it, ticket by ticket in dependency order, with TDD at the spec's seam. Skip tickets already marked `Status: done`. Mark each ticket `done` when its tests pass, then commit. Do **not** run `/code-review` per ticket. When all tickets are done, set STATUS `g1`.

**g1 (static)** → produce the probe notebook by running the patch script on the line's `base_notebook` from `config.yaml` (both lines: E006 v1 `casmi26-e006-window-40-ice-fallback-true.ipynb`; an explore idea grafts its module onto it). Run `compare.py base probe`. PASS iff only the cells named in the spec's G1 row change. Re-check the spec's hard-rule line against the diff. Append a row to `gates.md`. Then set STATUS `g2`, or `dead` with the reason on STATUS line 2.

**g2 (local CPU)** → run the new tests, then the line's suite **in the worktree**: `python research/bin/run_tests.py <line> --idea <idea>` (it sets cwd to the worktree and logs a `G2 suite` row). Never run `build/run_all_tests.sh` from the main checkout: it tests the wrong code and rewrites notebooks there. A suite failure that also fails on the base commit is pre-existing: note it and continue. Run any CPU dry-run the spec names on Kaggle CPU if it needs more than ~2 min or ~500 MB (**Heavy compute runs on Kaggle** (CLAUDE.md §3, human rule 2026-10-17)). Log it in `gates.md`; set STATUS `g3a` or `dead`.

**g3a (Colab smoke)** → `python research/bin/nb_to_colab.py <idea> <probe.ipynb> --kind smoke`, then `python research/bin/colab_run.py <idea> colab/smoke.py --kind smoke`. Read `colab_smoke.log`. Check the spec's G3a thresholds: it ran, the named counters moved, and the projected full time is under 8 h (L24). Write the verdict row in `gates.md` (colab_run already logged the run). colab_run marks a run FAIL when the output holds a Python error even if the CLI exits 0, and returns 4 once the retries in `gates.colab_infra_retries` are used up. If colab_run returns 7 (`LOGIN NEEDED`), it has already written the login request to questions.md and started no session: go straight to g4 (the Kaggle draft run is G3a). Never try to log in yourself. If the failure is Colab infrastructure (no session, Drive not mounted), not the notebook, do **not** debug Colab: write one line in questions.md, mark G3a `INFRA FAIL -> Kaggle draft`, and go to g4; the Kaggle draft run is then G3a. If the notebook itself fails, fix it (back to implementing) or set `dead`. Set STATUS `g3b`, `g4`, or `dead`.

**g3b (Colab val)**: only if the spec defines it; otherwise skip to g4. Build with `--kind val` for **both** the base and the probe notebooks, run both, and compare MRR@25 on the same cohort. BLOCK only on what `gates.g3b_block_if` lists (crash, drop > 0.02, counters unmoved, time > 8 h). "No gain" is not a block (L07). Set STATUS `g4` or `dead`.

**g4 (review, card, push)** → in this order:
1. Run the `code-review` skill at `low` on the worktree branch diff vs its base. Fix only correctness findings (re-run G2 if you change code); record the rest in `gates.md`.
2. Fill `submit_card.md`: the kernel slug, what changed, the paired control (for exploit, the line's `keep_bar`; if it is null, "submit the unchanged control first"), the KEEP bar, the expected effect bound, and a reminder to **paste the kernel slug into Kaggle's submission description** (poll_lb.py matches on it).
3. `python research/bin/kaggle_push.py <idea> <probe.ipynb> --counter "<regex per spec counter>"` with **no** `--wait`. G4 requires the number printed after each counter to be > 0; prefix a counter with `?` only if the spec says it may be 0 on the draft run (e.g. `?\[X\] eligible`), and keep at least one value-checked counter. It refuses (rc 5) when the notebook loads a file the preset does not mount, and (rc 6) while the line's control has no G4 PASS. On rc 5 or 6, write the reason to questions.md and stop; do not edit the catalog or config. On success STATUS is `g4-pushed` and the push has already started a detached watcher (`<idea>/kaggle_watch/watch.log`). Stop there; do not wait or poll.
The watcher finishes G4 by itself: it scans the log for the `gates.g4_fail_markers` and the counters, appends the result to submit_card.md, and sets `ready-to-submit`, or on a FAIL writes `g4_fix_brief.md` and sets `g4-fix` (a fresh run of this skill starts), or `needs-human` once `gates.g4_fix_rounds` are used up. After 60 min it sets `g4-timeout`. `kaggle_push.py --check` (run by /daily-digest) is only the backup.

**g4-fix (a fresh agent after a G4 FAIL)** → you have no memory of the earlier run; the brief is your context.
1. Read `<idea>/g4_fix_brief.md` first. Open other files only where the brief points (the full log is in `<idea>/kaggle_output/`).
2. Write `<idea>/g4_fix_<n>.md` (n = STATUS `g4_fixes`): symptom → root cause, with the log line as evidence → the fix, step by step.
3. Apply the fix within the brief's **Fix scope**. For code: change the patch in the line worktree, commit there, rerun G1 (`compare.py base probe`) and G2 (`python research/bin/run_tests.py <line> --idea <idea>`), and rebuild the probe notebook. For a missing source: add `--source <owner/slug>` only if that slug is in the catalog. A `kind: control` idea may change only sources and metadata, never the notebook code. Log each step in `gates.md`.
4. Re-push with the brief's re-push command, **no `--wait`**, then stop. The new watcher takes over.
5. Out of scope (a dataset missing from the catalog, a fix that would change the idea itself, or a catalog edit): write to `questions.md`, set STATUS `needs-human`, stop.

**g4-pushed | g4-failed | g4-timeout | needs-human | ready-to-submit | revise | escalate | dead | paused-quota | submitted | scored** → do nothing; print the STATUS and stop.

## Output
End with one line: `RUN-IDEA <idea> -> <final STATUS>`. Keep your messages short; the files are the record.
