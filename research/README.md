# research/: the research loop (v3, 2026-10-13)

One grill → one **campaign** on the exploit or explore line. The loop then runs unattended up to `ready-to-submit`:
it ideates (scout → diagnose → ideate when its pool runs dry), specs, attacks, builds, measures every idea on the
multi-cohort **scorecard**, ships probes through G4, and hands the human one ranked `research/submit_queue.md`. The
human's decisions: the Kaggle submit, Colab/high-compute training requests, and promoting lesson/rule proposals.
Decisions live in code (`gate.py`, `poll_lb.decide()`, `preflight.py`, the G4 watcher, `gpu.py`). Design: grill
2026-10-13 (`~/.claude/plans/read-scratch-handoffs-loop-audit-md-the-ancient-cerf.md`); audit:
`research/audit/20261013-loop-audit.md`. Portable bootstrap for any agent or model: `.scratch/prompt.md`.

## Conductor (2026-10-18)
`python research/bin/conductor.py start|status|pause|resume|yield-claude` drives every campaign in `research/conductor/conductor.yaml` through `campaign.py --once --router` (multi-provider routing, `providers.py`), plus research desks (`desk.py`), auto-submit (`autosubmit.py`), necropsy campaigns (`Mode: necropsy`), the daily digest and the `research/inbox/` decisions written by `/campaign-brief`. Specs: `research/conductor/specs/`.

## Roles
| Role | Who | Writes |
|---|---|---|
| Strategist | tier H (Opus): `/campaign-grill` (interactive), `/replan`, `/ideate` | charter.md, state.json `current`/`ideate`, hypothesis.md, research/ideas/*-candidates.md |
| Spec + adversary | tier H: `/spec`, `/attack` | spec.md, attack.md (death conditions, blocking flags) |
| Builder + diagnostician | tier M (Sonnet): `/build <idea> harness\|ship`, `/diagnose`, `/report` | worktree code on `idea/<name>` branches, stress_result.json, harness_result.json (scorecard), probe notebook, card.md, research/ideas/*-diagnosis.md, research/proposals/*.md |
| Fetcher | tier L (Haiku, web settings): `/scout`, `/fetch` | research/scout/<date>.md, answers in research/requests/*.md |
| Code | campaign.py, gate.py, preflight.py, harness.py, poll_lb.py, kaggle_push.py, dossier.py, gpu.py, promote.py | STATUS transitions, gates, decisions, the submit queue |
| Human | you | the grill, the Kaggle Submit click (+ `poll_lb.py --claim`), Colab training approvals, `promote.py` on approved proposals, settings files |

## Idea STATUS (first line of `STATUS`)
```
proposed -> specced -> preflight-ok -> attacked -> harness-done -> build -> g4-pushed      (campaign.py)
specced -> spec-fix (one round) -> specced | dead          attacked/harness-done -> dead (stress test or harness bar)
g4-pushed -> ready-to-submit | g4-fix | needs-human | g4-timeout                           (G4 watcher)
ready-to-submit -> submitted -> scored-unpaired | keep-pending -> keep | revert           (poll_lb.decide)
kind: instrument (harness dumps) -> instrument-ready | needs-human; never submitted, never auto-fixed
any -> paused-quota; any -> dead (reason on a `reason:` line)
```
Legacy ideas from v1 (ready-for-judge, judging, approved, implementing, g1–g3b) still resume through `/run-idea`, which
also handles `g4-fix`.

## Gate (rank-only, gate_report 2026-10-13)
`harness-done` → `gate.py` reads `harness_result.json` (scorecard-1) + `stress_result.json`. **Dead** only on: target Δ ≤ 0
or target CI entirely < 0; a non-advisory cohort with CI upper bound < −0.004; rank-1 displacements > 1%; an attack
that died or a blocking attack left UNTESTED; a heavy-path idea without a measured ceiling > 0. Everything else is
built, shipped and **ranked** in `submit_queue.md` by the composite (equal C1/C2/C3 weights until ≥ 5 LB pairs refit
them; `research/harness/calibration.md`). `prior.md` never passes the gate. Override: `campaign.py <c> --override
<idea> --reason "..."` (recorded). G4 additionally needs the probe's draft to differ from the control's draft
(`gates.g4_draft_diff`), and pushes need the control's pinned image, a non-FAIL G2 and a clean preflight.

## Budgets
Model calls per campaign (`campaign.call_caps`), 2 retries per step, ≤ 2 ideation rounds; one Kaggle GPU pool of
30 h/week (`gpu.py`; ≤ 4 h per training run; 8 h reserved for G4). Colab training needs `training_request.md` +
STATUS `colab_train: approved` and a live `colab_run.py --check-mount`.

## Entry points
- `/campaign-grill` (Claude Code, interactive) → charter → `python research/bin/campaign.py research/campaigns/<id> --detach`.
- `python research/bin/campaign.py <campaign> --dry-run` prints the next transition; it writes nothing. `--resume` clears a stop.
- `python research/harness/harness.py compare --dump C2 --b "KNOB=value"`: paired ΔMRR@25, 90% CI, leak proxy, LB-slot verdict.
- `python research/harness/harness.py lossmap --dump C2 --dump C1_npx` → `research/harness/loss_map.md`.
- `python research/bin/preflight.py <idea>`; `--selftest` proves every rule still flags exactly its FLAGS.
- `python research/bin/poll_lb.py [--write]`: match submissions and decide KEEP/REVERT (paired control, counters > 0, replicate).
- `python research/bin/poll_lb.py --claim research/<line>/<idea>`: right after a submit, bind it (Kaggle lists no kernel name).
- `python research/bin/dossier.py [--digest <campaign>]`: submit queue + cards + cycle digest.
- `python research/harness/harness.py scorecard --a "<base cfg>" --b "<base cfg>,K=v" --target C2`: every dumped cohort.
- `python research/bin/promote.py research/proposals/<file>.md`: apply the lesson/rule items you marked `decision: approve`.
- `python research/bin/gpu.py`: this week's Kaggle GPU spend.
- `python research/bin/colab_run.py --check-mount`: live Drive mount check after a Colab re-login.
- `python research/bin/kaggle_push.py <idea> <nb> --counter "<regex>"`: G4 (counter values must be > 0; `?regex` may be 0).
- `python research/bin/new_idea.py <line> <slug>` (adds the date prefix itself).

## Harness (P1)
`build/patch_harness_dump.py` (exploit worktree) adds three cells to the untouched 0.433 notebook: val mode
(`VAL_SHARD`), a post-scoring state snapshot, and an in-run proof that `build/casmi_post.py` replays the base exactly
(`POST EQUIV ... mismatches 0`). Dumps: `research/exploit/20261010-harness-{draft,c2,c1npx}` → `harness.py fetch` →
`research/harness/dumps/` (gitignored). Read `research/harness/gate_report.md` before trusting a number.

## Rules (lessons as code)
`research/rules/R*.py`: `ID, TAGS, LESSON, SRC, FLAGS, check(idea)`. A new mistake = a new rule + a fixture in
`research/rules/fixtures/` (the files that broke). FLAGS pins every folder the rule flags. Judgment lessons that
code cannot check stay in `research/LESSONS.md`; briefs quote only the IDs a charter or spec cites.

## G4 watcher (L34)
A successful push (without `--wait`) starts `kaggle_push.py --watch <idea>` as a detached process, so it outlives the Claude session and spends no Claude tokens. It polls `kaggle kernels status` every 30 s with a heartbeat line in `<idea>/kaggle_watch/watch.log`. When the run finishes, it downloads only the log and fails on `gates.g4_fail_markers` or missing counters. Every verdict pops a Windows toast.
- **PASS** → `ready-to-submit`. A line control's PASS also queues that line's waiting probes (one `call_claude.py --drain --detach`).
- **FAIL** → `g4_fix_brief.md` and STATUS `g4-fix`, and a fresh `/run-idea` fix agent re-pushes. After `gates.g4_fix_rounds` (2) failed rounds → `needs-human` with a `questions.md` entry. A source Kaggle rejects as "not valid" goes straight to `needs-human`, because only a catalog edit fixes it.
- **No verdict after 60 min** → `g4-timeout`; `--check` finalizes it later.

A new push supersedes a running watcher through STATUS `push_id`.
- `python research/bin/run_tests.py <line>` runs the test suite inside the line worktree (G2).
- `python research/bin/poll_lb.py` matches Kaggle submissions to ideas.


## Rules every agent follows
1. Read only what your brief lists. Cite lesson IDs and rule IDs; never restate them.
2. Never read raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py`.
3. Code changes happen only in the line worktree (`config.yaml` -> `lines.<line>.worktree`).
4. Only the Reporter edits `EXPERIMENTS.md`; KEEP/REVERT comes from `poll_lb.decide()` only.
5. Only `research/bin/autosubmit.py` (conductor, when `autosubmit.enabled`) or the human runs `kaggle competitions submit`; agents never do. The human picks the final submissions.

## Colab login (Drive scope)
`colab drivemount` needs the CLI token to hold the **full** Drive scope (`https://www.googleapis.com/auth/drive`). With only
`drive.file`, the VM shows a Google "400. That's an error" page and `ValueError: mount failed`, while the CLI still exits 0.
`OAUTHLIB_RELAX_TOKEN_SCOPE=1` hides the missing scope at login. To fix it, move the token aside and log in again in WSL,
ticking every Drive permission on the consent screen:
`wsl -- bash -lc "mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.json.bak; OAUTHLIB_RELAX_TOKEN_SCOPE=1 ~/.local/bin/colab sessions"`
The loop checks this for free (`python research/bin/colab_run.py --check-login`, rc 7 = login needed): `/cycle` asks you during the grill, `colab_run.py` refuses to start a session without it (writing the request to the idea's questions.md and falling back to the Kaggle draft run), and `/daily-digest` lists it first under "Needs you".
Check by hand: `wsl -- bash -lc "python3 -c \"import json;print(json.load(open('/home/tayeb/.config/colab-cli/token.json'))['scopes'])\""` lists `.../auth/drive`.
