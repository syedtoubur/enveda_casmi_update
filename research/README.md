# research/: the research loop (v2, 2026-10-10)

One grill → one **campaign** of 3–5 autonomous cycles. Each idea is measured on the **replay harness** before it
may cost code review or an LB slot. Decisions live in code (`poll_lb.decide()`, `preflight.py`, the G4 watcher),
and every new mistake becomes a **rule** in `research/rules/`. Design: `~/.claude/plans/here-i-am-trying-wiggly-pine.md`.

## Roles
| Role | Who | Writes |
|---|---|---|
| Strategist | Claude Opus: `/campaign-grill` (interactive), `/replan` (headless) | campaigns/<id>/charter.md, state.json `current`, hypothesis.md |
| Spec + adversary | Claude Opus: `/spec`, `/attack` (headless) | spec.md, attack.md |
| Builder | Claude Sonnet: `/build <idea> harness\|ship` (headless) | casmi_post.py knob + test (worktree), stress_test.py, harness_result.json, probe notebook, submit_card.md |
| Fetcher | Gemini via `agy -p`: `/scout`, `/daily-digest` | research/scout.md, daily/YYYY-MM-DD.md |
| Code | campaign.py, preflight.py, harness.py, poll_lb.py, kaggle_push.py | STATUS transitions, gates, decisions |
| Human | you | the campaign grill, the Kaggle Submit click, needs-human items |

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

## Entry points
- `/campaign-grill` (Claude Code, interactive) → charter → `python research/bin/campaign.py research/campaigns/<id> --detach`.
- `python research/bin/campaign.py <campaign> --dry-run` prints the next transition, with no model call.
- `python research/harness/harness.py compare --dump C2 --b "KNOB=value"`: paired ΔMRR@25, 90% CI, leak proxy, LB-slot verdict.
- `python research/harness/harness.py lossmap --dump C2 --dump C1_npx` → `research/harness/loss_map.md`.
- `python research/bin/preflight.py <idea>`; `--selftest` proves every rule still flags exactly its FLAGS.
- `python research/bin/poll_lb.py [--write]`: match submissions and decide KEEP/REVERT (paired control, counters > 0, replicate).
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
5. Nobody runs `kaggle competitions submit`. The human submits.

## Colab login (Drive scope)
`colab drivemount` needs the CLI token to hold the **full** Drive scope (`https://www.googleapis.com/auth/drive`). With only
`drive.file`, the VM shows a Google "400. That's an error" page and `ValueError: mount failed`, while the CLI still exits 0.
`OAUTHLIB_RELAX_TOKEN_SCOPE=1` hides the missing scope at login. To fix it, move the token aside and log in again in WSL,
ticking every Drive permission on the consent screen:
`wsl -- bash -lc "mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.json.bak; OAUTHLIB_RELAX_TOKEN_SCOPE=1 ~/.local/bin/colab sessions"`
The loop checks this for free (`python research/bin/colab_run.py --check-login`, rc 7 = login needed): `/cycle` asks you during the grill, `colab_run.py` refuses to start a session without it (writing the request to the idea's questions.md and falling back to the Kaggle draft run), and `/daily-digest` lists it first under "Needs you".
Check by hand: `wsl -- bash -lc "python3 -c \"import json;print(json.load(open('/home/tayeb/.config/colab-cli/token.json'))['scopes'])\""` lists `.../auth/drive`.
