# research/: the multi-agent research bus

Each idea is one folder, `research/<line>/<YYYYMMDD-slug>/`, copied from `_template/`.
Agents talk only through the files in that folder. Never paste file contents into a prompt; pass the path.

## Roles
| Role | Who | Writes |
|---|---|---|
| Ideator | Antigravity agent (Gemini Flash 3.8), `/cycle <line>` | hypothesis.md, research.md, spike/, spike_result.md, spec.md |
| Critic | separate Antigravity agent (fresh context), `/critic <idea>` | critique_gemini.md |
| Judge + Implementer | Claude Code headless, `/run-idea <idea>` | verdict.md, tickets/, gates.md, submit_card.md, code in the line worktree |
| Reporter | Antigravity agent, `/daily-digest` | daily/YYYY-MM-DD.md, EXPERIMENTS.md rows, LESSONS.md lines |
| Human | you | the grill at cycle start; the Kaggle Submit click |

## STATUS state machine (first line of `STATUS`)
```
drafting -> spiking -> specced -> ready-for-judge -> judging
judging -> revise (back to Ideator; bounce n/3) | escalate (to you) | approved
approved -> implementing -> g1 -> g2 -> g3a -> g3b -> g4 (review, card, push) -> g4-pushed
g4-pushed -> ready-to-submit | g4-failed   (kaggle_push.py --check, outside Claude)
ready-to-submit -> submitted -> scored (poll_lb.py) -> keep | revert (Reporter)
any -> paused-quota (Claude usage limit hit; call_claude.py resumes it)
any -> dead (spike showed no signal, or a gate blocked; reason in STATUS line 2)
```
Lines after the first in `STATUS` are `key: value` (`mode: autonomous`, `bounces: 0`, `line: exploit`, `cu_used: 0.0`).

## Entry points
- `python research/bin/new_idea.py <line> <slug>` creates an idea folder.
- `python research/bin/call_claude.py research/<line>/<idea>` hands an idea to Claude (lock and queue, one Claude job at a time).
- `python research/bin/colab_run.py <idea> <script.py> --kind smoke|val` runs a script on a Colab T4 and always stops the session.
- `python research/bin/kaggle_push.py <idea> <notebook.ipynb>` pushes a Kaggle kernel version (no submit). It refuses when the notebook loads a file the preset does not mount (L30), or when the line's `control` has no G4 PASS yet (L31).
- `python research/bin/kaggle_push.py --check` finishes G4 for every `g4-pushed` idea whose kernel is done: it downloads only the log, then fails on `gates.g4_fail_markers` or missing counters.
- `python research/bin/run_tests.py <line>` runs the test suite inside the line worktree (G2).
- `python research/bin/poll_lb.py` matches Kaggle submissions to ideas.

## Rules every agent follows
1. Read `research/LESSONS.md` and the line's `research/<line>/BASE.md` before proposing, critiquing or judging. Cite lesson IDs (L07), never restate them. Any claim about what the base currently does cites BASE.md or a `scratch/*.py:line`.
2. Never read raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py`.
3. Code changes happen only in the line worktree (`config.yaml` → `lines.<line>.worktree`), never in the main checkout.
4. Only the Reporter edits `EXPERIMENTS.md` and appends to `LESSONS.md`.
5. Nobody runs `kaggle competitions submit`. The human submits.

## Colab login (Drive scope)
`colab drivemount` needs the CLI token to hold the **full** Drive scope (`https://www.googleapis.com/auth/drive`). With only
`drive.file`, the VM shows a Google "400. That's an error" page and `ValueError: mount failed`, while the CLI still exits 0.
`OAUTHLIB_RELAX_TOKEN_SCOPE=1` hides the missing scope at login. To fix it, move the token aside and log in again in WSL,
ticking every Drive permission on the consent screen:
`wsl -- bash -lc "mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.json.bak; OAUTHLIB_RELAX_TOKEN_SCOPE=1 ~/.local/bin/colab sessions"`
The loop checks this for free (`python research/bin/colab_run.py --check-login`, rc 7 = login needed): `/cycle` asks you during the grill, `colab_run.py` refuses to start a session without it (writing the request to the idea's questions.md and falling back to the Kaggle draft run), and `/daily-digest` lists it first under "Needs you".
Check by hand: `wsl -- bash -lc "python3 -c \"import json;print(json.load(open('/home/tayeb/.config/colab-cli/token.json'))['scopes'])\""` lists `.../auth/drive`.
