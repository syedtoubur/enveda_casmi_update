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
approved -> implementing -> g1 -> g2 -> g3a -> g3b -> g4-pushed -> ready-to-submit
ready-to-submit -> submitted -> scored (poll_lb.py) -> keep | revert (Reporter)
any -> paused-quota (Claude usage limit hit; call_claude.py resumes it)
any -> dead (spike showed no signal, or a gate blocked; reason in STATUS line 2)
```
Lines after the first in `STATUS` are `key: value` (`mode: autonomous`, `bounces: 0`, `line: exploit`, `cu_used: 0.0`).

## Entry points
- `python research/bin/new_idea.py <line> <slug>` creates an idea folder.
- `python research/bin/call_claude.py research/<line>/<idea>` hands an idea to Claude (lock and queue, one Claude job at a time).
- `python research/bin/colab_run.py <idea> <script.py> --kind smoke|val` runs a script on a Colab T4 and always stops the session.
- `python research/bin/kaggle_push.py <idea> <notebook.ipynb>` pushes a Kaggle kernel version (no submit).
- `python research/bin/poll_lb.py` matches Kaggle submissions to ideas.

## Rules every agent follows
1. Read `research/LESSONS.md` before proposing or judging. Cite lesson IDs (L07), never restate them.
2. Never read raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py`.
3. Code changes happen only in the line worktree (`config.yaml` → `lines.<line>.worktree`), never in the main checkout.
4. Only the Reporter edits `EXPERIMENTS.md` and appends to `LESSONS.md`.
5. Nobody runs `kaggle competitions submit`. The human submits.
