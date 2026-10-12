---
name: daily-digest
description: Daily report for the human. Polls Kaggle LB results, records them, and summarizes every idea, gate, notebook and submit card. Writes research/daily/YYYY-MM-DD.md. Use when the user types /daily-digest or asks what to submit today.
---

# Daily digest (Reporter)

0. Run `python research/bin/kaggle_push.py --check`. It is the backup for the detached G4 watcher (L34). It finishes G4 for every `g4-pushed` or `g4-timeout` idea whose kernel is done and that no live watcher owns: the result is `ready-to-submit`, `g4-fix` (a fix agent is queued) or `needs-human`, with the reason in gates.md. Under "Pipeline", list `g4-pushed` and `g4-fix` (fix round = STATUS `g4_fixes`) as in progress. A control in `needs-human`, `g4-timeout` or `g4-failed` blocks its whole line (L31): put it first under "Needs you".
1. Run `python research/bin/poll_lb.py --write`. It decides in code (`decide()`), never you: `scored-unpaired` (control not scored), `keep-pending` (LB ≥ control + 0.004 and counter values > 0: list it under "Submit today" as a replicate), `keep` (replicate mean still ≥ the bar) or `revert`. Copy its `DECIDE` lines; never set keep/revert yourself. For each idea it moved to `keep` or `revert`:
   - Record it with the update-experiments skill (one EXPERIMENTS.md row).
   - If it taught something new, append **one** cited line to `research/LESSONS.md` under "Appended by the Reporter", using the next free L-ID. Also read the `## Lesson candidates` of every verdict.md written since the last digest; turn a candidate into a lesson only if it is not already covered.
   - List the `UNMATCHED` submissions and ask the human which idea each one belongs to.
2. Collect every idea folder whose STATUS changed since the last digest (by mtime), plus all ideas in `ready-to-submit`, `escalate`, `paused-quota`, `needs-human` or `g4-timeout`, and every `questions.md`.
3. Write `research/daily/YYYY-MM-DD.md` with these sections:
   - **Submit today**, ordered controls > exploit > explore, with caps of exploit ≤ 3 and explore ≤ 2 (an unused slot moves to the other line). For each: kernel slug, what changed, KEEP bar, and the line "paste `<slug>` into the submission description".
   - **Results since yesterday**: idea, LB, Δ vs control, KEEP/REVERT.
   - **Needs you**: first, Colab login, if `python research/bin/colab_run.py --check-login` prints `LOGIN NEEDED` (copy its command, and list the ideas whose questions.md says "Colab login needed"); then escalations (≤ 5 lines each), `needs-human` and `g4-timeout` ideas (link the `g4_fix_brief.md`), questions.md items, paused-quota jobs (`python research/bin/call_claude.py --drain` resumes them).
   - **Pipeline**: ideas by STATUS for each line, Colab CU spent (STATUS `cu_used`), Claude runs.
   - **Dead today**: idea plus a one-line reason.
4. Keep it under 80 lines. Never recommend a submission without a G4 PASS in gates.md.
