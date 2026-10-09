---
name: daily-digest
description: Daily report for the human. Polls Kaggle LB results, records them, and summarizes every idea, gate, notebook and submit card. Writes research/daily/YYYY-MM-DD.md. Use when the user types /daily-digest or asks what to submit today.
---

# Daily digest (Reporter)

0. Run `python research/bin/kaggle_push.py --check`. It finishes G4 for every `g4-pushed` idea whose kernel is done (`ready-to-submit` or `g4-failed`, with the reason in gates.md). A `g4-failed` control blocks its whole line (L31): put it first under "Needs you".
1. Run `python research/bin/poll_lb.py --write`. For each idea that is newly `scored`:
   - Compare its LB with the paired control in submit_card.md. KEEP iff LB ≥ control + 0.004 **and** its counters fired (L05). Set STATUS to `keep` or `revert`.
   - Record it with the update-experiments skill (one EXPERIMENTS.md row).
   - If it taught something new, append **one** cited line to `research/LESSONS.md` under "Appended by the Reporter", using the next free L-ID. Also read the `## Lesson candidates` of every verdict.md written since the last digest; turn a candidate into a lesson only if it is not already covered.
   - If it was the exploit line's control, set `lines.exploit.keep_bar` = control + 0.004 in config.yaml.
   - List the `UNMATCHED` submissions and ask the human which idea each one belongs to.
2. Collect every idea folder whose STATUS changed since the last digest (by mtime), plus all ideas in `ready-to-submit`, `escalate` or `paused-quota`, and every `questions.md`.
3. Write `research/daily/YYYY-MM-DD.md` with these sections:
   - **Submit today**, ordered controls > exploit > explore, with caps of exploit ≤ 3 and explore ≤ 2 (an unused slot moves to the other line). For each: kernel slug, what changed, KEEP bar, and the line "paste `<slug>` into the submission description".
   - **Results since yesterday**: idea, LB, Δ vs control, KEEP/REVERT.
   - **Needs you**: first, Colab login, if `python research/bin/colab_run.py --check-login` prints `LOGIN NEEDED` (copy its command, and list the ideas whose questions.md says "Colab login needed"); then escalations (≤ 5 lines each), questions.md items, paused-quota jobs (`python research/bin/call_claude.py --drain` resumes them).
   - **Pipeline**: ideas by STATUS for each line, Colab CU spent (STATUS `cu_used`), Claude runs.
   - **Dead today**: idea plus a one-line reason.
4. Keep it under 80 lines. Never recommend a submission without a G4 PASS in gates.md.
