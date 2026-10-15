# Proposals: loop audit + v3 build (2026-10-13, interactive session)

Set `decision: approve` (or `reject`) under each item, then run `python research/bin/promote.py research/proposals/20261013-loop-audit.md`.
The code side of each item is already in place; these entries record the *why* in LESSONS.md.

## P1: pin the control's image
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L36)
### Text
A probe runs on its control's pinned Kaggle image (control STATUS `docker_image:`), otherwise the pair is not paired: engine2-fppair v1 crashed on Kaggle's cp313 image against E006's Python 3.12 assert. kaggle_push refuses a probe push when the control has no pinned image.
### Reasoning
The fix (`control_image()`, `push_preconditions`) exists; the lesson was only in Claude memory, so other agents and models never see it.
### Evidence
research/exploit/20261012-engine2-fppair/g4_fix_1.md:10-21; research/bin/kaggle_push.py push_preconditions.

## P2: a counter pattern must not swallow its own value
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L37)
### Text
A G4 counter regex that already matches the count (`engine lists \d+`) must be read from inside the match: the old reader took the next number on the line (the runtime), so `engine lists 0 5s` would have passed. counter_value now reads a capture group, then a trailing number inside a digit pattern, then the next number.
### Reasoning
Value-checked counters (L05) were silently checking the wrong number on every probe since E006.
### Evidence
research/bin/kaggle_push.py counter_value; test_counter_value_reads_the_matched_count_not_the_runtime.

## P3: G4 proves firing only against the control's draft
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L38)
### Text
A probe can complete G4 with every counter printed and still change nothing: promo-soften logged `promoted: 0` and passed. G4 now also needs the probe's draft submission.csv to differ from the control's draft (research/exploit/20261013-control-draft) on at least `g4_min_changed` molecules.
### Reasoning
In-run "control" files (submission_reference.csv, submission_e004_control.csv) are identical to the probe's own output, so they cannot prove firing.
### Evidence
research/exploit/20261012-promo-soften/kaggle_output (submission vs submission_reference: 0 of 400 changed); kaggle_push.firing_check.

## P4: a gate that accepts the author's own prior certifies itself
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L39)
### Text
The harness gate passed every BLIND or BASE-COUPLED idea that had a prior.md, and the spec agent wrote that prior itself, so the gate never refused such an idea. Gates read measured numbers only (gate.py, rank-only); a human override is a recorded `--override`, never a file the idea's author writes.
### Reasoning
Both 10-12 probes reached the LB this way and both reverted (0.429, 0.437).
### Evidence
research/bin/campaign.py harness_gate before 2026-10-13 (`(idea / "prior.md").exists()`); research/harness/gate_report.md 2026-10-13 rank-only block.

## P5: a harness gain on leak-hit molecules is not an LB gain
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L40)
### Text
promo-soften's harness gain sat on molecules whose truth BASE already ranked first (C2 leak-hit Δ +0.0204 vs other +0.0020), and its LB moved −0.003. Read the scorecard's leak-free "other" Δ before the overall Δ, and treat a gain that lives only on leak-hit molecules as an ASSUMED prior.
### Reasoning
This is the first harness-vs-LB pair with the split; it is a hypothesis until more rows land in calibration.md.
### Evidence
research/harness/calibration.md row 1; `harness.py scorecard` REL_PROMOTE=500 on C2.

## P6: one model step, one command per tool call
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L41)
### Text
Headless agents stall on permission shapes: a chained shell command (`cat …; echo …; git -C …`) matches no allow rule, and a relative `Edit(research/**)` rule stops matching after a `cd`. Skills tell agents to run one command per call with absolute paths, and to retry once in a simpler form before reporting a block.
### Reasoning
Two of the 10-12 campaign's stalls (spec 13:10, ship v1 15:20) came from this, and each cost a full model call.
### Evidence
research/exploit/20261012-promo-soften/claude_run.jsonl (ship v1); .claude/skills/build/SKILL.md.

## P7: stale LB history is a prior, not a death condition
- action: add
- kind: lesson
- decision: promoted 2026-10-13 (L42)
### Text
An attack may kill an idea on LB history only from scores on the same base pipeline (STATUS `lb_pipeline:`). promo-soften's A3 died on a pre-fusion delta (b416 vs base415), which no longer described the E006 pipeline.
### Reasoning
LB keys carried no pipeline tag, so attacks could not tell old evidence from current evidence.
### Evidence
research/exploit/20261012-promo-soften/prior.md:12-17; research/bin/poll_lb.py lb_pipeline.
