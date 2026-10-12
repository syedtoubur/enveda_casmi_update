---
name: build
description: Campaign step (Sonnet, headless): implement and measure (`harness` mode) or ship (`ship` mode) one CASMI idea in the exploit worktree. Invoked as `/build research/exploit/<idea> harness|ship --brief <path>`.
---

# Build

Code changes go only in the exploit worktree (`C:/casmi/lines/exploit`), committed there after each green step with `git -C <worktree>`. Run one command per Bash call and finish every command inside this session. Leave git push and Kaggle submission to the human.

## harness mode
1. Add the spec's knob to `CFG` in `build/casmi_post.py` (default = base), with its branch in `run()`, and add a unit test to `build/test_patch_harness_dump.py`. Done when `python build/test_patch_harness_dump.py` is green, including the unchanged-default case.
2. Write `<idea>/stress_test.py`, one function per attack.md row, reading the harness dumps (`research/harness/dumps/`) or fixtures. Run it and write `stress_result.md`: one row per attack, `survived` or `died`, with the number. If any attack died: STATUS `dead`, `reason: <attack> <number>`, and stop.
3. Run the spec's `python research/harness/harness.py compare ... --post C:/casmi/lines/exploit/build/casmi_post.py --json <idea>/harness_result.json`.

Done when harness_result.json exists and STATUS is `harness-done` (or `dead` with a reason).

## ship mode
1. Build the probe notebook. `python build/patch_harness_dump.py` gives the instrumented base; add a ship switch so the final cell writes `casmi_post.run(state, deps, <KNOB>=<value>)` to `submission.csv` and prints the spec's counters with their values. G1: `compare.py` base vs probe, where only the harness cells and the switch differ. Log G1 in gates.md.
2. G2: `python research/bin/run_tests.py exploit --idea <idea>`.
3. Write `submit_card.md`: the kernel slug, the change, `Paired control: syedtoubur/casmi-control-0433`, and "KEEP is decided by poll_lb.decide()".
4. Run `python research/bin/kaggle_push.py <idea> <probe.ipynb> --counter "<regex>"` with at least one value-checked counter and no `--wait`.

Done when STATUS is `g4-pushed`. The G4 watcher and poll_lb take over from there.
