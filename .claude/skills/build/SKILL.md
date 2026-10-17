---
name: build
description: Campaign step (tier M, headless): implement and measure (`harness` mode) or ship (`ship` mode) one CASMI idea in its line's worktree. Invoked as `/build research/<line>/<idea> harness|ship --brief <path>`.
---

# Build

Code changes go only in the line's worktree (the brief names it), on a branch `idea/<idea folder name>`, committed there after each green step with `git -C <worktree>`. Run exactly one command per Bash call (no `;`, `&&` or pipes into a second command); if a command is denied, retry once in a simpler single form before reporting it blocked. Leave git push and Kaggle submission to the human.

## harness mode
1. Implement the spec's switch, default = base: a `CFG` knob in the post module with its branch in `run()`, or the module plus its graft patch. Add a unit test that includes the unchanged-default case. Done when the worktree's tests are green.
2. Write `<idea>/stress_test.py`, one function per attack.md row, reading the harness dumps (`research/harness/dumps/`) or fixtures. Run it locally only if it finishes in < 2 min and < 500 MB; otherwise run it on Kaggle CPU (same kit as step 3). Write `stress_result.md` (one row per attack, with the number) **and** `stress_result.json`: `[{"id": "A1", "died": false|true|null, "status": "survived|died|UNTESTED", "blocking": true|false, "why": "..."}]`. If you ran a different check than the pre-registered one, add `"deviation": {"original": ..., "proxy": ..., "accepted_by": "builder"}` to that row. If any attack died: STATUS `dead`, `reason: <attack> <number>`, and stop.
3. Run the spec's `python research/harness/harness.py scorecard --a "<harness_base_cfg>" --b "<harness_base_cfg>,KNOB=value" --target <cohort> --line <line> --json <idea>/harness_result.json`. This command is heavy: push it as Kaggle kernel `syedtoubur/casmi26-harness-run` (refresh the kit with `sh E:/casmi/kaggle_kit/build_kit.sh`, set `JOBS` in `E:/casmi/kaggle_kit/kernel/run_harness_kit.py`, `kaggle kernels push -p E:/casmi/kaggle_kit/kernel`) and copy the fetched JSON in; never run it on this device. Heavy path: run the ceiling script (as a Kaggle kernel) and write `harness_result.json` = `{"heavy_path": true, "ceiling": <measured>, "family": "<family>"}`.

Done when harness_result.json exists and STATUS is `harness-done` (or `dead` with a reason). `campaign.py`'s gate decides from the numbers.

## ship mode
1. Build the probe notebook from the line's base notebook (config `lines.<line>.base_notebook`, E006 v1). For a post-scoring knob: `python build/patch_harness_dump.py --src <base_notebook>` gives the instrumented base; add a ship switch so the final cell writes `casmi_post.run(state, deps, **harness_base_cfg, <KNOB>=<value>)` to `submission.csv`. For a heavy-path or graft idea: `build/patch_<idea>.py --src <base_notebook>` edits only the cells the spec's G1 row names. Either way the notebook prints the spec's counters with their values. G1: `compare.py` base vs probe, where only the named cells differ. Log G1 in gates.md.
2. G2: `python research/bin/run_tests.py <line> --idea <idea>`.
3. Write `card.md`: the kernel slug, the change, the scorecard table (or the ceiling), the flags, `Paired control: <kernel of config lines.<line>.control>`, and "KEEP is decided by poll_lb.decide()". Set STATUS `g4_min_changed:` from the spec.
4. Run `python research/bin/kaggle_push.py <idea> <probe.ipynb> --counter "<regex>"` with at least one value-checked counter and no `--wait`.

Done when STATUS is `g4-pushed`. The G4 watcher (which also diffs the draft against the control's draft) and poll_lb take over from there.
