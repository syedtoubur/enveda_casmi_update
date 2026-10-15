# Headless job (Antigravity): ship lib-isomer-margin

You run unattended. Repo `E:\ENVEDA_casmi` (Windows; Git Bash or PowerShell 5.1). Read
`.scratch/handoffs/20261014-0815-lib-isomer-ship.md` fully first: it holds the context and the exact commands. Then do
the three ship steps and write a report.

## Do
1. **Dataset**: create `research/exploit/20261014-lib-isomer-margin/falsifier/inputs/dataset-metadata.json` as the
   handoff says, then `kaggle datasets create -p research/exploit/20261014-lib-isomer-margin/falsifier/inputs`.
   - Private: never pass `--public`.
   - If it already exists, use `kaggle datasets version -p ... -m "lib-isomer inputs"` instead.
   - Poll `kaggle datasets status syedtoubur/casmi-lib-isomer-inputs` every 60 s, at most 20 times, until `ready`.
2. **Falsifier push**:
   - Copy `research/exploit/20261014-lib-isomer-margin/falsifier/lib_isomer_falsifier.ipynb` and `kernel-metadata.json`
     into a new empty folder `research/exploit/20261014-lib-isomer-margin/falsifier/push/`.
   - Run `kaggle kernels push -p research/exploit/20261014-lib-isomer-margin/falsifier/push`.
3. **Build push (G4)**, exactly:
   `python research/bin/kaggle_push.py research/exploit/20261014-lib-isomer-margin E:/casmi/lines/exploit/casmi26-e006-lib-isomer.ipynb --counter "\"lib_sim_mols\": " --counter "?\"lib_isomer_swapped\": "`
   - If it refuses (for example a sources audit), do not work around it. Record the message in the report and go on.
4. **Wait**:
   - Every 10 minutes (sleep 600 between checks), run
     `kaggle kernels status syedtoubur/casmi-lib-isomer-falsifier` and
     `python research/bin/kaggle_push.py --check research/exploit/20261014-lib-isomer-margin`.
   - Stop waiting when both are finished (complete or error), or after 4 hours.
5. **Falsifier output**: when it has finished (even with an error), run
   `kaggle kernels output syedtoubur/casmi-lib-isomer-falsifier -p research/exploit/20261014-lib-isomer-margin/falsifier/kaggle_output`.
   - If `ceiling.json` is there, copy it to `research/exploit/20261014-lib-isomer-margin/ceiling.json`.
   - Find the `FALSIFIER {json}` line in the log.
   - Fill `research/exploit/20261014-lib-isomer-margin/spike_result.md` (replace the template) with: the question, the
     script, the runtime, the per-cohort numbers (fire rate, ΔMRR, up/down, displaced), F1/F2/ceiling/A3 and
     `verdicts.overall`. Mark every number MEASURED and cite the log.
6. **Report**: write `research/exploit/20261014-lib-isomer-margin/ship_report.md` containing:
   - for each step, the command run, the outcome and the key output lines;
   - the falsifier verdict and numbers;
   - the G4 result (gates.md rows, the `LIB_ISOMER {...}` line from `kaggle_output`, whether the draft differed from
     the control draft);
   - every error verbatim;
   - what was not done.
   Finish by printing the report's contents.

## Never
- Submit to the competition (`kaggle competitions submit`), or run `research/bin/campaign.py` or `poll_lb.py --claim`.
- Edit STATUS by hand. Edit code, notebooks, loop files, `CLAUDE.md`, configs or skills.
- Run git commits, resets, checkouts or cleans.
- Upload anything except the two JSON files.
- Retry a failed push more than once. If a G4 FAIL starts a fix agent, note it in the report; don't intervene.
- Write files outside `research/exploit/20261014-lib-isomer-margin/` (except what `kaggle_push.py` writes itself).
