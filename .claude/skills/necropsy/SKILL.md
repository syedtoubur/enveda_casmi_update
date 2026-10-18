---
name: necropsy
description: Campaign step (tier M, headless): write the cause-coded necropsy.md for one dead CASMI idea so its failure teaches the next idea. Invoked as `/necropsy research/<line>/<idea>`.
---

# Necropsy

A necropsy is a side file next to a dead idea. It records why the idea died, what that rules in or out, and which revivals are worth a measurement. It sets nothing in `STATUS`.

1. Run `python research/bin/necropsy.py --skeleton research/<line>/<idea>`. It writes `necropsy.md` only if absent and pre-fills `cause:` with a regex guess marked `# guess, confirm`.
2. Read ONLY that idea's `spec.md`, `attack.md`, `gates.md`, `harness_result.json`, `result.md` or `card.md`, and `STATUS`. Do not read other ideas, raw `.ipynb`, or the repo at large.
3. Fill the file:
   - `cause:` one of `bug | leak-only | non-discriminative | pool-reach | noise-level | c1c2-harm | infra | unknown`. Confirm or replace the guess and drop the `# guess, confirm` comment once confirmed.
   - `evidence:` file:line citations only (gates.md, harness_result.json, result.md, stress_result.json). No uncited numbers.
   - `what_was_learned:` one or two sentences on what this rules in or out.
   - `revival_paths:` each as `change -> why it addresses the cause -> cheapest measurement`.
   - `combine_with:` other dead or near-miss ideas acting on a DIFFERENT cohort or stratum, with the numbers. Leave it empty rather than guess.
4. Rules:
   - Never propose re-running the same measurement.
   - A revival must change mechanism, data, or target stratum. A knob re-tune is not a revival (R12).
   - Do not reuse a signal listed in `research/necropsy/dead_signals.yaml` unless the revival cites a new measurement.
   - Cite `research/LESSONS.md` lines and rule IDs instead of restating them.
   - Follow `research/README.md`, "Rules every agent follows".

Done when `cause:` is confirmed, every claim cites file:line, and `STATUS` is untouched.
