---
name: spec
description: Campaign step (Opus, headless): turn a CASMI hypothesis into a measurable spec with a casmi_post knob and pre-registered bars. Invoked as `/spec research/exploit/<idea> --brief <path>`.
---

# Spec

Read the brief first. If `preflight.md` exists, this is the one fix round: resolve every finding in it.

1. **Ground the premise**: every claim about what the base does cites `scratch/casmi-notebook-0-433.py:<line>`, `build/casmi_post.py:<line>` or `BASE.md §n`, and you have opened and checked the citation. A claim you cannot ground is cut or marked ASSUMED.
2. **Shape one change** as a `casmi_post.CFG` knob whose default reproduces the base. If the change sits upstream of the dump (ranker, features, pool, simulator inputs), mark it `heavy-path: yes`, together with the val-mode run it needs.
3. **Write `spec.md`** from `research/_template/spec.md`. The CASMI section holds:
   - the knob name and its default;
   - the target slice (C2 / C1_npx) and the reach in its loss-map bucket;
   - the exact `harness.py compare --dump <slice> --b "<KNOB>=<value>"` command;
   - the LB-slot bar (gate_report.md);
   - value-checked counters for the shipped notebook;
   - the ceiling, with every factor labelled MEASURED (with its source) or ASSUMED.
4. Run `python research/bin/preflight.py <idea>` and fix until it is clean, or write in `questions.md` why a finding is wrong.

Done when preflight is clean (or disputed in writing) and STATUS is `specced`.
