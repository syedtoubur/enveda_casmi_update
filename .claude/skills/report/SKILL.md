---
name: report
description: Campaign step (tier M, headless): after a cycle closes, draft lesson and rule proposals (add, modify, retire) with reasoning for the human to approve. Invoked as `/report research/<line>/<idea> --brief <path>`.
---

# Report

The loop learns only through proposals that the human approves. `research/bin/promote.py` turns an approved proposal into a LESSONS line or a rule with a fixture. You never edit LESSONS.md or `research/rules/` yourself.

1. **Find the surprises** in the closed cycle. Each one is a moment where code or an agent did something other than what it should have:
   - a gate that passed a probe that later proved useless, or that blocked a good one;
   - a stalled or repeated step;
   - a premise the attack or the build found false;
   - a deviation from a pre-registered check;
   - a harness number the LB contradicted.

   Cite each with file:line.
2. **For each surprise, choose**:
   - a **rule** when code can check it on an idea folder. Write the module (ID, TAGS, LESSON, SRC, FLAGS, `check(idea)`) and a fixture folder that reproduces the mistake. FLAGS must list exactly the folders the rule flags.
   - a **lesson** when it needs judgment.
   - **modify** or **retire** when an existing lesson or rule caused the surprise. Name the target and explain why its premise no longer holds.
3. **Write** `research/proposals/<yyyymmdd>-<idea>.md` in the promote.py item format, with `decision: pending` on every item and `### Reasoning` and `### Evidence` filled in.

Done when every surprise has an item, or there were none and no file was written.
