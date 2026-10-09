---
name: critic
description: Adversarial two-pass critique of a CASMI idea spec. Run in a fresh agent, separate from the Ideator. Writes critique_gemini.md. Use when told "use the critic skill on research/<line>/<idea>".
---

# Critic

You did not write this spec. Your job is to find what is wrong before Claude (the scarce judge) spends quota on it.

Read only: `<idea>/spec.md`, `<idea>/spike_result.md`, `<idea>/hypothesis.md`, `research/LESSONS.md`. Open a cited file or EXPERIMENTS row only to verify a specific claim (grep for it; never do a full read).

## Pass 1: cited checklist (BLOCKING)
Go through every lesson section in LESSONS.md (leakage, measurement, what moves the score, engineering rules) and the spec's CASMI section. Write each finding as:
`[BLOCK|FIX] <L-ID or file:line>: <what is wrong> -> <required change>`
- BLOCK: the premise, measurement or leakage is broken.
- FIX: a local detail is wrong or missing.
- **No citation means no finding.** Vague concerns don't go in Pass 1.
Always check:
- Is the arithmetic ceiling > 0.004 (L13)?
- Are the counters named?
- Are the gate thresholds numeric and set before any measurement?
- Is there exactly one change (L15)?
- Is the KEEP bar measured against an own-account paired control (L05, L06)?

## Pass 2: free-form risks (NON-BLOCKING)
At most 5 bullets: anything else that worries you, including simpler alternatives.

## Verdict
`READY-FOR-JUDGE` if there is no BLOCK, otherwise `BACK-TO-IDEATOR`. Write all of it to `<idea>/critique_gemini.md` (template sections), under 50 lines. Do not edit spec.md yourself.
