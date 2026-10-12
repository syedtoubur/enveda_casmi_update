---
name: critic
description: Adversarial two-pass critique of a CASMI idea spec. Run in a fresh agent, separate from the Ideator. Writes critique_gemini.md. Use when told "use the critic skill on research/<line>/<idea>".
---

# Critic

You did not write this spec. Your job is to find what is wrong before Claude (the scarce judge) spends quota on it. A critique that only says "Verified" is a failed critique: the first real cycle passed a false premise, an inconsistent spike and an unsafe guard this way.

Read: `<idea>/spec.md`, `<idea>/spike_result.md`, `<idea>/hypothesis.md`, `research/LESSONS.md`, and `research/<line>/BASE.md` (what the base really does). To verify a claim, grep the cited file or the base code (`scratch/casmi-notebook-0-433.py` for exploit, `scratch_pkg/casmi_scratch/` for explore) for the exact flag or function. Never read EXPERIMENTS.md or a raw .ipynb in full.

## Pass 1: cited checklist (BLOCKING)
Write each finding as:
`[BLOCK|FIX] <L-ID or file:line>: <what is wrong> -> <required change>`
- BLOCK: the premise, measurement or leakage is broken.
- FIX: a local detail is wrong or missing.
- **No citation means no finding.**

Then write one line per mandatory check below, as `[OK] <check>: <evidence>`. The evidence is a file:line you opened or a number you recomputed; "verified" alone is not evidence.
1. **Premise vs base**: every "the base currently does / does not X" claim matches BASE.md or the code line. A false premise is a BLOCK.
2. **Numbers agree**: the molecule counts, cohorts and ceilings in hypothesis, spike_result and spec are the same numbers. Recompute the ceiling yourself (L13).
3. **Measured vs assumed**: each input of the ceiling is labelled MEASURED (with its source) or ASSUMED. A ceiling built only on ASSUMED rank gains is a FIX that must be stated in the spec.
4. **Data provenance**: counts from local `test.parquet` or a draft run are placeholder data (L04), not the hidden test.
5. **Failure modes**: what happens when an upstream stage returns empty or fails (e.g. `GL_SCORES = {}` for every molecule when GLACIER fails, BASE.md §2)? A guard that silently changes behaviour on failure is a BLOCK.
6. **One change** (L15), **counters named** and able to fire on the 12-molecule draft run (L14), **gate thresholds numeric and pre-registered**, **KEEP bar vs own-account paired control** (L05, L06).

An empty Pass 1 (no findings and no evidence lines) is invalid.

## Pass 2: free-form risks (NON-BLOCKING)
At most 5 bullets: anything else that worries you, including simpler alternatives.

## Verdict
`READY-FOR-JUDGE` if there is no BLOCK, otherwise `BACK-TO-IDEATOR`. Write all of it to `<idea>/critique_gemini.md` (template sections), under 60 lines. Do not edit spec.md yourself.
