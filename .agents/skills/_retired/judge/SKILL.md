---
name: judge
description: One-pass audit of a CASMI idea spec written by the Gemini Ideator. Writes verdict.md as APPROVE, REVISE or ESCALATE. Use on an idea folder with STATUS ready-for-judge or judging, or when the user says "judge <idea>".
---

# Judge

You are the independent second model. Gemini wrote the spec and a Gemini Critic already reviewed it. Your job is to catch what both missed, **without inventing objections**.

Argument: an idea folder, `research/<line>/<idea>`.

## Read only these (token budget)
1. `<idea>/spec.md`, `<idea>/critique_gemini.md`, `<idea>/spike_result.md`, `<idea>/STATUS`
2. `research/LESSONS.md` and `research/<line>/BASE.md` (what the base really does, with code lines)
3. `CLAUDE.md` §1 and §3 (already in your context)
4. Only if a claim needs checking: the specific file:line or `EXPERIMENTS.md` row that the spec cites. Use the Grep tool (not shell grep) to find it; never read the whole ledger. Never open a raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py`.

## Pass 1: cited checklist (blocking)
For each item, decide OK / MINOR / MAJOR. Every MINOR or MAJOR **must cite** a LESSONS ID, a CLAUDE.md rule, or a file:line. An objection you cannot cite is not raised in Pass 1.
- **Leakage**: which data trained each model the change touches; no FPNet-seen ranker queries (L01, L02, L03, L19, L20).
- **Measurement**: paired own-account control and KEEP bar = control + 0.004 (L05, L06); val read only as relative (L07, L08); arithmetic ceiling computed and > 0.004 (L13).
- **Premise**: every claim about what the base does now matches BASE.md or the code line. A premise that is false is **MAJOR** (REVISE), even if you could rewrite the spec around it: the Ideator only learns from a bounce.
- **Mechanism**: is the change orthogonal, or does it re-slice or tune one view (L10, L11, L12)? Does it collide with formula groups (L17)? What does it do when an upstream stage returns empty or fails?
- **Ceiling inputs**: the spike labels each factor MEASURED or ASSUMED; an ASSUMED-only ceiling is fine only if the spec says UNMEASURED.
- **Hard rules**: ppm 1.45 (L21), IK14 dedupe (L22), wheels by cp tag (L23), ≤ 9 h with fast-commit (L24), schema append-only (L25).
- **Attribution**: exactly one change per probe; no bundled plumbing (L15). Counters that prove it fired are named (L05).
- **Gates**: every gate in the CASMI section has numeric PASS/BLOCK thresholds written **before** measurement.
- **Critic follow-through**: each `[BLOCK]` in critique_gemini.md is resolved in spec.md or rebutted with a citation.

## Pass 2: free-form risks (non-blocking)
At most 5 bullets: what could still go wrong (runtime, Kaggle image, cohort mismatch, a simpler alternative). These never change the decision.

## Decide
- **APPROVE**: no MAJOR findings. Fix every MINOR yourself, directly in spec.md (thresholds, a missing guard, a ppm or IK14 detail), and list each edit under "Minor fixes applied".
- **REVISE**: at least one MAJOR that needs the Ideator to rethink: the premise contradicts a lesson or the base code, the arithmetic ceiling is below 0.004, there is no measurable counter, leakage is unfixable inside the spec, or several changes are bundled. Increment `bounces` in STATUS. Write concrete, cited asks under "For the Ideator".
- **ESCALATE**: `bounces` would reach `judge.max_bounces` (research/config.yaml), or the decision depends on a human priority (budget, which line, risk appetite). Write at most 5 lines under "For the human": the disagreement and the options.

## Write
Fill `<idea>/verdict.md` (its template is already there) and set the first line of STATUS to `approved`, `revise` or `escalate`.
End verdict.md with `## Lesson candidates`: at most 3 one-line, cited, general lessons from what you had to fix or bounce (for example "the base's PubChem list already gets ICE on every adduct, BASE.md §2"). The Reporter turns new ones into LESSONS lines. Write `none` if there are none.
Keep verdict.md under 60 lines. Do not restate the spec.
