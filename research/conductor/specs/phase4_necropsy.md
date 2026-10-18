# Phase 4 spec: necropsy (learn from dead ideas, gate revivals)

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decision 13).
Builder: Sonnet subagent on its own git worktree. Touch ONLY the files listed here.

## Goal
Dead ideas (`STATUS` first line `dead`, `status: dead`, `killed-falsifier`, `revert`) are currently abandoned.
Give each one a cause-coded necropsy, let an H-tier step propose revivals and combinations, and stop revivals
that only re-tune a knob or reuse a signal already measured dead.

## Files
- NEW `research/bin/necropsy.py`: deterministic inventory + skeleton writer (no model calls).
- NEW `research/bin/test_necropsy.py`.
- NEW `.claude/skills/necropsy/SKILL.md` (M tier: fills one idea's `necropsy.md`).
- NEW `.claude/skills/necropsy-synth/SKILL.md` (H tier: reads all necropsies → `research/ideas/<date>-revivals.md`).
- NEW `research/rules/R12_revival_needs_fix.py` + fixtures `research/rules/fixtures/R12_knob_retune/`,
  `R12_dead_signal/`, `R12_shared_stratum/` (each a minimal `hypothesis.md`, and STATUS if the rule reads it).
- NEW `research/necropsy/dead_signals.yaml`: the measured-dead list the rule reads.
- EDIT nothing else. (Wiring into brief.py/campaign.py is Phase 2's job; leave a note in your report.)

## necropsy.py
`python research/bin/necropsy.py --list` → table of dead ideas: path, line, STATUS, `reason:` (from STATUS, else
`gates.md` last FAIL line, else campaign history `reason`), has_necropsy (bool).
`python research/bin/necropsy.py --skeleton research/<line>/<idea>` → writes `necropsy.md` if absent:
```
# Necropsy: <idea>
cause: <one of: bug | leak-only | non-discriminative | pool-reach | noise-level | c1c2-harm | infra | unknown>
evidence:            # file:line citations only (gates.md, harness_result.json, result.md, stress_result.json)
- 
what_was_learned:    # what this rules in/out, in one or two sentences
revival_paths:       # each: change -> why it addresses the cause -> cheapest measurement
- 
combine_with:        # other dead/near-miss ideas acting on a DIFFERENT cohort/stratum, with the numbers
- 
```
Pre-fill `cause:` with a deterministic guess (regex over reason: `leak` → leak-only; `fire|FPR|discrimin|non-discriminating`
→ non-discriminative; `reach` → pool-reach; `noise|within noise|CI` → noise-level; `C1|C2 .*(harm|drop|negative)` →
c1c2-harm; `Traceback|FAIL.*G4|not valid` → infra; else unknown) and mark it `cause: <x>  # guess, confirm`.
`--all-skeletons` does it for every dead idea lacking one. `--json` prints the inventory as JSON (Phase 6 uses it).

## Skills
`necropsy` (M): procedure = run `--skeleton`, then read ONLY that idea's spec.md, attack.md, gates.md,
harness_result.json, result.md/card.md, STATUS; confirm/replace the cause, cite evidence as file:line, fill
learned/revival/combine. Rules: never propose re-running the same measurement; a revival must change mechanism,
data, or target stratum; cite LESSONS/rule IDs instead of restating; follow `research/README.md` "Rules every agent
follows". Set nothing in STATUS (necropsy is a side file).
`necropsy-synth` (H): read every `necropsy.md` + `research/necropsy/dead_signals.yaml` + `research/harness/gate_report.md`
headers; write `research/ideas/<date>-revivals.md` with ≥ 3 ranked candidates, each with `revives:`, `cause:`, `fix:`,
falsifier, cheapest harness measurement, and expected Δ vs the ±0.004 LB noise floor. Combination candidates must
show the parts act on different cohorts or leak-hit/other strata, with the numbers. Must append any new measured-dead
signal it finds to `dead_signals.yaml` only with a citation.

## R12 (preflight rule; format = existing `research/rules/R11_*.py`)
Applies when an idea's `hypothesis.md` or `spec.md` contains a `revives:` line.
- FLAG if `cause:` or `fix:` line is missing.
- FLAG if the `fix:` text only changes knob values: matches `^(fix:\s*)?(raise|lower|tune|change|set|increase|decrease)\b.*\b(threshold|weight|margin|window|k|slot|cap|alpha|beta)\b`
  and contains no mechanism word (`new signal|new feature|different cohort|stratum|model|data|detector|pool|generator`).
- FLAG if the text names a signal listed in `dead_signals.yaml` (case-insensitive word match) without the phrase
  `MEASURED` on the same line.
- FLAG if a `combines:` line lists ≥ 2 ideas and no line mentions distinct cohorts/strata (`cohort|stratum|leak-hit|other|C1|C2|C3`)
  for them (keep it simple: require the word `stratum` or `cohort` within the combines block).
`dead_signals.yaml` seed (cite each):
```yaml
- {signal: S_nl, src: research/c3/20261018-signal-audit/result.md, why: "C1/C2 fire 0.47"}
- {signal: S_mult, src: research/c3/20261018-signal-audit/result.md, why: "C1/C2 fire 0.53"}
- {signal: S_frag, src: research/c3/20261018-signal-audit/result.md, why: "C1/C2 fire 0.71"}
- {signal: knapsack depth 3, src: research/rules/R10_truth_reference_spectrum.py, why: "gain was a truth-fragment leak"}
- {signal: ice margin route, src: memory casmi-ice-loss-is-fused-gl-branch, why: "fused margin is not a class signal"}
```
Verify each `src` exists; drop or fix any that does not, and say so in the report.
FLAGS = the three fixtures exactly. `python research/bin/preflight.py --selftest` must pass (no existing real idea
may start being flagged; if one does, narrow the rule).

## Tests
- cause-guess regex on 6 real reason strings taken from `research/campaigns/*/state.json` history.
- skeleton not overwritten when present.
- `--json` inventory finds ≥ 10 dead ideas in this repo (there are ~29 dead-type STATUS files).
- R12 on the three fixtures + one clean revival fixture that passes (not in FLAGS; put it under a name the selftest
  treats as clean — check how preflight's selftest handles non-flagged fixtures and follow it).

## Done checklist
- `python research/bin/test_necropsy.py`, `python research/bin/preflight.py --selftest`, `python research/bin/test_bin.py` green.
- `python research/bin/necropsy.py --list` prints the inventory (paste its first 15 lines into your report).
- Commit on your worktree branch; message ends `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Report: files, test tails, deviations.
