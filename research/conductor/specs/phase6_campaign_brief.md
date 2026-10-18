# Phase 6 spec: campaign-brief (side-chat explainer + decision inbox)

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decision 16).
Builder: Sonnet subagent on its own git worktree. Touch ONLY the files listed here.

## Goal
While the autonomous loop runs, the human opens a side chat (Antigravity free tier, Claude Code, or Qoder), types
`/campaign-brief`, sees every campaign in 2 lines, picks one, and gets a plain-language, layered explanation: what
the idea is, why we do it, what was tested, what the numbers mean, why things failed, what is waiting on them. Their
decision is written to an inbox file the conductor applies later. The chat never edits loop state.

## Files
- NEW `research/bin/campaign_status.py`: deterministic, read-only JSON/markdown summariser (no model calls).
- NEW `research/bin/test_campaign_status.py`.
- NEW `.agents/skills/campaign-brief/SKILL.md` (Antigravity + Qoder read `.agents/skills/`).
- NEW `.claude/skills/campaign-brief/SKILL.md` (same content; Claude Code).
- NEW `research/inbox/README.md` (inbox file format; the conductor parses it in Phase 2).
- NEW `research/GLOSSARY.md` (plain-language definitions the skill links: MRR@25, InChIKey14, Class 1/2/3, cohort,
  C2/C1_npx/C2_hard/pseudo_c3, leak-hit vs other, harness, scorecard, gate, G4, paired control, LB noise floor
  ±0.004, keep_bar, FPNet, ICE/ICEBERG, RRF fusion, V22/E006 base, STATUS names). Take definitions from
  `research/README.md`, `CLAUDE.md`, `research/harness/gate_report.md`; ≤ 2 sentences each, no invention — if unsure,
  write `TODO(human)`.

## campaign_status.py
`python research/bin/campaign_status.py` → markdown list, 2 lines per campaign (newest first), for every
`research/campaigns/<id>/` with a `state.json`:
`<n>. <id> [line] — <running|stopped: reason> · cycle X · current: <idea or none> (<STATUS>)`
`   best: <best measured Δ / LB with idea> · needs you: <count> · last activity <date from log.md>`
`--json` → full structured object per campaign; `--campaign <id> [--json]` → deep detail:
- charter: goal, line, stop rules (from `charter.md` sections; use `brief.section()` from `research/bin/brief.py`).
- state: cycle, lb_probes, stop, current, calls per model/route.
- ideas: every idea in `history` + `current` + any `research/<line>/*` whose STATUS `campaign:` equals the id —
  each with STATUS, reason, gate line, harness Δ per cohort incl. leak-hit/other split if present in
  `harness_result.json` (be defensive: schemas vary; print what exists), LB if present, `necropsy.md` cause if present,
  open `questions.md` headings.
- log tail: last 15 lines of `log.md`.
- needs-human: ideas with STATUS needs-human/g4-timeout/paused-quota + open questions + `research/inbox/*` pending for it.
- gpu: `python research/bin/gpu.py` output if cheap (import its function if one exists; else skip).
Handle the non-standard schema of `research/campaigns/20261017-antigravity-sota-exploit/state.json` (has
`portfolio`, `active_idea`, no `history`): map portfolio entries to ideas. Never crash on a missing file; emit `null`.

## Skill (`SKILL.md`, identical in both folders; keep it ≤ 150 lines)
Front matter: `name: campaign-brief`, description that triggers on "campaign brief", "what is the loop doing",
"explain campaign".
Procedure:
1. Run `python research/bin/campaign_status.py` and show the list. Ask the human to pick one (use the host's
   clickable question tool if it has one, else a numbered list).
2. Run `python research/bin/campaign_status.py --campaign <id> --json`. Read additionally only: the campaign's
   `charter.md`, the current idea's `hypothesis.md`/`spec.md` (first 80 lines), and `research/GLOSSARY.md` as needed.
3. Answer in layers:
   - **In 5 lines**: goal · where it stands · best number so far and whether it beats the ±0.004 noise floor /
     keep_bar · what is blocked · what it needs from you.
   - Then offer sections the human can expand (ask which): **The idea in plain words** (with one chemistry or ML
     analogy) · **Why we are trying it** (which novelty class / loss bucket) · **What was tested and how** (harness,
     cohorts, gates, in plain words) · **Results and what they mean** (each number vs noise floor; leak-hit vs other) ·
     **Why it failed / what we learned** (cause codes from necropsy.md) · **Decisions open for you**.
   - Define every jargon term on first use (link GLOSSARY). No claims beyond the status JSON and the files read; say
     "not measured" when a number is missing.
4. Offer decision options (clickable if possible): pause campaign · resume · kill idea · revive idea (needs a reason)
   · steer next idea (free text) · raise a cap · approve a pending request · nothing. On a pick, write
   `research/inbox/<yyyymmdd-hhmm>-<campaign>.md` in the README format. Never edit STATUS, state.json, config, or run
   the loop.

## Inbox format (`research/inbox/README.md`)
```
---
campaign: <id>
idea: <research/<line>/<idea> or empty>
action: pause | resume | kill | revive | steer | raise-cap | approve | note
args: <free text, e.g. "cap claude:opus 20" or the steering text>
reason: <human's words>
status: pending          # the conductor sets applied/rejected + applied_at + result
created: <iso time>
---
```

## Tests
- status list on the real repo campaigns (≥ 5 found, no exception), including the antigravity schema.
- `--campaign 20261017-c3-iso --json` has ideas with STATUS and reasons.
- missing files → nulls, no crash (tmp fixture dir).

## Done checklist
- `python research/bin/test_campaign_status.py` and `python research/bin/test_bin.py` green.
- Paste the output of `python research/bin/campaign_status.py` into your report.
- Commit on your worktree branch; message ends `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Report: files, test tails, deviations.
