---
name: campaign-grill
description: Start a CASMI research campaign on the exploit or explore line: grill the human once, then write the charter the autonomous cycles run on. Use when the user says "new campaign", "campaign grill" or /campaign-grill.
---

# Campaign grill (interactive, tier H)

The human names the line (exploit = post-scoring and pipeline changes to E006; explore = Class-3 de novo, new Class-2 models and literature ideas, grafted onto E006). After this grill the loop runs unattended up to `ready-to-submit`; the human's only per-probe decision is the Kaggle submit, made from `research/submit_queue.md`.

1. **Prepare**. Read:
   - `research/harness/loss_map.md`, `research/harness/calibration.md`, and `research/harness/gate_report.md` (the newest PRE-REGISTERED blocks);
   - the newest `research/campaigns/*/report.md` and `digest_cycle*.md`;
   - the DECIDE lines of `python research/bin/poll_lb.py`;
   - the newest `research/ideas/*.md` candidates file (if none from this week, run the `ideate` skill first, which runs `scout` and `diagnose` when their notes are stale);
   - `research/proposals/` (lesson and rule drafts waiting for the human);
   - `python research/bin/preflight.py --selftest`.

   Build 4–8 candidate ideas from the ideas file. Each has: target cohort and reach, mechanism with a code cite, the switch, a heavy-path flag, a prior with its source (MEASURED or ASSUMED), a novelty tag, and an EV.
2. **Grill** with AskUserQuestion in rounds (use the grilling skill). Settle:
   - the line, the seed portfolio and its branches (the loop re-ideates on its own when the portfolio runs dry);
   - scope and out-of-scope;
   - the stop rules (default `max_cycles: 5`, `max_lb_probes: 4`, `dead_streak: 2`, `quota_pause_h: 12`) and the call caps;
   - pending lesson and rule proposals: approve, edit or reject each (approved ones are promoted with `research/bin/promote.py`).

   The gate is the rank-only scorecard gate (gate_report 2026-10-13); change it only with a new dated PRE-REGISTERED block. Re-read free-text answers and clarify anything ambiguous.
3. **Write** `research/campaigns/<YYYYMMDD-name>/charter.md` with exactly these sections: `## Scope` (its first line is `- Line: exploit|explore`), `## Portfolio` (table: slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV), `## Bars` (cite the gate_report block), `## Stop rules` (`key: value` lines), `## Out of scope`.
4. **Start it**: `python research/bin/campaign.py research/campaigns/<id> --detach`, and tell the human where `log.md`, the cycle digests and `research/submit_queue.md` will appear.

Done when the charter has all five sections and the detached driver has logged its first step.
