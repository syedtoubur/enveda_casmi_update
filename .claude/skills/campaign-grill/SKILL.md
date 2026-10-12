---
name: campaign-grill
description: Start a CASMI research campaign: grill the human once, then write the charter that 3-5 autonomous cycles run on. Use when the user says "new campaign", "campaign grill" or /campaign-grill.
---

# Campaign grill (interactive, Opus)

1. **Prepare**. Read:
   - `research/harness/loss_map.md` and `research/harness/gate_report.md` (Results);
   - the newest `research/campaigns/*/report.md`;
   - the DECIDE lines of `python research/bin/poll_lb.py`;
   - `research/scout.md` if it is from this week (otherwise run `agy -p "/scout"` first);
   - `python research/bin/preflight.py --selftest` (rules added since the last charter).

   Build 4–8 candidate ideas. Each has: loss-map bucket and reach, mechanism with a code cite, a harness experiment (the knob), a heavy-path flag, a prior with its source, and an EV.
2. **Grill** with AskUserQuestion in rounds (use the grilling skill). Settle:
   - the portfolio and its branches;
   - scope and out-of-scope;
   - the bars (default: gate_report.md);
   - the stop rules (default `max_cycles: 5`, `max_lb_probes: 4`, `dead_streak: 2`, `quota_pause_h: 12`);
   - vetoes on new rules (a vetoed rule is deleted together with its fixture).

   Re-read free-text answers and clarify anything ambiguous.
3. **Write** `research/campaigns/<YYYYMMDD-name>/charter.md` with exactly these sections: `## Scope`, `## Portfolio` (table: slug | branch | slice | claim | knob | heavy-path | prior | EV), `## Bars`, `## Stop rules` (`key: value` lines), `## Out of scope`.
4. **Start it**: `python research/bin/campaign.py research/campaigns/<id> --detach`, and tell the human where `log.md` and `report.md` will appear.

Done when the charter has all five sections and the detached driver has logged its first step.
