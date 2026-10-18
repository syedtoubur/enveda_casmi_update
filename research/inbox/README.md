# research/inbox/: human decisions waiting for the conductor

A side chat (`/campaign-brief`) writes one file per decision here: `research/inbox/<yyyymmdd-hhmm>-<campaign>.md`.
The chat never edits loop state (STATUS, state.json, config) and never runs the loop. The conductor reads these files
(Phase 2), applies each one, and updates the front matter.

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

## Fields
- `campaign`: campaign id, the folder name under `research/campaigns/`.
- `idea`: the idea folder the action targets (`kill`, `revive`, `approve`); empty for campaign-wide actions.
- `action`: `pause` and `resume` the campaign; `kill` or `revive` one idea (`revive` needs a `reason`); `steer` sets the
  next idea or direction (text in `args`); `raise-cap` raises a model-call or GPU cap (`args`: `cap <model:route> <n>`);
  `approve` a pending request (training, submit, proposal); `note` records a comment with no effect.
- `args`: free text.
- `reason`: the human's own words, copied verbatim.
- `status`: written as `pending`. The conductor changes it to `applied` or `rejected` and adds `applied_at` and `result`.
- `created`: ISO 8601 time.

Files are never deleted; `campaign_status.py` counts `status: pending` files toward "needs you".
