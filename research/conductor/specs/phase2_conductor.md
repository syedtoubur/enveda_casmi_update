# Phase 2 spec: conductor daemon

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decisions 1, 2, 4, 8, 9, 11, 16).
Builder: Sonnet subagent working DIRECTLY in E:/ENVEDA_casmi (no worktree; other agents are writing
`research/bin/autosubmit.py` and `research/bin/desk.py` at the same time; do not touch those or any file not listed).

## Already in place (read, reuse, do not rewrite)
- `research/bin/providers.py`: `Router(cfg, ledger_path)`, `yield_claude(seconds)`, ledger `research/provider_ledger.json`.
- `research/bin/campaign.py`: `python research/bin/campaign.py <campaign> --once --router` advances one transition;
  `Campaign(path, readonly=True).describe()` gives the next transition; `campaign.lock` refuses a second driver (exit 6).
- `research/bin/dossier.py`: `write_queue()`, `write_digest(campaign)`.
- `research/bin/necropsy.py --json` (dead-idea inventory), `research/bin/campaign_status.py --json`.
- `research/inbox/README.md`: inbox file format (front matter: campaign, idea, action, args, reason, status, created).
- `research/config.yaml` blocks `conductor:`, `autosubmit:`, `desks:`, `routing:` (already added; read them, do not edit config).
- `call_claude._pid_alive`, `call_claude.Lock` for locks; detached Popen pattern in `campaign.py main()` (`--detach`).

## Files
- NEW `research/bin/conductor.py`
- NEW `research/bin/test_conductor.py` (fake subprocess/clock; no real CLIs, no network)
- NEW `research/conductor/conductor.yaml`:
  ```yaml
  campaigns:            # active campaigns, driven in this order of priority
    - research/campaigns/20261017-c3-iso
    - research/campaigns/20261017-antigravity-sota-exploit
  desks: [chem, ml]
  paused: false
  ```
  (The necropsy campaign is added in Phase 7.)
- EDIT `research/bin/run_tests.py`: add `test_conductor.py` to the repo test list (same way test_providers was added).

## CLI
`python research/bin/conductor.py start` (detached daemon; refuses if `research/conductor/conductor.lock` is held by a live pid),
`run` (foreground, same loop; `--ticks N` for testing), `status`, `pause`, `resume`, `yield-claude [hours=4]`, `tick` (one tick, foreground).
`status` prints: daemon pid/alive, paused, last tick time, per campaign the `describe()` next transition + running worker pid,
provider ledger today (calls/done/paused/error per route, paused_until for paused routes in hours), autosubmit today count,
pending inbox files, last 5 lines of `research/conductor/conductor.log`.

## One tick (in this order; every step try/except-logged so one failure never kills the daemon)
1. **Inbox**: for each `research/inbox/*.md` with `status: pending` (skip README.md): apply, then rewrite its front matter
   with `status: applied|rejected`, `applied_at`, `result`. Actions:
   - `pause` / `resume`: add/remove the campaign in `conductor.yaml` key `paused_campaigns:` (list).
   - `kill`: `write_status(idea, "dead", reason="human (inbox): <reason>")` — only if the idea exists.
   - `revive`: create a new idea with `python research/bin/new_idea.py <line> <slug>-rev` and write a `hypothesis.md`
     stub with `revives: <idea>`, `cause:` (from its necropsy.md if any) and `fix: <args>`; reject if `args` is empty.
   - `steer`: append `## Human steer (<date>)\n<args>` to the campaign's `charter.md` under a new section (append only).
   - `raise-cap`: args `"<route> <n>"` → write into `research/conductor/caps_override.yaml` (the Router does not read
     it yet; log a TODO line — do NOT edit config.yaml).
   - `approve`: append the reason to the idea's `questions.md` as `## Approved by human (<date>)`.
   - `note`: append to the campaign `log.md`.
   - unknown → `rejected`.
2. **Workers**: reap finished worker processes (record rc in the log). If not paused, for each campaign in priority order
   that is not paused, has no `stop` in its `state.json`, has no live worker and whose `campaign.lock` is free: start
   `python research/bin/campaign.py <c> --once --router` as a child (stdout/stderr appended to `<c>/campaign_driver.log`),
   until `conductor.max_workers` are running. A campaign whose last 3 `--once` runs changed nothing in `state.json`
   AND whose `describe()` did not change gets a backoff of 30 min (record in `research/conductor/state.json`).
3. **Desks**: if `research/bin/desk.py` exists, run `python research/bin/desk.py due` (prints JSON list of due jobs) and
   start at most one `python research/bin/desk.py run <job>` worker when a worker slot is free. If desk.py is missing, skip.
4. **Auto-submit**: if `research/bin/autosubmit.py` exists, run `python research/bin/autosubmit.py --once` (it decides
   everything itself and respects `autosubmit.enabled`). Log its stdout's last line.
5. **Queue + digest**: `dossier.write_queue()` every tick; once a day after `digest_hour`, write
   `research/daily/<yyyy-mm-dd>.md` = provider ledger summary (calls per route, pauses with reset times, paid usd), per
   campaign `describe()` + new log.md lines since the last digest, submissions made today, needs-human items (from
   `campaign_status.py --json`), and a "Final-pick recommendation" placeholder section (filled by Phase 3's helper if
   available: `autosubmit.final_pick_note()`; else omit).
6. **Toast**: when a new needs-human item appears (idea STATUS needs-human / g4-timeout, campaign stop, inbox rejected),
   pop a Windows toast. Reuse the toast helper in `research/bin/kaggle_push.py` if one exists (grep for "toast"); else a
   minimal PowerShell BurntToast-free fallback (`[Windows.UI.Notifications]` via powershell -NoProfile) wrapped in try/except.
7. **State commit**: only if `conductor.commit_state` is true AND the current branch of E:/ENVEDA_casmi equals
   `conductor.state_branch`: `git add research/` (respecting .gitignore; never `-f`) and commit
   `conductor: tick <time>` if anything is staged. Otherwise do nothing. Never push.

## Safety
- Never run more than `max_workers` agent processes (desk + campaign workers share the budget).
- Never delete files; never `git reset`/`checkout`/`clean`.
- Heavy compute is not the conductor's job (Kaggle kernels are pushed by the build step).
- `yield-claude H` calls `Router.yield_claude(H*3600)`.
- `pause` sets `paused: true` in conductor.yaml; running workers finish their single step.

## Tests (`test_conductor.py`)
- inbox: each action on a tmp repo copy (monkeypatch REPO/RESEARCH paths) → expected file effects + front matter rewritten.
- workers: fake Popen; max_workers respected; paused campaign skipped; stopped campaign skipped; backoff after 3 no-ops.
- desk/autosubmit hooks skipped when the scripts are missing; invoked when present (fake).
- commit step: no git call when commit_state false or branch mismatch (fake runner).
- `status` runs on the real repo without crashing.

## Done checklist
- `python research/bin/test_conductor.py`, `python research/bin/test_providers.py`, `python research/bin/test_campaign.py` green;
  `python research/bin/test_bin.py` no new failures (67/68 baseline, the poll_lb one pre-exists).
- `python research/bin/conductor.py status` output pasted in the report. Do NOT run `start` or `run` for real.
- Do not commit (the orchestrator commits). Report: files, test tails, deviations.
