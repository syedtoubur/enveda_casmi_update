# Phase 1 spec: providers + router (multi-provider campaign steps)

Plan: `C:/Users/h/.claude/plans/pasted-content-id-c01c-here-what-distributed-clarke.md` (decisions 3, 3a, 3b, 9, 10).
Builder: Sonnet subagent on its own git worktree. Touch ONLY the files listed here.

## Goal
`campaign.py` today sends every model step to `claude -p` (`call_claude.run_prompt`). Make it send each step to the
first available route of a per-step, ordered list across four CLIs, falling back immediately when a route is
paused, rate-limited, capped or unhealthy. No real model calls in tests.

## Files
- NEW `research/bin/providers.py`: adapters, Router, ledger, slot locks.
- NEW `research/bin/test_providers.py`: unit tests (pattern: `research/bin/test_campaign.py`, fake subprocess).
- NEW `research/conductor/opencode.json`: opencode permission config (copied into place by the adapter, see below).
- EDIT `research/bin/campaign.py`: router integration (small, surgical).
- EDIT `research/config.yaml`: add a `routing:` block (append; do not change other keys).
- EDIT `research/bin/run_tests.py` only if it enumerates loop tests explicitly (add test_providers).

## Adapters (`providers.py`)
Common: `run(prompt: str, model: str, log: Path, worktree: Path | None, web: bool) -> "done" | "paused" | "error"`.
Each appends JSONL to `log` (start event with provider/model/cmd, raw stdout lines, end event with rc), exactly like
`call_claude.run_prompt`. Strip `ANTHROPIC_API_KEY` from env for every provider. `stdin=DEVNULL`. Timeout per call:
`routing.timeout_s` (default 2400) → "error".

| class | command | paused when |
|---|---|---|
| `ClaudeCLI` | delegate to `call_claude.run_prompt(cfg, prompt, log, model, worktree, settings)`; web → `cfg.claude.web_settings` | its own return |
| `AgyCLI` | `[agy_exe, "-p", prompt, "--model", model, "--output-format", "json", "--print-timeout", f"{t}s"]` (+ `--add-dir worktree`); exe from `routing.bin.agy` | JSON `status != SUCCESS` and text matches QUOTA_RE, or non-JSON stderr matches QUOTA_RE |
| `QoderCLI` | `[qoder_exe, "-p", prompt, "-m", model, "-o", "json", "--permission-mode", "dont_ask", "--no-session-persistence"]` (+ `-w worktree`) | QUOTA_RE or "Not logged in" → `paused` with ledger note `login` |
| `OpencodeCLI` | `["opencode", "run", "--pure", "--format", "json", "-m", model, "--dir", str(worktree or REPO), prompt]`; NEVER `--auto` | QUOTA_RE (covers "rate-limited upstream", "429", "quota") |

`QUOTA_RE = re.compile(r"usage limit|limit reached|rate[-_ ]?limit|429|quota|exhausted|hit your limit|temporarily rate-limited", re.I)`.
Test every regex on real strings: agy JSON `{"status": "SUCCESS", ...}`; opencode error line
`Error: [Google AI Studio] google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly`;
qodercli `Not logged in. Run \`qodercli login\` to authenticate.`

**opencode safety (measured 2026-10-18: `opencode run` executes shell WITHOUT asking).** Before each opencode call the
adapter ensures `<dir>/opencode.json` exists with the content of `research/conductor/opencode.json` (write it only if
absent or different; never overwrite a file that lacks the marker key `"_conductor": true`; in that case → "error"
with a log line). Permission block (opencode config schema `permission`):
```json
{"$schema": "https://opencode.ai/config.json", "_conductor": true,
 "permission": {"edit": "allow", "external_directory": "deny", "webfetch": "allow",
   "bash": {"*": "deny", "python research/*": "allow", "python -m pytest*": "allow", "python build/*": "allow",
            "kaggle kernels*": "allow", "kaggle datasets*": "allow", "git status*": "allow", "git diff*": "allow",
            "git log*": "allow", "git add*": "allow", "git commit*": "allow", "ls*": "allow", "cat *": "allow"}}}
```
(Verify key names against https://opencode.ai/docs/permissions if network allows; if the schema differs, follow the
docs and note it in the PR summary.) Paid OpenRouter models (any id without `:free` and not `openrouter/free`) are
refused unless `run(..., paid=True)`.

**rc is not success** (opencode returned rc=0 on a failed tool call): adapters return `done` on rc 0, but the
campaign's existing `progressed()` check (STATUS/artefact change) is what decides progress. Do not weaken it.

## Prompt translation
Campaign steps are Claude slash-commands: `/spec research/c3/x --brief research/campaigns/y/briefs/spec.md`.
Claude gets it verbatim. Every other provider gets:
```
You are a headless research agent in the repo E:/ENVEDA_casmi. Do NOT run claude, agy, qodercli, opencode,
campaign.py, call_claude.py or conductor.py. Your procedure is the file .claude/skills/<step>/SKILL.md (read it
first; where it names Opus/Sonnet/Haiku or says headless, that is you). Arguments: <args>. Read the brief at
<brief path> and only the files it lists. Write the artifacts the procedure names and set STATUS as it says.
Heavy compute goes to Kaggle kernels, never this PC. Finish with one line: STEP-RESULT: <new STATUS or 'none'>.
```
`translate(prompt) -> str` is a pure function; unit-test it.

## Router
Config (append to `research/config.yaml`):
```yaml
routing:
  timeout_s: 2400
  bin: {agy: C:/Users/h/AppData/Local/agy/bin/agy.exe, qoder: C:/Users/h/.qoder/bin/qodercli/qodercli}
  slots: {claude: 1, agy: 1, qoder: 2, opencode: 2}
  families:      # provider:model -> family (cross-family review)
    claude: anthropic
    agy:claude-opus-4-6-thinking: anthropic
    agy:claude-sonnet-4-6: anthropic
    agy:gemini-3.8-flash-high: google
    agy:gemini-3.8-flash-medium: google
    opencode:openrouter/nvidia/nemotron-3-ultra-550b-a55b:free: nvidia
    opencode:openrouter/nvidia/nemotron-3-super-120b-a12b:free: nvidia
    opencode:openrouter/openrouter/free: openrouter
    qoder: qwen
  tiers:
    H: [claude:opus, agy:claude-opus-4-6-thinking, opencode:openrouter/nvidia/nemotron-3-ultra-550b-a55b:free]
    M: [claude:sonnet, agy:claude-sonnet-4-6, opencode:openrouter/nvidia/nemotron-3-super-120b-a12b:free, qoder:qwen3.8-flash]
    L: [agy:gemini-3.8-flash-medium, qoder:qwen3.8-flash, opencode:openrouter/openrouter/free]
  paid_last_resort: {route: "opencode:openrouter/deepseek/deepseek-v4", tiers: [H], usd_per_call_est: 0.02, usd_day: 0.25, usd_total: 2.0}
  steps: {replan: H, spec: H, attack: H, ideate: H, build: M, diagnose: M, report: M, scout: L, fetch: L}
  web_steps: [scout, fetch]
  caps_day: {"claude:opus": 12, "claude:sonnet": 20}
```
Family lookup: exact `provider:model` key, else bare `provider` key, else the provider name.
(The paid route id and the qoder model id are placeholders; the human checks them. Keep them config-only.)

`Router(cfg, ledger_path, clock=time.time)`:
- `routes(step, exclude_family=None) -> list[str]` from `steps` → tier list; `build-harness`/`build-ship` map to `build`.
- `pick(step, exclude_family)`: first route that is not over `caps_day` today, not paused (ledger `paused_until` > now),
  not marked unhealthy today, and whose family != `exclude_family`. None → try `paid_last_resort` if the step's tier is in
  its `tiers` and the ledger's paid spend today < usd_day and total < usd_total (passes `paid=True`). Else None.
- `run(step, prompt, log, worktree, exclude_family=None) -> (result, route)`: loop over picks; on `paused` mark the route
  paused for `routing.pause_s` (default 1800) and continue to the next route immediately; on `done`/`error` return.
  All routes paused → `("paused", None)`.
- Takes a per-provider slot lock before launching (`research/queue/<provider>.<n>.lock`, n < slots; reuse
  `call_claude.Lock`; skip to next route if all slots busy for that provider — do not block). Claude keeps using
  the existing `claude.lock_file` as its only slot (G4 fix agents and `/run-idea` drains take that lock today).
- Ledger `research/provider_ledger.json`: `{date: {route: {calls, done, paused, error, usd}}, "paused_until": {route: t},
  "unhealthy": {date: [route]}, "yield_claude_until": t}`. `yield_claude_until > now` removes every `claude:*` route.
  Atomic writes (tmp + replace).

## campaign.py integration (surgical)
- `Campaign.__init__(..., router=None)`; when `router` is given, `call()` uses it instead of `self.models`/`_claude`.
- In `call()` with a router: the cap check uses the router; for `attack`, pass `exclude_family` = the idea's STATUS
  `spec_by:` value; after a successful `spec`, write `spec_by: <family>` into STATUS via `write_status(idea, st, spec_by=...)`
  (keep the current STATUS value). Record `self.state["calls"][route] += 1`.
- `("paused", None)` from the router keeps the existing quota-pause loop (sleep 30 min, `quota_pause_h`).
- `fetch`/`scout` with a router: allowed (the web gate `web_ok()` is Claude-only; with a router use
  `routing.web_steps` routes, all of which can browse).
- `main()`: new `--router` flag → `Campaign(path, router=Router(load_config(), REPO/"research/provider_ledger.json"))`.
  Without the flag nothing changes (all existing tests must stay green).

## Tests (`test_providers.py`, no network, no real CLIs)
1. Command building for each adapter (exact argv), incl. `--add-dir`/`-w`/`--dir`.
2. QUOTA_RE / paused mapping on the three real strings above; agy SUCCESS JSON → done.
3. opencode.json: written when absent; refused (error) when a foreign `opencode.json` without `_conductor` exists.
4. Paid refusal without `paid=True`.
5. Router: fallback order on paused; cap exhaustion skips `claude:opus`; `exclude_family="anthropic"` picks nemotron for
   H; all paused → paid last resort only for H and only under budget; yield_claude removes claude routes; slot busy →
   next provider.
6. `translate()` keeps `--brief` path and step name.
7. campaign.py with a fake router: attack gets `exclude_family` from `spec_by`; existing `test_campaign.py` passes.

## Done checklist
- `python research/bin/test_providers.py` and `python research/bin/test_campaign.py` and `python research/bin/test_bin.py` green.
- `python research/bin/campaign.py research/campaigns/20261017-c3-iso --dry-run` unchanged output.
- Commit on your worktree branch with a message ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Report: files changed, test output tail, anything in this spec you had to deviate from and why.
