---
name: qoder-ide-loop
description: Run the Enveda CASMI autoresearch loop inside Qoder Desktop/IDE with a lean orchestrator and per-stage subagents (default Qwen flash ~0.1x, strong model only for D3 judgment stages). Portable: works on any line in research/config.yaml. Use when the user types /qoder-ide-loop [line] [campaign-dir], or asks to run the research loop in the IDE while keeping main-agent context and credits low.
---

# Qoder IDE research loop (lean orchestrator + subagents)

You are the **Lean Orchestrator**. You do not do stage thinking in your own context. For every stage you:
generate a compact brief with `brief.py`, spawn **one fresh subagent** that reads only that brief and the idea
folder, let the subagent write the stage's artifacts, then you run the deterministic Python gate and update state.
Decisions stay in code (`research/bin/*.py`); you never invent a verdict the scripts can compute.

Everything runs inside this repo. Repo root `E:\ENVEDA_casmi`, Git Bash + PowerShell 5.1, Windows.

## 0. Non-negotiable guardrails (each has cost the project before)
- **Qwen/Qoder agents ONLY. No external model CLI, ever.** Never run `claude`, `claude -p`, `research/bin/call_claude.py`, `research/bin/claude_run.py`, `agy`, Gemini, Antigravity, or `research/bin/campaign.py` / `run-idea` (those spawn Claude). All reasoning in this loop happens in **Qoder subagents** (the Agent tool: `Explore` / `general-purpose` / `Plan`, default Qwen flash). The only executables you run are the deterministic decision scripts (`research/bin/*.py`, `research/harness/harness.py`) which do no model calls.
- **When a referenced `.claude/skills/<step>/SKILL.md` says "headless", "tier H", "invoked by campaign.py as `/step`", or names Opus/Sonnet/Haiku** — that is the *procedure* for you (a Qoder subagent) to perform now, not a CLI to launch. Map Opus→Qoder strong model, Sonnet/Haiku→Qoder flash. `campaign-grill` step 4 ("start `campaign.py --detach`") is **overridden here**: do not start it; you drive the stages yourself via §4.
- **Only the human runs `kaggle competitions submit`.** You may push kernels via `research/bin/kaggle_push.py`. Never submit unless the human explicitly grants session-scoped submit autonomy in chat.
- **Never open raw `.ipynb`.** Use `python .agents/skills/analyze-notebook/scripts/compare.py <nb>` or the line's `.py` export (`base_export` in config, e.g. `scratch/casmi26-e006.py`).
- **Code changes go only in the line worktree** (`config.yaml -> lines.<line>.worktree`) on `idea/<slug>` branches. Never `git reset --hard`, `git clean -fd`, or `git checkout -- .` there; other branches hold live work. Loop code (`research/`, skills) is not edited in a campaign — propose changes in the idea's `questions.md`.
- **Only the Reporter edits `EXPERIMENTS.md`** (via `/update-experiments`); KEEP/REVERT comes only from `poll_lb.decide()`.
- **Pre-registered text is never edited.** To change a bar, append a dated `### YYYY-MM-DD: <name>, PRE-REGISTERED (supersedes ...)` block to `research/harness/gate_report.md` **before** measuring. `backtests.yaml` and `LESSONS.md` are append-only.
- **Never stage/commit** `.npz`, `.pt`, `.parquet`, `.zip`, `.env` or secrets. Ranker training never samples structures the FPNet saw.
- **Shared files**: another agent works in this repo. Never touch `research/harness/dumps/`, `research/exploit/*harness-*`, another campaign's folder, `research/queue/claude.lock`, or any `*.lock`. Check mtime before appending to `EXPERIMENTS.md`, `LESSONS.md`, `calibration.md`.
- **Heavy compute runs on Kaggle, never on this device** (human rule 2026-10-17; the machine has 7.7 GB RAM and the human works on it). Scorecards, suites, lossmaps, `compare`, stress tests that replay dumps, candidate generation/scoring, ICEBERG, training, or anything over ~500 MB RAM / ~5 min: push a Kaggle kernel (CPU default; GPU only on the human's go). Harness jobs use the kit (see §4.2 Scorecard). Fetch only logs + small JSON and analyse locally. Locally: edits, unit tests on small fixtures, smoke runs < 2 min. Never start a local `multiprocessing.Pool` or a long detached local job, and never run two heavy jobs in parallel on this machine. Subagent prompts must repeat this rule.
- **PowerShell 5.1**: never round-trip UTF-8 through `Get-Content`/`Set-Content`; use the edit tool or Python.

## 1. Step 1: probe the runtime, write `research/runtime.md`
Determine by running, not assuming:
- Which Qoder model you are now (orchestrator) and its tier: **H** (frontier reasoning), **M** (solid coder), **L** (fast/cheap). Qwen flash ≈ **L/M**. Note the strong model you can escalate to.
- Services: `python --version`, `kaggle --version`, `git -C <worktree> status`, `python research/bin/gpu.py` (this week's Kaggle GPU spend).
Write/append `research/runtime.md` (append a dated `## Qoder IDE` section if the file exists; never overwrite). This is the honest tier map that drives §3.

## 2. Pick the line and campaign
- **Line** = first arg, else ask. Read `config.yaml -> lines.<line>` for: `worktree`, `base_notebook`, `base_export`, `post_module`, `harness_base_cfg`, `keep_bar`, `control`, `control_draft`, `g4_required`. All later commands use those values; never hardcode a line.
- **Campaign** = second arg, else the newest `research/campaigns/<date>-*` matching this line that has a `state.json`. If none, run the campaign-grill step (`.claude/skills/campaign-grill/SKILL.md`) with the human to create `charter.md` + `state.json` + `log.md`.
- Read only: the campaign's `state.json`, `log.md`, `charter.md`, and any `questions.md`. That is the whole resumption context; you do not need README/LESSONS/EXPERIMENTS (the briefs carry cited lines).

## 3. Tier routing + credit policy (this is why the loop is cheap)
Default every stage to the **cheapest model that meets its demand**. Qwen flash is ~0.1x and free in the IDE; a strong model is a higher multiplier.

| demand | stages | model / agent |
|---|---|---|
| D1 mechanical | run tests/scripts, poll LB, grep, format logs, `new_idea.py`, `dossier.py`, `brief.py`, scorecard runs (pushed to Kaggle, §4.2), G4 `--check` | **You (main), inline** — no subagent, no thinking budget |
| D1 scout/fetch | literature scan, answer `research/requests/*.md` | flash subagent (web) |
| D2 build | implement knob+test in worktree, stress tests, fix failing test, push kernel | flash `general-purpose` subagent |
| D3 judgment | **replan, ideate, spec, attack, diagnose-critical, interpret harness-vs-LB** | **strong model** subagent (or you escalate) |

Rules:
- **Strong model ONLY for D3 stages**, or after **two** failed attempts at a stage, or right before a pre-slot measurement would kill an idea. Nothing else earns a strong call.
- Single-model fallback: if no strong model is available, do the D3 stage in a fresh flash subagent, mark the output file's first line `tier: L`, then re-read it in **another** fresh subagent that only opens that file and attacks it before you act. A `tier: L` D3 output never books an LB slot on its own say-so — its `card.md` states the weakest assumption and the human decides.
- **Per-campaign caps**: honor `charter ## Stop rules` (`max_cycles`, `max_lb_probes`, `dead_streak`) and `state.json` counters. When credits run low, stop at a stage boundary and do §7 End first.

## 4. The loop — one stage at a time
For each stage: **(a)** `python research/bin/brief.py research/campaigns/<id> <step> [research/<line>/<idea>]` (prints the brief path), **(b)** spawn a fresh **Qoder** subagent (Agent tool, Qwen flash; strong model only per §3) whose prompt = "You are a Qoder agent. Do NOT run `claude`, `agy`, `campaign.py`, `call_claude.py` or any external model CLI — you ARE the model. Read `<brief path>` and `.claude/skills/<step>/SKILL.md` as your procedure (map Opus→strong, Sonnet/Haiku→flash; ignore any 'headless/invoked by campaign.py' lines); do only this stage; write the named artifacts to the idea folder; report the STATUS line you set. Do not open other files.", **(c)** you run the deterministic script and update `state.json` + `log.md`.

| # | step (`<step>` for brief.py) | subagent proc | output | you-then (deterministic) | done when |
|---|---|---|---|---|---|
| 1 | replan | `.claude/skills/replan` | `new_idea.py <line> <slug>` + `hypothesis.md`; or state `ideate`/`stop` | update `state.json.current` | current/ideate/stop set |
| 2 | scout *(only if `ideate` set AND `research/scout/` >7d old)* | `.claude/skills/scout` | `research/scout/<date>.md` | — | ≥10 sources, each "plugs in at" |
| 3 | diagnose | `.claude/skills/diagnose` | `research/ideas/<date>-diagnosis.md` | — | every >0.01 headroom bucket has a falsifiable hypothesis |
| 4 | ideate | `.claude/skills/ideate` | `research/ideas/<date>-candidates.md` | clear `ideate`, re-run replan | ≥5 ranked, ≥2 `novelty: new`, each with falsifier |
| — | fetch *(any open `research/requests/*.md`)* | `.claude/skills/fetch` | answers in request files | — | no request `status: open` |
| 5 | spec | `.claude/skills/spec` | `spec.md` | `python research/bin/preflight.py research/<line>/<idea>` | 0 findings (else one spec-fix round) |
| 6 | attack | `.claude/skills/attack` | `attack.md` (death conds, blocking flags) | — | ≥3 runnable attacks |
| 7 | build | `.claude/skills/build` (harness mode) | worktree `idea/<slug>`; `stress_result.json`; `harness_result.json` | set STATUS `harness-done` | harness-done |
| 8 | gate | **code, no subagent** | — | `gate.evaluate` (§4.2) | `ok:False`→`dead` (+why); `ok:True`→`build`; log to `gates.md` |
| 9 | ship | `.claude/skills/build` (ship mode) | G1; G2 `python research/bin/run_tests.py <line>`; push; detached G4 watcher; `card.md` | `kaggle_push.py --check <idea>` once, later | G4 PASS **and** draft differs from control draft |
| 10 | queue | **code** | `research/submit_queue.md`, `card.md` | `python research/bin/dossier.py --digest research/campaigns/<id>` | written |
| 11 | submit | **human only** | LB bound to idea | `python research/bin/poll_lb.py --claim research/<line>/<idea>` (human runs, then you) | LB bound |
| 12 | report | `.claude/skills/report` | `research/proposals/<date>-<idea>.md` | human approves → `promote.py` | every surprise has an item |

When a skill says "Opus/Sonnet/Haiku", read it as tier H/M/L (strong / flash / flash-web).

### 4.2 Commands that are easy to get wrong (run these yourself, inline — D1)
- **Scorecard** (harness mode; config strings are `KEY=value,...`, NOT json):
  ```
  python research/harness/harness.py scorecard --a "<harness_base_cfg>" --b "<harness_base_cfg>,<KNOB>=<value>" --target <cohort> --line <line> --json research/<line>/<idea>/harness_result.json
  ```
  **Run it on Kaggle, not locally** (§0): `sh E:/casmi/kaggle_kit/build_kit.sh "<idea>"` (re-tars harness + dumps + the line's `build/` and versions dataset `syedtoubur/casmi26-harness-kit`; extend it if your line is not c3/exploit), put the args above in `JOBS` of `E:/casmi/kaggle_kit/kernel/run_harness_kit.py` (`--json` under `/kaggle/working/out/`), `kaggle kernels push -p E:/casmi/kaggle_kit/kernel`, wait with `kaggle kernels status syedtoubur/casmi26-harness-run`, then `kaggle kernels output syedtoubur/casmi26-harness-run -p <tmp> --file-pattern "*.json|*.log"` and copy the JSON to `research/<line>/<idea>/harness_result.json`.
  Heavy-path ideas (ranker, features, pool, simulator inputs) do not replay: they write `{"heavy_path": true, "ceiling": <measured>, "family": "<family>"}` from a measured ceiling script.
- **Gate**: `gate.py` has no CLI — `python research/bin/gate.py` does nothing. Use:
  ```
  python -c "import sys; sys.path.insert(0,'research/bin'); import gate; from _common import read_status, load_config; from pathlib import Path; i=Path('research/<line>/<idea>'); print(gate.evaluate(i, load_config().get('harness') or {}, read_status(i)[1]))"
  ```
- **Kernel push** (starts the detached zero-token watcher):
  ```
  python research/bin/kaggle_push.py research/<line>/<idea> <notebook> --counter "<regex, value >0>"
  ```
  Do **not** poll the watcher. Later, run `python research/bin/kaggle_push.py --check research/<line>/<idea>` once to finalize G4.
- **New idea folder**: `python research/bin/new_idea.py <line> <slug>` (adds the date prefix).

## 5. Gate & battery rules (apply the decided policy, don't re-litigate)
- **Rank-only gate** (gate_report 2026-10-13 block): an idea dies only if (1) target Δ ≤ 0 or its 90% CI upper < 0; (2) a non-advisory cohort CI upper < −0.004; (3) rank-1 truth displacements > 1% of target cohort; (4) an attack died or a blocking attack is UNTESTED; (5) heavy-path with no measured ceiling > 0. Everything else is queued and ranked by the composite (equal C1/C2/C3 until ≥5 LB pairs refit). `prior.md` never passes.
- **Judge by "other" Δ, not leak-hit Δ.** BASE has seen these structures, so C2 deltas read high. Each scorecard splits Δ into leak-hit vs other; a gain only on leak-hit is the promo-soften trap (C2 +0.0185 but other +0.0020, LB −0.003) — do not promote it.
- **Zero-blocking dumps**: never wait for a cohort dump. `harness.py scorecard` scores every cohort with a `post_dump/state.pkl` in `research/harness/dumps/`. `C2_np` is advisory (reported, never vetoes).
- **Battery changes under you**: when a new non-advisory cohort lands, re-score every idea still at `harness-done`/`build`/`ready-to-submit` before comparing — ideas scored on different batteries are not comparable and a new cohort can kill one.

## 6. Parallel waves + session hygiene
- **Multi-idea wave**: fan the independent build/measure stages out as parallel `general-purpose` subagents (one per idea folder), each with its own brief; you collect their `harness_result.json` and gate each in code. Keep orchestrator context to brief paths + one-line outcomes.
- **Small-context rules**: read files by line range; a subagent opens ~8 files max; one focused command per call; hand off between stages through files, not memory. No polling loops (the G4 watcher is detached).
- **Session start**: read `research/runtime.md`, newest `.scratch/handoffs/*.md` you own, campaign `state.json`/`log.md`/`questions.md`; run `python research/bin/test_bin.py` and `python research/bin/preflight.py --selftest`; state in one line where the loop stands.
- **Session end**: append to campaign `log.md` (date, stage, idea, outcome); write `.scratch/handoffs/<yyyymmdd-hhmm>-qoder-ide.md` (what changed, verified-vs-hypothesis, next step, open questions) so any model resumes cold.

## 7. Do this now
1. Probe the runtime and write `research/runtime.md` (§1).
2. Resolve the line + campaign (§2); if no charter exists, run campaign-grill with the human.
3. Run the next stage from `state.json` via §4: `brief.py` → spawn the subagent → run the deterministic script → update state.
4. Stop at each human decision (submit, Colab/high-compute approval, settings, proposals, changing a bar).
