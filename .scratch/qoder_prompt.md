# Autoresearch bootstrap: Enveda CASMI 2026 (new campaign, run by hand in Qoder Desktop)

You are taking over an autonomous research loop inside the **Qoder Desktop App**. You run every stage yourself with
Qoder's models (300 trial + 100 daily credits). Do **not** call `claude -p` or `research/bin/campaign.py` (both spend
Claude tokens). You still write every file in the exact format `campaign.py` reads, so a Claude session can resume
your campaign cold.

Repo root: `E:\ENVEDA_casmi` (Windows; PowerShell 5.1 and Git Bash). Exploit worktree: `E:\casmi\lines\exploit`
(branch `line/exploit`).

---

## 0. The mission in one paragraph
Predict 2D molecular structures from LC-MS/MS spectra. The score is **MRR@25** on tautomer-canonical InChIKey14,
scored per `molecule_id`. The current best is **E006 v1, LB 0.440** (one run, not replicated). The loop finds ideas
that move the LB by more than the noise (±0.004), measures each one on the replay harness before it costs an LB slot,
and turns every mistake into a rule written as code. Only the human submits to Kaggle, **unless the human grants
session-scoped submit autonomy as described in §10**.

---

## 1. Step 1: probe your runtime, then write `research/runtime.md`
- **Model**: record the Qoder model you are running on (name and family) and its context limit. Rate it **H**
  (frontier reasoning), **M** (solid coder) or **L** (fast/cheap). Be honest: this decides §4.3.
- **Tools**: shell (PowerShell 5.1 or Git Bash), file read/edit, web search/fetch.
- **Services**: `python --version`, `kaggle --version`, `git -C E:/casmi/lines/exploit status`,
  `python research/bin/gpu.py` (this week's Kaggle GPU spend).
Write `research/runtime.md`: the date, each item, and the command you used to check it. If the file already exists
(from a Claude session), append a dated `## Qoder` section; do not overwrite it.

---

## 2. Ground truth: read these before planning (only the sections you need)
1. `CLAUDE.md`: domain rules and hard guardrails.
2. `research/README.md`: roles, the idea STATUS machine, entry points.
3. `research/config.yaml`: lines, `harness_base_cfg`, cohorts, GPU pool.
4. `EXPERIMENTS.md`: SOTA, history, **dead ends**.
5. `research/LESSONS.md`: lessons L01 onward (append-only).
6. `research/harness/gate_report.md`: read the block `### 2026-10-13: rank-only gate + multi-cohort scorecard`
   (line ~355). It supersedes the earlier 0.008 effect-size bar.
7. `research/harness/loss_map.md`: where the misses are.
8. `research/ideas/2026-10-13-candidates.md`, **especially "Pre-slot measurements" (line ~93)**: the four seeds of
   the last charter were already measured there. Do not re-measure them.

### What the loss map does and does not say (read this before you pick anything)
On C2 (n = 400, MRR 0.631): rank 1 is 41.5%; **rank 2–5 is 51.0% (+0.298 headroom, a *re-ordering* problem: the
truth is already in the top 5)**; rank 6–25 is 4.0% (+0.035); "not in top 25 but in some channel list" is 0.5%
(+0.005); retrieval/pool misses are 3.0% (+0.030). An idea can only claim the headroom of the bucket its mechanism
actually reaches: a tail backfill reaches the +0.005 bucket, not the +0.298 one.

### The harness reads C2 deltas high (BASE leaks)
BASE (the FPNet and the np-fold ranker plus popularity) has seen these structures, so a reorder that trusts BASE
more looks good offline. Every scorecard splits each cohort's Δ into **leak-hit** (BASE rank 1 = truth) and **other**.
Judge an idea by its **other** Δ. Calibration so far (`research/harness/calibration.md`): promo-soften had C2
+0.0185 but only **+0.0020 on "other"**, and LB −0.003; engine2-fppair LB −0.011.

---

## 3. Stage 0: a new campaign charter
The previous charter, `research/campaigns/20261013-e006-structural/`, was written but never started (it has no
`state.json`), and its seeds have since been measured (§3.2). Your campaign **supersedes** it. Do not edit or run
that folder; name it in your charter's `## Scope`.

### 3.1 Create the directory
`research/campaigns/<YYYYMMDD>-qoder-<slug>/` (e.g. `20261014-qoder-structural`).

### 3.2 Write `charter.md`
Follow `.claude/skills/campaign-grill/SKILL.md` step 3. The parsers are strict, so use **exactly** these five
sections:

1. `## Scope`. Its **first line must be `- Line: exploit`** (plain text, no backticks: `campaign.py` reads it with
   the regex `Line\s*:\s*(\w+)`). Then: base notebook `casmi26-e006-window-40-ice-fallback-true.ipynb`, cite code
   lines from `scratch/casmi26-e006.py` in the worktree; posture: structural mechanisms, not ±0.004 knob tweaks;
   "supersedes 20261013-e006-structural".
2. `## Portfolio`. A table with the skill's columns:
   `slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV`.
   The `prior` column states MEASURED or ASSUMED and its source. Seed it with what is still alive after the
   2026-10-13 pre-slot measurements:

   | seed | measured state (candidates file, line ~97–100) | how to carry it |
   |---|---|---|
   | `refspectra-lib` | MassBank covers 16.8% of C2 truths, but 62 of 67 are already rank 1 → **5 targets, ceiling ≈ +0.012**; 81.2% of C1_npx (17 targets). The only measured non-zero effect left; a new *channel* | heavy-path, priority 1; exactly one probe, no tuning follow-up |
   | `xattn-rerank` | cheap form: **no signal**; cross-attention form: **unmeasured**. A folder already exists: `research/exploit/20261013-xattn-rerank` (preflight R02 flags it) | heavy-path, priority 2; **reuse that folder**, fix the R02 finding first; GPU training needs a model-search request and §6 approval |
   | `formula-window` | **reach-starved**: truth formula absent from every candidate for only 10 C2 molecules (2.5%), 0 on C1_npx | keep only if the claim is narrowed to those 10 (max ≈ +0.025 on C2); otherwise list under Out of scope |

   The portfolio is thin. That is expected: when no live idea is left, `/replan` sets `ideate` and you run
   scout → diagnose → ideate (§4).
3. `## Bars`. Cite the gate_report 2026-10-13 rank-only block: dead only on the five conditions in §5. KEEP at submit
   (`poll_lb.decide`): LB ≥ **0.444** (E006 0.440 + 0.004) with the idea's counters firing with values > 0. G4 firing
   proof: the probe's draft `submission.csv` differs from the control draft
   (`research/exploit/20261013-control-draft`) in ≥ 1 molecule.
4. `## Stop rules`, as `key: value` lines:
   ```
   max_cycles: 5
   max_lb_probes: 4
   dead_streak: 2
   quota_pause_h: 12
   ```
5. `## Out of scope`, the measured dead ends:
   - `fusion-tail` (cap 25→40: ΔMRR −0.0001, ceiling **+0.0000**; the discarded primary tail holds no truths);
   - per-molecule cross-spectrum weighting (already done in `casmi/engine.py`; audit 2026-10-13);
   - +1.45 ppm centring as a probe (a diagnostic only, revived only if > 2% of molecules change window);
   - TOPN 60→100 and PubChem-depth widening; FPNet swaps or a second FPNet family; random bit-subspace RRF;
   - generated candidates at ranks 1–5 (−0.064); uncalibrated simulated-spectrum scores; pure Class-3 de novo
     without grafting onto E006;
   - anything else EXPERIMENTS.md lists as a dead end.

### 3.3 Write `state.json` in `campaign.py`'s schema
```json
{
  "cycle": 0,
  "lb_probes": 0,
  "branch_streak": {},
  "closed_branches": [],
  "current": null,
  "stop": null,
  "ideate": null,
  "history": [],
  "calls": {},
  "stalls": {},
  "ideation_rounds": 0
}
```
Rules for updating it, so a Claude session can resume:
- `current` = the idea folder path relative to the repo (e.g. `research/exploit/20261014-refspectra-lib`).
- When an idea closes, append `{"idea": <path>, "status": <STATUS>, "branch": <branch>}` to `history`, then
  `cycle += 1` and `current = null`.
- A harness death adds 1 to `branch_streak[<branch>]`. At `dead_streak`, add the branch to `closed_branches`.
- A probe pushed for the human to submit adds 1 to `lb_probes`.
- **A pre-slot measurement kill is not a harness death**: record it in `spike_result.md`, not as STATUS `dead`, or
  it closes the branch (`campaign.py` counts every `dead`).
Also create `log.md` and append one line per stage: date, stage, idea, outcome.

---

## 4. The loop (cycle 1 onward)

### 4.1 Per stage: brief first, then the skill
For every stage, generate the compact brief and read **it** instead of README/LESSONS/EXPERIMENTS:
```
python research/bin/brief.py research/campaigns/<id> <step> [research/exploit/<idea>]
```
It prints the brief's path. Then follow `.claude/skills/<step>/SKILL.md` with that brief. Where a skill says
"Opus" or "Sonnet", read tier H or M.

| # | stage | skill | output | done when |
|---|---|---|---|---|
| 1 | Replan | `replan` | `python research/bin/new_idea.py exploit <slug>` + `hypothesis.md` (with `novelty:`); or `state.json` `ideate`/`stop` | `current`, `ideate` or `stop` is set |
| 2 | Scout (only if `ideate` is set and `research/scout/` is > 7 days old) | `scout` | `research/scout/<date>.md` | ≥ 10 new sources, each with "plugs in at" |
| 3 | Diagnose | `diagnose` | `research/ideas/<date>-diagnosis.md` | every bucket with headroom > 0.01 has a falsifiable hypothesis |
| 4 | Ideate | `ideate` | `research/ideas/<date>-candidates.md` | ≥ 5 ranked candidates, ≥ 2 `novelty: new`, each with a falsifier; then clear `ideate` and replan |
| — | Fetch (whenever a step filed `research/requests/*.md`) | `fetch` | answers in the request files | no request left `status: open` |
| 5 | Spec | `spec` | `spec.md`; `python research/bin/preflight.py research/exploit/<idea>` | 0 findings |
| 6 | Attack | `attack` | `attack.md` (death conditions, blocking flags) | ≥ 3 runnable attacks |
| 7 | Build + measure | `build` (harness mode) | worktree branch `idea/<slug>`; `stress_result.json`; `harness_result.json` | STATUS `harness-done` |
| 8 | Gate | code (§4.2) | `build` or `dead` | gate printed and recorded in `gates.md` |
| 9 | Ship | `build` (ship mode) | G1, G2 (`python research/bin/run_tests.py`), `kaggle_push.py`, detached G4 watcher | G4 PASS **and** the draft differs from the control draft |
| 10 | Queue | code | `python research/bin/dossier.py --digest research/campaigns/<id>` | `research/submit_queue.md` and `card.md` written |
| 11 | Submit | **human only**, or you under §10 when granted | `python research/bin/poll_lb.py --claim research/exploit/<idea>` | LB bound to the idea |
| 12 | Report | `report` | `research/proposals/<date>-<idea>.md` | every surprise has an item; the human approves, then runs `promote.py` |

### 4.2 Commands that are easy to get wrong
- **Scorecard** (harness mode; the config strings are `KEY=value,...`, not JSON):
  ```
  python research/harness/harness.py scorecard --a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40" --b "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40,<KNOB>=<value>" --target C2 --line exploit --json research/exploit/<idea>/harness_result.json
  ```
  Heavy-path ideas (ranker, features, pool, simulator inputs) do not replay. They write
  `{"heavy_path": true, "ceiling": <measured>, "family": "<family>"}` from a measured ceiling script instead.
- **Gate**: `gate.py` has **no command line**; running `python research/bin/gate.py` does nothing. Use:
  ```
  python -c "import sys; sys.path.insert(0,'research/bin'); import gate; from _common import read_status, load_config; from pathlib import Path; i=Path('research/exploit/<idea>'); print(gate.evaluate(i, load_config().get('harness') or {}, read_status(i)[1]))"
  ```
  `ok: False` → STATUS `dead` (with the `why`); `ok: True` → STATUS `build`.
- **Kernel push**: `python research/bin/kaggle_push.py research/exploit/<idea> <notebook> --counter "<regex>"`. It
  pins the control's image and starts a detached watcher. Do not poll it yourself: run
  `python research/bin/kaggle_push.py --check research/exploit/<idea>` once, later.

### 4.3 Tier rules (single-model mode)
- D3 work (replan, ideate, spec, attack, interpreting a harness-vs-LB result) done by a tier **L or M** model:
  write `tier: L` (or `M`) on the file's first line. Before acting on it, re-read it in a fresh step that loads only
  that file, and attack it.
- A tier L/M D3 output never earns an LB slot on its own: its `card.md` states the weakest assumption, and the human
  decides (under an active §10 grant, the §10 eligibility checks stand in for that decision).
- After **two** failed attempts at a step, stop that idea and write a `questions.md` entry for the human; do not loop.

---

## 5. Decided policy (grill 2026-10-13): apply it, don't re-litigate it

### Battery: the cohorts available now
- On disk in `research/harness/dumps/`: **C2** (n 400, Class 2), **C1_npx** (n 245, Class 1), **C2_np** (advisory:
  reported, never vetoes).
- **About to land (Claude's ICE-only completion run)**: **C2_hard** (n 200, Class 2) and **C1_xlib** (n 300,
  Class 1). Pending later: `pseudo_c3`, `c2_displace`.
- **Zero-blocking rule**: never wait for a dump. `harness.py scorecard` scores every cohort that has
  `post_dump/state.pkl`, so run it now on what exists.
- **The battery changes under you.** Every `harness_result.json` lists its cohorts. When C2_hard/C1_xlib appear,
  **re-run the scorecard of every idea still at `harness-done`/`build`/`ready-to-submit` before the human compares
  them**. Ideas scored on different batteries are not comparable, and a new non-advisory cohort can kill an idea.

### Gate (rank-only): an idea dies only if
1. target-cohort Δ ≤ 0, or its 90% CI upper bound < 0;
2. a non-advisory cohort's 90% CI upper bound < −0.004;
3. rank-1 truth displacements > 1% of the target cohort;
4. an attack died, or a blocking attack is UNTESTED;
5. it is a heavy-path idea without a measured ceiling > 0.
Everything else is queued and ranked by the composite (equal C1/C2/C3 weights). `prior.md` never passes the gate.

---

## 6. Hard guardrails and concurrency
- **Claude is working in the same repo.** Never edit, clean or cancel anything in `research/exploit/*harness-*`,
  `research/harness/dumps/`, or another campaign's folder. Never delete or overwrite `research/queue/claude.lock`
  or any `campaign.lock`/`watch.lock`. Before touching shared files (`EXPERIMENTS.md`, `LESSONS.md`,
  `calibration.md`), check their mtime; append, never rewrite.
- **Worktree**: notebook and `casmi_post.py` changes go only in `E:\casmi\lines\exploit` on `idea/<slug>` branches.
  Never `git reset --hard`, `git clean -fd`, or `git checkout -- .` there; other branches hold live work. Loop code
  (`research/`, skills) is not yours to change in this campaign; propose changes in `questions.md`.
- **Commits**: only on `idea/<slug>` worktree branches. No commits in the main repo.
- **GPU**: one pool of 30 GPU-h per ISO week, 8 h reserved for G4 (`python research/bin/gpu.py` before any heavy
  run; at most 4 h per training run). Before any training, file a model-search request in `research/requests/`.
  Colab training needs `training_request.md` and the human's approval (STATUS `colab_train: approved`).
- **Kaggle**: only the human runs `kaggle competitions submit`, except under an active §10 grant. You may push
  kernels via `kaggle_push.py`.
- **Pre-registered text** is never edited. To change a bar, append a dated `### YYYY-MM-DD: <name>, PRE-REGISTERED
  (supersedes ...)` block to `gate_report.md` **before** measuring. `backtests.yaml` and `LESSONS.md` are
  append-only; lessons and rules change only through approved proposals.
- `.claude/settings.headless.json` is human-only.
- **Data**: never stage `.npz`, `.pt`, `.parquet`, `.zip`. Ranker training never samples structures the FPNet saw.
- **PowerShell 5.1**: never round-trip UTF-8 files through `Get-Content`/`Set-Content`; use the edit tool or Python.

---

## 7. Credit and context efficiency
- Read files by line range; open at most ~8 files per step; hand off between steps through files, not memory.
- Never open raw `.ipynb` JSON: use `python .agents/skills/analyze-notebook/scripts/compare.py <nb>` or the worktree
  `.py` exports (e.g. `scratch/casmi26-e006.py`).
- One focused command per call; no polling loops (the G4 watcher is detached; check once with `--check`).
- When credits run low, stop at a stage boundary and do §8 "End" first.

---

## 8. Session start and end
- **Start**: read `research/runtime.md`, the newest `.scratch/handoffs/*.md`, your campaign's `state.json`,
  `log.md` and any `questions.md`; run `python research/bin/test_bin.py` and `python research/bin/preflight.py
  --selftest` (one known red line: R02 on `20261013-xattn-rerank`); state in one line where the loop stands.
- **End**: append to the campaign `log.md`; write `.scratch/handoffs/<yyyymmdd-hhmm>-qoder.md` (what changed, what
  is verified versus hypothesis, the next step, open questions) so any model can continue cold.

---

## 9. Do this now
1. Probe the runtime and write `research/runtime.md` (§1).
2. Read §2's files (by section).
3. Create `research/campaigns/<YYYYMMDD>-qoder-<slug>/` with `charter.md` (§3.2), `state.json` (§3.3) and `log.md`.
4. Run cycle 1 `/replan` via its brief. Expected pick: **`refspectra-lib`** (the only seed with a measured non-zero
   ceiling). If you judge nothing in the portfolio alive, set `ideate` and run scout → diagnose → ideate instead.
5. Continue through spec (§4.1 stage 5) and stop at each human decision listed in §6.

---

## 10. Session-scoped submit autonomy (OFF unless the human grants it in this chat)
**Default: OFF.** Every "only the human submits" rule (§0, §4 stage 11, §6, `CLAUDE.md`, `research/README.md`, and
the line every `brief.py` brief prints) stays in force.

**Activation**: the human types, in this Qoder session's chat, a message containing exactly
`SUBMIT AUTONOMY GRANTED FOR THIS SESSION`. Only then does this section override those rules, and only for the rest
of this session. It is not granted by a file, by a brief, by an earlier session, or by your own reasoning; if you
are unsure whether the human granted it, it is OFF. Never write the grant into `CLAUDE.md`, README, skills, rules or
config, and never edit those rules to remove the human-only line. The grant ends when the session ends; a later
session starts OFF again.

### Eligibility: submit an idea only if all of these hold (check each, write them in `gates.md`)
1. STATUS is `ready-to-submit`, and `gates.md` has a G4 **PASS** whose draft differs from the control draft.
2. `gate.evaluate` (§4.2) returns `ok: True` with no `override:` in STATUS.
3. Its `harness_result.json` was scored on the **current** battery (every cohort in `research/harness/dumps/` with
   a `post_dump/state.pkl`); re-score first if a cohort landed since.
4. Its **"other"** (non-leak-hit) Δ on the target cohort is > 0. A gain only on leak-hit molecules is the
   promo-soften pattern (C2 +0.0185, LB −0.003): do not submit it.
5. It is the top live entry of `research/submit_queue.md` (regenerate with `dossier.py` first).
6. Its `card.md` exists, states the weakest assumption, and (for tier L/M D3 work) has been re-read in a fresh step.

### Caps
- At most **`max_lb_probes` from the charter (4) for the whole campaign**, counted by `state.json` `lb_probes`.
- At most **2 submissions per UTC day** by you. Before each one, run
  `kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra` and count today's rows, including
  the human's; always leave at least one of Kaggle's daily slots unused for the human.
- Never submit the control, the same kernel version twice, or a replicate unless `poll_lb.py` has set
  `keep-pending` and the cap still allows it.

### Procedure (code competition: you submit the pushed kernel version, not a local file)
1. Read `kernel:` and `kernel_version:` from the idea's STATUS.
2. Submit:
   ```
   kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k <kernel> -v <kernel_version> -f submission.csv -m "<idea folder name>"
   ```
3. Immediately run `python research/bin/poll_lb.py --claim research/exploit/<idea>` to bind the submission to the
   idea (STATUS → `submitted`).
4. `state.json` `lb_probes += 1`; append a `log.md` line: date, idea, kernel version, "submitted by Qoder under
   session grant".
5. Later, `python research/bin/poll_lb.py` scores it against the 0.444 KEEP bar. Record the result (with
   `/update-experiments`) and add the harness-vs-LB pair to `research/harness/calibration.md` (append only).

### Stop
Stop submitting, and leave a `questions.md` entry for the human, on any of these:
- a submission errors or scores below **0.436** (control 0.440 − one noise unit);
- two submitted ideas in a row land as `revert`;
- any eligibility check above cannot be verified.
