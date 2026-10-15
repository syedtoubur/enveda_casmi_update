# Autoresearch bootstrap: Enveda CASMI 2026 (portable, model-adaptive)

You are taking over an autonomous research loop. You may be a strong model, a weak or fast one, or one of several. This
prompt tells you how to find out which you are, how to split the work by difficulty, and what to do first.
Repo root: `E:\ENVEDA_casmi` (Windows; PowerShell 5.1 and Git Bash). Exploit worktree:
`E:\casmi\lines\exploit`.

## 0. The mission in one paragraph
Predict 2D molecular structures from LC-MS/MS spectra. The score is MRR@25 on tautomer-canonical InChIKey14, scored
per `molecule_id`. The current best is **E006 v1, LB 0.440** (one run, not replicated). The loop's job is to find ideas
that move the LB by more than the noise (±0.004), measure each idea cheaply before it spends one of the few LB slots,
and turn every mistake into a rule written as code. A human submits to Kaggle, and is the only one who does.

## 1. Step 1: probe your runtime, then write `research/runtime.md`
Find out, by trying things rather than assuming:
- **Models**: which model you are, and whether you can call other models. Try `claude --version` (headless Claude:
  `claude -p --model <opus|sonnet|haiku>`), `agy --version` (Gemini), and any subagent or Task tool your app offers.
- **Tools**: shell (which one), file read/edit, web search/fetch, subagents, and how long your context window is.
- **Services**: `kaggle --version`, `python --version`, and `git -C E:/casmi/lines/exploit status`.
Write `research/runtime.md` with: the date, each available model with its tier (**H** = frontier reasoning, **M** =
solid coder, **L** = fast/cheap), the tools, and what is missing. Done when every row says how you checked it.

## 2. Tier routing: who does what
Every task has a **demand** level. Route it to the cheapest tier that meets that demand.

| demand | tasks | min tier | why |
|---|---|---|---|
| D1 mechanical | run tests and scripts, poll LB/Kaggle, grep, format logs, fetch data, literature **scout** (collect and summarise papers) | L | Pass/fail is visible from the output |
| D2 build | implement a knob plus its test in the worktree, write stress tests, fix a failing test, push a kernel | M | The code must be correct, and the tests check it |
| D3 judgment | **ideate**, **spec**, **attack** (falsify), **replan**, audit the loop, write a pre-registration, interpret a harness-vs-LB result | H | Errors here are silent and cost LB slots |

Routing rules:
1. With several tiers, give each task to its minimum tier. Escalate one tier after **two** failed attempts or one
   ambiguous result.
2. **Single-model mode** (e.g. only Qwen Flash): do everything yourself, but for D3 work:
   - Work in small steps. Write each output to a file, then re-read it **in a fresh context** (new chat, or a step that
     reads only that file) and attack it before acting on it.
   - Mark every D3 file with `tier: L` (or `M`) on its first line.
   - A D3 output from tier L or M **never spends an LB slot on its own say-so**. It goes to the human as a checkpoint
     with its weakest assumption stated.
3. When a stronger model becomes available later, read `research/runtime.md`, update it, and send every open
   `tier: L` D3 file to the stronger model for review before building on it.
4. Small-context hygiene for every tier: read files by line range, open at most about 8 files per step, use one shell
   command per call, and hand off between steps through files, not through memory.

## 3. Ground truth: read these, never guess
Read them in this order, and only the sections you need:
1. `CLAUDE.md`: domain rules and hard guardrails.
2. `research/README.md`: roles, the idea STATUS machine, entry points.
3. `research/config.yaml`: lines, base, bars, gates.
4. `EXPERIMENTS.md`: SOTA, history, dead ends.
5. `research/LESSONS.md`: lessons L01 onward, append-only.
6. `research/harness/gate_report.md`: pre-registered bars; read before trusting any harness number.
   `research/harness/capability_map.md` lists which families the harness can judge.
   `research/harness/loss_map.md` shows where the misses are.
7. `.scratch/handoffs/loop_audit.md` and `loop_audit_evidence.md`: the weak-point ledger W1–W19.
8. Optional, if readable: `C:\Users\h\.claude\projects\E--ENVEDA-casmi\memory\MEMORY.md`
   (one-line project facts, each linking to a file).

Skills are plain markdown procedures in `.claude/skills/<name>/SKILL.md` (campaign-grill, replan, spec, attack,
build, run-idea, update-experiments). Any model can follow them by reading them. Where a skill says "Opus" or
"Sonnet", read that as tier H or M.

## 4. Hard guardrails (each one has cost us before)
- Only the human runs `kaggle competitions submit`. You may push kernels (`research/bin/kaggle_push.py`).
- Never open raw `.ipynb` files. Use `python .agents/skills/analyze-notebook/scripts/compare.py <nb>` or the `.py`
  exports (e.g. worktree `scratch/casmi26-e006.py`).
- Pre-registered text is never edited. To change a bar, append a dated block
  `### YYYY-MM-DD: <name>, PRE-REGISTERED (supersedes <section @ line>)` to `research/harness/gate_report.md`, and
  write it **before** the measurement. `backtests.yaml` and `LESSONS.md` are append-only.
- Notebook and `casmi_post.py` code changes go only in `E:\casmi\lines\exploit` (branch `line/exploit`). Loop code
  (`research/`, skills) lives in the repo (branch `specs-scratch-r0`). Commit only when the human asks. Never commit
  `.npz`, `.pt`, `.parquet` or `.zip` files.
- `.claude/settings.headless.json` is edited by the human only. Propose the exact JSON lines for them to paste.
- Ranker training never samples structures the FPNet was trained on (leakage cost 0.05 LB once).
- Another agent works in this repo. Check file mtimes and running `campaign.py` processes before you touch campaign
  state, and keep your scratch files in your own folder.
- PowerShell 5.1: never round-trip UTF-8 files through `Get-Content`/`Set-Content`; use the file-edit tool or Python.
- `campaign.py --dry-run` is read-only since 2026-10-13; a stopped campaign resumes with `--resume`.

## 5. The loop (loop v3, 2026-10-13: stages, owners, completion criteria)
The human chooses the line (exploit = changes to E006; explore = Class-3 de novo, new Class-2 models and
literature ideas, built as modules grafted onto E006). After that, everything up to `ready-to-submit` runs
unattended.

| # | stage | tier | output | done when |
|---|---|---|---|---|
| 0 | **Grill** the human once per campaign (`/campaign-grill`): line, seed portfolio, stop rules, pending proposals | H | `research/campaigns/<id>/charter.md` | the human picked the line and the seeds |
| 1 | **Replan**: pick the next idea from the portfolio and `research/ideas/`; if none is left, set state `ideate` | H | idea folder + `hypothesis.md` (with `novelty:`) | state.json holds `current`, `ideate` or `stop` |
| 2 | **Scout** (only when `ideate` is set and `research/scout/` is > 7 days old) | L, web | `research/scout/<date>.md` | ≥ 10 new sources, each with "plugs in at" |
| 3 | **Diagnose** where the truth is lost | M | `research/ideas/<date>-diagnosis.md` | every bucket with headroom > 0.01 has a falsifiable hypothesis |
| 4 | **Ideate** | H | `research/ideas/<date>-candidates.md` | ≥ 5 ranked candidates, ≥ 2 `novelty: new`, each with a falsifier |
| — | **Fetch** (any time a step filed `research/requests/*.md`) | L, web | answers in the request files | no request left `status: open` |
| 5 | **Spec** to a default-off switch and a scorecard command | H | `spec.md`; `preflight.py` clean | preflight reports 0 findings |
| 6 | **Attack** | H | `attack.md` (death conditions, blocking flags) | ≥ 3 runnable attacks |
| 7 | **Build + measure** | M | worktree branch `idea/<name>`; `stress_result.json`; `harness_result.json` from `harness.py scorecard` (or a heavy-path ceiling) | STATUS `harness-done` |
| 8 | **Gate** (`research/bin/gate.py`, rank-only) | code | `build` or `dead` | see §6 |
| 9 | **Ship**: G1, G2 (`run_tests.py`, keep-going), push, G4 watcher | M | probe kernel, `card.md` | G4 PASS **and** the draft differs from the control draft |
| 10 | **Queue**: `research/submit_queue.md` (ranked), per-idea `card.md`, `digest_cycle<N>.md` | code | dossier.py | written at every G4 PASS and cycle close |
| 11 | **Submit** + `python research/bin/poll_lb.py --claim research/<line>/<idea>` | human | LB score bound to the idea | — |
| 12 | **Report**: lesson and rule proposals (add, modify, retire) | M | `research/proposals/<date>-<idea>.md` | every surprise has an item; the human approves, then runs `promote.py` |

Driver: `python research/bin/campaign.py research/campaigns/<id> [--once|--detach]` (lock, retry cap, per-model call
caps; `--dry-run` reads only; `--resume`; `--override <idea> --reason`). Without headless sub-models, run each stage
yourself by following `.claude/skills/<stage>/SKILL.md`, with the tier rules from §2.

## 6. Decided policy (grill 2026-10-13): apply it, don't re-litigate it
- **Gate = rank-only** (gate_report.md, 2026-10-13 block). An idea dies only if:
  - target Δ ≤ 0, or its CI lies entirely below 0;
  - a non-advisory cohort's CI upper bound is below −0.004;
  - rank-1 displacements exceed 1%;
  - an attack died, or a blocking attack is UNTESTED;
  - it is a heavy-path idea without a measured ceiling > 0.

  Everything else is queued and ranked. `prior.md` never passes the gate.
- **Battery**: one scorecard over every dumped cohort (C2, C1_npx now; C2_hard, C1_xlib, pseudo_c3 and c2_displace
  are pending; C2_np is advisory). Each cohort is split leak-hit vs other. The composite uses equal C1/C2/C3 weights,
  refit by regression once ≥ 5 LB pairs sit in `research/harness/calibration.md`.
- **G4** also requires the probe's draft to differ from the control's draft
  (`research/exploit/20261013-control-draft`). A push requires the control's pinned image, a non-FAIL G2 and a
  clean preflight.
- **GPU**: one Kaggle pool of 30 GPU-h per week (`gpu.py`): ≤ 4 h per training run, 8 h reserved for G4. Before any
  training, file a model-search request; reuse an open-licence, competition-legal, leak-audited model when one
  exists. Colab training needs `training_request.md` (what, why, evidence, cost), human approval (STATUS
  `colab_train: approved`) and a passing `colab_run.py --check-mount` after the human's Drive re-login.
- **Commits**: only on worktree branches. Main-repo commits happen only when the human asks.
- **Learning**: LESSONS.md and rules change only through approved proposals.
- **Control**: the single E006 run (0.440) stays the control; no replicate.

## 7. State at handoff (2026-10-13)
- Loop v3 is built and tested: `research/bin/test_bin.py` 62/62, `test_campaign.py`, the `research/harness/test_*.py`
  files, and `preflight.py --selftest`. The selftest has one red line: R02 flags another session's draft idea
  `20261013-xattn-rerank`. That is a real finding about that idea, not a regression.
- Audit with every W/N row and its fix: `research/audit/20261013-loop-audit.md`.
- Still open:
  - the battery dumps for C2_hard and C1_xlib (VAL_SHARD only), and for pseudo_c3 and c2_displace (port the
    patch_c3cal pool-ban into the worktree's `build/patch_harness_dump.py` VAL_CELL first);
  - the human's settings paste (web settings file, narrower python allow rule);
  - the human's verdict on `research/proposals/20261013-loop-audit.md`;
  - the `20261013-e006-structural` charter still cites the superseded 0.008 bar.
- The first harness-vs-LB pairs are in `calibration.md`. promo-soften: +0.0185 on C2, but only +0.0020 on the
  leak-free "other" molecules, and LB −0.003. engine2-fppair: LB −0.011.

## 8. Every session: start and end
- **Start**:
  - read `research/runtime.md`, the newest `.scratch/handoffs/*.md` you own, and the active campaign's
    `state.json`, `log.md`, digests and every `questions.md`;
  - run the test suites;
  - state in one line where the loop stands.
- **During**: ask the human only for decisions that are theirs: line, seed portfolio, settings, Colab approvals,
  proposals, submits. Use sensible defaults for everything else, and record each default.
- **End**:
  - append to the campaign `log.md`;
  - write `.scratch/handoffs/<yyyymmdd-hhmm>-<you>.md` (what changed, what is verified versus hypothesis, the next
    step, the open questions), so a different model can continue cold.
