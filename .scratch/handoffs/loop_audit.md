# Handoff: audit and harden the autoresearch loop
**Written:** 2026-10-12 ~15:45. **For:** a fresh session. **Evidence:** `.scratch/handoffs/loop_audit_evidence.md`
(timeline, harness-vs-LB table, weak points W1–W19). Read that ledger second, after this file.

## 1. Goal
The loop (`research/`) has completed its first supervised campaign: 2 cycles, 2 probes pushed. The human stepped it
with `campaign.py --once` and decided at every checkpoint. Several hurdles came from **the loop's own rules and
gates**, which the previous session was not allowed to change. Your job:
1. **Audit** every stage of the loop: what it claims, what it really measures, how it fails silently, and who decides.
   Verify each ledger row (W1–W19) against the code. Add any weak points you find; mark the unverified ones as
   hypotheses.
2. **Rank** the weak points by (cycles or LB slots lost) × (how often) ÷ (cost to fix).
3. **Grill** the human with AskUserQuestion rounds (re-read free-text answers, clarify ambiguity). Settle which
   rules, gates and pre-registrations change, and what replaces them.
4. **Implement** the approved changes with tests (rules come with fixtures), then log them:
   - append to `research/harness/gate_report.md`;
   - LESSONS via the Reporter path;
   - `/update-experiments` if a gate result changes.

## 2. Hard rules
- CLAUDE.md applies. Never read raw `.ipynb`; use `python .agents/skills/analyze-notebook/scripts/compare.py` or the
  exports (WT `scratch/casmi26-e006.py`).
- **Pre-registered text is never edited.** To change a gate or bar, append a dated block
  `### YYYY-MM-DD: <name>, PRE-REGISTERED (supersedes <section @ line>)` to `research/harness/gate_report.md`, written
  **before** any measurement under it. `research/harness/backtests.yaml` is append-only. `LESSONS.md` is append-only.
- `.claude/settings.headless.json`: **the human edits it** (the auto-mode classifier refuses Claude edits as
  self-modification). Propose the exact JSON lines and let the human paste them.
- The human alone submits. Agents may push Kaggle kernels.
- Code changes to the notebook or `casmi_post` go only in the worktree `E:/casmi/lines/exploit` (branch
  `line/exploit`). Loop code (`research/bin`, `research/harness`, `research/rules`, skills) lives in this repo
  (branch `specs-scratch-r0`). Commit only when the human asks.
- Out of scope: `.scratch/smart_gate` (REVERT, separate track), and ideas themselves (the campaign is paused; don't
  run cycle 3).
- Another agent works in this repo (memory: concurrent agents; `.scratch/handoffs/latest.md` is theirs). Check mtimes
  and live `campaign.py` processes before touching campaign state.

## 3. Component map (read in this order; line ranges only)
| layer | file:lines | what to check |
|---|---|---|
| design | `research/README.md` (all, 64 lines) | the STATUS machine, roles, entry points |
| driver | `research/bin/campaign.py:1-21` (docstring), `:102-146` (stop rules, `close_cycle`, `harness_gate`), `:148-192` (`step`), `:194-232` (`finish`, `main`, `--dry-run`) | W3, W4, W14, W15 |
| briefs | `research/bin/brief.py:20-64` (STEPS: reads and output contract per step), `:77-93` | W16; what each step is allowed to read |
| headless | `research/bin/call_claude.py:92-123` (command, `--settings`, `--add-dir`, cwd=REPO) + `.claude/settings.headless.json` | W9 |
| rules | `research/bin/preflight.py:23-70`, `research/rules/R01–R08` (11–26 lines each), `research/rules/fixtures/` | W10, W18: text-only checks |
| G4 | `research/bin/kaggle_push.py:60-97` (metadata + `control_image`), `:107-145` (source audit, `control_passed`), `:168-203` (`counter_value`, `check_log`), `:375-412` (`after_g4`), `:429-474` (watch), `:497-590` (main) | W6, W7 |
| LB | `research/bin/poll_lb.py:41-62` (`match`, `decide`), `:81-135` | W13, W17, W19 |
| instrument | `research/harness/harness.py:88-110` (`load_dump`), `:157-214` (`parse_set`, `compare`), `:227-295` (families, verdicts, coupling), `:308-379` (suite) | W1, W12 |
| pre-reg | `research/harness/gate_report.md`: section index :7 scope, :18 gates, :36 LB-slot bar, :146 back-test PRE-REG, :298 BASE-coupling PRE-REG, :326 latest suite result | which bars to supersede |
| maps | `research/harness/capability_map.md:5-13` (verdicts), `:65-70` (channels), `research/harness/loss_map.md` | W1 |
| skills | `.claude/skills/{campaign-grill,replan,spec,attack,build,run-idea}/SKILL.md` (each ≤ 30 lines) | prescribed commands vs the real CLIs (W12); checkpoint hooks (W4) |
| config | `research/config.yaml:6-29` (lines; exploit = E006 v1), `:33-47` (gates), `:61-71` (campaign, harness) | |
| lessons | `research/LESSONS.md` (L01–L35) | W18 |

Campaign artefacts:
- `research/campaigns/20261012-e006-engine2-pubchem/{charter.md,log.md,state.json}`;
- idea folders `research/exploit/20261012-{engine2-fppair,promo-soften,control-e006}/`. Read `STATUS`, `gates.md`,
  `stress_result.md`, `prior.md` and `attack.md`; skip the stub files `verdict.md`, `critique_gemini.md`,
  `research.md` and `spike_result.md` (W16).
- Headless transcripts: `claude_run.jsonl` in each idea folder (0.3–1 MB). Read only the tail, with a script that
  prints tool calls and denials, never the whole file.

## 4. Audit checklist (one row per stage; answer from code plus the ledger)
For each stage, answer four questions: (a) what does it claim to establish? (b) what does it actually measure? (c) how
can it pass while wrong, or block while right? (d) who decides, and is that decision recorded in code?
1. Campaign grill → charter (is the portfolio fixed? are reshapes allowed?)
2. /replan (EV inputs; is ASSUMED vs MEASURED enforced?)
3. /spec + preflight (W10: data claims unchecked; R04 bare-filename false positives)
4. /attack (W2: stale LB keys; W11: deviations)
5. /build harness + `harness_gate` (W1, W3, W5, W12)
6. /build ship: G1/G2 (W8) and the push (W7)
7. G4 watcher plus fix agent (W6; the fix agent behaved well, see the engine2 `g4_fix_1.md`)
8. LB → `poll_lb.decide` (W13, W17, W19)
9. Report/lessons (W18), orchestration safety (W4, W14, W15), brief budget (W16)

## 5. First commands
- `python research/bin/poll_lb.py`: are the two probes scored yet? Kernels `syedtoubur/casmi-engine2-fppair` (v2) and
  `syedtoubur/casmi-promo-soften`. They are the **first harness-vs-LB data points**; put them into the ledger table.
  Don't run `--write` without the human.
- Tests (all green at handoff):
  - `python research/bin/test_bin.py` (52/52);
  - `python research/bin/test_campaign.py`;
  - `python research/bin/preflight.py --selftest`;
  - `python research/harness/test_suite.py`, `test_harness.py`, `test_metrics.py`.
- Do **not** run `campaign.py --dry-run` on the live campaign: it writes `stop` plus `report.md` (W15). Copy the
  campaign folder to the scratchpad first.

## 6. Output
- `research/audit/2026101x-loop-audit.md`: the verified ledger, the ranking, and the grill decisions.
- Code, rules (with fixtures), skill edits and pre-registration blocks as approved, with the tests green.
- A short list of the `settings.headless.json` lines the human must paste.
- A fresh handoff for the next campaign grill, which must include W1's calibration data from the ledger table.
