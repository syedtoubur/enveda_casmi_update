# Handoff: campaign `20261014-e180-npool-mutator` (2026-10-14, ~17:30)

Self-contained resume point for a fresh session. Repo `E:\ENVEDA_casmi`; exploit worktree `E:/casmi/lines/exploit`.
Read first: `research/campaigns/20261014-e180-npool-mutator/charter.md`, its `log.md`, and gate_report block
`### 2026-10-14: e180-npool-mutator campaign, PRE-REGISTERED` (`research/harness/gate_report.md`).

## How this campaign is run (user's choices)
- Supervised: step `python research/bin/campaign.py research/campaigns/20261014-e180-npool-mutator --once` (run it in
  the background, read `log.md` and the idea folder after each step). Checkpoints: after replan, at harness-done,
  before ship. Never `--detach`. The human submits.
- **Headless build steps get Bash denied** (don't-ask mode; the agent works from the worktree so relative rules miss,
  L41). The user asked the interactive session to **run the measurements itself**: `stress_test.py`, the
  `harness.py scorecard` command from the spec, then `gate.py` (via `python -c "import gate, campaign; ...
  gate.evaluate(idea, campaign.load_config()['harness'], {})"` from `research/bin`) and set STATUS with
  `campaign.write_status(idea, 'dead'|'harness-done', reason=...)` + a log line.
- After each cycle's `/report`, review `research/proposals/<new>.md`: recommend, ask the user (AskUserQuestion), set
  `decision: approve|reject`, run `python research/bin/promote.py <file>`, then `python research/bin/preflight.py
  --selftest` must print only `ok` lines. If a dead idea newly trips a rule, add it to that rule's `FLAGS` with a comment.

## State right now
- state.json: cycle 2, lb_probes 0, current = `research/exploit/20261014-pseudo-c3-dump` (STATUS `proposed`),
  stalls `pseudo-c3-dump:spec` = 1, calls opus 8 / sonnet 3. Stop rules 5/4/2/12.
- **Interrupted**: the spec retry for pseudo-c3-dump (started 17:21) was killed by Claude Code for low system memory,
  not by the step. Attempt 1 earlier died on the Opus usage limit. The stall counter did not count the killed run, so
  the next `--once` is attempt 2/2.
- **Stale lock**: `research/campaigns/20261014-e180-npool-mutator/campaign.lock` (pid 20596, dead). Confirm the pid
  is gone (`tasklist /FI "PID eq 20596"`), then delete the lock before the next `--once`.
- Exploit worktree is checked out on `idea/20261014-e180-drop` (commit `fbd206f`: `E180_DROP`/`E180_KEYS` knob, default
  off, plus lib-isomer-margin's previously uncommitted edits, `LIB_ISOMER_MARGIN=None` default). The next build should
  branch from `line/exploit` (or knowingly from this). Stray untracked file to delete:
  `E:/casmi/lines/exploit/scratch_dbg_e180.py`.

## Next steps
1. Clear the lock; run `--once` → `/spec research/exploit/20261014-pseudo-c3-dump` (Opus).
2. Review the spec. It must make the pseudo_c3 truths unlistable in **all four** E006 candidate channels: Engine 1
   v2-pool, Engine 2 `build_pool` (COCONUT+BIO+train), the PubChem tier (105.9M), and the DreaMS-NP tail fill.
   The partial spike (`research/exploit/20261014-pseudo-c3-dump/spike.log`) only checked base/pc/eng lists. Inputs:
   `p1_novel/pseudo_c3.parquet` (600 mols, `pool_ban_keys`, `lib_ban_keys`), `build/patch_c3cal.py` (ban logic for the
   old v4n notebook), the dump pattern `build/patch_harness_dump.py --shard <name>` (see instrument folder
   `research/exploit/20261013-harness-c2hard`: kind instrument, pinned docker_image, ICE completed separately via
   `20261013-harness-iceonly`). Pre-registered pass: dump replays, E006 cfg reproduces, truth in no channel list for
   100% of molecules. A Kaggle run needs human approval (GPU ~3 h).
3. Decision point for the user: the replan scored the whole mutator chain at **EV 0.0002** (25% assumed C3 share ×
   +0.010 ceiling × P 0.15). Ask before spending the GPU dump whether to continue or stop the campaign.
4. If continued: c3-detector (precision gate: fires ≥ 30% pseudo_c3, ≤ 2% of C2∪C2_hard truths at ranks 1–5, ≤ 1% C1;
   relative signals only, thresholds fixed on C2_np, no sweep) → mutator-c3v2 ([2,3,4] if detector passes, else ranks
   21–25; `'|gen'` tag; dead on any rank-1 displacement or C2/C2_hard CI upper < −0.004).

## Results so far (all in log.md)
| cycle | idea | result |
|---|---|---|
| 1 | e180-drop | **dead** (gate.py: attack A4). C2 +0.0038 [+0.0006, +0.0076], 7 up / 0 down, but leak-free "other" +0.0003 vs leak-hit +0.0042; C2_hard other +0.0012. A1(b') survived (0/250 np-examples carry the enveda-180 bit), A2/A5/A6 survived, A3 UNTESTED (`pool_smiles.parquet` not local). A1(b) was amended by the human (instrument share is 1.0 for every Enveda library). |
| 2 | engine2-npx | **dead** (human, after spec). 4,748 NP-table structures absent from Engine 2 (2,629 from both engines; `spike/reach_flags.md`); "reach > 0" true by construction; val reach 0/1,390; arithmetic ceiling ≈ +0.002; row 32 VOID. |
| 3 | pseudo-c3-dump | spec interrupted (see above) |

## Done in this session (do not redo)
- Old campaign `20261014-qoder-structural` stopped via `finish()` (report.md written). lib-isomer-margin is submitted;
  its own spike FAILED (F1 5% vs 50%, oracle −0.0088). **Still to do**: `python research/bin/poll_lb.py --claim
  research/exploit/20261014-lib-isomer-margin` once its LB score is known.
- Lessons promoted: L44 (split recall reach), L45 (gate ignores "other" Δ), L46 (blocking-attack inputs must be
  local), L47 (scorecard family reads `capability_map.json`), L48 (reach rule must be able to return zero). Rule R09
  (spec cites a spike → spike_result.md filled). Rejected: e180 P4 (brief length).
- Selftest FLAGS updated: R02 (+xattn-rerank, lib-isomer-margin, refspectra-lib), R04 (+formula-window,
  refspectra-lib), R08 (+ice-margin-route, e180-drop).
- `backtests.yaml` `e006_overlay` += `E180_DROP, E180_KEYS` (human-approved). **Open**: `capability_map.json` is
  stale, so new knobs show `unknown:` on the scorecard; refresh with `harness.py suite` when convenient.
- Memory: `casmi-campaign-e180-npool-mutator.md` (index line in MEMORY.md).

## Watch-outs
- Judge by the leak-free "other" Δ (L40/L45); gate.py alone returns QUEUE on leak-hit gains.
- Pre-registered conditions are changed only by a dated human amendment kept next to the original (as in
  `research/exploit/20261014-e180-drop/attack.md`).
- PowerShell 5.1 corrupts UTF-8 rewrites; edit with the Edit tool or Python. Do not read raw `.ipynb`.
