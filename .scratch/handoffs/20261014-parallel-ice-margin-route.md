# Handoff 2026-10-14 (Claude): parallel idea `ice-margin-route`, spec → attack → build → gate (harness only)

Start a **fresh Claude Code session in `E:\ENVEDA_casmi`** and paste: "Read
`.scratch/handoffs/20261014-parallel-ice-margin-route.md` and do it."

## Why this runs in parallel, and what is happening elsewhere (don't touch it)
- Campaign `research/campaigns/20261014-qoder-structural`, cycle 3. Its current idea is
  `research/exploit/20261014-lib-isomer-margin` (STATUS `attacked`).
- An **Antigravity chat agent is shipping lib-isomer-margin right now**
  (`.scratch/handoffs/20261014-agy-ship-job.md`):
  - it creates the Kaggle dataset, pushes the falsifier kernel, and pushes the build notebook with `kaggle_push.py`;
  - `kaggle_push` sets that idea's STATUS to `g4-pushed` and starts a detached G4 watcher;
  - a G4 FAIL can start a `/run-idea` fix agent **inside `E:/casmi/lines/exploit`**.
- So for this session:
  - **Never** run `research/bin/campaign.py` (any flag). Stepping it would close cycle 3 before the falsifier verdict.
  - **Never** edit `state.json`, nor anything under `research/exploit/20261014-lib-isomer-margin/`.
  - **Never** touch the `E:/casmi/lines/exploit` worktree: it holds lib-isomer's uncommitted work on
    `idea/lib-isomer-margin`. You may read files there.
  - Run this idea **manually**, outside the driver: the skills by hand, in your own worktree (below).

## The idea: `ice-margin-route` (candidate C, rank 3 in `research/ideas/20261014-candidates.md`)
Read that file's §3 (and §6 `ice-libprotect` = D), and `research/ideas/2026-10-14-libgate-diagnosis.md` §3.
- **Measured motivation** (diag `.scratch/libgate_diag/diag.json`). The ICE sign flips by class: removing E006's
  ICE everywhere gives C1_npx +0.011 and C1_xlib +0.013, but C2 −0.069.
  - D, the absolute gate (skip ICE when `lib_max ≥ 0.9`), gave C1_npx +0.0054 [+0.001, +0.011], C1_xlib +0.0033,
    C2 −0.0005 [−0.0035, +0.002] and C2_hard +0.001.
  - That is small, because `lib_max` is a weak class signal (C2_np false-lock rate is 24.5% at 0.7).
- **Mechanism**: in the metric overlay, the ICE fallback (`E:/casmi/lines/exploit/build/casmi_post.py:372-373`,
  `elif c['ICE_FALLBACK'] and ice and not gl:` → `ice_rerank(...)`) fires only when the fused list is **uncertain**.
  - Gate signal: the relative margin between the top-1 and top-2 `metric_fusion` scores (the `scores` returned at
    `casmi_post.py:361`, from `metric_fusion` at `:144-161`), e.g. `(s1 - s2) / s1`. This is within-molecule and
    normalised, NOT an absolute score (graveyard row "Absolute f·z gate").
  - A confident top-1 skips the ICE rerank.
  - New default-off knob `ICE_ROUTE_MARGIN=None`. `None` must reproduce E006 byte for byte.
- **Why it's cheap**: a pure `casmi_post` knob, replayable on the dumps that already exist
  (`research/harness/dumps/{C1_npx,C1_xlib,C2,C2_hard,C2_np}/post_dump/state.pkl`). The fused scores are recomputed
  at replay time, so you need no new dump field, no GPU and no Kaggle.
- **Pre-registered falsifier** (from the candidates file; keep it, append to it, never edit it):
  - The margin is fixed **a priori** as the median top-1/top-2 relative RRF gap on C2_np. Compute it once, write it
    into `spec.md` and the `gate_report.md` pre-reg block **before** running the scorecard. No tuning, no sweep (L08).
  - Dead if C1_npx Δ ≤ +0.0054 (it does no better than D), **or** the C2 or C2_hard 90% CI upper bound < −0.004,
    **or** the standard rank-only gate fails.
- **Stack note**: if it lives, it is lib-isomer's stack partner (it replaces D). Both edit `casmi_post.py`, in
  different functions (`_metric_overlay` vs the final `_lib_isomer` step), so a later merge is mechanical.

## Setup (do first)
1. **Your own worktree** (new branch from `line/exploit` @ 052c96f; nothing in the existing worktree moves):
   ```
   git -C E:/ENVEDA_casmi worktree add E:/casmi/lines/exploit-b -b idea/ice-margin-route line/exploit
   ```
   - The committed `build/casmi_post.py` at 052c96f is E006's version without lib-isomer (lib-isomer is uncommitted
     in the other worktree, +50 lines). Confirm with `git -C E:/casmi/lines/exploit-b diff line/exploit --stat`
     (empty).
   - `scratch/casmi26-e006.py` (the base export) is untracked, so it is **not** in the new worktree. Cite and read it
     at `E:/casmi/lines/exploit/scratch/casmi26-e006.py` (read only).
2. **Idea folder**: `python research/bin/new_idea.py exploit ice-margin-route`, which creates
   `research/exploit/20261014-ice-margin-route/` with STATUS `drafting`.
   - Write `hypothesis.md` from candidate C, with `novelty: variant`, branch `scorer`, and a `tier: H` first line if
     you are Opus.
   - Add `worktree: E:/casmi/lines/exploit-b` to STATUS with `write_status(..., worktree=...)` from
     `research/bin/_common.py:38`. Never write the file by hand.
3. **Every harness call takes `--post`**, because the harness defaults to the *other* worktree's `casmi_post.py`
   (`research/harness/harness.py:34,113`):
   `--post E:/casmi/lines/exploit-b/build/casmi_post.py`.

## Steps (the qoder loop by hand: `.scratch/qoder_prompt.md` §4.1–§4.3, §5)
For each stage, generate the brief and read **it**, not README/LESSONS/EXPERIMENTS:
`python research/bin/brief.py research/campaigns/20261014-qoder-structural <step> research/exploit/20261014-ice-margin-route`.
Then follow `.claude/skills/<step>/SKILL.md`. Wherever a brief or skill says the worktree is `E:/casmi/lines/exploit`,
use `E:/casmi/lines/exploit-b` for writing (cite line numbers from either; the two are identical apart from lib-isomer).

1. **Spec** (`.claude/skills/spec/SKILL.md`):
   - knob `ICE_ROUTE_MARGIN`; target cohort **C1_npx**;
   - scorecard command (exact):
     ```
     python research/harness/harness.py scorecard --a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40" --b "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40,ICE_ROUTE_MARGIN=<m>" --target C1_npx --line exploit --post E:/casmi/lines/exploit-b/build/casmi_post.py --json research/exploit/20261014-ice-margin-route/harness_result.json
     ```
   - G4 counters: `ice_route_skipped` > 0 and `g4_min_changed: 1`;
   - the a-priori margin rule;
   - then `python research/bin/preflight.py research/exploit/20261014-ice-margin-route` until it has 0 findings.
     Set STATUS `specced` (write_status).
2. **Pre-register**: append a dated block `### 2026-10-14: ice-margin-route, PRE-REGISTERED` to
   `research/harness/gate_report.md` (append only) with the margin rule, the computed margin value, and the kill bars
   above. Compute the C2_np median gap with a small script in `.scratch/ice_route/` that loads the C2_np dump and
   replays E006 (copy the approach of `.scratch/libgate_diag/diag.py`). Do this **before** the scorecard.
3. **Attack** (`.claude/skills/attack/SKILL.md`): ≥ 3 runnable attacks in `attack.md`. Suggested attacks:
   - **A1**: the gain is just D under another name. Check how many C1_npx molecules the margin routes differently
     from `lib_max ≥ 0.9`.
   - **A2**: C2_np routes are mostly confident-wrong lists, i.e. ICE was rescuing them. Check by measuring Δ on the
     C2_np skipped set.
   - **A3**: ICE coverage differs on the Kaggle GPU vs the dumps (memory: GPU vs CPU changes ICE coverage). Check the
     share of molecules where `ice` exists in each dump.
   - Set STATUS `attacked`.
4. **Build, harness mode** (`.claude/skills/build/SKILL.md`, harness mode ONLY), in `E:/casmi/lines/exploit-b` on
   `idea/ice-margin-route`:
   - add the knob, a counter `stats['ice_route_skipped']`, and a pure helper with unit tests (follow the style of
     `lib_isomer_rows` and its tests in the other worktree);
   - prove the `None` default is a no-op: on the C1_npx dump, `--b` with only the default gives Δ = 0 exactly;
   - run any stress test the skill asks for (`stress_result.json`);
   - run the scorecard above, then STATUS `harness-done`.
5. **Gate**: run the python one-liner from `.scratch/qoder_prompt.md` §4.2 with `i=Path('research/exploit/20261014-ice-margin-route')`.
   - Add the falsifier bars from the pre-reg on top: dead if C1_npx Δ ≤ +0.0054.
   - Record the result in `gates.md` (`log_gate` from `_common.py`).
   - STATUS `dead` (with the why) or `build`.
   - Also report the `d_lib_leak` reading. On C1 the "other" column hides library gains (open loop proposal in the
     campaign `questions.md`), so give leak-hit and other separately.
6. **STOP before ship.** Don't run `kaggle_push.py`. It would run G1/G2 in the config worktree (`E:/casmi/lines/exploit`,
   the wrong one) and spend a Kaggle slot. Write `research/exploit/20261014-ice-margin-route/result_summary.md` with
   the scorecard table per cohort (Δ, CI, displacements, leak/other split), the gate verdict, and the stack proposal
   with lib-isomer. Shipping is for the human to decide after lib-isomer's falsifier verdict.
7. Append one line to `research/campaigns/20261014-qoder-structural/log.md`:
   `| parallel (manual) | research/exploit/20261014-ice-margin-route | <verdict, numbers> | outside the driver, worktree exploit-b`.

## If it dies early
If the a-priori margin or the attack kills it, and only then, run the same procedure on **candidate B
`test-network-vote`** (candidates file §2). Its falsifier step (i) is a CPU ceiling script on the val parquet and
the post_dump lists; it needs no GPU. It gets its own worktree, `E:/casmi/lines/exploit-c` on
`idea/test-network-vote`. Stop at the same point (no ship), and stop sooner if a step needs GPU or Kaggle.

## Hard rules (CLAUDE.md, the qoder prompt, the user)
- Only the human submits; submit autonomy is OFF. No Kaggle pushes or GPU in this session.
- No commits in the main repo. Commits only on `idea/ice-margin-route` in `exploit-b`, and only if the build skill
  asks.
- Never run `git reset --hard`, `clean` or `checkout --`. Never stage `.npz/.pt/.parquet/.zip`.
- Never edit loop code (`research/bin`, `research/harness`, `research/rules`, skills, `CLAUDE.md`, config,
  `.claude/settings*`). Proposals go in `questions.md`.
- Pre-registered text is append-only. Never read raw `.ipynb` JSON (use `compare.py`).
- After two failed attempts at a step, write `questions.md` and stop.

## Useful facts (verified 2026-10-14)
- Harness base cfg: `METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40` (`research/config.yaml:13`).
  The CLI takes `KEY=value`, not JSON.
- `harness.py scorecard` flags: `--a --b --target --cohort --post --line --json` (`harness.py:637-639`).
- Dumps on disk: C1_npx (245), C1_xlib (300), C2 (400), C2_hard (200), C2_np (advisory).
- E006 replay MRR reference: C1_npx 0.9355 (diag).
- `metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40)` returns `(cands, scores, collapsed)`, with
  scores sorted in descending order (`casmi_post.py:144-161`).
- The ICE fallback in the overlay is `casmi_post.py:372-373`; the base-path fallback is `:286-303`, and it is
  inactive when `METRIC_FUSION` is on.
- Memories that apply: "BASE leaks on every val shard" (read the leak-hit/other split), "LB noise floor ±0.004",
  "Exploit base = E006 v1", "Headless loop pitfalls" (the CLI is `KEY=value`; compound Bash is denied in headless
  runs).

## Done when
`result_summary.md` exists, STATUS is `dead` or `build`, the gate is recorded in `gates.md`, and the log line is
appended. Tell the human the verdict in ≤ 10 lines.
