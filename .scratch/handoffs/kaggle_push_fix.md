# Handoff: a detached Kaggle watcher that closes G4 with no one waiting

Written 2026-10-10 by a Claude Code session after a grill with the user. **The decisions below are the user's: implement them, don't re-litigate them.**

---

## 1. Background you need (read first)

- **Repo:** `C:\Users\h\OneDrive\Desktop\ENVEDA_casmi` (Windows 11; PowerShell primary, Git Bash available). Branch `specs-scratch-r0`; last commit `a4da9f5` "research/: fix the loop after the first cycle (L30-L33)".
- **Competition:** Enveda CASMI 2026 on Kaggle, slug `enveda-CASMI26-molecule-id-mass-spectra`, Kaggle user `syedtoubur`.
- **Multi-agent research loop** (see `research/README.md`, `CLAUDE.md` §4):
  - Gemini, in Antigravity, ideates: `/cycle`, `/critic`, `/daily-digest`.
  - `research/bin/call_claude.py` starts headless Claude (`claude -p "/run-idea research/<line>/<idea>"`, Pro subscription, no API key). One Claude job runs at a time, enforced by a lock and a queue.
  - `/run-idea` (`.claude/skills/run-idea/SKILL.md`) runs judge → tickets → implement → gates G1–G4 → submit card.
  - **Only the human submits on Kaggle.**
- **Idea folder:** `research/<line>/<idea>/`.
  - Line 1 of `STATUS` is the state; the other lines are `key: value` meta. Read and write it via `_common.read_status` / `write_status`.
  - Gate history goes in `gates.md` via `_common.log_gate(idea, gate, result, evidence, cu="")`.
  - Questions for the human go in `questions.md`.
- **G4** = push the probe notebook as a private Kaggle kernel version. This is a **draft save version**: no `KAGGLE_IS_COMPETITION_RERUN`, so the fast-commit path runs on about 12 molecules. It takes **about 10–25 min** from push to COMPLETE; the control's last run took 03:19 → 03:41, mostly queue and installs.
  - `kaggle_push.py` builds `kernel-metadata.json` from the preset in `specs/Dataset_intro/dataset_catalog.json` and refuses to push in two cases (L30/L31):
    - **rc 5:** the source audit fails;
    - **rc 6:** the line's control has no G4 PASS.
  - After the run, `finalize()` downloads only `*.log`, `*.csv` and `*_work/*meta.json`. It FAILs on any `gates.g4_fail_markers` hit, a missing `--counter`, or a missing line `g4_required` regex. It sets `ready-to-submit` or `g4-failed` and appends the result to `submit_card.md`.

## 2. The problem being fixed

After `/run-idea` pushes, nothing finishes G4:
1. **The first run** started `kaggle_push --wait` as a `claude -p` background job. It died when the session ended (L33), so `research/exploit/20261010-control-0433` sat at `g4-pushed` until it was finalized by hand.
2. **The L33 fix** made `/run-idea` push without `--wait` and stop at `g4-pushed`. Only `kaggle_push.py --check` turns that into a verdict, and it runs only from `/daily-digest` or by hand. **A state that only a human-run command can leave will stall.**
3. **`--wait` prints nothing while sleeping 60 s per loop,** so in a terminal it looks hung.
4. **A finished verdict notifies nobody.**

## 3. Decisions (user, grill 2026-10-10)

| Topic | Decision |
|---|---|
| Who waits | A **detached OS process** that `kaggle_push.py` starts right after a successful push, using the same flags as `call_claude.py --detach` (`0x00000008 \| 0x00000200` = DETACHED_PROCESS \| CREATE_NEW_PROCESS_GROUP, see `call_claude.py` ~line 191). It outlives Claude. No Claude tokens are spent waiting. `--check` stays as the backup, and `/daily-digest` runs it. |
| Polling | Every **30 s**, printing a heartbeat line each poll. **Give up at 60 min**: STATUS `g4-timeout`, a gates row and a toast. |
| On PASS | A **Windows toast**. If the idea is the line's **control** (`kind: control` in STATUS), also **auto-queue that line's waiting probes** through `call_claude.py <idea> --detach`, one at a time under the existing lock. |
| On FAIL | A toast, then hand off to a **fresh, independent Claude agent** with a new context, not a resumed one. The watcher writes a compact **`g4_fix_brief.md`**. The new agent reasons step by step from the brief and the Kaggle log, fixes and re-pushes. |
| Fix rounds | At most **2** automatic fix → re-push rounds. On the 3rd FAIL: `needs-human`, a `questions.md` entry and a toast. |
| Fix scope | The fix agent may change the patch code in the line worktree (`C:/casmi/lines/<line>`), then rerun G1 and G2. It may also add push sources **that already exist in `dataset_catalog.json`**. A dataset missing from the catalog, or a fix that would change the idea itself, means: write to `questions.md`, set `needs-human`, stop. For a **control** idea only sources and metadata may change, never the notebook code. The catalog itself is never edited. |
| E2E test | After building, **re-push control v4 with all 16 sources** through the new watcher. The last PASS (v3) was pushed before the user fixed the catalog, so it lacked the DreaMS source. |

## 4. Implementation

### 4.1 `research/bin/kaggle_push.py`
Read the whole file first. The relevant existing pieces:
- `kaggle(args)` is the CLI runner, injected as `run=` for tests. `main(argv, run, sleep)` takes the same injections.
- `parse_status(text)` and `DONE = ("complete","error","cancelacknowledged")`.
- `finalize(cfg, idea, kernel, status, run) -> 0 PASS / 2 FAIL`.
- `check(cfg, ideas, run)` is one poll per `g4-pushed` idea.
- `check_log`, `grep_counters`, `log_text_of(path)` (a JSON-aware log reader) and `control_passed(cfg, line)`.
- The tail of `main` is the `--wait` loop: `deadline`, `sleep(60)`, then `finalize`.

Add:
1. **`--watch <idea>` mode → `watch(cfg, idea, run, sleep, clock=time.time)`**
   - Takes `watch.lock` (see 3).
   - Loops `parse_status(run(["kernels","status",kernel]).stdout)` every `cfg["kaggle"]["watch_interval_s"]` (30) and prints `[watch] <kernel> <status> t+<s>s` with `flush=True`.
   - On DONE: `rc = finalize(...)`, then `after_g4(cfg, idea, rc == 0, run)`.
   - At `watch_timeout_s` (3600): `write_status(idea, "g4-timeout")`, `log_gate(idea, "G4", "TIMEOUT", ...)`, toast.
   - In a `finally`, remove the lock.
   - Treat a transient CLI failure (non-zero rc or status `unknown`) as "keep polling", not as a FAIL.
2. **`spawn_watcher(idea)`:** called at the end of a successful push when `--wait` is 0. It replaces the bare `return 0` after `G4 PUSHED`.
   - Starts `[sys.executable, __file__, "--watch", rel]` with `cwd=REPO`, the detach flags, and stdout/stderr going to `<idea>/kaggle_push/watch.log`. Open the file and pass the handle. Use `python -u` or `flush=True` so the heartbeat shows live.
   - Make it injectable (`spawn=` param on `main`) so tests don't start processes.
3. **`watch.lock`** in `<idea>/kaggle_push/`, holding the pid and start time:
   - A second `--watch` for the same idea exits 0 with a message while the lock is younger than `watch_timeout_s + 300`.
   - An older lock is stale; overwrite it.
   - Note: `main` currently does `shutil.rmtree(stage)` on each push. That is fine, because a new push should supersede an old watcher. The old watcher must then notice that the STATUS `kernel` or state changed and exit quietly: check `read_status` each loop and stop if the state is no longer `g4-pushed`.
4. **`after_g4(cfg, idea, ok, ...)`**, called from `watch` and from `check` after `finalize`, so the backup path behaves the same:
   - **PASS:**
     - Toast: `"<kernel> G4 PASS: ready to submit"`.
     - If `meta.kind == "control"`: for each `research/<line>/*/STATUS` other than the control whose state is in `call_claude.WAITING_STATES`, run `call_claude.py <rel> --detach` (injectable `queue=`).
     - Include the list of queued probes in the toast and in a gates row.
   - **FAIL:**
     - `n = int(meta.get("g4_fixes", 0))`.
     - If `n < cfg["gates"]["g4_fix_rounds"]`: `build_fix_brief(...)`, `write_status(idea, "g4-fix", g4_fixes=n+1)`, queue `call_claude.py <rel> --detach`, and send the toast "G4 FAIL, fix agent round n+1 started".
     - Otherwise: `write_status(idea, "needs-human")`, append to `questions.md` (reason and brief path), and toast.
   - The push-time FAIL ("push dropped sources", currently sets `g4-failed` in `main`) should also go through `after_g4(ok=False)` so a fix agent can add the source.
5. **`build_fix_brief(cfg, idea, log_text, reasons)`** writes `<idea>/g4_fix_brief.md`, short and self-contained:
   - the kernel, version/attempt, `kind`, line and worktree path;
   - the one-change summary: the first section of `spec.md`, capped at about 40 lines (skip for a control);
   - the fail reasons, as returned by `check_log` plus the missing `g4_required` entries (`finalize` already computes these; refactor so they are returned, not only logged);
   - every log line matching a fail marker or `Traceback`, with **±20 lines of context**, deduped and capped at about 300 lines;
   - the **last 80 log lines**;
   - the `gates.md` history;
   - for non-control ideas, `python .agents/skills/analyze-notebook/scripts/compare.py <base_notebook> <pushed nb>` output, capped at about 200 lines. **Never read raw `.ipynb`** (CLAUDE.md §3).
   - the fix-scope rules from §3 above, verbatim.
6. **`toast(title, body)`**:
   - Best effort, so an exception is only logged. Uses `powershell.exe -NoProfile -Command` with the WinRT `Windows.UI.Notifications.ToastNotificationManager`, with the PowerShell AppId `{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe`.
   - Pass the text through `-EncodedCommand` (base64 UTF-16LE) or env vars to avoid quoting bugs.
   - Make it injectable so tests never pop a toast.
7. **`--wait`:** add the same `flush=True` heartbeat. Keep it working; it is still handy by hand.
8. Update the module docstring.

### 4.2 Other files
- **`research/bin/call_claude.py`:** add `"g4-fix"` to `WAITING_STATES`. The command stays `/run-idea <rel>`, and every call is already a fresh `claude -p` with a fresh context.
- **`research/config.yaml`:**
  - add a top-level `kaggle:` block with `watch_interval_s: 30` and `watch_timeout_s: 3600`;
  - add `gates.g4_fix_rounds: 2`.
  - `research/bin/test_bin.py`'s fixture copies the real config, so set or check these values there.
- **`.claude/skills/run-idea/SKILL.md`:**
  - Change the g4 step 3 text: the push starts the watcher automatically, and `--check` is only a backup.
  - Add a **g4-fix** state:
    1. Read `g4_fix_brief.md` first. Open other files only where the brief points.
    2. Write `g4_fix_<n>.md`: symptom → root cause, with the log line as evidence → fix, step by step.
    3. Apply the fix within scope. For code: commit in the worktree, rerun G1 (compare.py) and G2 (`python research/bin/run_tests.py <line> --idea <idea>`), and rebuild the probe notebook.
    4. Re-push with **no `--wait`**, then stop.
    5. Out of scope: write to `questions.md`, set `needs-human`, stop.
  - Add `g4-timeout` and `needs-human` to the "do nothing" list.
  - **No `run_in_background` ever (L33).**
- **`research/README.md`:** update the state machine:
  ```
  g4-pushed → ready-to-submit | g4-fix → g4-pushed … | needs-human | g4-timeout
  ```
  Add a short "G4 watcher" paragraph: watch.log, toasts, the 2-round limit, and that a control PASS auto-queues probes.
- **`.agents/skills/daily-digest/SKILL.md`:** show `g4-fix` (in progress), `g4-timeout` and `needs-human` (under Needs you).
- **`research/LESSONS.md`:** add L34: "A wait needs an owner process: a state that only a human-run command can leave will stall (G4 sat at g4-pushed). The push now starts a detached watcher." Keep the existing style (read L30–L33).

## 5. Tests: `research/bin/test_bin.py`
It is a plain-script custom runner, not pytest: `python research/bin/test_bin.py`. It currently passes **33/33**. Read its fixture (temp research dir, patched modules, `lines.exploit.control = None`, `g4_required = []`, stubbed `colab_run.login_problem`) and follow the existing fake-runner style. Add these tests, with fake `run`, `sleep`, `clock`, `spawn`, `queue` and `toast`:
- `watch` finalizes on the first DONE poll, prints a heartbeat per poll, and removes the lock.
- `watch` keeps polling through a transient CLI error.
- `watch` times out to `g4-timeout` and sends a toast.
- `watch` exits quietly when STATUS is no longer `g4-pushed` (it was superseded).
- A fresh lock refuses a second watcher; a stale lock is overwritten.
- FAIL with `g4_fixes` unset writes a brief containing the marker context and the log tail, sets `g4-fix` with `g4_fixes: 1`, and queues the idea.
- FAIL with `g4_fixes: 2` sets `needs-human` and adds a `questions.md` entry; nothing is queued.
- A control PASS queues a waiting probe on the same line only. A non-control PASS queues nothing.
- A push without `--wait` calls `spawn` once; a push with `--wait` doesn't.
- `--check` also runs `after_g4`.

All 33 old tests must still pass.

## 6. End-to-end verification (the user approved this run)
1. **Control v4.** Use `casmi-notebook-0-433.ipynb` (repo root) unchanged. In `research/exploit/20261010-control-0433/STATUS`, set the state to `g4` and keep `kind: control`. Run:
   ```
   python research/bin/kaggle_push.py research/exploit/20261010-control-0433 casmi-notebook-0-433.ipynb
   ```
   It must return within seconds. The source audit should show all **16** sources, including DreaMS; `dreams_np_mass.npy` used to be unresolved.
2. **Watch.** Tail `research/exploit/20261010-control-0433/kaggle_push/watch.log`. You should see 30 s heartbeats. **Do not keep a Claude session waiting on it.** Check back with one-off reads, or tell the user it is running.
3. **Expected result** about 20 min later:
   - a G4 PASS row in gates.md;
   - a toast;
   - `submit_card.md` mentions the new version;
   - `research/exploit/20261010-adduct-iceberg-rerank` (STATUS `g2`) is queued through `call_claude --detach`. The user wants this.
4. If it FAILs instead, the fix-agent path runs for real. That is also a valid test, so report what happened.

## 7. Hard rules
- **Never** run `kaggle competitions submit`. Only the human submits.
- **Never** stage `.npz/.pt/.parquet/.zip`. Don't commit the `kaggle_output/` or `kaggle_push/` folders.
- **Never** read raw `.ipynb`; use `compare.py`.
- **No `--dangerously-skip-permissions`.**
- **Commit only when the user asks; never push git.**
- **Edit Python with the Edit or Write tools, not heredoc or sed.** Last session, backslashes in regexes got mangled (backspace characters, literal ESC).
- In YAML, write regexes single-quoted and escape-free (e.g. `'Traceback .most recent call last.'`).
- Another agent may edit files in this repo concurrently. Leave alone the untracked files you didn't create (`.agents/hooks.json`, `.agents/scripts/`, `.agents/skills/smart-zone-handoff/`, `specs/Dataset_intro/`, `research/queue/`).
- The 4 `casmi_simple_sim_*.ipynb` files have stray 1-line diffs from an earlier run. Don't touch them.

## 8. Key files

| File | Why |
|---|---|
| `research/bin/kaggle_push.py` | The main change |
| `research/bin/call_claude.py` | Detach pattern, `WAITING_STATES`, lock and queue |
| `research/bin/_common.py` | `REPO`, `RESEARCH`, `idea_dir`, `load_config`, `read_status`, `write_status`, `log_gate` |
| `research/bin/test_bin.py` | Tests (plain script) |
| `research/config.yaml` | New `kaggle:` block and `gates.g4_fix_rounds` |
| `.claude/skills/run-idea/SKILL.md` | g4 text, plus the new g4-fix state |
| `research/README.md`, `.agents/skills/daily-digest/SKILL.md`, `research/LESSONS.md` | Docs |
| `research/exploit/20261010-control-0433/` | The E2E subject. gates.md shows its history: v1 and v2 ERROR because of missing sources, v3 PASS. |
| `research/exploit/BASE.md` | 0.433 cell map, counters and markers, if you need to understand log lines |
