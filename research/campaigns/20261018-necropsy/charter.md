# Campaign charter: 20261018-necropsy

Grill 2026-10-18 (conductor plan, decision 13): dead ideas are not abandoned. Learn why each died and revive the ones
a real change can fix.

## Scope
- Line: necropsy
- Mode: necropsy
- Base: V22 = `casmi26-v22-rel150-e2r2-clean` (keep_bar 0.451, paired control `research/c3/20261017-control-v22`);
  worktree `E:/casmi/lines/necropsy`, branch `line/necropsy`.
- Idea pool: `research/ideas/*-revivals.md`, written by `/necropsy-synth` from the `necropsy.md` of every dead idea
  (all lines). When the pool is dry the driver autopsies up to 5 more dead ideas and re-synthesises.
- Each revival is a NEW idea folder under `research/necropsy/`, whose hypothesis.md carries `revives:`, `cause:`, `fix:`
  (rule R12). Combinations are allowed only when the parts act on different cohorts or leak-hit/other strata.
- Ideas from exploit (E006 base) that need porting to V22 say so in spec.md; the port is part of the build.
- Heavy compute runs on Kaggle (CPU default; GPU within the 30 h/week pool, autonomous since 2026-10-18).
- Isolation: never write in another line's worktree or another campaign's idea folders; necropsy.md files in dead
  idea folders are the only exception (side files, STATUS untouched).

## Bars
- Gate: the rank-only gate (`research/harness/gate_report.md`, 2026-10-13 block); judge by the "other" Δ, not leak-hit.
- KEEP at submit: `poll_lb.decide` (paired V22 control + 0.004, counters > 0, one replicate).

## Stop rules
- max_cycles: 6
- max_lb_probes: 3
- dead_streak: 3

## Out of scope
- Re-running a measurement that already killed an idea without a mechanism change (R12).
- Signals in `research/necropsy/dead_signals.yaml` unless re-MEASURED (R10, R11).
- Edits to the loop code, skills, CLAUDE.md, or other campaigns' charters.
