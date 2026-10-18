---
name: necropsy-synth
description: Campaign step (tier H, headless): read every necropsy.md and propose ranked revivals and combinations of dead CASMI ideas. Invoked as `/necropsy-synth research --brief <path>`.
---

# Necropsy synth

Dead ideas are only worth reviving when the fix addresses the recorded cause. You turn the necropsies into ranked, falsifiable candidates.

1. Run `python research/bin/necropsy.py --list`. Read every `necropsy.md`, `research/necropsy/dead_signals.yaml`, and the headers of `research/harness/gate_report.md`. Nothing else unless a necropsy cites it.
2. Write `research/ideas/<yyyy-mm-dd>-revivals.md` with at least 3 candidates, ranked. Each has:
   - `revives:` the dead idea path(s)
   - `cause:` the necropsy cause it addresses
   - `fix:` what changes in mechanism, data, or target stratum (a knob re-tune is refused by R12)
   - falsifier: the number that kills it, pre-registered
   - cheapest harness measurement, run on Kaggle per CLAUDE.md
   - expected delta against the +/-0.004 LB noise floor
3. A combination candidate adds `combines:` and must show, with numbers, that the parts act on different cohorts or on leak-hit versus other strata. If the parts act on the same stratum, drop it.
4. Never name a signal from `dead_signals.yaml` unless the same line says `MEASURED` and cites a new measurement.
5. If you find a new measured-dead signal, append it to `research/necropsy/dead_signals.yaml` only with `src:` pointing at an existing file that shows the measurement.
6. When a candidate becomes an idea folder, `python research/bin/preflight.py <idea>` checks it against R12.

Do not edit any idea's `STATUS`, `necropsy.md`, or `EXPERIMENTS.md`. Done when the ideas file exists with >= 3 candidates and each cites its necropsies.
