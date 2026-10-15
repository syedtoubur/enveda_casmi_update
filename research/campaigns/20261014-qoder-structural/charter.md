# Campaign charter: 20261014-qoder-structural

Run by hand in Qoder Desktop (single model, recorded tier M in `research/runtime.md` ## Qoder), following
`.scratch/qoder_prompt.md`. Files are in `campaign.py`'s format so a Claude session can resume cold.

## Scope
- Line: exploit
- Base notebook: `casmi26-e006-window-40-ice-fallback-true.ipynb` (E006 v1, LB 0.440, one run, not replicated) in the
  worktree `E:/casmi/lines/exploit` (branch `line/exploit`); cite base lines from `scratch/casmi26-e006.py`. Paired
  control: `research/exploit/20261012-control-e006`; control draft: `research/exploit/20261013-control-draft`.
- Posture: structural mechanisms (new evidence channels, new scorers), not ±0.004 knob tweaks.
- Supersedes `20261013-e006-structural` (charter written, never started; its seeds were measured pre-slot on
  2026-10-13, `research/ideas/2026-10-13-candidates.md` "Pre-slot measurements"). That folder is not edited or run.
- Harness: `--a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"`; the battery is every cohort with a
  `post_dump/state.pkl` (at charter time: C2, C1_npx, C2_hard, C1_xlib; C2_np advisory). Judge by the **other**
  (non-leak-hit) Δ (L40, calibration.md).
- The loop code (`research/bin`, `research/harness`, `research/rules`, skills) is a constant; changes go to
  `questions.md` as proposals.
- Submit: the human, or Qoder under a session-scoped grant (qoder_prompt §10) with its eligibility checks written in
  the idea's `gates.md`.

## Portfolio
| slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV |
|---|---|---|---|---|---|---|---|---|
| refspectra-lib | evidence-channel | C1_xlib / C2 (truths whose structure has a measured MassBank spectrum but is not at rank 1) | The tier-1 library channel only reads the competition's own train spectra (`Library('work')`, `casmi/engine.py:313`). Measured MassBank spectra (public Kaggle dataset `samartalwar/casmi-2026-spectral-library-massbankharmonized`, 18,316 IK14) matched against the query spectra are orthogonal evidence (L10) for candidates already in the lists | new default-off `REFLIB` channel fused after the E006 overlay; spec picks the fusion form | yes | MEASURED reach (`.scratch/refspectra/reach.py`, 2026-10-13): covers 16.8% of C2 truths, 62 of 67 already rank 1 → 5 targets, ceiling ≈ +0.012; 81.2% of C1_npx (17 targets). ASSUMED: test-set reach (MassBank overlaps train sources; reach on hidden test unknown) | new | priority 1; exactly one probe, no tuning follow-up |
| xattn-rerank | scorer | C2 rank 2–5 | A candidate-conditioned cross-attention scorer over (spectrum, structure) pairs is a different evidence type than FPNet f·z | new rerank stage over the fused 40-window | yes | cheap form MEASURED no signal (`research/exploit/20261013-xattn-rerank/spike_result.md`: learned model denied BASE order −0.0218 [−0.057, +0.014]); cross-attention form UNMEASURED | new | priority 2; reuse that folder, fix its R02 finding first; GPU training needs a model-search request + human approval |
| formula-window | candidate-recall | C2, narrowed to the 10 molecules (2.5%) whose truth formula is absent from every candidate | Adding candidates of a missing formula (formula posterior over `P.window(target, ppm_win)`) rescues truths no reranker can reach | formula window / `sf_*` block (`casmi/engine.py:66-80`) | partly | MEASURED reach (`.scratch/fusion_tail/formula_reach.py`): 10 C2 molecules, 0 on C1_npx → max ≈ +0.025 C2, realistic far less | variant | priority 3; low |

When no live idea is left, `/replan` sets `ideate` and the loop runs scout → diagnose → ideate.

## Bars
- Gate: the rank-only gate, `research/harness/gate_report.md` block `### 2026-10-13: rank-only gate + multi-cohort
  scorecard` (enforced by `research/bin/gate.py`). Dead only on: target Δ ≤ 0 or target CI upper bound < 0; a
  non-advisory cohort's CI upper bound < −0.004; rank-1 displacements > 1% of the target cohort; an attack died or a
  blocking attack is UNTESTED; a heavy-path idea without a measured ceiling > 0. Everything else is queued and ranked
  by the composite (equal C1/C2/C3 weights). `prior.md` never passes the gate.
- KEEP at submit (`poll_lb.decide`): LB ≥ **0.444** (E006 0.440 + 0.004) with the idea's counters firing with values > 0.
- G4 firing proof: the probe's draft `submission.csv` differs from the control draft
  (`research/exploit/20261013-control-draft/kaggle_output/submission.csv`) in ≥ 1 molecule.
- Session-grant submits additionally need the qoder_prompt §10 eligibility checks, incl. target-cohort "other" Δ > 0.

## Stop rules
max_cycles: 5
max_lb_probes: 4
dead_streak: 2
quota_pause_h: 12

## Out of scope
- `fusion-tail` (cap 25→40: ΔMRR −0.0001, ceiling +0.0000; the discarded primary tail holds no truths).
- Per-molecule cross-spectrum weighting (already done in `casmi/engine.py`; audit 2026-10-13).
- +1.45 ppm centring as a probe (a diagnostic only, revived only if > 2% of molecules change window).
- TOPN 60→100 and PubChem-depth widening; FPNet swaps or a second FPNet family; random bit-subspace RRF (L10, L11).
- Generated candidates at ranks 1–5 (−0.064); uncalibrated simulated-spectrum scores; pure Class-3 de novo without
  grafting onto E006 (L18).
- Fusion/λ retuning, popularity as a window selector (L12), promotion `REL_PROMOTE`/`SLOTS_*` tweaks, the Class-1
  smart library gate (row 55), absolute-score gates hijacking ranks 1–5, and every other EXPERIMENTS.md graveyard row.
- Edits to loop code, `CLAUDE.md`, README, skills, rules, config or `.claude/settings*`.
