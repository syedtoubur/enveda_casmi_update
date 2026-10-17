---
name: spec
description: Campaign step (tier H, headless): turn a CASMI hypothesis into a measurable spec with a default-off switch and a scorecard experiment. Invoked as `/spec research/<line>/<idea> --brief <path>`.
---

# Spec

Read the brief first; it names the line, its worktree, the base export, the post module and the scorecard cohorts. If `preflight.md` exists, this is the one fix round: resolve every finding in it.

1. **Ground the premise**: every claim about what the base does cites the base export (`<base_export>:<line>`), the post module (`build/casmi_post.py:<line>`) or `BASE.md §n`, and you have opened and checked the citation. A claim about data (array shapes, columns, file contents, counts) cites a spike output you ran, not a card or a memory. A claim you cannot ground is cut or marked ASSUMED.
2. **Shape one change** behind a switch whose default reproduces the base: a `casmi_post.CFG` knob (exploit), or a module plus a graft patch onto the E006 notebook (explore, the line's `code_form`). If the change sits upstream of the dump (ranker, features, pool, simulator inputs, a new model), mark it `heavy-path: yes` and name the script that will measure its **ceiling** on real data.
3. **Write `spec.md`** from `research/_template/spec.md`. The CASMI section holds:
   - the switch name and its default;
   - the target cohort (from the brief's list) and the reach in its loss-map bucket;
   - where each measurement runs: anything that loads dumps, scores candidates or trains is a Kaggle kernel (**Heavy compute runs on Kaggle** (CLAUDE.md §3, human rule 2026-10-17)); name the kernel and its inputs;
   - the exact scorecard command from the brief (`--a` = the line's `harness_base_cfg`, `--b` = that plus the knob, `--target <cohort>`), or for a heavy-path idea the ceiling measurement plus the G1 row and G4 counters that prove the change fired;
   - value-checked counters for the shipped notebook, and `g4_min_changed`: how many draft molecules the change should alter (G4 diffs the probe's draft against the control's draft);
   - the ceiling, with every factor labelled MEASURED (with its source) or ASSUMED.
   The gate is rank-only (`research/bin/gate.py`): don't invent an LB-slot bar, and don't write a prior.md to pass it.
4. **Training**: if the idea needs a trained model, first record in `spec.md` whether an existing open-licence, competition-compliant model does the job (from the model-search notes, or write a `questions.md` request for one). A Kaggle training run is at most 4 h and comes out of the weekly GPU pool. A Colab or bigger run is a `training_request.md` for the human (what, why, evidence, cost); the idea waits on it.
5. Run `python research/bin/preflight.py <idea>` and fix until it is clean, or write in `questions.md` why a finding is wrong.

Done when preflight is clean (or disputed in writing) and STATUS is `specced`.
