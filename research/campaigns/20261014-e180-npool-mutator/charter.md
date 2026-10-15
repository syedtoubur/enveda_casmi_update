# Campaign charter: 20261014-e180-npool-mutator

Grill 2026-10-14 (interactive, Opus). The human's seeds were:
- carry the scratch line's candidate pool (`casmi_scratch_r3_full1_r2gl`, LB 0.346: new NP structures, with the
  enveda-180 structures excluded) into E006's Engine 2;
- build a separate mutator channel from `specs/track4_priors_probes`.

**Supervised run.** The human steps `campaign.py --once` and decides at three checkpoints: after /replan, at
harness-done, and before /build ship.

What the grill research established (it reshaped the seeds):
- **The scratch pool adds little to E006.** E006's Engine 2 already pools COCONUT + BIO + train
  (`.scratch/nb2_e006.py:232-235`), and E006 also has a 105.9M-structure PubChem channel. Beyond those, the scratch
  pool adds only about 2.6k NP-table (LOTUS/NPAtlas) rows (`specs/specs_scratch/r1_pool_plan.md:79-88`). The scratch
  +0.035 came from the full1 FPNet, which E006 already has.
- **The R1b lever was the drop of 180,649 enveda-180-only structures.** These are synthetic drug-like decoys, and the
  hidden test is about 400 NPs. E006 drops none of them.
- **The harness cannot see pool misses (L07, L44).** A pool expansion is heavy path, and its reach is measured
  outside val. Row 32 (the same expansion on the 0.415 code) came out VOID.
- **Every mutator LB was at or below noise** (rows 22, 23, 25, 26, 34: 0.000, −0.002, −0.064, 0.000, 0.393). c3cal
  failed CC.3 and CC.4. There is no pseudo_c3 dump, so the C3 column of the scorecard is empty.

## Scope
- Line: exploit
- Base and paired control: **E006 v1** = `syedtoubur/casmi26-e006-window-40-ice-fallback-true` v1, LB 0.440
  (`research/exploit/20261012-control-e006`); control draft `research/exploit/20261013-control-draft`.
  - Worktree: `E:/casmi/lines/exploit` (branch `line/exploit`). Cite base lines from `scratch/casmi26-e006.py`.
  - Post-scoring knobs go in `build/casmi_post.py`.
- Harness: `--a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40"` over every dumped cohort (C2, C1_npx,
  C2_hard, C1_xlib; C2_np advisory; pseudo_c3 once this campaign dumps it). Judge by the target cohort's leak-free
  "other" Δ (L40).
- Supersedes `20261014-qoder-structural` (stopped 2026-10-14). lib-isomer-margin there is submitted; bind its LB with
  `poll_lb.py --claim`.
- Order is the portfolio's priority column.
  - **Pool branches:** e180-drop first, then the engine2-npx reach spike. Build only if the spike shows any non-zero
    reach.
  - **Mutator chain:** pseudo-c3-dump → c3-detector → mutator-c3v2. Each is gated on the previous one.
  - **Mutator placement:** slots [2,3,4] on routed molecules if the detector passes its precision gate; otherwise the
    mutator falls back to ranks 21–25 only.
- New Kaggle datasets (the e180 key list; Engine 2 fingerprint rows for npx) and any GPU or Colab job are approved
  by the human at a checkpoint.
- The loop code (`research/bin`, `research/harness`, `research/rules`, skills) is a constant. Changes to it go to
  `questions.md` as proposals.

## Portfolio
| slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV |
|---|---|---|---|---|---|---|---|---|
| e180-drop | decoy-prune | C2 / C2_hard (31.5% of C2 molecules carry e180 decoys in the E006 top 25) | Synthetic drug-like enveda-180-only structures are decoys for an NP test. Dropping them from the fused list before the 25 cut lifts truths at ranks 2–25 and lets rank 26+ candidates in | `E180_DROP` (default off) in `build/casmi_post.py`. Key list = R1b `e180_only` (`scratch_pkg/casmi_scratch/pool.py:91`) over `syedtoubur/casmi26-scratch-pool` flags, shipped as a small Kaggle dataset | no | MEASURED demote-form ceiling (`.scratch/e180/ceiling.json`): C2 +0.0038 [+0.0007, +0.0078], C2_hard +0.0060, C1 ≤ +0.0006, 0 molecules worse, no truth e180-only. Scratch r1a_gl → r1b_gl +0.002 (rows 43–44). ASSUMED: the hidden-test decoy share is like val's | variant (port of scratch R1b) | priority 1; one probe |
| engine2-npx | candidate-recall | hidden-test NP truths missing from Engine 2's pool | NP-table structures absent from COCONUT + BIO + train enter Engine 2 and get scored by its two-ranker blend instead of reaching the list only through PubChem's gated merge slots | Engine 2 pool append (`casmi_engine.build_pool`, `.scratch/nb2_e006.py:232-235`). Needs prvsiyan fingerprint and feature rows for the new structures (a new dataset). Spike first on cohort P | yes | ASSUMED. Row 32 (the same idea on the 0.415 code) VOID. Cohort P: only 21.3% in the tier (EXPERIMENTS B4) | variant | priority 2; spike, then build iff reach > 0 |
| pseudo-c3-dump | instrument | — | Port the `patch_c3cal` pool-ban into VAL_CELL and dump `research/harness/dumps/pseudo_c3/post_dump/state.pkl` (600 NPs, truth banned from retrieval), so the scorecard gains its C3 column | `harness.cohorts.pseudo_c3` (already in config) | — | prerequisite; Kaggle CPU/GPU run approved at a checkpoint | infra | priority 3 |
| c3-detector | class3-route | pseudo_c3 vs C2 / C2_hard / C1 | Relative, within-molecule signals (lib_max, best analog similarity, the fused list's own margin) separate novel molecules from retrievable ones well enough to route them | `C3_ROUTE` (default off) in casmi_post | no | MEASURED c3cal (stage-1 form, absolute f·z): CC.3 3.5% fire on C2 FAIL, CC.4 2.5% FAIL (`specs/track4_priors_probes/gate_report.md:77-84`). The ice-margin-route result: the fused margin is not a class signal by itself | variant | priority 4 |
| mutator-c3v2 | class3-route | pseudo_c3 | c3v2 analog-edit mutator (`specs/track4_priors_probes/c3v2_plan.md`): Engine 2 analogs, ≤ 2 palette moves within ±3 mDa, FPNet f·z scored, tagged `'\|gen'` (L17). Placed at [2,3,4] on routed molecules if the detector passes, otherwise ranks 21–25 | `MUT_SLOTS`. Reuse `build/delta_m_mutator.py`, `build/mut_runner.py`, `build/eval_mutator_cohort.py`. The spec prefers offline generation on val with injection into the replay, so it is scoreable | partly | Every past mutator LB at or below noise (rows 22–26, 34). Ceiling ≈ +0.010 at a 25% C3 test share (`EXPERIMENTS.md:156`) | variant | priority 5 |

When no live idea is left, `/replan` sets `ideate` and the loop runs scout → diagnose → ideate.

## Bars
- Gate: the rank-only gate, `research/harness/gate_report.md` block `### 2026-10-13: rank-only gate + multi-cohort
  scorecard` (enforced by `research/bin/gate.py`). Per-idea bars are in the block `### 2026-10-14: e180-npool-mutator
  campaign, PRE-REGISTERED`, written before any measurement:
  - the e180-drop counters;
  - the engine2-npx spike rule (reach > 0);
  - the pseudo-c3-dump instrument check;
  - the c3-detector precision gate (≥ 30% of pseudo_c3; ≤ 2% of C2 ∪ C2_hard truths at ranks 1–5; ≤ 1% of C1);
  - the mutator vetoes (C2 or C2_hard CI upper bound < −0.004; any rank-1 displacement).
- Heavy-path ideas (engine2-npx): dead unless a measured ceiling > 0. `prior.md` never passes a gate (L39).
- KEEP at submit (`poll_lb.decide`): LB ≥ **0.444** (E006 0.440 + 0.004), the idea's counters firing with values > 0,
  and the draft submission.csv differing from the control draft (L38).

## Stop rules
max_cycles: 5
max_lb_probes: 4
dead_streak: 2
quota_pause_h: 12

## Out of scope
- A full scratch-pool swap into either engine; Engine 1 pool changes; NP-table rows into Engine 1.
- R2 NPClassifier class vote and R4 second-FPNet family fusion (noise once an FPNet channel is present, rows 47–50).
- FPNet or ranker retraining and any FPNet swap without its matching rows (L01, L02).
- TOPN 60→100, PubChem-depth widening, fusion-window or tail-cap changes (`fusion-tail` dead).
- Popularity voice changes, popularity as a window selector, NP-membership bonuses (L12; rows 29–33).
- Absolute-score gates (L14); generated candidates at rank 1; uncalibrated simulated-spectrum scores.
- Pure Class-3 de novo (DreaMS SELFIES decoder) without grafting onto E006 (L18).
- Re-proposing anything already closed: refspectra-lib, formula-window, ice-margin-route(-gl), lib-isomer-margin
  variants, promotion `REL_PROMOTE` / `SLOTS_*` tweaks, the smart library gate (row 55), λ retuning (L08).
- Edits to loop code, `CLAUDE.md`, README, skills, rules, config or `.claude/settings*`.
- Any `kaggle competitions submit`: the human submits.
