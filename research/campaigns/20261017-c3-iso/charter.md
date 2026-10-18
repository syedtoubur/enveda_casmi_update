# Campaign charter: 20261017-c3-iso

Grilled 2026-10-17 (Opus, plan `~/.claude/plans/scratch-handoffs-handoff-20261017-c3-ta-sleepy-kernighan.md`; handoff
`.scratch/handoffs/handoff-20261017-c3-target.md`). Goal: lift Class 3 on top of the V22 base, which targets only C1/C2.
The previous campaign (c3-gen-slot2) proved the generators reach Class 3 (mutator top-1 h 0.12 on pseudo_c3) but a
separate C3 detector died (K2 v0). This campaign uses **relative, within-molecule triggers only** (no detector).

Human decisions: a new `c3` line on V22; seed portfolio = c3-slot-curve, gen-rank1-keep (ICE-agreement guard), and the
human's idea isomer-mut-sim; isomer-reach kill bar 10%; supervised and CPU-first; the three CPU parts run in parallel
(wave 1) with ONE checkpoint when all three finish or one dies; every GPU job needs a costed plan and the human's go.

## Scope
- Line: c3
- Base: V22 = `casmi26-v22-rel150-e2r2-clean` (LB 0.447–0.451); export `scratch/baseline/casmi26_baseline.py` in the worktree.
- Worktree `E:/casmi/lines/c3`, branch `line/c3` = `idea/20261015-c3-gen-slot2` (faeedf0: novel_p4*, c3_detector,
  delta_m_mutator, patch_harness_dump) + V22 parity merged from `antigravity-exploit` 38a42a4 (commit dab59b0). Idea
  branches `idea/20261017-<slug>` off `line/c3`, one worktree each.
- Harness: `harness.py scorecard --line c3 --a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40,REL_PROMOTE=150.0,FRAG_GAP=true,E2R2=true"`,
  target pseudo_c3; C1_npx, C1_xlib, C2, C2_hard judged on the leak-free "other" Δ (L40); C2_np advisory.
- Paired control: an own-account V22 replicate pushed in the same window as the first LB probe (L05/L06); keep_bar = control + 0.004.
- Assets (measured): pseudo_c3 dump (K0), K1 generator output `E:/casmi/c3_gen/p4_raw.pkl`, `p4_inputs.pkl`,
  `novel.pkl`, the K2 feature table `k2_table.pkl`.
- Wave 1 (parallel, CPU): S1 c3-slot-curve, R1 gen-rank1-keep, I1 isomer-mut-sim reach. Opus writes spec.md + attack.md
  per idea; Sonnet builders run in parallel; gate.py recomputes each verdict; one human checkpoint after the wave.
  Then `campaign.py --once` steps ship / I2 (GPU, after a go) / replan.
- Resources (human rule 2026-10-17, CLAUDE.md §3): **every heavy job runs as a Kaggle CPU kernel, never locally**:
  harness scorecard/suite (kit dataset `syedtoubur/casmi26-harness-kit`, kernel `syedtoubur/casmi26-harness-run`;
  rebuild the kit tar when line code changes), candidate generation/scoring, ICEBERG. Locally only edits, unit tests on
  small fixtures, smoke runs < 2 min, and reading fetched outputs. Colab balance ≈ 157 CU; no GPU without a costed
  plan and a human go.
- Isolation: never write in `E:/casmi/lines/exploit`, `E:/casmi/lines/antigravity-exploit` or their idea folders
  (read-only copies are fine); stage only this campaign's files (EXPERIMENTS.md and `.scratch/handoffs/latest.md`
  carry other agents' edits).
- Open portfolio: the human (or /replan) may add Class-3 ideas mid-campaign without a new grill; each new idea needs a
  dated PRE-REGISTERED block in gate_report.md before its first measurement.
- Known: `preflight.py --selftest` FAILs R04/R08 on fixture-list drift from other agents' idea folders; not blocking
  (campaign.py uses per-idea `preflight.run`); not fixed here. `test_e2r2_row_pure_invariants` fails on 38a42a4 itself (inherited).

## Portfolio
| slug | branch | target cohort | claim | switch | heavy-path | prior | novelty | EV |
|---|---|---|---|---|---|---|---|---|
| c3-slot-curve | c3-route | pseudo_c3 + C1/C2 other | some fixed slot s pays for the mut-first top-1 without any detector | instrument (`build/slot_curve.py`) | no | ASSUMED: slot 2 negative (0.25·0.125/2 < 0.75·0.06); mid slots ≈ +0.002 | instrument | priority 1 (cheap; sets s*) |
| gen-rank1-keep | c3-gen1 | pseudo_c3 | V22 post-scoring demotes correct generated BASE #1 (44/55 under E006); keeping it when ICEBERG agrees recovers them at ≈ 1–2% C1/C2 fire | `GEN1_KEEP` | no | MEASURED fire (generated BASE #1) 2/245 C1_npx, 1/300 C1_xlib, 6/400 C2, 4/200 C2_hard (K0); ICE-guarded precision UNMEASURED | new | +0.003–0.010 LB at π 0.25 |
| isomer-mut-sim | c3-iso | pseudo_c3 | C3 truths are isomers of fused candidates: isomer edits of the fused top 10, CPU frag pre-screen, then ICEBERG on retrieved + mutants; a mutant that beats every retrieved candidate goes to rank 2, others to 21–25 | `ISO_MUT` | partly (I2 GPU) | ASSUMED; reach UNMEASURED; reference: K1 analog-anchored mutator h 0.12 | new (mass-preserving operators on the fused list; human's idea) | +0.005–0.015 at π 0.25 if reach_20 ≥ 10% |

## Bars
- Gate: the rank-only gate (`research/harness/gate_report.md`, block `### 2026-10-13: rank-only gate + multi-cohort
  scorecard`), plus the block `### 2026-10-17: c3-iso campaign, PRE-REGISTERED` (P0 parity, S1, R1, I1 reach ≥ 10%, I2, I3, ship).
- KEEP at submit (`poll_lb.decide`): probe ≥ paired own-account V22 replicate + 0.004, value-checked counters
  (`gen1_kept`, `iso_fired`), draft differs from the control draft only where the knob fired (L38); replicate before it becomes the base.

## Stop rules
max_cycles: 5
max_lb_probes: 4
dead_streak: 2
quota_pause_h: 12

## Out of scope
- A separate Class-3 detector (K2 v0 dead); absolute-score gates (L14); λ retuning (L08).
- Generated candidates at rank 1, except R1's ICE-guarded keep of Engine 1's own generated BASE #1.
- FPNet-based scoring or triggers for novel candidates (pseudo_c3 is FPNet-leaky).
- Knapsack / MetFrag-lite scorers that read anything but the query's measured spectrum (R10); c3v2 depth 3 (K-D3).
- Tail-only (21–25) as the main lever (ceiling ≈ +0.001); it may ride along.
- Retraining FPNet or the rankers; any FPNet swap without its rows (L01, L02).
- The 5 pending proposals in `research/proposals/` (deferred to a separate review).
- Edits to CLAUDE.md, README, skills, `.claude/settings*`, and the loop code (`research/bin`, `research/rules`), except
  the c3 line entry in `research/config.yaml` (done in the grill).
- Any `kaggle competitions submit`: the human submits.

## Human steer (2026-10-18)
New direction for /replan. Propose idea 20261018-iso-plausibility-prior (c3 line): add TWO orthogonal, spectrum-blind chemistry features to the EXISTING ranker, applied ONLY to low-spectral-confidence candidates (generated isomer-mutants + retrieved-without-spectra / Class-2-3); NEVER to Class-1 tau>=0.75 library matches. (A) Feasibility/strain: RDKit MMFF/UFF single-point minus relaxed energy -> strain z-score as one ranker FEATURE (not a hard gate; caveat: formation energy != stability, LBL 2020). (B) NP-membership/likeliness: binary in-COCONUT/LOTUS + continuous Ertl NP-likeness, calibrated on pseudo_c3. PRE-REGISTER a reach rule BEFORE measuring: reach = share of pseudo_c3 molecules whose candidate rank can change; if reach < ~5% KILL at the desk (Track 4b died from ~0.001 reach, rows 36/38). Then rank-only gate on pseudo_c3 "other" delta; KEEP only if delta_other > keep_bar with iso_fired / prior-fired counters > 0. Evidence: strain-as-MS-ranker is an unoccupied literature gap (arXiv 2308.07413; bioRxiv 2021 docking strain "elusive"); NP-citation popularity prior already paid +0.006/+0.010 (rows 24/27); COCONUT already in the pool (row 41); peers on THIS Kaggle use a "candidate filter/prior before final ranking" (discussion 741745). Do NOT re-run ICEBERG GPU for the killed isomer-mut-ice.
