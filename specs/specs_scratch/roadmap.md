# Roadmap

| Step | Content | Status |
|---|---|---|
| S0 | Spec folder, `casmi_scratch` package, tests (`scratch_pkg/tests/run.py`, incl. a notebook dry run), generator | done 10-05 |
| S1a | **R0** Kaggle twin `scratch_pkg/notebooks/casmi_scratch_r0.ipynb` (+ `-nosim` control) → LB | LB 0.290 (10-07): R0.4 FAIL by 0.002, compare.py diff vs the base pending |
| S1b | Validation harness (slices I + C, ban arms) on Colab, R0 baseline readings | design discussion pending |
| S2 | **R1** unified pool (`r1_pool_plan.md`): build on Colab → `casmi26-scratch-pool`; coverage audit + class-vote diagnostic (gates R1.1–R1.3, CV); LB twins R1a (union) / R1b (− enveda-180-only) + paired R0 | v1 built 10-06 (778,095 structures, R1.1 PASS); Kaggle upload + audit next. GLACIER twins R1a/R1b-GL (`sim_model="glacier"`, gate R1-GL) built 10-06. **LB 10-07**: r1b 0.301, r1a_gl 0.301, r1b_gl 0.303 → R1.LB PASS, R1-GL neutral (not kept); r1a pending |
| S3 | **R2** class vote in tier-2 ranking (grilling 10-07; gates R2.0–R2.LB in `gate_report.md`): NPClassifier labels for all 778k structures → pool v2; vote = delta-list analog hits' top-1 superclass; tier-2 key `sv + 0.10·conf·P(voted)`; twins R2-ICE / R2-GL on r1b | code done 10-07; next: Colab `casmi_scratch_r2_npc_colab` (R2.0 spike + R2.1) → upload pool v2 → `casmi_scratch_r2_audit_colab` (R2.2, R2.3 kill gate) → LB |
| S4 | Deferred from R2: NP_Score prior, class-level (hierarchical) match, GLACIER-vs-ICEBERG choice | todo |
| S5 | **R3** FPNet channel (`r3_fpnet_plan.md`, grilling 10-08): own `casmi-fpnet` ×4 and Ahmed `fpnet_full1`; tier-2 RRF `1/(3+r_sv)+1/(3+r_fp)`; probes on r1b_glacier and r2_gl; C2+C2_hard audit in parallel | **LB 10-08: PASS ×4.** own_gl 0.322, own_r2gl 0.320, full1_gl 0.341, **full1_r2gl 0.346** (best scratch). Colab audit pending |
| S6 | **R4** FPNet family fusion (`r4_famfuse_plan.md`, grilling 10-08): full1 (A) × prvsiyan v2 (P), tier-2 key `1/(3+r_sv)+1/(3+r_A)+0.6/(3+r_P)`, stage C ordered by FP; same-day pair vs `r3_full1_r2gl` | **LB 10-08: 0.345 vs 0.346 → DROP (neutral).** R4.run PASS. The best scratch run stays `r3_full1_r2gl` |
| S7 | **R5** spectral-analogue channel (`pssm.md` v2, grilling 10-08): open ±200 Da analogues → PSSM log-odds (16,520 raw bits) or Tanimoto kernel (Morgan r2); 3rd tier-2 RRF list, w = 1.0; train structures get the neutral term; 4 probes `r5_{pssm,tani}[_novote]`, no control, KEEP ≥ 0.354; slice I audit alongside | code + notebooks done 10-08 (gates in `gate_report.md`; handoff `r5_pssm_tani.md`). R5.0 fast commits PASS for pssm and tani; next: novote fast commits → LB ×4; Colab `casmi_scratch_r5_audit_colab` in parallel |

## R0 — port with fixes
Behaviour of `build/simple_sim_submit.py` (LB 0.289) with `MAX_PEAKS=100` (LB 0.296), except:
1. `MAX_PEAKS` drives library similarity, MetFrag-lite **and** the ICEBERG similarity `sv` (was hard-coded 64).
2. No global state; all settings in `casmi_scratch.config.Config`.
3. The pool skips the fingerprint arrays nobody used (mass from RDKit `ExactMolWt` for train structures, as before).
4. Fast commit runs 20 molecules deep (was 30).
5. Stage-C top-up deduplicates on ik14 like stages A/B.

## Harness — open design questions
Settled: two slices (I: 245 enveda-np-examples timsTOF molecules; C: ~400 NP-library molecules, CE known, ≥ 2 spectra,
adduct-matched, MSG-unseen preferred), per-query arms C1/C2/C3 by banning, 40/35/25 weighting at report time,
pool recall + truth-in-simulated + MRR per arm + lever reach + MSG-seen split, ~400 molecules ≤ 1.5 h T4, cache by
(pool, simulator) version. Still to decide with the user before building: slice C's library mix, how a slice-C query's
non-timsTOF spectra are corrected (ppm offset), and the pass/fail rule per arm.
