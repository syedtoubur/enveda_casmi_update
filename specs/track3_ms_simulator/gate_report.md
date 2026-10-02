# Phase 0 — ICEBERG gate report

*Run 2026-09-26. Sources: `phase_0_colab_gate_results.json` (Colab, A100-40GB) and
`phase0_kaggle_g5.json` (Kaggle, T4, internet off).*

## Verdict (first run): **HOLD — the mechanics pass, but two checks need revising before GO**

*Superseded by the revision 2 re-run below: **PASS → GO to Phase 1**.*

| gate | result | status |
|---|---|---|
| G1 licence | `msg_all` + `msg_simulation`, MIT, SHA1 recorded; both have adduct, CE and instrument embeddings | PASS |
| G2 coverage | best variant `msg_all` / CE in **eV** / **QTOF**. 7 adducts pass the rule, 3 fail it (below) | PASS, but the rule is too lenient (see finding 2) |
| G3 CE | median sim(pred@10 eV, pred@60 eV) = 0.350, so CE conditions the output. `[M+H]+` MRR 0.971 with eV vs 0.927 with NCE | PASS |
| G4 throughput | A100: 58.5 mol/s (bs 128) → **3.4 GPU-h per condition** for 712k structures. CPU: 2.4 mol/s | measured |
| G5 offline | installs offline in 177 s, CUDA OK, SHA1 OK, 20/20 predicted, median sim to Colab 1.000 (min 0.729) | PASS |
| G6 sanity | 0.863 < 0.98 | **FAIL — rule mis-specified (finding 1)** |

`msg_all` beats `msg_simulation` on almost every adduct. This is expected: it trained on more of
MassSpecGym. Both likely saw many of these common natural products, so absolute fidelity is
optimistic.

### Best-variant table (`msg_all_ev_QTOF`)

| adduct | n | true | shuffled | gap | top1 | rule |
|---|---|---|---|---|---|---|
| [M+H]+ | 40 | 0.425 | 0.022 | 0.403 | 0.950 | covered |
| [M-H2O+H]+ | 40 | 0.261 | 0.000 | 0.261 | 0.725 | covered |
| [M-H]- | 40 | 0.187 | 0.004 | 0.183 | 0.750 | covered |
| [M-2H2O+H]+ | 40 | 0.132 | 0.000 | 0.132 | 0.550 | covered |
| [M+NH4]+ | 40 | 0.106 | 0.011 | 0.094 | 0.425 | covered |
| [M-H2O-H]- | 20 | 0.078 | 0.000 | 0.078 | 0.400 | covered |
| [M+Na]+ | 40 | 0.055 | 0.000 | 0.055 | 0.350 | covered (barely) |
| [M+K]+ | 40 | 0.028 | 0.000 | 0.028 | 0.325 | uncovered |
| [M+CH2O2-H]- | 40 | 0.027 | 0.000 | 0.027 | 0.250 | uncovered |
| [M+Cl]- | 40 | 0.021 | 0.000 | 0.021 | 0.250 | uncovered |

## Findings

**1. G6 checked raw output against a rule the pipeline never applies.** There are no worker
errors: all 380 molecules were predicted in every variant. The sanity failures therefore come from
peaks above precursor + 2 Da. The inten checkpoint has `include_unshifted_mz = True`, so ms-pred
emits every fragment twice: once with the adduct offset and once *without* it (neutral − e). For
water-loss adducts the unshifted root is M − e, which sits ~17 Da (`[M-H2O+H]+`), ~35 Da
(`[M-2H2O+H]+`) or ~19 Da (`[M-H2O-H]-`) **above** the precursor. Those three adducts are 100 of
the 380 rows, and 52 rows failed. This is consistent with the explanation but **not yet confirmed
per adduct**. The scoring path (`prep`) already drops peaks above precursor + 2 Da, the same
cleaning the organisers applied to the test spectra, so the G2 numbers are unaffected.

**2. G2's rule over-states coverage.** Decoys inside an adduct group have different masses, so a
prediction that merely lands in the right mass range ranks first often. That is why `[M+K]+` with a
median fidelity of 0.028 still reaches 32% top-1. Tier 2 ranks **isobaric** candidates inside a
±8.5 ppm window, and mass is no help there. Adducts with median true fidelity ≤ ~0.1 (`[M+NH4]+`,
`[M-H2O-H]-`, `[M+Na]+`) have not shown they can separate isomers.

**3. Colab vs Kaggle predictions are not bit-identical.** One of 20 molecules scored 0.729 against
its Colab reference. This is probably fragment-selection ties that flip across GPUs. It is harmless
for ranking, but precomputed and on-the-fly spectra of the same structure can differ slightly.

**4. Budget signal for Phase 1.** 7 adducts × 4 CEs × 3.4 h ≈ 95 A100-hours for a full-pool
precompute. The Kaggle smoke run's 15 mol/s came from one 20-molecule batch and is not a throughput
figure.

## Revision 2 re-run (2026-09-27, `new_gate_results.txt`) — **VERDICT: PASS → GO to Phase 1**

The best variant is unchanged: `msg_all` / eV / QTOF. The G2b truth fidelities reproduce the first
run exactly (for example `[M+H]+` 0.425). G1, G3, G4 and G5 are carried from the first run.

**G6 — PASS, and the explanation is confirmed.** Sanity after the cut is 1.000 (threshold 0.98),
with no empty spectra. Peaks were dropped *only* in the three water-loss adducts, at exactly the
unshifted-root offset:

| adduct | rows with drops | median dropped intensity | median offset | expected |
|---|---|---|---|---|
| [M-H2O+H]+ | 97.5 % | 0.3 % | +17.003 | +17.003 |
| [M-2H2O+H]+ | 100 % | 2.0 % | +35.013 | +35.013 |
| [M-H2O-H]- | 75 % | 1.1 % | +19.020 | +19.018 |

The dropped intensity is tiny (≤ 2 %), so the cut costs nothing.

**G2b — isobaric coverage** (21 candidates per query: the truth plus 20 decoys from its ±8.5 ppm
window)

| adduct | true fid. | top1 | chance | lift | p | MRR (chance) | covered |
|---|---|---|---|---|---|---|---|
| [M+H]+ | 0.425 | 0.900 | 0.049 | 18.6 | <0.001 | 0.943 (0.176) | **yes** |
| [M-H]- | 0.187 | 0.575 | 0.052 | 11.0 | <0.001 | 0.712 (0.184) | **yes** |
| [M+Na]+ | 0.055 | 0.350 | 0.048 | 7.3 | <0.001 | 0.499 (0.175) | **yes** |
| [M+Cl]- | 0.021 | 0.225 | 0.049 | 4.6 | <0.001 | 0.384 (0.176) | **yes** |
| [M+CH2O2-H]- | 0.027 | 0.225 | 0.050 | 4.5 | <0.001 | 0.405 (0.180) | **yes** |
| [M+NH4]+ | 0.106 | 0.200 | 0.050 | 4.0 | 0.001 | 0.371 (0.179) | **yes** |
| [M-H2O+H]+ | 0.257 | 0.200 | 0.086 | 2.3 | 0.014 | 0.429 (0.240) | no (p) |
| [M+K]+ | 0.028 | 0.150 | 0.053 | 2.8 | 0.017 | 0.279 (0.182) | no (p) |
| [M-2H2O+H]+ | 0.132 | 0.125 | 0.054 | 2.3 | 0.060 | 0.274 (0.187) | no (p) |
| [M-H2O-H]- | 0.091 | 0.211 | 0.086 | 2.5 | 0.064 | 0.384 (0.239) | no (n = 19, p) |

**Covered (tier 2 by simulation):** `[M+H]+`, `[M+NH4]+`, `[M+Na]+`, `[M-H]-`, `[M+CH2O2-H]-`,
`[M+Cl]-`. **Uncovered → MetFrag-lite:** `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+K]+`, `[M-H2O-H]-`.

### Findings from the re-run

5. **Fidelity is not the same as discrimination.** G2's rule flipped in both directions:
   - The water-loss adducts have high fidelity (0.26 / 0.13) but separate isomers weakly. The
     fragments they match (water-loss ladders) are shared by the decoys.
   - `[M+Cl]-`, `[M+CH2O2-H]-` and `[M+Na]+` have near-zero median fidelity, yet the truth wins
     22–35 % of the time. It usually wins because a single correct fragment beats decoys that score
     exactly 0.

   That second signal is real (p < 0.001) but fragile, because it rests on single peaks.
6. **Absolute numbers are optimistic.** The truths are train structures, and many are likely in
   MassSpecGym, on which `msg_all` was trained. Decoys are mostly COCONUT structures the model never
   saw. The real Class-2 targets resemble the decoys, not the truths, so `[M+H]+` top-1 0.90 is an
   upper bound. Phase 3 should split fidelity by MassSpecGym membership.
7. **Real windows are larger than the test's.** The median full ±8.5 ppm window is 20–115
   candidates (`[M+H]+` 99, `[M-H]-` 91, `[M+Na]+` 115.5), not 21. Rank quality at window size ~100
   will be lower than this table.
8. **Pool: 711,690 unique ik14.** Only 509 rows were duplicates or unparseable, so the COCONUT keys
   are real ik14s and COCONUT∩train is tiny. This answers roadmap Phase 1 action 3.
9. **Throughput:** 67.6 mol/s on an A100 at batch 128, over 7,551 real window candidates. That gives
   711,690 / 67.6 ≈ **2.9 A100-h per (adduct, CE) condition**. 6 covered adducts × 4 CEs ≈ 70 A100-h.
   Kaggle T4 throughput is still unmeasured.

## Required before GO (first run) — both done in revision 2

1. **G6, per-adduct:** report the share of rows and of intensity above precursor + 2 Da per adduct.
   Redefine sanity as *after* the precursor + 2 Da cut, and make the worker drop those peaks at
   source.
2. **G2b, isomer test:** for each sampled structure, predict up to 10 **same-formula** decoys from
   the pool and rank the truth. An adduct is covered only if its isomer top-1 clearly beats chance.
   This measures the tier-2 task itself.

## Phase 4 — first LB submission (2026-09-27)

| notebook | channels | LB |
|---|---|---|
| `casmi_simple_sim_submission` v2 | library tier 1 (τ 0.75) → ICEBERG `sv` (6 covered adducts) → MetFrag-lite → ppm | **0.289** |
| `casmi_simple_sim_nosim` (control) | the same, `CFG.USE_SIM = False`: library tier 1 → MetFrag-lite → ppm | **0.254** |
| reference: phase1-enhanced / own FPNet+ranker / tier1-patched | library + analog + MetFrag + FPNet → GBDT | 0.319 / 0.323 / 0.337 |

0.289 alone does not measure ICEBERG: against the references it both *adds* ICEBERG and *removes*
FPNet, the analog channel and the GBDT. The control differs in one line, so
**ICEBERG's contribution = 0.289 − LB(control)**. Both runs score the same molecules and differ only
in tier-2 order, so the paired difference is much less noisy than either absolute score.

Read-out rule (set before the control is scored):
- control ≤ 0.279 (ICEBERG ≥ +0.010): the simulator carries real signal → plan `sv` as a new
  ranker feature (append-only column 31) in the GBDT pipeline.
- control within ±0.005 of 0.289: no usable signal in this form.
- control ≥ 0.294: ICEBERG ordering is worse than MetFrag-lite ordering → ICEBERG is hurting.

**Result: control 0.254 → ICEBERG contributes +0.035 LB** (threshold for "real signal" was
+0.010). The simulator works as a tier-2 scorer.
