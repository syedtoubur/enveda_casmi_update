tier: H
# Diagnosis: the Class-1 library gate (LIB_TAU 0.9, smart gate row 55), 2026-10-14

Human seed for campaign `20261014-qoder-structural`, ideation round 1. Pre-registered in
`research/harness/gate_report.md` (block "2026-10-14: libgate diagnosis"). Instrument: `.scratch/libgate_diag/diag.py`
(offline replay of the five battery dumps, E006 cfg; output `diag.json`, `per_mol_<cohort>.csv`). 0 LB slots, 0 GPU.

## The question
The cross-instrument libmax eval (`.scratch/kaggle_libmax_eval`, output v1–v3) found that same-structure timsTOF vs
Orbitrap/QTOF similarity rarely reaches 0.9: at 0.9, recall is ~2% on single pairs and 1.75% on window-max entropy.
So `LIB_TAU = 0.9` looked wrong for Class 1. The smart gate built on that eval (pin/lock on a masked, filtered entropy
score at tiers down to 0.40) lost −0.011 on the LB. Where does Class 1 actually lose MRR, and why did the gate lose?

## Measured (MEASURED unless marked)

### 1. The test-domain pair: C1_npx vs C2_np
C1_npx and C2_np are the **same 245 Enveda timsTOF queries**. In C1_npx the truth's other library spectra remain;
230 of the 245 have **only non-timsTOF** references left (gnps/massbank/mona/…; `refs = other_only`). In C2_np every
truth spectrum is banned. This pair is therefore the hidden test's Class-1 vs Class-2 situation on the engine's own
score, `lib_max` (`state['base'][m][2]`).

| τ on `lib_max` | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| C1_npx (truth in library, cross-instrument): share ≥ τ | 94.3% | 91.8% | 82.0% | 64.9% | 43.7% | **17.1%** | 0.8% |
| C2_np (truth banned, so a false lock): share ≥ τ | 53.1% | 41.2% | 24.5% | 12.2% | 5.7% | **2.9%** | 0.0% |
| C2 (gnps/riken/… queries): false lock | 34.8% | 23.0% | 16.8% | 11.0% | 8.5% | 6.0% | 1.8% |
| C2_hard: false lock | 21.0% | 15.0% | 9.5% | 5.5% | 4.5% | 3.5% | 2.0% |

- **Your observation holds on the engine scale.** 83% of true cross-instrument Class-1 matches score below 0.9.
- **The false-lock side is far worse than the eval said.** On the same timsTOF queries with the truth removed, 24.5%
  of molecules still reach 0.7, and 53% reach 0.5. The library is full of near-isomers of NP truths.
  - The eval's Phase-3 negatives were mostly enveda-180 synthetic windows, with a smart-filter FLR of 0.58% at 0.40.
    They do not represent this.
  - No grid τ gets the C2 ∪ C2_hard false-lock rate ≤ 1% (it is 5.2% at 0.9), so the pre-registered τ* fell back to
    0.9.

### 2. What the threshold actually controls, and where Class 1 loses
`LIB_TAU` only gates the PubChem merge (`build/casmi_post.py:225`). BASE already puts the library match first whatever
the threshold. On C1_npx, BASE rank 1 = truth for 89.6% of molecules, and E006 scores **0.9355**.

Loss decomposition of E006 on C1 (molecules not at final rank 1):

| cohort | BASE wrong | BASE right, demoted by the base pipeline (C8 rerank / PubChem merge / promotion) | demoted by E006's ICE fallback |
|---|---|---|---|
| C1_npx | 18 mol, **−0.052** | 6 mol, −0.013 | 0 |
| C1_xlib | 28 mol, **−0.065** | 5 mol, −0.009 | 3 mol, −0.006 |

- **The PubChem-merge threshold reaches at most the middle column** (≈ 0.01). That matches the old back-test: 0.9 → 0.7
  gives C1_npx +0.004, C2 flat (`capability_map.md:51`).
- **Most of the C1 loss is BASE ranking an isomer above the truth.** Of the C1_npx molecules where BASE is wrong
  (26 including those E006 later fixes), **17 have the truth at BASE rank 2** and 7 at rank 3–4; median `lib_max` is
  0.78. This is a *comparative* problem: which of two library-backed isomers matches better. It is not a threshold
  problem.

### 3. Protect library hits from the overlay (pre-registered counterfactual)
If `lib_max ≥ 0.9`, take E006 with ICE scores removed; else E006 (`protect` → `tau0.9_noice` in `diag.json`):

| cohort | fired | Δ MRR [90% CI] | up/down | rank-1 displaced |
|---|---|---|---|---|
| C1_npx | 42 | **+0.0054** [+0.001, +0.011] | 4/0 | 0 |
| C1_xlib | 242 | +0.0033 [−0.002, +0.009] | 3/2 | 1 |
| C2 | 24 | −0.0005 [−0.0035, +0.002] | 5/3 | 1 |
| C2_hard | 7 | +0.0010 [−0.0001, +0.003] | 2/1 | 0 |
| C2_np (adv.) | 7 | −0.0007 [−0.003, +0.001] | 2/2 | 0 |

- It **passes the pre-registered lead bars** (C1_npx CI upper bound > 0; C2 and C2_hard CI upper bound ≥ −0.004;
  displacements ≤ 1%).
- It is **small**: about +0.005 on Class 1, so about +0.001–0.002 on an LB with ~20–30% Class 1. That is ASSUMED
  share and below the ±0.004 noise floor.
- The overlay-off variant (`tau0.9_ctl`) is ~0.
- ICE's sign flips by class. With ICE removed everywhere: C1_npx +0.011, C1_xlib +0.013, but C2 −0.069, C2_hard
  −0.054, C2_np −0.093. ICE must stay for Class 2.

## Smart-gate post-mortem (HYPOTHESIS: the hidden-rerun counters are not recoverable; the saved output is the visible placeholder run)
- **Upside was capped.** On the test-like C1_npx, E006 is already 0.9355. The reachable library-protection headroom
  is the "demoted" columns, ≈ 0.013. Pinning cannot fix the dominant "BASE wrong" bucket unless the pin picks the
  truth over its isomer, and the gate's absolute score has no isomer margin calibrated on NP families.
- **Downside was large.** The gate locked (pin at rank 1, PubChem kept out) at S ≥ 0.65 and kept T3 down to 0.40,
  calibrated on synthetic decoys.
  - On the engine scale, 25–53% of timsTOF Class-2 NP queries have a library neighbour at 0.5–0.7. For those, BASE
    rank 1 is still the truth 88% of the time (C2_np, `lib_max ≥ 0.7`).
  - So a false lock displaces a correct rank 1 (the 1/1 → 1/2 loss), and on Class 2 there are many more such
    molecules than Class-1 rescues.
  - The masked, filtered S is stricter than `lib_max`, but there is no NP-isomer decoy set behind its FLR.
- Falsifier: replay `smart_gate.evaluate_molecule` on C2_np/C2 (it needs the train spectral library, so a Kaggle CPU
  instrument kernel). If its lock rate on C2_np at the T1–T3 bars is < 2%, this hypothesis is wrong.

## Hypotheses for /ideate (falsifiable)
1. **H1 `lib-isomer-margin` (target: the C1 "BASE wrong, truth at rank 2" bucket, C1_npx 17 molecules, ≈ −0.04).**
   - Claim: when BASE's top-2/3 are both library-backed isomers, reorder them by a *comparative* per-candidate library
     score, using each candidate's own best reference spectrum and masked/high-m/z-filtered entropy, not by
     `lib_max`.
   - This is not an absolute lock, so the C2_np false-lock rate does not apply. On C2 the truth has no reference
     spectra, so the rule fires only when ≥ 2 candidates have references.
   - Needs per-candidate library similarities, which the dumps lack: `base[m][3]` holds fused ranker scores
     (e.g. 4.74, 3.13, 0.87), not library similarities. So it is heavy-path: a new dump field, or a Kaggle CPU
     instrument over the train library.
   - Falsifier: on C1_npx, the truth's own reference similarity exceeds the BASE-rank-1 isomer's in < 50% of the 17
     rank-2 cases.
2. **H2 `ice-libprotect` (target: C1 molecules demoted by ICE).**
   - Claim: skip the ICE fallback rerank when `lib_max ≥ 0.9`. This is a new default-off `casmi_post` knob, fully
     replayable.
   - Measured above: C1_npx +0.0054, C2 neutral.
   - Falsifier: the scorecard's C2 or C2_hard CI upper bound < −0.004, or a C1_npx Δ ≤ 0 on the knob as built.
   - Expected LB effect is below noise; only worth an LB slot stacked with another C1 lever.
3. **H3 (closed by this diagnosis): lowering `LIB_TAU` or any absolute library lock below 0.9.**
   - On the engine scale, false locks on timsTOF Class-2 queries (C2_np 12% at 0.8, 24.5% at 0.7) outnumber the
     Class-1 gains unless the Class-1 share is > ~50%.
   - Out of scope unless a decoy-calibrated *comparative* score (H1) replaces the absolute one.

## Loop note (also filed in the campaign's `questions.md`)
On C1 cohorts, "leak-hit" (BASE rank 1 = truth) mostly means the library matched, which is genuine Class-1 evidence
and not the FPNet leak. `diag.py` splits it into a library-driven part (`lib_max ≥ τ`) and an FPNet part:
- every H2 gain is library-driven (`d_lib_leak`);
- by the current "other Δ" reading (L40) it would look like zero.
