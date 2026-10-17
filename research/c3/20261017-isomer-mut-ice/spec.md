# Spec: isomer-mut-ice (I1b), human decision at the wave-1 checkpoint 2026-10-17

Gate block: gate_report `### 2026-10-17: c3-iso I1b (isomer-mut-ice), PRE-REGISTERED`. I1 (`../20261017-isomer-mut-sim`)
stays DEAD (reach_20 0.073 with the CPU frag top 20; reach_all 0.167 at cap 300, a lower bound: 474/600 capped).

## I1b.0 (CPU, this step)
- Same operators (`build/isomer_mut.py` from branch `idea/20261017-isomer-mut-sim`, c9c5fb2), cap **600** children,
  deadline **120 s** per molecule (check it inside the operator loop, not only between parents), Pool ≤ 4.
- Same pipeline as `build/isomer_reach.py` (V22 fused top 10 parents, plain-IK14 dedupe, drop retrieved, CPU frag score
  vs the query's measured spectrum), but freeze the **top 150** per molecule (and keep the frag score) to disk before
  loading truths. Also freeze, per molecule, the full child list (SMILES + operator + frag score) in a compact pickle under
  `E:/casmi/c3_iso/` — I1b.1 (GPU) will read it.
- Report: reach_all (cap 600) vs the I1 cap-300 figure, reach_150, reach_50, reach_20 by frag (for comparison), the
  512 "truth not in V22 top 25" subset, acyclic/ring, timsTOF, children/molecule, capped count, CPU s/molecule.
- **DEAD iff reach_150 < 0.12 on pseudo_c3.** Else STATUS `needs-human` (GPU go for I1b.1).
- Also produce the I1b.1 request size: number of (child, adduct, CE) ICEBERG requests for the 150 + fused top 10 per
  molecule, using the query spectra's adduct/CE (count distinct (adduct, CE) per molecule as the base worker would).

## I1b.1 (GPU, NOT in this step)
ICEBERG on the frozen 150 + fused top 10; DEAD iff reach_20 by ICE < 0.10. Plan only after I1b.0.
