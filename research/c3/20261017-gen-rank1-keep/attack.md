# Attack: gen-rank1-keep

1. **A1 the trigger reads FPNet (blocking).** Engine 1 orders its list by FPNet features, so "BASE #1" is FPNet-chosen;
   that is allowed (it is the base's own output), but the keep decision must add only ICE. Death: any `fz_*`, `rel`,
   `S` or `pc_fz` read in the GEN1_KEEP code path. Test: code review + grep in the diff.
2. **A2 ICE is FPNet-blind but ICEBERG-trained on MassSpecGym (blocking).** pseudo_c3 excludes MSG structures, so ICE is
   leak-free there; on C1/C2, MSG overlap favours the retrieved truth over the generated one, which makes C1/C2 harm
   read low. Death: none by itself; report fires on C1/C2 split by truth ∈ MSG if the list is available
   (`p1_novel` / harness truths), else record UNTESTED-with-proxy (`deviation`).
3. **A3 rank-1 displacement (blocking).** A fire on a molecule whose V22 rank 1 is the truth displaces it. Death: rank-1
   displacements > 1% of pseudo_c3 (rank-only rule), or any non-advisory C1/C2 cohort CI upper bound < −0.004.
4. **A4 knob-off parity (blocking).** Death: knob off changes any row on any cohort (scorecard a = b must give Δ 0).
5. **A5 test transfer (flag).** On the hidden test FPNet has not seen the spectra, so Engine 1's generated #1 precision
   will be lower than on pseudo_c3; the ICE guard is the only protection. Flag; the LB decides.
