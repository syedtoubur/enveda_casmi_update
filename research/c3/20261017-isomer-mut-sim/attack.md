# Attack: isomer-mut-sim (I1)

1. **A1 truth leak into selection (blocking).** Death: the truth (SMILES, IK14, formula or a truth-derived spectrum)
   is read anywhere before the reach count (R10, the knapsack lesson: biomet.md's gain was a truth-fragment leak).
   Test: the reach script loads truths only after the top-20 lists are frozen to disk.
2. **A2 the pool ban is undone (blocking).** pseudo_c3 bans the truth from every channel; a mutant equal to the truth
   is a legitimate hit ONLY if it was generated from a non-truth parent. Death: any parent in the fused top 10 equals
   the truth IK14 on pseudo_c3 (would mean the K0 ban leaked) → stop and report.
3. **A3 formula drift (blocking).** Death: any child whose formula differs from its parent (unit test + a runtime
   assert counted in the result).
4. **A4 reach is trivially high because children are many (blocking for the bar).** reach_all with ~300 children is
   not the claim; the bar is on reach_20 after the CPU pre-screen. Death: verdict computed on reach_all.
5. **A5 frag pre-screen is the weak link (flag).** If reach_all ≥ 10% but reach_20 < 10%, report it: the operator set
   reaches but the pre-screen loses it (a GPU ICEBERG pre-screen might recover it; human decides at the checkpoint).
6. **A6 domain shift (flag).** 15 timsTOF only; report the split.
7. **A7 runtime (flag).** Report CPU s per molecule; the hidden test has ~400 molecules within the 9 h budget.
