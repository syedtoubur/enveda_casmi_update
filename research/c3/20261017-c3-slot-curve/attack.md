# Attack: c3-slot-curve

1. **A1 placeholder cost is too optimistic (blocking).** On C1/C2 a real inserted generator candidate could equal the
   truth (then it is a gain, not a cost); the placeholder treats every insert as wrong, which is the conservative side.
   Death: none; report this as the conservative bound. Status: accepted by design.
2. **A2 insert duplicates (blocking).** If x already sits in the row, inserting without dedupe double-counts.
   Death condition: any output row with a duplicate IK14 → result void. Tested by the unit test.
3. **A3 FPNet leak in x (blocking).** The mut-first top-1 must come from `p4_raw.pkl` (decoder log-p beam, no FPNet,
   gate amendment 2026-10-15). Death: any use of `fz_*` / `gen_rank_fz` in choosing x → void.
4. **A4 domain shift (flag).** pseudo_c3 is 15/600 timsTOF; report the curve on that subset too (n tiny, flag only).
5. **A5 π unknown (flag).** The curve is reported at three π; s* is chosen at 0.25 once.
