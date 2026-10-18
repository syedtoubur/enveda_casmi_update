# iso-rank fast-ship runtime (K decided by runtime only)

STATUS 2026-10-18: DECIDED K = 150 from the draft run of syedtoubur/casmi26-iso-rank v1 (T4, 12 molecules, 1760 scored candidates).

Inputs (measured, I1b.1 / I1b.0, `harness/gate_report.md`): V22 hidden 4.5 h (human); mutation 21.9 s/mol CPU (cap 600);
ICEBERG 5.99 cand/s on a T4, 482/600 = 80 % of molecules covered; hidden ~400 molecules.

| term | K = 150 | K = 50 |
|---|---|---|
| V22 | 4.50 h | 4.50 h |
| mutation 400 x 21.9 s / 4 workers | 0.61 h | 0.61 h |
| ICE: 400 x 0.80 x (10 parents + K) / 5.99 | 320 x 160 / 5.99 = 8 550 s = 2.37 h | 320 x 60 / 5.99 = 3 200 s = 0.89 h |
| subtotal without GLACIER | 7.48 h | 6.00 h |
| room for GLACIER under 8.5 h | 1.02 h = 3 670 s | 2.50 h |
| GLACIER cand/s needed (51 200 / 19 200 cand) | >= 14 cand/s | >= 2.1 cand/s |

Rule (brief): K = 150 iff projected <= 8.5 h, else 50. K = 150 passes only if GLACIER >= 14 cand/s on [M+H]+/[M+Na]+ spectra.
The draft run of the pushed kernel prints `ISO_RATES ice_cand_per_s ... gl_cand_per_s ...`; fill the measured values below.

Guards in the kernel regardless of K: stage starts after E2R2; molecules not started before T0 + 8.3 h keep their V22 row
(`iso_skipped_time`); ICE budget = remaining - 1 h, GLACIER budget = remaining to 8.3 h - 10 min, both stop themselves and
molecules without scores simply get no fire (tail still rides along).

## Measured (draft log, v1)
- ice_cand_per_s 15.89 (1760 cand in 110 s; includes load overhead; I1b.1 measured 5.99 on a larger pseudo_c3 set, used below as the pessimistic rate)
- gl_cand_per_s 28.31 (1760 cand in 62 s); gen_s 18 s for 12 molecules (1.5 s/mol, far below the 21.9 s/mol hidden estimate).
- Projection K=150, hidden 400 molecules, 51 200 ICE/GL candidates: V22 4.50 + mutation 0.61 + ICE 2.37 (at 5.99/s; 0.90 h at 15.9/s)
  + GLACIER 0.50 (at 28.3/s) = 7.98 h pessimistic, 6.5 h at the draft rate. <= 8.5 h, so K = 150 stays.
- Safety: the 8.3 h wall-clock guard and the ICE/GLACIER budgets still bound the run if the rates are lower on hidden.
