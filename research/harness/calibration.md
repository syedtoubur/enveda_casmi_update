# Harness ↔ LB calibration ledger (append-only)

One row per scored probe, paired with its line control (gate_report 2026-10-13 rank-only block). Once there are ≥ 5
rows, the composite's class weights are refit by regressing LB Δ on the cohort Δs, and the refit is pre-registered in
gate_report.md before it is used. "other" = Δ on molecules where BASE's rank 1 is **not** the truth (no BASE leak).

| date | probe (kernel vN) | knobs | C2 Δ [90% CI] (leak-hit / other) | C1_npx Δ (leak-hit / other) | composite | LB | control | LB Δ | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-12 | promo-soften (`casmi-promo-soften` v1) | REL_PROMOTE 500 | +0.0185 [+0.0115, +0.0265] (+0.0204 / +0.0020) | +0.0070 (+0.0053 / +0.0209) | +0.0127 | 0.437 | 0.440 | −0.003 | C2 gain sits on leak-hit molecules; the honest "other" Δ is +0.002 |
| 2026-10-12 | engine2-fppair (`casmi-engine2-fppair` v2) | Engine 2 fppair rows + w_pv 0.50 (heavy path) | not replayable; ceiling +0.002 C2 | ceiling +0.021 | — | 0.429 | 0.440 | −0.011 | bundle of rows + weight (L15); unbundled as engine2-pv88 |

Earlier sign-only points (back-test suite, `backtests.yaml`): e006 +0.0005 C2 / +0.0027 C1_npx → LB +0.007; ice_fallback
→ +0.004; final_pass_off → −0.008; cell19_base_ice −0.044 / −0.050 → −0.015.

## Observations (hypotheses until ≥ 5 rows)
- 2026-10-13: the first two pairs both read positive on the harness and negative on the LB. On promo-soften, the
  whole C2 gain came from leak-hit molecules. Candidate predictor to test as rows arrive: the **"other" Δ**, not the
  overall Δ.
