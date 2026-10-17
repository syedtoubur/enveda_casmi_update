# Campaign report: 20261015-c3-gen-slot2

Stopped 2026-10-17 by the human at checkpoint 2: K2 v0 (the Class-3 detector) was DEAD on the fire-rate kills, and the
human chose not to spend GPU on a v1 detector. Nothing was pushed to Kaggle as a probe; nothing was submitted.

Cycles 3 (K0, K1, K2), probes pushed 0, submitted 0, Colab 45.2 CU (K1 only), closed branches: class3-route.

| idea | closed as | gate | branch | reason |
|---|---|---|---|---|
| 20261014-pseudo-c3-dump | instrument-ready | K0 PASS | instrument | POST EQUIV 600/600, banned truth in no channel; E006 MRR@25 on pseudo_c3 0.0612 (all via Engine 1 `generate=True`) |
| 20261015-c3-gen-union | K1 PASS (asset) | K1 PASS | class3-gen | top-1 exact h: union 0.077, decoder 0.073, mutator 0.120, mut-first 0.125; acyclic 0.27 vs ring 0.02 |
| 20261017-c3-detector | dead | K2 v0 DEAD | class3-route | C1 fire 0.171 (bar 0.01), C2 rank-2–5 fire 0.603 (bar 0.02); AUC 0.656 (clean), 0.81 with FPNet features, all DEAD |
| c3-slot2-route | not started | — | class3-route | gated on K2 |

## What the campaign established
1. **A Class-3 instrument exists.** `research/harness/dumps/pseudo_c3` replays exactly under the E006 cfg with the truth
   banned from every channel. E006 scores MRR@25 0.061 there (12 at rank 1, 88 in the top 25, 512 misses).
2. **Generators can hit Class 3 at rank 1.** With predicted formulas and no FPNet, the c3v2 mutator puts the truth at top-1
   on 12% of pseudo_c3 (decoder 7.3%). Hits concentrate on acyclic molecules (h 0.27). On the 512 molecules BASE
   misses entirely, the union has 37 top-1 and 94 top-25 hits.
3. **Gated insertion via a separate detector does not work with the signals we have.** A slot-2 insert is cheap per
   misfire (L ≈ 0.03–0.06), so break-even precision is easy; the binding constraint is false fires on C1/C2. Dump
   features separate C3 from C1/C2 weakly (AUC 0.66). FPNet-derived features mostly separate FPNet-seen from unseen
   spectra (leak-hit negatives look like pseudo_c3 on f·z).
4. **Measurement traps found:** only 133 leak-free C1/C2 negatives exist (L40 removes ~89%); Engine 1's generated
   rank-1 is FPNet-ranked (leaky on pseudo_c3); the Track 2 decoder trained on 89% of the C1_xlib/C2/C2_hard truths.

## Agenda for the next grill (Class-3 targeting)
- Use pseudo_c3 as the target cohort, judged together with the C1/C2 "other" Δ on the harness (rank-only gate).
- Candidates: routing without a separate detector (e.g. interleaving generated candidates into the existing ranker on
  one leak-free scale), the acyclic sub-class, improving the mutator's anchors.
- Open: the test.parquet fire rate / domain shift (pseudo_c3 is mostly non-timsTOF; test is 100% timsTOF).
