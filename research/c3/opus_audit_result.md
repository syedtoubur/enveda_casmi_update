# Referee report: c3-iso campaign and iso-rank

**Verdict:** The five DEAD verdicts all stand. The "don't push above the parent" decision is right, but the −0.205 figure is an upper bound, not a projection. The proposed nl ∧ mult veto rests on a number that no file in the repo contains, and it can't be tested on half B, which has already been used.

**What I could and couldn't check.** I read every result file in the idea folders, `spec.md`, `brief_stage_a.md`, `stage_a_result.json`/`stage_a_frozen.json`, the nl-sim-rda stress result and the 13:20 handoff. I couldn't open the Stage A code: access to the worktrees `E:/casmi/lines/c3-iso-rank{,-a}` was denied. So `build/iso_rank.py` and the per-molecule tables are unaudited. All arithmetic below was done by hand from the JSON.

---

## 1. Post-mortems of the five approaches

**c3-slot-curve: holds, with one qualification.**
- The pre-registered variant takes the generated candidate out of the row and re-inserts it at slot *s*. That demotes candidates V22 had already ranked high: the leak-hit stratum loses −0.118 at s=7. The best slot, s*=7, scores −0.0009.
- The deployable "keep-higher" variant peaks at **+0.0018** (s=7, π=0.25).
- Also, the C1/C2 cost was computed with a placeholder "sure-wrong" candidate, so it is an upper bound.
- Honest summary: no fixed slot pays *above the ±0.004 noise floor*. "No slot pays" is literally true only for the variant that demotes.

**isomer-mut-sim: the pre-screen bottleneck is confirmed; "blind to regio-isomers" is only suggested.**
- On the same 150 children, ICE lifts reach_20 from 36 to 61 (I1b.1). That is direct evidence that MetFrag-lite is the bottleneck.
- Operator loss rates from reach_all to reach_20:

| operator | reach_all → reach_20 | lost |
|---|---|---|
| subst_shift | 61 → 25 | 59% |
| chain_iso | 16 → 10 | 38% |
| group_move | 11 → 4 | 64% |

- Two more signs of a coarse scorer: raising the cap from 300 to 600 children *lowered* frag reach_20 (7.3% → 6.0%), and the mean rank of the hits was 8.3.
- So regio-isomers are lost more often, but group_move is lost just as often. The data support "frag can't tell isomers apart", not regio-isomers specifically. Nobody measured tie rates.

**gen-rank1-keep: right conclusion, partly circular argument.**
- The +0.0100 is entirely in the leak-hit stratum: 0.107 × 56/600 = 0.00999; the other stratum is exactly 0.
- But a rule that only keeps BASE #1 at rank 1 can *only* gain when BASE #1 is the truth, so that stratum split is true by construction. It doesn't prove leakage by itself.
- The leakage argument actually rests on A5: the generator is conditioned on an FPNet that saw the pseudo_c3 spectra.
- **Design flaw:** the best-case LB gain was π × 0.010 ≈ 0.0025, below the +0.004 keep bar. The LB could never have passed the probe. And 0.446 vs 0.447 against an unpaired control carries no information either way.

**isomer-mut-ice I3: the conclusion holds, but the 74% vs 72% comparison is wrong.**
- The 74% (356/482) counts only ICE-covered molecules. The 72% (44/61) includes 7 hits in uncovered molecules, which cannot fire because they have no ICE scores.
- Like-for-like: **44/54 = 81% vs 74%**. That is a likelihood ratio of about 1.1, so the trigger is still useless.
- The mechanistic reason is an extreme-value effect. The best of about 150 noisy isomer cosines beats the best of 10 retrieved candidates almost always, whether or not the truth is among the children. The retrieved candidates are wrong anyway in about 87% of pseudo_c3.

**iso-rank: the verdict holds, and there are two findings the post-mortem missed.**
- **Possible zero true positives on half B.** On half B, `delta_fire_only` equals `delta_wc` exactly (−0.000297, also equal in fast_ship_B). On half A they differ (+0.0027 vs −0.0014). So either no fired child was the truth in any of the ~66 half-B fires, or that key is mis-wired. One check on the per-molecule table settles which.
- **The 0.012 bar was out of reach for this mechanism.** The truth is in the child pool for 49 of 298 molecules (16%), with within-pool MRR@20 = 0.319. So P(top child = truth) ≤ ~5%. With placement at ≥ rank 2 and a 22% fire rate that isn't enriched for truth, the ceiling is ≈ 0.22 × 0.05 × 0.5 ≈ 0.006. Half A's +0.0027 fits that ceiling.
- **Why pos(p)+1 hurts C1/C2.** It never touches a rank-1 truth. But in C2, about 45% of truths sit at rank 2–5 behind a wrong rank-1 parent, and each inserted child shifts them down. That gives −0.021 at a 64% fire rate.
- **The handoff contradicts its own data.** It says the tail ride-along has "a hard ceiling of < +0.001". The data say the tail contributed **+0.0018** on half B and +0.0023 on half A.

## 2. "Push above the parent / to rank 1": right direction, magnitude overstated

I reconstructed the −0.205 as 0.75 × mean[0.92 × 0.796 × 0.5, 0.55 × 0.656 × 0.5] = 0.75 × 0.273 = 0.205. The fire rates are the frag-only means for C1 and C2. Problems with it:

1. **Exact only for a rank-1 push.** For a push to pos(p), only fires whose parent is the rank-1 candidate cost 0.5. The 64–84% is the rate at which the top child beats *its own* parent, which can be any of the top 10. Weak evidence suggests the parent is usually near the top (C1_npx: 8 of 124 down at pos(p)+1), but this fraction was never measured.
2. **The fire rates are upper bounds.** The ICE conjunct was dropped on C1/C2. On pseudo_c3, adding it cut the fire rate to 22–32%.
3. **The rank-1 rates (0.92, 0.55) have no source.** They look like MRR used as hit@1, and MRR is always at least hit@1. Val also reads above the LB. If C3 ≈ 0, an LB of 0.447 implies a C1/C2 MRR of about 0.60 on the LB, versus 0.75 on val. Scaling the loss accordingly gives about **−0.16**, not −0.205. The "0.242" figure isn't defensible.
4. **Two omissions that roughly cancel:** extra losses from shifting truths at ranks ≥ 2, and any C3 gain (≤ ~+0.01).

**Useful replacement number:** at break-even, 0.75 × 0.367 × FPR equals π × the C3 gain. Even with an optimistic C3 gain of 0.05, the C1/C2 false-fire rate must be **≲ 4.5%** before any above-parent push can pay. Every measured rate is 15× higher than that. So the decision stands.

## 3. The six signals and the dual-veto rule

1. **"nl and mult have ~4–8% false alarms" has no source.** The C1/C2 cohorts in Stage A were scored *frag-only*. No file contains per-signal C1/C2 fire rates. `audit_clean_signals.py` only prints the existing JSON, and `audit_signals.py` doesn't exist. The A3 kernel table on Kaggle could measure these rates, but no one has done it.
2. **"Raw ice has 64–84% false alarms" is also unsupported.** ICE was never run on C1/C2 ("UNTESTED" in I1b.1; no ICE in Stage A). The 74% figure is from pseudo_c3. There is also a bias to watch: ICE and GLACIER are MassSpecGym-trained and overlap the val cohorts, so any false-alarm rate measured on val will look better than on the hidden test.
3. **The evidence that does exist goes against nl:**
   - same-formula pairwise win rate is only 58.5%;
   - nl-sim-rda failed A3 (C2 displaced 1.8%; C1_npx CI lower bound −0.028);
   - it is the *weakest* single list on pseudo_c3 (within-pool MRR@20 = 0.100).

   The only way it gets a low false-alarm rate is if "Δnl > 0" is rare because of ties. That would cut true positives just as much.
4. **mult can't separate C3 from C1/C2.** It ignores the spectrum: it only counts how many routes in the parents' mutation graph reach the child. It is also one of the RRF lists, so the RRF-top child is pre-selected for high mult, and "mult ≥ 2" will be close to always true. The two signals aren't independent either, so their false-alarm rates can't simply be multiplied.
5. **The rule is underspecified.** It doesn't say where the child goes (pos(p) or pos(p)+1). With pos(p), a parent that is the truth at rank k still loses 1/k − 1/(k+1), which is 0.167 at rank 2 and common in C2.
6. **Methodology: half B is used up.** This rule was proposed after half-B results were read. Validating it on half B would be selecting on the test set. It needs a fresh scaffold-disjoint holdout or a new pseudo_c3 draw, with thresholds pre-registered in `gate_report.md`.
7. **Minor points:**
   - nl entered forward selection on a +0.0011 rise in within-pool MRR, with n_pool = 49. That is noise.
   - The Δ shrank from +0.0050 on half A to +0.0015 on half B, which is what selection inflation looks like.
   - GLACIER's "70% top-1" isn't sourced anywhere in the repo.

## Recommended next steps (cheap, CPU)
1. From the A3 per-molecule table on Kaggle, measure the C1/C2 fire rate of each signal on its own and of Rule A, plus how often the parent is at rank 1 when a fire happens. If Rule A's C1/C2 fire rate is above ~4.5%, drop above-parent placement for good.
2. Settle the `delta_fire_only == delta_wc` question on half B: count fired children that equal the truth.
3. Correct the handoff: the tail figure, the source of the ice false-alarm rate, the 0.242 projection, and the I3 denominators.

I saved these conclusions to memory (`casmi-iso-rank-referee-audit.md`).
