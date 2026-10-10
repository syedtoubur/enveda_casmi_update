# specs_scratch R4: FPNet family fusion + FP-ordered stage C (grilling outcome, 2026-10-08)

Status: **closed 2026-10-08: DROP (neutral).** LB 0.345 vs 0.346 (see Results).

## Grilling log (3 rounds, clickable `AskUserQuestion`)
0. **Rejected: random 1k-bit subspaces of one model** (10–20 channels, then RRF). With p = σ(z), the Bernoulli
   log-likelihood of a candidate's bits is `f·z + Σ log(1−p)`, and the second term is the same for every candidate. So
   `f·z` already is the model's exact likelihood ranking. A random subset sum is a noisy partial copy of the same
   evidence: nothing is trained per subspace, so this is not bagging, and the mean of the subset sums is the full sum
   again. RRF over those copies helps only if a few redundant or miscalibrated bits dominate the score. The user agreed;
   what has paid is fusion of **different models/families**.
1. Goal: grow the scratch pipeline (own code); no zenith fusion this phase. Levers: FP family fusion, plus FP on
   stage C. LB protocol: a paired same-day control.
2. Families: **A = full1, P = prvsiyan v2** (not our own `casmi-fpnet`). Key `sv + A + 0.6·P`. Stage C is bundled
   (with a counter). Control + 1 probe.

## Why this lever
- The ledger's repeatable gain is RRF of systems that differ in FP family and bit basis: zenith Engine 1
  (full1, 10,226 bits) × Engine 2 (prvsiyan, 6,930 bits) gave +0.016 and +0.015.
- Swapping one model inside a family gave −0.002.
- In scratch, the two families scored 0.341 (A) and 0.322 (P, own) alone, so each carries real signal.

## Settled decisions
| # | Topic | Decision |
|---|---|---|
| 1 | Families | **A** = `casmi26-fpnet-full1/fpnet_full1.pt`, `fp_bits.npy` 10,226 (`casmi26-v2-pool`). **P** = `prvsiyan/casmi26-fp-models-v2` (`fp_single_s2`, `fp_merged_m1`), `fp_bits.npy` 6,930 next to `coco_fp.npy` (as zenith Engine 2) |
| 2 | Tier-2 key (stages A, B) | `1/(3+r_sv) + 1.0/(3+r_A) + 0.6/(3+r_P)`. r = average rank, descending. NaN ranks last within its own family. Tie-breaks fr, then \|ppm\| |
| 3 | Stage C | Mass-nearest top-up ordered by `1/(3+r_A) + 0.6/(3+r_P)`, then \|Δmass\|; counters `n_stagec_mol` / `n_stagec_slots` |
| 4 | Config | `fp_banks=("ahmed","prv")`, `fp_weights=(1.0,0.6)`, `fp_stagec=True`. `fp_channel` alone still means R3 exactly (stage C stays mass-ordered), so the control is unchanged |
| 5 | LB | Same day: `r3_full1_r2gl` (control) + `r4_famfuse`. KEEP at Δ ≥ +0.006; a Δ in [+0.004, +0.006) gets one repeat pair |
| 6 | Audit | None (no honest cohort for either family) |

## Facts the build relies on (verified)
- Zenith Engine 2 (`scratch/v18_zenith.py`, `ENG_FILES`):
  - `load_fp_models` builds `FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'])` and splits single from merged by filename.
  - `molecule_logits` = mean(single per-spectrum, merged on merged peaks at CE 25). Our `kind="own"` path is a port of this.
  - Its `fp_bits.npy` comes from the directory of `coco_fp.npy`: R3's own run found it in `prvsiyan/coconut-casmi26-candidates`, sha1 `4686f545…`.
- prvsiyan's checkpoints have no `fp_bits_sha1.txt`, so `own_ckpt_paths` skips them. Kind `"prv"` finds them by the dataset path `casmi26-fp-models-v2`.

## Build log (2026-10-08)
- `config.py`: `fp_banks`, `fp_weights`, `fp_stagec`; `fp_kinds()` / `fp_w()` (fp_channel alone → one bank, weight 1).
- `rank.py`: `fp_rrf(fps, weights, k, idx)`; `rank_two_tier(..., fp=list, fp_w=)`. A single array gives the R3 order (test).
- `fpnet.py`: kind `"prv"` (family P, `P_KINDS`), found by the `casmi26-fp-models-v2` path; fp_bits = the coco-adjacent file of matching length.
- `pipeline.py`:
  - `plan["fpz"]` = one z per bank;
  - stages A/B pass the list;
  - `fill_mass_nearest(..., fp_fn, fp_w, k)`;
  - counters `n_fp_*_by`, `n_stagec_mol`, `n_stagec_slots`.
- `r0_kaggle.py`: loads every bank, with parity and self-rank for each. Fast-commit asserts parity (ahmed, prv), self-rank, and that each family fires.
- Notebook `casmi_scratch_r4_famfuse.ipynb`. Its source diff vs `r3_full1_r2gl` is only `CONFIG_OVERRIDES` + `RUN_TAG`.
- Tests: `test_r4.py` (6, incl. a dry run with two tiny banks in which stage C fires). The full suite is green.

## Run order (user-side)
1. Kaggle `casmi_scratch_r4_famfuse`: attach r3_full1_r2gl's inputs + `prvsiyan/casmi26-fp-models-v2` +
   `prvsiyan/coconut-casmi26-candidates` (COCONUT arrays: `fp_bits.npy`, `coco_fp.npy`, `coco_meta.pkl`).
   The fast commit must print parity and self-rank for both banks and PASS R4.run (a)–(c).
2. Same day: submit `r4_famfuse` and re-run `casmi_scratch_r3_full1_r2gl` (control).
3. Record both scores, `total_s` and the counters in `gate_report.md` R4, then `/update-experiments`.

## Risks
- **Unknown training split for prvsiyan:** its quality on hidden NP structures is untested in scratch. The LB pair is the only judge.
- **Vocab mismatch:** prvsiyan's adduct/instrument vocab may differ from our family-P port. Parity and self-rank catch only gross errors.
- **Runtime:** one more featuriser (Morgan r2/r3, RDKitFP, MACCS) per candidate, cached per key. Cheap next to GLACIER.

## Results (2026-10-08)
- **R4.run PASS.** Both families: parity 50/50 and self-rank 5/5. `n_fp_reorders_by` [20, 19]. Stage C fired 0/20.
- **R4.LB DROP.** `r4_famfuse` **0.345** vs `r3_full1_r2gl` 0.346 (the earlier score; the control was not re-run). Δ −0.001.
- **Reading.**
  - A second FPNet family is live (it reorders the top 25) but adds no information: both models solve the same spectrum→FP task on the same spectra and overlapping training libraries, so their errors are correlated.
  - Zenith's two-engine gain fused **whole systems** (pool, rankers, analog features), not just a second FPNet.
  - Next levers should read different evidence: popularity prior, PubChem candidates with promotion, a ranker.
