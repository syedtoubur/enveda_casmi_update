# Handoff: target Class 3 (novel structures) with the pseudo_c3 instrument

Written 2026-10-17 after the human stopped campaign `20261015-c3-gen-slot2` at checkpoint 2. The goal is still Class 3.
What changed is that we can now **measure** it offline. Start with a grill (`/campaign-grill` or a plain grill), not with
code. The user prefers step-by-step progress with simple explanations and clickable grill questions.

## Read first
1. `research/campaigns/20261015-c3-gen-slot2/report.md`: what the campaign established, in one page.
2. `research/harness/gate_report.md`: the last five blocks (c3-gen-slot2 PRE-REGISTERED, K0 MEASURED, K1 MEASURED,
   K2 amendment, K2 MEASURED).
3. `EXPERIMENTS.md` §2: the rows K0, K1 and K2 v0; §3: the "Gated slot-2 novel insert" graveyard row.
4. Memory: `casmi-campaign-c3-gen-slot2.md`, `casmi-base-leaks-every-val-shard.md`, `casmi-knapsack-depth3-dead.md`.

## Assets you can use (all measured)
| asset | where | what it gives |
|---|---|---|
| pseudo_c3 dump (K0) | `research/harness/dumps/pseudo_c3/post_dump/state.pkl`; `harness.load_dump("pseudo_c3", post=...)` | 600 Class-3 molecules, truth banned from every channel, exact E006 replay. E006 MRR@25 **0.0612** (rank 1: 12, ranks 2–5: 56, 6–24: 20, miss 512) |
| truths | `harness.truths()` (pseudo_c3 from `p1_novel/pseudo_c3.parquet`, ids `C3_*`) | IK14 + SMILES |
| generator output (K1) | `E:/casmi/c3_gen/p4_raw.pkl` (`raw[mid]` = `dec` [(smi, logp, formula)], `mut` [(smi, logp, ...)]), `p4_inputs.pkl`, union top 25 in `novel.pkl` | top-1 exact h: mutator **0.120**, mut-first **0.125**, decoder 0.073, union 0.077; hit@25 ≈ 0.20; acyclic h 0.27 (n 129), ring 0.02 (n 471) |
| K2 feature table | `E:/casmi/c3_gen/k2_table.pkl` (cached rows of `build/c3_detector.py`) | per-molecule dump features + E006 truth rank + leak flag for pseudo_c3 and C1_npx/C1_xlib/C2/C2_hard (1,745 rows) |
| code | worktree `E:/casmi/lines/c3-commit`, branch `idea/20261015-c3-gen-slot2` (tip `faeedf0`) | `novel_p4*` (generation, scoring), `c3_detector.py`, `patch_harness_dump.py --shard pseudo_c3` |

## What we learned (constraints for any Class-3 idea)
- **pseudo_c3 is FPNet-leaky.** The queries are train.parquet rows FPNet full1 trained on. Any f·z signal, and Engine 1's
  generated candidates (ranked with `fz_*` / `gen_rank_fz`), are inflated there. Novel scoring must not use FPNet.
- **A separate C3 detector failed** (K2 v0: C1 fire 17% at the break-even τ; the bar is 1%). The cost per misfire is small
  (L ≈ 0.03–0.06 for a slot-2 insert); false fires on C1/C2 were the problem. A v1 with generator features was not run:
  the decoder trained on 89% of C1_xlib/C2/C2_hard truths (`p1_novel/dec_split.parquet`), so its confidence on those
  negatives is memorised.
- **Only 133 leak-free C1/C2 negatives** ("other": BASE#1 ≠ truth). Any C1/C2 regression check runs on small n.
- **Domain shift:** pseudo_c3 is mostly non-timsTOF (15 timsTOF); the test is 100% timsTOF. On the 15 timsTOF, mut h 0.267
  (4/15), union 0.
- K0 observation (not pre-registered): BASE#1 is a generated candidate on 138/600 pseudo_c3 (55 correct), and E006
  post-scoring moves 44 of those 55 from rank 1 to ranks 2–7. That ranking is FPNet-driven (leaky), so this needs a
  leak-aware check before anyone builds on it.
- Dead already: knapsack/MetFrag-lite (truth leak), c3v2 depth 3 (K-D3), tail-only inserts (ceiling ≈ +0.001), rank-1
  inserts (bio2 −0.064), detector-gated slot 2 (K2).

## Directions to grill (not decided; the human picks)
1. **No-detector routing curve (cheap, CPU, first).** From the K0 rank distributions and the K1 per-molecule hits,
   compute the expected Δ of inserting the mut-first top-1 at a fixed slot s for EVERY molecule (s = 2..25), at C3 shares
   0.15/0.25/0.35: gain = π·h·1/s versus cost = (1−π)·L_s. This tells us whether any slot pays without a detector.
2. **One leak-free scale inside the ranker:** score generated candidates and retrieved ones together (decoder log-p,
   ICEBERG cosine), so novel candidates compete on evidence rather than through a gate.
3. **The acyclic sub-class:** h 0.27 on acyclic truths. Check whether "predicted acyclic" (formula DBE, decoder top) is a
   cheap, low-false-fire trigger.
4. **Better mutator anchors / per-step prior** (K-D3 lesson: reach comes from anchors, not depth).
5. **Instrument first:** a test.parquet feature dump (Kaggle) to measure domain shift before any LB slot.

## Ops lessons
- Long local jobs: detached `Start-Process python -u ... -RedirectStandardOutput <log> -WindowStyle Hidden`; background
  shells get reaped under memory pressure. The K2 run took ~6 min (pseudo_c3 replay 5 min).
- Tautomer-IK14 of every generated candidate took ~4 h; canonicalize only the top few per arm (`c3_detector.top1`).
- Never use `E:/casmi/lines/exploit` (another agent). Concurrent agents edit the main repo: stage only your own files
  (EXPERIMENTS.md and `.scratch/handoffs/latest.md` carry other agents' pending edits).
- Colab: `research/bin/colab_run.py`, Drive consent via `research/queue/drive_consent.md`; balance ≈ 157 CU (approval needed).
