---
name: diagnose
description: Campaign step (tier M, headless): turn where the truth is lost (loss map, scorecard misses, harness-vs-LB calibration) into falsifiable hypotheses. Invoked as `/diagnose research/campaigns/<id> --brief <path>`.
---

# Diagnose

Measure first, then hypothesise. Every number you write cites the command that produced it.

1. Refresh the loss map when it is older than the newest dump: `python research/harness/harness.py lossmap --dump C2 --dump C1_npx` (add every dumped cohort). **Heavy compute runs on Kaggle** (CLAUDE.md §3, human rule 2026-10-17): push it as Kaggle kernel `syedtoubur/casmi26-harness-run` (refresh the kit with `sh E:/casmi/kaggle_kit/build_kit.sh`, set `JOBS` in `E:/casmi/kaggle_kit/kernel/run_harness_kit.py`, `kaggle kernels push -p E:/casmi/kaggle_kit/kernel`) and fetch `loss_map.md`; never run it on this device.
2. For each loss bucket (pool miss / in some channel list but not top 25 / rank 6–25 / rank 2–5), write down:
   - the number of molecules and the headroom;
   - which channel holds the truth (BASE, Engine 2, PubChem), from the loss map or a short script over `research/harness/dumps/`.
3. Read `research/harness/calibration.md`. Note which cohort Δ (overall, leak-hit or other) has tracked the LB so far.
4. Give each bucket 1–3 hypotheses: "the truth is lost here because X; if X, then measurement M shows Y". M must be runnable on the dumps.
5. Write `research/ideas/<yyyymmdd>-diagnosis.md`.

Done when every bucket with headroom > 0.01 has at least one hypothesis with a named, runnable falsifying measurement.
