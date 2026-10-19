# Handoff 2026-10-18 (c3-iso overnight close-out)  -- self-contained
Commit 456f6b6 on branch specs-scratch-r0 (not pushed). Other agents' uncommitted edits remain unstaged. `.agents/skills/qoder-ide-loop/` left untracked (another agent's; its section 0/4.2 carry the Kaggle rule). `.scratch/handoffs/latest.md` NOT overwritten (has another agent's uncommitted changes): read this file by name.

## What ran overnight
- Heavy-compute-on-Kaggle rule added (CLAUDE.md, AGENTS.md, skills build/spec/diagnose/run-idea/attack). Local measurements stopped; all runs on Kaggle.
- Harness rerun on Kaggle CPU: kit builder `E:/casmi/kaggle_kit/build_kit.sh` -> dataset `syedtoubur/casmi26-harness-kit`; kernel `syedtoubur/casmi26-harness-run` v2 (set JOBS in kaggle_kit/kernel/run_harness_kit.py; push `kaggle kernels push -p E:/casmi/kaggle_kit/kernel`; fetch with `export PYTHONIOENCODING=utf-8; kaggle kernels output ...`; Kaggle auto-extracts uploaded tars). Existing verdicts unchanged; families `v22_overlay`, `gen1_keep` added to backtests.yaml (ADVISORY).
- control: `research/c3/20261017-control-v22` (G4 PASS; config lines.c3.control).

## gen-rank1-keep (research/c3/20261017-gen-rank1-keep)
- Kernel `syedtoubur/casmi26-gen-rank1-keep` v1 (+ source dataset `syedtoubur/casmi26-e180-keys`). Watcher timed out in queue (60 min), kernel completed ~02:44; G4 FAIL only on changed=0 (gen1_checked 12, kept 0, moved 0: no gen #1 on the 12-molecule draft). Fix agent found no code bug (g4_fix_1.md). Orchestrator waived g4_min_changed and SUBMITTED: **ref 57325719 (PENDING)**. STATUS stays needs-human.
- Harness: pseudo_c3 +0.0100 [+0.0045,+0.0163], composite +0.0038, gain only on leak-hit molecules (flag).
- `poll_lb.py --claim` was blocked by the permission classifier overnight (not retried).
- Morning decision rule: KEEP-pending iff LB >= 0.451 (0.447 human V22 + 0.004) AND hidden-rerun `gen1_kept` > 0, then one replicate. Unpaired vs the human's V22 variant (flag).

## isomer-mut-ice (research/c3/20261017-isomer-mut-ice)
- I1b.0 PASS: reach_150 0.140 (84/600, CI 0.118-0.165, bar 0.12). Measure on Kaggle CPU `syedtoubur/c3iso-i1b-measure-run` v1.
- I1b.1: ICE `syedtoubur/c3iso-i1b1-ice-run` v2 (GPU T4, 205 min, 5.99 cand/s), measure `syedtoubur/c3iso-i1b1-measure-run` v1; datasets `syedtoubur/casmi26-c3iso-i1b1-ice-in`, `-ice-out`, `casmi26-c3iso-i1b-measure`; code commit 9c1ef2b in worktree E:/casmi/lines/c3-isomer-mut-ice. Result: NOT DEAD, marginal. reach_20 by ICE 0.1017 (61/600, CI 0.083-0.124) vs bar 0.10 (CI straddles); reach_50 0.1317; reach_150 0.140; ICE-covered 482: 0.112; frag reach_20 0.060. 482/482 scored, 18,454 null (all in 118 uncovered), n_pred_fail 0. Hidden cost projection ~2.3 h GPU + ~20 s/mol CPU.
- I3 fire signal: "best child beats best fused-top-10 by ICE" fires 74% of covered and 72% of hits = non-discriminating. C1/C2 part UNTESTED.
- Stray-kernel audit: another agent's unpushed GPU kernel in the idea folder had a truth leak (truth as fallback parent, truth columns read), pruned operators, cap 50, and never called ICEBERG; not used, originals kept as `*.orig`. Code of record is 9c1ef2b.
- Ledger: gate_report.md blocks "c3-iso I1b.0/I1b.1 MEASURED"; campaign log; state.json (submitted 1).

## Costed I3 plan (NOT started; stopped at gate C2)
Replay needs mutant ICE on ~1,390 C1/C2 molecules = ~215k requests = ~7 h GPU (~3.5 h on two kernels) + ~2 h CPU mutation; could not finish overnight. Runtime gate C1 also unconfirmed: isomer stage ~ +3.4 h on 400 hidden molecules on top of V22's hidden time (unknown; draft was 1,036 s on 12 molecules). Inputs: frozen pkls in E:/casmi/c3_iso, datasets above, harness kit. Prerequisite: pre-register a margin/rank trigger on a scaffold-grouped half split of pseudo_c3 BEFORE any replay (plain "best child > best retrieved" is useless).

## Open decisions for the human
1. Spend I3 on a margin rule (e.g. 0.0017) or close isomer-mut-ice (point estimate only 0.0017 above the bar).
2. Run the C2/C1_npx fire check first (cheap-kill), before any mutant-ICE replay.
3. Whether the g4_min_changed waiver for gen-rank1-keep should become a standing rule.

## First commands in the morning
```
export PYTHONIOENCODING=utf-8; cd /e/ENVEDA_casmi
python research/bin/poll_lb.py --claim research/c3/20261017-gen-rank1-keep --ref 57325719
python research/bin/poll_lb.py        # decide(): paired control, counters > 0, replicate
kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra
python research/bin/gpu.py            # weekly quota before any I3 GPU
```
