# Brief: report for 20261018-iso-rank

**Goal**: Learn from the cycle that just closed: draft lesson and rule changes (add, modify, retire) for the human.

## Campaign scope (from charter.md; decisions already made, do not reopen)
- Line: c3
- Base: V22 = `casmi26-v22-rel150-e2r2-clean` (LB 0.447–0.451); export `scratch/baseline/casmi26_baseline.py` in the worktree.
- Worktree `E:/casmi/lines/c3`, branch `line/c3` = `idea/20261015-c3-gen-slot2` (faeedf0: novel_p4*, c3_detector,
  delta_m_mutator, patch_harness_dump) + V22 parity merged from `antigravity-exploit` 38a42a4 (commit dab59b0). Idea
  branches `idea/20261017-<slug>` off `line/c3`, one worktree each.
- Harness: `harness.py scorecard --line c3 --a "METRIC_FUSION=true,ICE_FALLBACK=true,FUSION_WINDOW=40,REL_PROMOTE=150.0,FRAG_GAP=true,E2R2=true"`,
  target pseudo_c3; C1_npx, C1_xlib, C2, C2_hard judged on the leak-free "other" Δ (L40); C2_np advisory.
- Paired control: an own-account V22 replicate pushed in the same window as the first LB probe (L05/L06); keep_bar = control + 0.004.
- Assets (measured): pseudo_c3 dump (K0), K1 generator output `E:/casmi/c3_gen/p4_raw.pkl`, `p4_inputs.pkl`,
  `novel.pkl`, the K2 feature table `k2_table.pkl`.
- Wave 1 (parallel, CPU): S1 c3-slot-curve, R1 gen-rank1-keep, I1 isomer-mut-sim reach. Opus writes spec.md + attack.md
  per idea; Sonnet builders run in parallel; gate.py recomputes each verdict; one human checkpoint after the wave.
  Then `campaign.py --once` steps ship / I2 (GPU, after a go) / replan.
- Resources (human rule 2026-10-17, CLAUDE.md §3): **every heavy job runs as a Kaggle CPU kernel, never locally**:
  harness scorecard/suite (kit dataset `syedtoubur/casmi26-harness-kit`, kernel `syedtoubur/casmi26-harness-run`;
  rebuild the kit tar when line code changes), candidate generation/scoring, ICEBERG. Locally only edits, unit tests on
  small fixtures, smoke runs < 2 min, and reading fetched outputs. Colab balance ≈ 157 CU; no GPU without a costed
  plan and a human go.
- Isolation: never write in `E:/casmi/lines/exploit`, `E:/casmi/lines/antigravity-exploit` or their idea folders
  (read-only copies are fine); stage only this campaign's files (EXPERIMENTS.md and `.scratch/handoffs/latest.md`
  carry other agents' edits).
- Open portfolio: the human (or /replan) may add Class-3 ideas mid-campaign without a new grill; each new idea needs a
  dated PRE-REGISTERED block in gate_report.md before its first measurement.
- Known: `preflight.py --selftest` FAILs R04/R08 on fixture-list drift from other agents' idea folders; not blocking
  (campaign.py uses per-idea `preflight.run`); not fixed here. `test_e2r2_row_pure_invariants` fails on 38a42a4 itself (inherited).

### Bars
- Gate: the rank-only gate (`research/harness/gate_report.md`, block `### 2026-10-13: rank-only gate + multi-cohort
  scorecard`), plus the block `### 2026-10-17: c3-iso campaign, PRE-REGISTERED` (P0 parity, S1, R1, I1 reach ≥ 10%, I2, I3, ship).
- KEEP at submit (`poll_lb.decide`): probe ≥ paired own-account V22 replicate + 0.004, value-checked counters
  (`gen1_kept`, `iso_fired`), draft differs from the control draft only where the knob fired (L38); replicate before it becomes the base.

### Out of scope
- A separate Class-3 detector (K2 v0 dead); absolute-score gates (L14); λ retuning (L08).
- Generated candidates at rank 1, except R1's ICE-guarded keep of Engine 1's own generated BASE #1.
- FPNet-based scoring or triggers for novel candidates (pseudo_c3 is FPNet-leaky).
- Knapsack / MetFrag-lite scorers that read anything but the query's measured spectrum (R10); c3v2 depth 3 (K-D3).
- Tail-only (21–25) as the main lever (ceiling ≈ +0.001); it may ride along.
- Retraining FPNet or the rankers; any FPNet swap without its rows (L01, L02).
- The 5 pending proposals in `research/proposals/` (deferred to a separate review).
- Edits to CLAUDE.md, README, skills, `.claude/settings*`, and the loop code (`research/bin`, `research/rules`), except
  the c3 line entry in `research/config.yaml` (done in the grill).
- Any `kaggle competitions submit`: the human submits.

## Rules preflight.py enforces on your output (do not restate them; satisfy them)
- R05: A pushed probe has at least one counter whose printed value must be > 0; presence alone proves nothing (L05).
- R06: A submit card names the paired own-account control kernel; KEEP is decided only by poll_lb.decide() (L05, L06).
- R08: A harness result counts only for families the capability map does not call BLIND or BASE-COUPLED; it must carry its family + verdict, and such a family's idea goes to the LB on a written prior.md (LB decides).

## Judgment lessons cited by this campaign or idea
- **L01** Ranker training queries must include structures the FPNet never trained on. Excluding the eval holdout is not enough: the FPNet becomes an oracle on seen queries, the GBDT drops the other channels, and the LB falls (0.337 → 0.287). Src: E#04.
- **L02** Never ship a new FPNet without regenerating `rank_train.npz`: 12 of the 31 ranker features are FPNet logits. A *better* FPNet under an old ranker scores *worse*. Src: E#04, E#16.
- **L05** The LB noise floor for identical code is ±0.004. KEEP only if LB ≥ the **paired own-account control** + 0.004 *and* the change's counters show it fired. Src: E#31, E#35.
- **L06** The base is our own account's score, never the author's headline. The "0.415" fork really scored 0.393 from our account (private vs public popularity arrays). Src: E#35.
- **L08** Tuning on local val overfits: −0.015 LB. Never tune hyperparameters to val MRR. Src: E#10.
- **L14** Calibrate gate thresholds on the full distribution, not on 12 fast-commit molecules: a hard gate hijacking ranks 1–5 cost −0.064. Src: E#25.
- **L38** A probe can complete G4 with every counter printed and still change nothing: promo-soften logged `promoted: 0` and passed. G4 now also needs the probe's draft submission.csv to differ from the control's draft (research/exploit/20261013-control-draft) on at least `g4_min_changed` molecules. Src: research/proposals/20261013-loop-audit.md.
- **L40** promo-soften's harness gain sat on molecules whose truth BASE already ranked first (C2 leak-hit Δ +0.0204 vs other +0.0020), and its LB moved −0.003. Read the scorecard's leak-free "other" Δ before the overall Δ, and treat a gain that lives only on leak-hit molecules as an ASSUMED prior. Src: research/proposals/20261013-loop-audit.md.

## You may open only
- E:/ENVEDA_casmi/research/c3/20261018-iso-rank/ (STATUS, gates.md, stress_result.md, harness_result.json, questions.md, card.md)
- E:/ENVEDA_casmi/research/campaigns/20261017-c3-iso/log.md (this cycle's lines)
- research/LESSONS.md (IDs + one-liners only)
- research/rules/ (the rule files; `python research/bin/preflight.py --selftest`)

## Output contract
research/proposals/20261018-20261018-iso-rank.md in the promote.py item format (`## Pn: title`, `- action: add|modify|retire`, `- kind: lesson|rule`, `- target:`, `- decision: pending`, `### Text`, `### Reasoning`, `### Evidence` with file:line, and for a rule a ```python module + ```file: fixture blocks that reproduce the mistake). Only surprises: a gate that passed or blocked wrongly, a stall, a wrong premise, a harness-vs-LB mismatch. Zero items is a valid result: then write no file. Never edit LESSONS.md or research/rules/ yourself.

Never read README, LESSONS or EXPERIMENTS in full, never read a raw .ipynb, never ask a question (write it to questions.md), never submit. Need one specific fact that is not in the repo (a paper detail, a model's licence, whether pretrained weights exist)? Write `research/requests/<yyyymmdd>-<slug>.md` (`status: open`, the question, why it matters) and continue; a web-enabled fetch step answers it before your next step.
