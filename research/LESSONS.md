# LESSONS: the Critic and Judge checklist

One line per lesson, plus its source. Cite lessons by ID (`L07`). This file is append-only: only the Reporter appends to it, after an LB result or a gate failure.
Sources: `EXPERIMENTS.md` rows (`E#21`) and specs files.

## Leakage and data boundaries
- **L01** Ranker training queries must include structures the FPNet never trained on. Excluding the eval holdout is not enough: the FPNet becomes an oracle on seen queries, the GBDT drops the other channels, and the LB falls (0.337 → 0.287). Src: E#04.
- **L02** Never ship a new FPNet without regenerating `rank_train.npz`: 12 of the 31 ranker features are FPNet logits. A *better* FPNet under an old ranker scores *worse*. Src: E#04, E#16.
- **L03** Stage-C/training queries must be retrievable by themselves: the window comes from the reported m/z (riken's 2 dp ≈ 12 ppm), so check that the truth is *in the window*, not only *in the pool*.
- **L04** Saved-version outputs on Kaggle are placeholder runs: `test.parquet` is a slice of train. `lib_max`, `rel`, merge stats and timings from a saved run mean nothing about the hidden test. Src: E#22–26 analysis.
- **L19** MassSpecGym-trained simulators (ICEBERG and GLACIER, MIT) are the only legal checkpoints, and their training data overlaps the val cohorts (gnps/riken/mona). Treat simulator gains on val as possibly leaked.
- **L20** FPNet families: P (own/prvsiyan, 6,930 bits) vs A (Ahmed/full1, 10,226 bits). Both trained on C1_npx, so audit FP-based changes on C2/C2_hard only. A wrong bit set gives noise with no error message.

## Measurement and decision rules
- **L05** The LB noise floor for identical code is ±0.004. KEEP only if LB ≥ the **paired own-account control** + 0.004 *and* the change's counters show it fired. Src: E#31, E#35.
- **L06** The base is our own account's score, never the author's headline. The "0.415" fork really scored 0.393 from our account (private vs public popularity arrays). Src: E#35.
- **L07** Offline val is blind to pool misses (truth is in the pool by construction). Read val only as a *relative* delta between configs on the same cohort, never as an LB forecast. Src: E#09, E#10.
- **L08** Tuning on local val overfits: −0.015 LB. Never tune hyperparameters to val MRR. Src: E#10.
- **L09** Two unrelated probes landing on the same score point to a shared base or dataset cause; VOID them and run a base control first. Src: E#32, E#33.
- **L13** Compute the arithmetic ceiling before shipping: (affected molecules × best rank gain) / 400. The mutator's best case was +0.0007, invisible on a 3-dp LB. Src: E#22, E#26.
- **L14** Calibrate gate thresholds on the full distribution, not on 12 fast-commit molecules: a hard gate hijacking ranks 1–5 cost −0.064. Src: E#25.
- **L15** Don't bundle plumbing changes (budgets, timeouts) with the idea; the result becomes unattributable. Src: E#34.
- **L16** Submit with the T4 GPU on, and check `device: cuda` in the ICE meta. On CPU, ICEBERG covered 90/400 molecules within its budget. Src: E#30 note.

## What moves the score
- **L10** Gains come from **orthogonal** sources or scorers fused by RRF (BIO pool +0.009, GLACIER +0.007, second engine RRF +0.016, union scoring +0.004). Tuning, deepening or swapping one view gives ≈ 0 or less (TOPN 60→100: 0.000; FPNet swap −0.002; λ 1→2: −0.034). Src: E#13–21.
- **L11** Splitting one model's output (random FPNet bit subspaces, a second FPNet of the same role) adds noise, not signal: f·z already is the likelihood. R4 family fusion was neutral. Src: E#50.
- **L12** The popularity prior is a tie-breaker *inside* the spectral top-N; using it to choose which candidates enter the window pulls in popular decoys. Total popularity voice can overshoot when stacked across lists. Src: E#29, E#30.
- **L17** Same-formula generated candidates collide with the truth's formula group in the forward rerank; tag them (`'|gen'`) so they get their own groups. Src: E#23, E#24.
- **L18** Class 3 (novel) has a 0% ceiling for retrieval-only ideas; mutator approaches hit a 53% unique-scaffold wall. Explore line: Class 3 is parked until the line reaches ≈ 0.400. Src: E#26, specs_scratch.

## Engineering hard rules
- **L21** timsTOF precursor offset: centre the query mass at +1.45 ppm (`CFG.PPM_CENTER`).
- **L22** Score on tautomer-canonical InChIKey14 with RDKit 2026.03.3; never emit duplicate IK14s in the top 25.
- **L23** Kaggle image is cp313: install RDKit and ICEBERG wheels **by cp tag**, never `whl[0]`. A mismatched wheel makes ICEBERG fail silently ("ICE FAILED -> ranker order kept"). Src: zenith incident.
- **L24** Inference ≤ 9 h on one T4, fully offline (`--no-index`). Fast-commit: 10–20 molecules on draft commits, all molecules when `KAGGLE_IS_COMPETITION_RERUN`. Src: E#03 (fast mode leaked into the full run: 0.032).
- **L25** The feature schema is append-only: columns 0–30 are frozen; inference truncates with `[:, :NFEAT]`.
- **L26** Pretrained fingerprint models (NPClassifier) need the RDKit they were trained with (≤ 2021.03). Add known-label structures to every spike.
- **L27** Long Colab jobs checkpoint to Drive every epoch or chunk, including all RNG states, and resume rather than restart.
- **L28** Never stage `.npz/.pt/.parquet/.zip` files. Never read raw `.ipynb`; use `compare.py`.
- **L29** Another agent may edit the same repo: work in the line worktree, write intermediates to your own idea folder, and re-check mtimes before you report numbers.

## Appended by the Reporter (newest last)
- **L30** A catalog preset is not proof that a notebook's data is mounted. The 0.433 preset listed 10 of 16 sources (one with the wrong owner), so control v1/v2 died at `ENGINE FAILED: sim_rank_rows_nofp.npz`. `kaggle_push.py` now audits every `find()`/`glob` file against the pushed sources and fails a push Kaggle answers with "not valid dataset sources". Src: research/exploit/20261010-control-0433/gates.md.
- **L31** A line's control must reach G4 PASS before any probe on that line is pushed: a broken base kills the probe too and wastes a cycle. `kaggle_push.py` refuses probe pushes until then. Src: 20261010-adduct-iceberg-rerank.
- **L32** In the 0.433 base, ICEBERG already re-ranks the PubChem-only list of every low-`lib_max` molecule for every adduct; it is off only on the base and fused lists, and GLACIER sees only `[M+H]+` (BASE.md §2). "Zero physical modelling" premises must cite BASE.md. Src: 20261010-adduct-iceberg-rerank/verdict.md.
- **L33** A headless Claude session must never wait in the background: its background jobs die when the session ends. Kaggle waits run outside Claude via `kaggle_push.py --check`. Src: 20261010-adduct-iceberg-rerank/claude_run.jsonl.
- **L34** A wait needs an owner process. A state that only a human-run command can leave will stall: G4 sat at `g4-pushed` until `--check` was run by hand. The push now starts a detached watcher (`kaggle_push.py --watch`) that finalizes G4 and hands a FAIL to a fresh fix agent (at most 2 rounds, then `needs-human`). Src: research/exploit/20261010-control-0433/gates.md.
