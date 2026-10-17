# Handoff: K2 (c3-detector), campaign 20261015-c3-gen-slot2

Written 2026-10-17 for a fresh session. K1 passed. This step is K2, the detector that decides on which
molecules the novel candidate is inserted at slot 2. **Analyse first, then propose a plan to the human
before any GPU spend or build.** The user prefers step-by-step progress with simple explanations.

## Read first (in this order)
1. `research/harness/gate_report.md`, the last three blocks:
   - `2026-10-15: c3-gen-slot2 campaign, PRE-REGISTERED`: the K0–K4 text. K2 is the line beginning "K2 (c3-detector)".
   - The amendment "leak-free re-scorer": there's no FPNet in the novel channel, and FPNet-derived detector features
     need a drop-test.
   - `K0 MEASURED` (the L values) and `2026-10-17: K1 MEASURED` (h, splits, marginal).
2. `research/campaigns/20261015-c3-gen-slot2/charter.md`: the scope, the out-of-scope list, and the three human
   checkpoints. K2 ends at checkpoint 2.
3. `research/exploit/20261015-c3-gen-union/k1_result.md` + `k1_marginal.json`.
4. `build/c3_gate.py` in the code worktree: the old detector (stage-1 thresholds, `classify_v2`). It failed CC.3/CC.4
   (3.5% / 2.5% false fires on C2; `specs/track4_priors_probes/gate_report.md:77-84`) with absolute f·z signals. Reuse
   the ideas, not the thresholds.
5. Memory: `casmi-campaign-c3-gen-slot2.md`, `casmi-base-leaks-every-val-shard.md`, `casmi-ice-loss-is-fused-gl-branch.md`
   (the fused margin is not a class signal by itself).

## The K2 gate (pre-registered, do not change it)
- **Features:** relative or within-molecule only, never an absolute f·z threshold (L14). Candidates are lib_max, rel, S,
  top_pop, the fused margin, ICE/GL coverage, `gen_ice_gap` (the best generated ICEBERG cosine minus the best over
  the molecule's ENG/PC/BASE candidates) and the decoder log-prob.
- **Model:** logistic regression or a depth-2 GBDT, with 5-fold CV grouped by Murcko scaffold.
  - Positives: pseudo_c3 (600).
  - Negatives: the "other" molecules of C1_npx, C1_xlib, C2 and C2_hard. Exclude leak-hit molecules (L40).
- **Operating point:** τ is the smallest p with p·h_fired/2 > (1−p)·L_fired, on out-of-fold predictions.
  - h_fired and L_fired are measured on the fired set, not assumed.
  - τ is chosen once and never re-swept against the LB.
- **DEAD iff:** no τ meets it, OR at τ C1 fire > 1%, OR it fires on > 2% of the C2 ∪ C2_hard molecules whose E006
  truth is at ranks 2–5.
- **Sanity check (not a kill):** if the predicted fire rate on test.parquet is outside [10%, 40%], flag domain shift.
- **Leak rule:** FPNet-derived features (rel, fz_top, S) may enter, but report their pseudo_c3 vs test.parquet KS
  shift. If the detector fails K2 when they're dropped, it's LEAK-DEPENDENT and isn't shipped.

## What K1 established (numbers the detector builds on)
- union h 0.0767; **mutator alone h 0.120** (decoder leaves outscore true mutants on one log-p scale, so the union is
  worse at rank 1).
  - Decide which arm defines h_fired. The per-molecule ranked lists for `dec`, `mut` and `union` can be rebuilt from
    `E:/casmi/c3_gen/p4_raw.pkl` with `build/novel_p4_score.py::score`. Caution: tautomer-IK14 canonicalization of
    everything took **~4 h**, so cache it or only canonicalize the top few per arm.
  - `E:/casmi/c3_gen/novel.pkl` holds only the union top 25 `(smiles, ik14, score, src)`.
- Acyclic h 0.27 (n 129) vs ring-bearing 0.02 (n 471). Acyclicity and formula features may be strong detector inputs,
  and they're a per-molecule predicted property, not a leak.
- Off-BASE marginal: on the 512 molecules whose truth isn't in BASE, the union gets 37 top-1 hits and 94 top-25 hits.
- L (cost of a slot-2 insert on a non-C3 molecule, E006): C1_npx 0.058, C1_xlib 0.088, C2 0.048, C2_hard 0.048
  (small n). Break-even h_fired: about 12% at p 0.5, about 5% at p 0.7. With h ≈ 0.08–0.12 you need **p ≳ 0.6**.
- From K0 (not pre-registered): BASE#1 is a *generated* candidate (pid −1) on 138/600 pseudo_c3 vs ≈1% of C1/C2. It's a
  very strong class signal.
  - Check whether it's leaky: does Engine 1 `generate=True` use FPNet, which trained on the pseudo_c3 spectra? If it
    does, it falls under the drop-test.

## The open problems to analyse first (my read; verify)
1. **The generator features exist only for pseudo_c3.** The decoder log-prob and `gen_ice_gap` need the generator run on
   the negative cohorts too (C1_npx 245, C1_xlib 300, C2 400, C2_hard 200, or just their "other" subsets).
   - That's a new Colab/GPU job, and **the charter needs human approval for it.** K1 cost 45 CU on an A100 for 600
     molecules. Beam 256 adds nothing, so use beam 64 and skip dec256: about 12 s/mol on the A100.
   - It also needs `p4_inputs.pkl` for those cohorts (`build/novel_p4_inputs.py`, written for pseudo_c3; the mutator
     anchors need the dump's Engine 2 analog export, check that the C1/C2 dumps have it).
   - Cheap first step: **K2-v0 with dump-only features** (no generator features) to see how far the BASE/PC/ICE
     signals get. If v0 already meets the bar, the GPU run may be unnecessary. If it's far off, that's evidence the
     generator features are needed.
2. **`gen_ice_gap` needs ICEBERG spectra of generated candidates.** It wasn't computed in K1 (the ICE re-score was
   skipped). ICEBERG runs on Kaggle (msg_all weights; `casmi-simulator-checkpoint-licence.md`).
3. **The FPNet leak** on pseudo_c3: every f·z-derived feature is inflated on positives. Run the drop-test from the
   start, not at the end.
4. **Cohort shift:** pseudo_c3 queries are train.parquet rows (non-timsTOF mostly, 15 timsTOF), while test is 100%
   timsTOF. Instrument-specific features (peak counts, intensity shapes) could separate the cohorts by instrument, not
   by novelty. Prefer features that are invariant to instrument, and report the test.parquet fire rate.

## Where things are
- **Code worktree: `E:/casmi/lines/c3-commit`, branch `idea/20261015-c3-gen-slot2`** (tip `3fe7693`, pushed). Write
  detector code here (e.g. `build/c3_detector.py` + `build/test_c3_detector.py`).
  - **Never use `E:/casmi/lines/exploit`**: another agent owns it (idea/20261015-isdb-cfm-fallback; leave stash@{0}).
  - `research/harness/harness.py` defaults WORKTREE to exploit, so always pass
    `post=Path("E:/casmi/lines/c3-commit/build/casmi_post.py")` to `harness.load_dump`.
- **Main repo `E:/ENVEDA_casmi`, branch `specs-scratch-r0`** (tip `8f3608d`, pushed): gate_report, idea folders.
  - Create the idea folder `research/exploit/20261017-c3-detector/` (STATUS `specced`, `kind: offline`, hypothesis.md,
    gates.md) following `research/exploit/20261015-c3-gen-union/`. Valid STATUS values are listed in
    `research/README.md` § Idea STATUS. "measured" isn't one, so record gate results on a `k2:` line.
- **Dumps** (`research/harness/dumps/<cohort>/post_dump/state.pkl`; the ident cache is warm):
  C1_npx, C1_xlib, C2, C2_hard, C2_np, *_a6 variants, pseudo_c3.
  - `harness.load_dump(shard, post=...)` returns `(state, deps, cp)`.
  - `state` keys: base, pc, eng, ice, gl, te_mids, sample_ids, targets, consts, shard, rdkit.
  - `state['base'][mid]` = (smiles list, ik14 list, float [lib_max?, verify], score list, formula list).
  - `state['pc'][mid]` keys: pc, pc_fz, pc_keys, fz_top, S, top_pop, best_pool_fz, target, pc_form.
  - `state['eng'][mid]`: smiles, keys. `state['ice'][mid]`: {smiles: cosine}.
  - The leak-hit / "other" split: `harness.py` ~line 278 / 457 (`leak = BASE#1 ik14 == truth`).
- Truths: `harness.truths()` (pseudo_c3 from `E:/casmi/c3cal_check/pseudo_c3.parquet`).
- K1 artefacts: `E:/casmi/c3_gen/{p4_inputs.pkl, p4_raw.pkl, novel.pkl}`; Drive `casmi_2026/_novel/p4/`.

## Ops lessons from K1 (save time)
- **Long local jobs:** Claude Code's background shells get killed under memory pressure (it happened 3 times). Run
  anything over ~10 min as a detached `Start-Process python -ArgumentList "-u",... -RedirectStandardOutput <log>
  -WindowStyle Hidden`, and print progress inside the script.
- **Colab** (only if the human approves a generator run on the negatives):
  - Run `python research/bin/colab_run.py <idea> colab/<script>.py --kind val --gate K2 --gpu A100 --cu-cap N
    --cu-per-hour 12 --until <DONE_MARKER> --fetch <drive path>:<local>`. Launch it detached.
  - Drive consent is asked on every new session. The link lands in `research/queue/drive_consent.md`. The human ticks
    every box, then you create `research/queue/drive_consent.ok`.
  - **Never patch colab auth.py scopes** (Google blocks it). Don't run `colab update`. Stop sessions after use and check
    `colab sessions` / `colab usage`. Balance ≈ 156.8 CU.
  - The generation script template is `build/novel_p4_colab.py` (resumable chunks, `CASMI_TIME_BUDGET_S`). Build it with
    `build/mk_novel_p4_colab.py --py OUT`.
- Never read raw `.ipynb` JSON (use `.agents/skills/analyze-notebook/scripts/compare.py`).
- Never stage .pkl/.npz/.pt/.parquet/.zip. `colab_val.log` is gitignored. Run `git checkout` on modified
  build/__pycache__ pyc files. Concurrent agents edit the main repo: stage only your own files.
- Commit trailer: the Co-Authored-By / Claude-Session lines from the session's system reminder.

## Suggested order
1. Read the above. Check the leak status of the "BASE#1 is generated" feature and of Engine 1 `generate=True`.
2. Write the K2 feature extractor over the dumps (dump-only features), plus tests. Build the positive/negative table
   with scaffold groups.
3. Fit K2-v0 (logistic + depth-2 GBDT, grouped 5-fold). Compute τ, h_fired (pick the arm: union, or mut-first routing),
   L_fired, C1 fire, C2∪C2_hard rank-2–5 fire, the FPNet drop-test, and the test.parquet fire rate (needs the test
   features: check which dump or kernel provides them).
4. Report v0 to the human with a recommendation: ship v0 as K2 / run the generator on the negatives (CU estimate) /
   kill. Wait for the decision.
5. Record a "K2 MEASURED" block in gate_report.md, the idea gates.md + STATUS, and commit + push both repos. Then
   **checkpoint 2: the human reads K2 before K3** (`NOVEL_SLOT` knob in casmi_post.py, harness replay).
