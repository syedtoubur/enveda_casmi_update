# Track 4c — High-influence class vote on the zenith fused list

Deliverable doc: `specs/track4_priors_probes/np_class_sig_high_influence.md` (this plan, copied there as step 0).

## Context
- **Validation of `iceberg_bug_comparison.md` (done 10-06).** Its claim that ICEBERG crashed silently in supernext is
  **false for our runs**. Our saved supernext log shows the Python 3.12 image, ICE 11/11 and GLACIER 11/11 `status ok`.
  The crash only happens on the Python 3.13 image used by the zenith author's b415–b417 runs.
  - Zenith = our v29 code (0.393 on our account), plus four changes: base-list ICE+GL rerank off (`BASE_ICE=False`),
    ICE off in the fused rerank (`CELL19_ICE=False`), the promotion rule, and the cp313 preseed.
  - Its DreaMS tail is dead code (`name 'Path' is not defined`).
  - Building on supernext was not a mistake. Zenith is the base now because we are shipping on it.
- **Why Track 4b had no reach.** nplike and classcons only re-sorted the PubChem-only list, which feeds a few merge
  slots and never rank 1. Realism check R.1 also showed the old class term carried no information beyond "the candidate
  is a labelled NP".
- **Goal.** Ship fast with LB-first validation, while specs_scratch builds its pool. One signal: a **spectral class vote**
  (the specs_scratch R2 idea) applied to zenith's **fused list**, where it can reach rank 1.
  - It must not break zenith: library matches are untouched, failures are a no-op, and the GLACIER rerank still runs after it.
- **Complementary to specs_scratch.** Scratch tests coverage (R1 pool) and measures class-vote purity (CV gate) in our
  own pipeline. Track 4c tests the class vote as a ranking term in the borrowed stack.
  - Shared code is scratch's `analog.py` and the `np_labels` kNN recipe.
  - Track 4c does not touch pool expansion. Scratch does not touch PubChem or popularity.

## Decisions (grilling 10-06)
| # | Decision |
|---|---|
| Base | `enveda-casmi-2026-v18-sovereign-zenith.ipynb`, unchanged otherwise (DreaMS bug left as is). Control = the user's one zenith LB run (Z0) |
| Injection | Fused list: after `fuse2`, before the GLACIER rerank: `score = z(RRF) + μ·term`, re-sort, then zenith's GL rerank as before |
| Voters | All train-library spectra; a hit votes only if its structure has a label |
| Labels | Exact NP-table scorer_key match, else runtime kNN to the NP-table bank (Morgan r2 2048, Tanimoto ≥ 0.7), cached per key; else none (abstain). Same labels for fused candidates |
| Search | Delta list ±10 ppm (query neutral mass ± 13 NP biosynthetic deltas, both signs), shift-aware entropy (scratch `analog.candidates/analog_hits`). Top 10 hits with sim ≥ 0.5, similarity-weighted superclass vote (`analog.class_vote`) |
| Term | Signed agreement, with s = the vote share of the candidate's superclass. If that label got votes: `+s`. If it is labelled with an unvoted class: `−(winner share)`. Unlabelled: `0` |
| Fire gate | `lib_max < 0.9` **and** ≥ 2 labelled voters **and** winner share ≥ 0.5. Otherwise zenith's order is untouched |
| μ | Two probes: μ 0.5 and μ 1.0 |
| NP-likeness | Out of scope (later, if the vote pays) |
| Reach proof | Dry-reach diagnostic in commit mode. The vote is run with the lib_max gate ignored, and counters are printed **without applying** it. Real counters go to `run_manifest.json` |
| Build | New `build/patch_zenith.py` (the `must_replace` pattern of `patch_supernext.py`). It embeds the source of scratch `analog.py` plus the kNN labeller into the notebook; no new dataset. Tests in `build/test_patch_zenith.py` |
| Keep bar | vs Z0 (single control): keep the better μ iff ≥ Z0 + 0.004 **and** its dry-reach counters are non-zero. A pass is a candidate new base, not proof (single control on the noise floor) |

## Steps
0. **Docs and pre-registration (before any run).**
   - Copy this plan to `specs/track4_priors_probes/np_class_sig_high_influence.md`.
   - Append a "Track 4c" register to `specs/track4_priors_probes/gate_report.md`: the keep bar, fire gate, μ grid and
     dry-reach counters above.
   - Add a row to `specs/README.md`.
   - Add a note to `iceberg_bug_comparison.md` saying its supernext-crash claim is refuted by our own log, with the evidence lines above.
1. **Adapter probe.** Zenith's library object is `L = Library('work')` from `casmi26-v4b-models` (no local copy).
   - Scratch `analog_hits` needs `off, mz, it, nm, ik, pol` and `window(target, tol)`.
   - Inspect the v4b `casmi/library.py` (pull the dataset code with the Kaggle CLI, read-only) and write a thin adapter.
     Note that `del V, E, bank` in the ICE cell must not free `L`.
   - Fallback: read the `work/` spec cache arrays that `build_spec_cache` writes.
2. **`build/patch_zenith.py`.**
   - Source: the zenith notebook (read only). Outputs: `casmi26-t4c-classvote{μ}.ipynb`, with cleared outputs.
   - Fusion cell: insert a `CLASS_VOTE` block before `if ENG:`, holding the embedded analog kernels, the labeller and the vote per molecule.
     - It reads test spectra from `te`/`test.parquet` and uses the 1.45 ppm centring of the precursor.
     - Polarity follows the spectrum.
   - Hook after `fsm, fk, fsc = fuse2(...)`: if the gate passes, apply `z(fsc) + μ·term`, re-sort fsm/fk/fsc, and pass
     the new scores into the existing `ice_fuse.rerank` / `gl_fuse.rerank_multi` calls.
   - Everything is wrapped in try/except with a no-op notice, and honours `KAGGLE_IS_COMPETITION_RERUN`.
   - Counters (printed and added to `run_manifest.json`): molecules searched, with ≥ 1 hit, voted, fired, top-1 changed,
     mean/p99 |term|, labelled-voter share, kNN cache size, seconds.
   - Commit mode also prints the dry-reach line (gate ignored, not applied).
3. **Tests** `build/test_patch_zenith.py`, wired into `build/run_all_tests.sh`.
   - Every anchor hits exactly once.
   - `compare.py` delta = the fusion cell only.
   - The term follows sign rules on a toy vote: agree, disagree, unlabelled, gate off.
   - A μ=0 variant produces an order identical to zenith.
   - The embedded `analog.py` source equals the scratch file (drift check).
4. **Kaggle.**
   - Commit runs for μ 0.5 and μ 1.0 (attach `natural_products.parquet`).
   - Check the dry-reach counters are > 0 and the runtime delta fits 9 h.
   - Submit both. Log with `/update-experiments`.

## Critical files
- `enveda-casmi-2026-v18-sovereign-zenith.ipynb` (base; fusion cell with `fuse2`/`CELL19_ICE`)
- `build/patch_supernext.py` (pattern, and the `_nlab` kNN code to reuse), `build/test_patch_supernext.py`
- `scratch_pkg/casmi_scratch/analog.py` (`candidates`, `analog_hits`, `class_vote`, `NP_DELTAS`), `np_labels.py` (`KNN_THR`)
- `specs/track4_priors_probes/gate_report.md`, `EXPERIMENTS.md`, `specs/README.md`

## Verification
- `sh build/run_all_tests.sh` passes. `python .agents/skills/analyze-notebook/scripts/compare.py` shows the zenith → probe
  diff only in the fusion cell.
- In the Kaggle commit log:
  - `CLAW STAGE SCORED ice=11 gl=11` is unchanged.
  - The class-vote block shows no FAILED line.
  - Dry-reach: voted > 0 and top-1 changed > 0.
  - `submission.csv` is (400, 2) with ≤ 25 unique ik14 per row.
- On the LB, apply the keep bar vs Z0. Feed the result back to specs_scratch as evidence for R2 alongside its CV purity gate.

## Status (2026-10-06): built and tested, not yet run on Kaggle
- Steps 0–3 done.
  - Register T4C-* added to `gate_report.md`.
  - Row added to `specs/README.md`.
  - Correction note added at the top of `iceberg_bug_comparison.md`.
- `build/patch_zenith.py --classvote μ` → `casmi26-t4c-classvote0.5.ipynb`, `casmi26-t4c-classvote1.ipynb`.
  `compare.py` shows that only the fusion cell (C10) differs from zenith.
- `build/test_patch_zenith.py`: 30 checks, wired into `run_all_tests.sh` (suite green apart from the known `test_stage_d_colab`). It covers:
  - the delta being one cell, with the hook after `fuse2` and before the GLACIER rerank
  - NP_DELTAS drift against scratch
  - the vote, term and sign rules
  - an end-to-end toy library: fire, library-match gate, μ 0, dry reach, a single voter not firing, missing NP table, and hook failure
- **Deviation from step 1/2.** The search uses zenith's own in-memory library `L` (`casmi.library.Library`) and its
  shift-aware kernel `casmi.spectra.search_shift` (`hybrid_sim`: a peak matches directly or shifted by Δneutral mass).
  It does not embed scratch's `analog.py` kernels.
  - Same idea, no second copy of 2.5M spectra, and it is already numba-compiled in the notebook.
  - Shared with scratch: `NP_DELTAS` (embedded, drift-tested) and the vote rule.
  - Voters are library spectra whose structure is known (`sid ≥ 0`), with the same polarity as the query spectrum.
    The target is centred by −1.45 ppm.
- **Next (user).** Attach the NP knowledge table dataset (`natural_products.parquet`) to both notebooks, and run the commit.
  - Check the commit log: no `TRACK4c classvote FAILED`; the `TRACK4c dry-reach` line shows voted > 0 and top1_changed > 0;
    `CLAW STAGE SCORED ice=11 gl=11`; and `secs` in the `TRACK4c classvote` line (the budget is 1800 s).
  - Then submit both, and give me Z0 and the two LB scores to log with `/update-experiments`.
