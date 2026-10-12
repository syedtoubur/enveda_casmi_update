# Smart Library Gate → `casmi26-e006-window-40-ice-fallback-true.ipynb` (final plan)

## Context
E006 (LB ~0.440) protects a molecule from PubChem insertion only when the engine's `lib_max ≥ 0.9`.
`.scratch/handoffs/smart_gate.md` proposes a 4-tier library gate. The goal is to protect cross-instrument
Class-1 hits that score below 0.9 and to stop PubChem and the simulators from displacing them.
I audited that blueprint against the real notebook code (`.scratch/nb2_e006.py`), the `casmi` package
(`.scratch/casmi_v4/casmi/`) and all three eval runs (`.scratch/kaggle_libmax_eval/output{,_v2,_v3}`).
As written, the blueprint would either crash the hidden rerun or silently remove the existing 0.9 protection.
This plan fixes those defects and keeps all three tiers active, as you chose. All new files go in `.scratch/smart_gate/`
and nothing touches `research/`.

## A. Defects found in the handoff blueprint

### Silent errors (wrong results, no crash)
1. **`L.window(target, tol_da)` takes ppm, not Da** (`library.py:57`: `tol = target*ppm*1e-6`). Passing
   `target*10e-6` (~0.004) gives a 0.004 ppm window, which is empty for almost every molecule. Every
   molecule then lands in Tier 4.
2. **The gate replaces the 0.9 rule instead of extending it.** Step 4 deletes `elif lib_max >= LIB_TAU`. Combined
   with defect 1, every molecule the old code left untouched would get PubChem slots and promotion. That is a
   regression on the strongest Class-1 hits, and nothing would be printed.
3. **Stale `rel` in Cell 22.** The Tier-3 branch never assigns `rel` but then tests `rel > 250.0`. At notebook
   top level, `rel` is the previous molecule's value, so promotion is decided from another molecule's data.
   If the first molecule is Tier 3, this is a NameError instead.
4. **Metric mismatch.** The blueprint code uses Phase-2 sqrt-cosine, but the calibrated operating curve is
   Phase-3 masked entropy with N≥4 and high-m/z filters. Tier cut-offs only mean what they meant in the
   eval if the gate uses the eval's own scoring function, byte for byte.
5. **Pin identity.** `L.ik` does not exist. `L.struct_key` is tautomer-canonical but not stereo-stripped
   first, unlike `metric_identity`. The pin's fallback `lib_hit = candidates[0]` is the already-reranked top,
   so an identity mismatch turns the lock into a silent no-op.
6. **Tiers 1 and 2 skip the filters.** Tier 1 has no fragment-count filter and Tier 2 needs only N≥2. That
   admits exactly the "one/two-peak wonder" decoys from the Phase-2 autopsy. Those tiers were never measured.
7. **Precursor mass and polarity.** Masking with `sub.precursor_mz.median()` across mixed adducts masks the
   wrong peak. The window is not filtered by mode or adduct, so positive queries are scored against negative
   spectra. Rows with `L.sid < 0` map to a bogus mass.

### Crashes
8. **`sp.mzs` / `sp.ints` do not exist.** Cell 18 spectra are dicts (`mz`, `it`, `prec`, `mode`, `adduct`).
   Placed inside the engine `try`, the error empties every BASE list and increments `n_err`, so Cell 27's
   `assert n_err == 0` fails.
9. **The Step-5 pin inside `rerank_row` breaks E006's control reconstruction.** `run_overlay` calls
   `rerank_row` twice and asserts that the first call reproduces the control byte for byte
   (`nb2_e006.py:1252`). A pin applied in both calls fails that assert, and the hidden rerun dies.

### Spec gaps
10. "Tier 3 keeps the hit in the top 2" and "dampen simulators" are not implemented by any step.
11. The claimed runtime of under 2 s and the gain of +0.008 to +0.015 are unsupported. LB noise is about ±0.004.

### Calibration limits of Phase 3 (accepted, but constrained in the design)
- The eval used **one query spectrum per molecule**. The notebook takes the max over 3–5 spectra, so the
  false-lock rate (FLR) scales by roughly ×k. At 0.40 that is about 0.58% → 2–3%.
- Windows were on precursor m/z (effectively same adduct) and contained **non-timsTOF libraries only**.
  The notebook's library also holds enveda-180 (1.15M timsTOF spectra) and np-examples, which are uncalibrated decoys.
- The negatives were mostly enveda-180 synthetic compounds. Plant-NP isomer families (glycoside positional
  isomers and similar) produce near-identical spectra, so their real FLR is likely higher. The design uses a
  rival-margin rule to cover this.
- The 15% Class-1 prior used in the "precision" column is assumed, not measured.

## B. Final gate design

**Score.** `S` = Phase-3 masked spectral entropy. `clean_and_prep_spectrum` and `compare_spectra` are copied
verbatim from eval cell 5, with `mask_precursor=True` and the query's own precursor m/z used for both sides,
exactly as in the eval.

**Reference window**, per query spectrum `q`:
`L.window(target*(1-1.45e-6), 10.0, mode=q.mode)`, keeping only rows with `L.sid ≥ 0`,
`L.adduct_ix == adduct_index(q.adduct)`, and `L.lib ∉ {enveda-180, enveda-np-examples}`.
This matches the calibration domain. Intra-instrument (np-examples) hits stay covered by the unchanged legacy `lib_max ≥ 0.9` rule.

**Per structure (sid):** `S_sid` = max over its reference spectra and query spectra, counting only pairs that pass
`n_matched ≥ 4` and `has_high_mz` (the filtered "Smart" curve). `support_sid` = the number of query spectra on which that
structure passed the filters with S ≥ 0.40.

**Tiers**, applied to the best structure:
| tier | condition | Cell 20 (ICE PubChem cands) | Cell 22 (PubChem merge) | final overlay |
|---|---|---|---|---|
| T0 legacy | `lib_max ≥ 0.9` and not T1/T2 | skipped (as today) | untouched (as today) | unchanged |
| T1 | S ≥ 0.85 | skipped | untouched | pin lib structure at rank 1 (insert if absent) |
| T2 | 0.65 ≤ S < 0.85 | skipped | untouched | pin at rank 1 |
| T3 | 0.40 ≤ S < 0.65, support ≥ min(2, n_query_spectra) | kept | slots [15,18,21], **no promotion**; if lib_max ≥ 0.9 → T0 wins | ensure lib structure rank ≤ 2 |
| T4 | otherwise | today's logic | today's logic, byte-identical | unchanged |

**Rival margin.** If the best rival structure (a different `metric_identity`) is within 0.05 of S, demote one tier:
T1/T2 → T3, and T3 → T4. Locking one of two near-identical isomers is a coin flip. The 0.05 value is uncalibrated, so the run logs it.

**Superset property.** No molecule that today's code leaves untouched ever gets PubChem slots. T4 output is
identical to today's.

**Fail-safe.** The gate cell runs outside the engine `try`, with its own `try`, a 900 s budget, and per-molecule
time checks. If it fails or runs out of time, `SMART_GATES` is empty for the remaining molecules and they keep legacy behaviour.
`SG_HEALTH` is printed and written into `run_manifest.json`. The gate never touches `n_err`.

**Pin/lock in the overlay.** `run_overlay` applies it to the **treatment** result only, after
`result, audit = rerank_row(..., window=window, ice_fallback=ice_fallback, ...)`. The control-reconstruction call
stays unchanged. The lock matches by `metric_identity(L.struct_smiles[sid])`, inserts that SMILES if the structure
is absent, and then runs `safe_extend` to dedupe and cap the list at 25. Counters: `sg_pinned_moved`, `sg_inserted`,
`sg_top2_moved`, `sg_identity_fail`.

## C. Files (all new, under `.scratch/smart_gate/`)
- `smart_gate.py` is the single source of truth: `prep`/`compare` (verbatim from Phase 3), `evaluate_molecule(L, spectra, target, lib_ix_excl, adduct_index)`,
  `decide_tier`, `merge_decision(...)` (rel computed before branching), and `apply_lock(result, dec, identity, safe_extend)`.
  It is pure: no notebook globals and no RDKit at import time.
- `test_smart_gate.py` (pytest) covers:
  1. Parity: exec eval cell 5 from the eval ipynb (parsed via json, code only) and check that `compare` matches on 200 random spectrum pairs.
  2. `L.window` is called with ppm, checked with a fake Library that records its args.
  3. Tier boundaries, filters, the support rule and margin demotion.
  4. `merge_decision` reproduces the old Cell-22 loop exactly on T0 and T4 for randomized inputs (superset property), and `rel` is defined in every branch.
  5. `apply_lock`: pin, insert-when-absent, T3 rank ≤ 2, no duplicate identities, length ≤ 25.
  6. Dict-spectra input shape (`mz`/`it`/`prec`/`mode`/`adduct`).
- `build_nb.py` loads the base ipynb as JSON. It patches by **unique anchor strings** and asserts each matches exactly once:
  - Insert a new cell after Cell 18 (anchor `BASE_TAIL_CANDIDATES = {`). It embeds the `smart_gate.py` text and computes `SMART_GATES` with `SG_HEALTH` and timings. `L`, `te` and `mols` are still alive there; `del V, E, bank` comes later and does not delete `L`.
  - Cell 20, anchor `BASE[m][2] < LIB_TAU:` → `... < LIB_TAU and not sg_locked(m):` (T1/T2 only).
  - Cell 22: replace the `for mid in te.molecule_id.unique():` loop body with a `merge_decision` call. Keep the `stats` keys and add `t0, t1, t2, t3, sg_margin_demoted`. Add `smart_gate=` to the `run_manifest.json` dump.
  - Cell 30, `run_overlay`: add the treatment-only `apply_lock` after the treatment `rerank_row` call and add the SG counters to `report`.
  - Clear outputs, then write `casmi26-e006-smartgate.ipynb`, and run `python .agents/skills/analyze-notebook/scripts/compare.py` base vs new as a semantic-diff check.
- `gate_report.md`: pre-registered bars, written before any run (see E).
- `kernel-metadata.json`: copied from the E006 kernel (`kaggle kernels pull <e006 slug> -m`) with a new slug, the same 17 sources, GPU T4 and internet off.

## D. Verification
1. Local: `pytest .scratch/smart_gate -q` passes. `build_nb.py` patches with every anchor at count 1, and each code cell `compile()`s.
2. Kaggle visible commit (12-molecule smoke): the notebook completes and prints `SG_HEALTH ok`, the gate seconds,
   `merge stats` with the tier counts, `FACTORIAL_VERIFICATION` (meaning control reconstruction passed), and `E004_VERIFICATION`.
   Smoke molecules come from train, so only plumbing is checked here. Tier counts from smoke mean nothing; see the memory note on placeholder runs.
3. The hidden rerun and the LB submission are done by you. Read `merge stats` (t1/t2/t3 counts), `sg_pinned_moved`, and
   `sg_inserted` from the run log, plus the gate seconds, which must stay ≤ 900.

## E. Pre-registered bars (`gate_report.md`, written before the push)
- Runtime: gate cell ≤ 900 s on the hidden rerun, and the total notebook stays inside the 9 h limit.
- Mechanism alive: on hidden, t1+t2+t3 ≥ 10 and sg_pinned_moved + sg_inserted + sg_top2_moved ≥ 1. If not, any LB delta is noise by construction.
- Keep: LB ≥ E006's own score + 0.005, compared against our E006 run, not the author's headline.
  Between −0.004 and +0.005 counts as neutral, so do not stack on it. Below −0.004 is a regression. Then T3 is the first
  suspect, because it has the highest uncalibrated FLR. The follow-up arm is T3 audit-only.
- Optional afterwards: log the result with `/update-experiments`.
