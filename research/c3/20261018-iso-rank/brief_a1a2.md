# Brief: iso-rank A1 + A2 (common prerequisite), Claude Code Sonnet builder

Read first: `research/c3/20261018-iso-rank/spec.md` (the contract) and the gate block
`### 2026-10-18: c3-iso iso-rank, PRE-REGISTERED` at the end of `research/harness/gate_report.md`. Do not change either.

## Where
- Worktree `E:/casmi/lines/c3-iso-rank`, branch `idea/20261018-iso-rank` (off 9c1ef2b; it contains `build/isomer_mut.py`,
  `build/frag_rescore.py`, `build/iceberg_worker.py`, `build/casmi_post.py`, `build/test_casmi_post.py`). Write code only there.
  Commit there when the tests pass (message ends with the Co-Authored-By/Claude-Session lines from your system prompt).
- Read-only sources: `E:/ENVEDA_casmi/research/antigravity-exploit/nl-sim-rda/nl_stoich_rda.py` (`score_molecule`,
  `rda_fragments`, `_match_spectrum_nl`), `E:/casmi/lines/c3/build/patch_gen1_keep.py` (a model of a last-step
  notebook patch, for the later ship agent only — not your job).

## Build
1. **A1 `build/iso_nl.py`**: a self-contained copy of `nl_stoich_rda.py`'s scorer (keep its maths and tolerances unchanged;
   strip anything that reads files or truths). API: `nl_scores(spectra, cand_smiles) -> np.ndarray` (one float per
   candidate, higher is better), `spectra` in the same format `frag_rescore.py` takes. Measured spectrum only (R10).
2. **A2 `build/iso_rank.py`**: pure functions, no I/O, no truth.
   - `parent_map(children)`: children = list of dicts `{smiles, ik14, parent_ik14, parent_rank, op}` (one per generated
     edge) → per-IK14 record `{smiles, ik14, parent_ik14 (best-ranked parent), parent_rank, multiplicity = number of
     distinct (parent_ik14, op)}`.
   - `rrf(records, scores, k=10)`: `scores` = `{list_name: {ik14: value}}` with a direction per list (ice, frag, nl, gl,
     mult higher-better; parent_rank lower-better); average ranks on ties; a list missing for an ik14 is skipped for it.
     Returns ik14s sorted by RRF desc (stable tie-break by ik14).
   - `apply(row, rec, cfg) -> (new_row, info)`: `row` = the final top-25 list of IK14/SMILES entries (the format
     `casmi_post.run` returns per molecule); `rec` = `{children, scores, parent_scores: {ik14: {ice, frag}}}`. It
     implements the fire rule, placement (directly below the parent's position in `row`, never rank 1, no fire if the
     parent is not in `row`), IK14 dedupe, cut at 25, and the tail fill of ranks 21–25 with the next RRF children not in the
     row. cfg keys: `ISO_M_ICE` (default 0.0), `ISO_RRF_K` (10), `ISO_LISTS` (default all six names), `ISO_TAIL` (True).
     `info` = `{fired: bool, tail: int}`.
3. **Knob in `build/casmi_post.py`**: `ISO_MUT=False` in CFG plus the cfg keys above. When on and `state.get('iso')`
   (`{molecule_id: rec}`) exists, apply `iso_rank.apply` as the **last** step on every row (after E2R2 / final pass).
   stats: `iso_children` (molecules with ≥ 1 child), `iso_fired`, `iso_tail`. With `ISO_MUT=False` the output must be
   byte-identical to today's (existing tests unchanged and passing).

## Tests (`build/test_iso_rank.py`, `build/test_iso_nl.py`; pytest, small fixtures, < 2 min total)
- Real SMILES only (e.g. salicylic acid vs 4-hydroxybenzoic acid; hand-built child sets from `isomer_mut.mutate` on 2–3
  real parents). Assert non-empty rows (lesson: toy SMILES are dropped silently, so the tests become vacuous).
- Cases: fire / no fire (ICE lower, frag lower, ICE missing, parent absent); placement below a parent at rank 1 → rank 2;
  dedupe; the cut at 25; the tail; RRF with missing lists and ties; parent_map multiplicity; `ISO_MUT=False` parity
  on the existing casmi_post fixtures; `iso_nl` scores differ between two regioisomers on a hand-made spectrum with a
  diagnostic loss.
- Truth-blind audit test: grep-style assertion that `iso_rank.py`/`iso_nl.py` never reference `truth`, `ik14_true`,
  `inchikey` columns of a key file, or `val_key`.

## Rules
- Heavy compute is Kaggle-only (CLAUDE.md §3): no harness dumps, no multiprocessing pools, nothing over ~2 min locally.
- No FPNet anywhere. Do not edit files outside the worktree, except to write `research/c3/20261018-iso-rank/a1a2_result.md`
  (what you built, the test output summary, the commit hash, and open questions).
- Never push Kaggle kernels or submit.
