# specs_scratch R1 — unified candidate pool (grilling outcome, 2026-10-05)

## Context
R0 (port of the Track 3 0.296 notebook) retrieves from COCONUT + train built at runtime (`scratch_pkg/casmi_scratch/pool.py:45`
`build_r0_pool`). Track 4b proved the bottleneck for pool-missing NPs is **coverage, not ranking**, and that a lever's
**reach** must be estimated before trusting an offline gain. R1 adds coverage only: one deduplicated pool built once on
Colab and published as our own Kaggle dataset. It carries NP metadata for later rungs, which R1 itself does not use.

Prerequisite: R0 has passed its LB gate (R0.4 ≥ 0.292). If it hasn't, fix the port first.

## Decisions (settled)
| # | Decision |
|---|---|
| Sources | **Union** of v2-pool (710,701) + R0's coco_meta+train + BIO (ChEBI/LIPID MAPS `bio_*`) + NP knowledge table rows. **No PubChem tier, no popularity columns** in R1 (PubChem is a later rung; NP flags exist for that). |
| Key | Recompute RDKit 2026.03.3 tautomer-canonical ik14 (scorer_key) for **every** row. Log disagreements with source ik14. |
| Hygiene | Drop rows RDKit can't parse or that give no key. Otherwise keep SMILES **verbatim** (no salt stripping). Log multi-fragment/charged counts. |
| Dedup winner | First SMILES in priority train > v2-pool/COCONUT > BIO > NP table. Mass recomputed with `ExactMolWt` from that SMILES. Provenance bits OR'd. |
| Columns | mass, key, smiles, provenance bitmask (per train library incl. **enveda-180-only**, COCONUT, v2, R0-coco, BIO, NP-table, LOTUS, NPAtlas), NP labels (pathway/superclass/class, `label_source` ∈ {table, knn, none}), NP-table metadata (kingdoms, n_organisms), Ertl NP_Score (f16). |
| Labels | Exact scorer_key match to NP table, else kNN to nearest labelled LOTUS/NPAtlas structure (Morgan r2 2048, Tanimoto ≥ 0.7, Track 4b recipe). Unlabelled = neutral. Chunked and resumable on Colab. |
| Layout | Mass-sorted `pool_mass.npy` (f64), `pool_key.npy` (S14), `pool_smiles.parquet`, `pool_flags.npy` (bitmask), `pool_np.parquet` (labels, metadata, NP_Score), `MANIFEST.json` (counts per source, overlaps, set differences v2 vs R0-coco, RDKit version, sha1s). |
| Sim priority | **Unchanged from R0** (lv → fr → \|ppm\|). The NP flag has no effect inside an all-NP pool. R1 is a pure coverage test. R0's time-budget mechanism (`sim.rows_that_fit`) sets how many requests are simulated, so K is measured on a T4. |
| Probes | **R1a** = union pool. **R1b** = R1a minus enveda-180-only structures (in no NP source), a config switch on the same dataset. Same day, with a paired R0 control. Library spectra are untouched in both. |
| Class vote | **R2 candidate**, not built into ranking now. A pre-registered diagnostic runs with the coverage audit (below). |

## Pre-registered gates (write into `specs/specs_scratch/gate_report.md` before measuring)
- **R1.1** No loss: every R0 pool key ∈ R1a pool (exact superset, 0 lost).
- **R1.2** Truth-in-window recall on slice I (245 enveda-np-examples molecules, 1.45 ppm centring, R0 `ppm_win`): R1a ≥ R0. Report the gain, and R1b's recall too.
- **R1.3** Median and p95 window-size growth reported. The R1a run fits 7.5 h on a T4. Truth-in-simulated rate logged.
- **LB** Keep R1a / R1b iff ≥ paired-control mean + 0.004 and its counters are non-zero (pool size, windows grown, e180 rows dropped).
- **CV (class-vote diagnostic)**, slice I with own-ik14 library spectra banned, same polarity. Hits come from two separately reported variants:
  (a) direct ±10 ppm + NP biosynthetic delta list (CH₂, O, H₂O, acetyl, CO₂, malonyl, pentose, deoxyhexose, hexose,
  glucuronide, coumaroyl … each ±10 ppm); (b) open ±200 Da. Similarity = **shift-aware entropy** (a fragment matches
  directly or shifted by Δprecursor). Top 10 hits with sim ≥ 0.5, similarity-weighted superclass vote.
  PASS per variant iff purity@superclass ≥ majority-class baseline + 0.10 **and** coverage ≥ 50%. Purity@pathway also reported.
  If both pass, R2 uses (a). If neither passes, the idea is dropped before R2.

## Build steps
1. **Package (local, tests first)** in `scratch_pkg/casmi_scratch/`:
   - `pool_build.py`: source loaders (v2 `pool_meta.parquet`, `coco_meta.pkl`/`coco_mass.npy`, train, `bio_meta.pkl`/`bio_mass.npy`,
     `natural_products.parquet`), scorer_key, hygiene, dedup-with-priority, flag bits, writer + MANIFEST.
     Reuse `pool._exact_mass` and the `build_r0_pool` COCONUT/train logic.
   - `pool.py`: `load_pool(dir, drop_flags=None)` → `CandidatePool` (flags already supported; add NP side-table lazily).
   - `np_labels.py`: exact-match + kNN labelling. Port from `build/np_signals_core.py` (kNN labels, NP_Score); don't rewrite it.
   - `analog.py`: shift-aware entropy numba kernel next to `kernels.entropy_sim`, the delta-list / open-window library search, and the vote.
   - Config keys: `pool_dir`, `pool_drop_e180`.
   - Tests in `scratch_pkg/tests/test_core.py`: dedup priority, flag OR, key recompute on a tautomer pair, hygiene drops, mass-sort invariants, superset check, shift-aware kernel on a hand-built +162 pair, and the vote on a toy case.
2. **Colab driver** `scratch_pkg/drivers/r1_pool_colab.py` (generated notebook via `tools/mk_notebook.py`), CPU High-RAM,
   per-stage Drive checkpoints (keys, labels in chunks) so it resumes rather than restarts. Output goes to Drive, then the user publishes it as
   our private Kaggle dataset `casmi26-scratch-pool` (v1).
3. **Coverage audit + CV diagnostic** (Colab): R1.1–R1.3 and CV on slice I. Write the numbers into `gate_report.md`.
4. **Kaggle twins**: `r0_kaggle.py` reads the pool from `pool_dir` when set (fallback = R0 runtime build). Generate
   `casmi_scratch_r1a` and `casmi_scratch_r1b` (`--set pool_drop_e180=True`). Submit with a paired R0 control and log via `/update-experiments`.
5. Docs: update `roadmap.md` (S2 = R1a/R1b), `data_contract.md` (new dataset), `kb_index.md` (class vote as an R2 candidate, with the
   CV result), `summarize.md`.

## Verification
- `python scratch_pkg/tests/run.py` passes (including the stale-notebook check after regenerating).
- The Colab printout shows MANIFEST counts per source, overlaps, key-disagreement and hygiene counts, R1.1 = 0 lost, R1.2/R1.3 numbers, and the CV table.
- Kaggle commit run (fast mode): the pool loads from the dataset, counters print, and runtime fits. On the LB, the submission is (400, 2) with ≤ 25 unique ik14 per row.

## Status (2026-10-06): code done, nothing measured
Built and tested (`python scratch_pkg/tests/run.py`: test_r1 15 checks, plus an R1b dry run of the Kaggle notebook):
- `casmi_scratch/pool_build.py`: source loaders, `smiles_props` (scorer key), `run_chunked` (hash-named checkpoints,
  resumable), `merge_rows`, `superset_check`, `write_pool`, and `build()` (the whole Colab sequence). Single- and
  multi-worker builds give byte-identical files.
- `np_labels.py` (exact + kNN labels, NP_Score), `analog.py` (shift-aware entropy kernel, delta/open search, vote),
  `audit.py` (slice I, coverage, growth, truth-in-sim, CV).
- `pool.py`: `load_pool`, `e180_only`, `load_keymap`, `load_np_labels`. `library.py`: `pol`, `remap_keys`.
- Notebooks (`python scratch_pkg/tools/mk_notebook.py --all`): `casmi_scratch_r1_pool_colab`,
  `casmi_scratch_r1_audit_colab`, `casmi_scratch_r1a`, `casmi_scratch_r1b`.

Deviations from the plan above:
- Config switch `pool_source` ("r0" | "r1") instead of `pool_dir`: the notebook finds the dataset by file name.
- **Key-space fix:** R1 pool keys are scorer keys but the library's `inchikey14` is the plain key, so `lv` would miss
  any structure where they differ. The build writes `train_keymap.parquet`, and R1 runs remap the library.
- R1.1 counts R0 rows RDKit parses but cannot key. A plain "keys ⊆" test passes by construction.
- R0 notebooks were regenerated. Behaviour is unchanged (new config defaults, one extra report counter `cand_median`).

### v1 build result (2026-10-06, Colab CPU, 6.4 h, `MANIFEST.json`)
| | |
|---|---|
| R1a pool | **778,095** structures; R0 has 706,289 scorer keys → +71,806 (+10.2%). **R1.1 PASS** (0 lost) |
| New structures by winning source | BIO 62,404 · NP table 2,629 · COCONUT-arrays-only 2,361 (rest: v2 436,515, train 274,186) |
| R1b pool | 778,095 − 180,649 enveda-180-only = **597,446**, i.e. 108,843 fewer structures than R0 (−15%). Only 2,292 of the 182,941 enveda-180 structures appear in any NP source |
| Key disagreements (source key ≠ scorer key) | v2 0 (already scorer keys) · COCONUT 45,573 · BIO 2,796 · **train 4,370** (the cases the `train_keymap` remap fixes) · NP table 39 |
| Tautomer collapse | train 275,810 plain ik14 → 274,186 scorer keys; COCONUT arrays 436,389 → 432,648 |
| Hygiene | 3 NP-table rows unkeyable (dropped); 33 multi-fragment, 17,247 charged (kept verbatim) |
| Labels | table 140,136 · kNN 125,258 · none 512,701 (34% labelled); top superclasses Glycerolipids 36,545, Triterpenoids 20,468, Diterpenoids 18,968 |

Readings:
- The R1a gain is mostly BIO, which is largely LIPID MAPS lipids (big isomer families). Watch R1.3 window growth in the lipid
  mass range. If R1a fails on the LB, a "minus BIO-only" probe is the obvious follow-up (an R1 config switch, not a new build).
- R1b removes far more than R1a adds. It is mainly a decoy-pruning probe, so its LB result does not tell us anything about coverage.

Run order (user):
1. Colab, CPU High-RAM: `casmi_scratch_r1_pool_colab.ipynb`. Drive `casmi_2026/` needs train.parquet,
   `pool_meta.parquet` (v2-pool), `coco_meta.pkl`, `bio_meta.pkl` and `natural_products.parquet`. Paste the printout.
2. Upload `casmi_2026/casmi26-scratch-pool/` as the private Kaggle dataset `casmi26-scratch-pool`.
3. Colab: `casmi_scratch_r1_audit_colab.ipynb` → R1.1–R1.3 and CV verdicts → `gate_report.md`.
4. After R0.4 passes: submit R1a + R1b (attach `casmi26-scratch-pool`, not the COCONUT arrays) with a paired R0.

## GLACIER twins (2026-10-06): code done, nothing measured
`casmi_scratch_r1a_glacier` / `casmi_scratch_r1b_glacier` = R1a / R1b with `sim_model="glacier"` (one override line;
`test_notebook` checks this). Gate R1-GL is in `gate_report.md`.
- `sim.setup_sim` dispatches on `sim_model`. `setup_glacier` finds `gl_runner.py`, checks `ckpt/glacier.pt` against
  `MANIFEST.json`, and reuses `preseed_ice_site` (casmi26-iceberg wheels + the cp-tag RDKit 2025.3.6). Then `GlacierWorker`
  runs `resources/glacier_worker.py`, which uses gl_runner's `setup_env` (site `none`), `load_model`, `Predictor`,
  `smiles_ok`, `instr_token` and `ce_bucket` (5 eV) and writes the ICEBERG NPZ schema. Scheduler, retry (GL batch 32 → 8)
  and `spectrum_sv` are shared.
- `pipeline.plan_molecule` keeps only `sim.sim_adducts(cfg)`: `[M+H]+` / `[M+Na]+` for GLACIER, all six for ICEBERG
  (unchanged). New report counters: `sim_model`, `n_covered_spec`.
- R0 / R1a / R1b notebooks were regenerated; their behaviour is unchanged (`sim_model="iceberg"` default).

Kaggle inputs for the GLACIER twins: competition data, `casmi26-scratch-pool`, **`casmi26-glacier`**, `casmi26-iceberg`
(for its wheels only), `eveliaveldrine/rdkit-wheels`. GPU T4, internet off. Run them the same day as r1a / r1b.
