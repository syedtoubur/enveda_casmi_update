# specs_novel — Phase 1 plan (handover)

*Written 2026-09-29 at the end of the P0 session, so a fresh session can start P1 directly. Read this
file first, then `roadmap.md` (P1 section), `plan.md` §5 (exclusion sets) and `gate_report.md` (G0
outcome). Plan approved by the user in outline; the two open design choices were answered (§3).*

---

## 1. Where the project stands

**specs_novel** adds a generator to Approach B (`build/simple_sim_submit.py`, LB 0.289): a SELFIES
decoder conditioned on frozen DreaMS embeddings, searched with a formula-constrained DFS. Every phase
has a gate pre-registered in `specs_novel/gate_report.md` **before** it runs (roadmap protocol 1).

### P0 — done, CLOSED BY USER DECISION (not PASS)

| gate | result |
|---|---|
| G0.1 | Official ↔ vendored DreaMS exact (cos 1.000000); Kaggle T4 ↔ Colab CPU 0.99999999999. Cache leg failed; closure check **G0.1-R FAILED**. |
| G0.2 | Cell-7 replay matches `corpus_meta.parquet` row by row (1,323,878 rows); `corpus_ce.parquet` written. PASS |
| G0.3 | SELFIES round-trip 1.0000; 79 excluded (disallowed elements); `MAX_LEN` = 171. PASS |
| G0.4 | Tracker parity 100 %; 0 count/EOS violations; `DBE_PRUNE = False` (139 DBE violations). PASS |
| G0.5 | massspecgym imports; tokenizer ids = ours (81 tokens); `msg_ik14.npy` 28,923 ik14. PASS |

**G0.1-R diagnosis:** the DreaMS network was never the problem. `dreams_peaks` cut spectra to the
top 100 peaks with `np.argsort(it)[-100:]`, which is unstable. When intensities tie at the cut,
different numpy builds and CPUs kept different peaks. Spectra with fewer than 40 peaks are identical
between the old and new caches; 0.80 % of rows fall below cosine 0.999, the worst at 0.546.
**User decision (2026-09-29):**

- keep `_novel/emb_dbec3a0/` (re-embedded on an A100) as the decoder-training cache;
- make peak selection deterministic for future inference. Now fixed: `np.lexsort`, and a tie at the
  cut goes to the higher m/z;
- proceed to P1;
- accepted risk: rows with ties can differ between the training vectors and the test-time vectors.

### Fixed values carried into P1+

`MAX_LEN = 171`, `DBE_PRUNE = False`, `CE_BINS = [15, 23, 36, 44]` + "unknown" (23.9 % of corpus rows
have no CE), vocab 81 tokens (`selfies_vocab.json` SHA1 `3046040b…`), selfies 2.1.2,
`dreams_embed.pt` SHA1 `a28488b2…`, DreaMS commit `dbec3a0b514a99e5056cfccde4559fda8cfe8129`.
`T_embed` 6.68 ms/spectrum (T4), `T_novel_install` 3.7 s.

### Data on Google Drive (`/content/drive/MyDrive/casmi_2026/`)

| path | content |
|---|---|
| `enveda-CASMI26-molecule-id-mass-spectra/train.parquet`, `test.parquet` | competition data |
| `_dreams_out/corpus_meta.parquet` | the corpus: 1,323,878 rows (ingest_lib, inchikey14, normalized_smiles, molecular_formula, ionization_mode, adduct, precursor_mz, oob_mass, …) |
| `_dreams_out/emb/` | **old** cache — do not use for novel training |
| `_novel/emb_dbec3a0/emb_{0000..0006}.npy` | **the training cache**: row `i` = `corpus_meta` row `i`, 200,000 rows/chunk, 1024-d float32, `manifest.json`, `cos_new_vs_old.npy` |
| `_novel/p0/corpus_ce.parquet` | per corpus row: `row`, `train_row` (index into train.parquet), `ce_list`, `n_ce`, `ce_mean`, `instrument_type` |
| `_novel/p0/msg_ik14.npy` | MassSpecGym ik14 (U14), ICEBERG `msg_all` exclusion set |
| `_novel/p0/selfies_vocab.json`, `dreams_embed.pt`, `parity_200.npz`, `phase0_colab_gate.json`, `reembed_gate.json` | P0 artefacts |

A local copy of the re-embed report is at `specs_novel/re_embed_gate.json`.

### Repository state (branch `specs-novel-p0`) — **committed in 8bdc4dd**

§6 step 1 is **done**. The P0 work below was committed in 8bdc4dd, without `__pycache__`:

- modified: `build/dreams_infer.py` (lexsort tie-break), `build/test_novel_p0.py`,
  `specs_novel/gate_report.md`, `roadmap.md`, `tech_stack.md`
- new: `build/dreams_official_worker.py`, `build/mk_novel_p0.py`, `build/novel_p0_colab.py`,
  `build/novel_p0_kaggle.py`, `build/novel_reembed_colab.py`, the three generated
  `casmi_novel_*.ipynb`, `specs_novel/re_embed_gate.json`
- do **not** commit `build/__pycache__/`; check `.gitignore` (untracked) covers it.

`python build/test_novel_p0.py` currently prints `test_novel_p0: OK`.

---

## 2. P1 goal and gate

**Question:** which molecules can we evaluate on that none of the three learned components has
learned, and that retrieval cannot reach? The three components are the decoder (ours), ICEBERG
`msg_all` and the DreaMS head. Then fix the decoder's training split.

**Gate G1** (thresholds from roadmap P1; must be written into `gate_report.md` before the run):

- `pseudo_c3` has ≥ 300 molecules, ≥ 30 of them timsTOF (enveda-np-examples);
- `c2_displace` has ≥ 300 molecules;
- the training split keeps ≥ 90 % of the corpus structures (unique ik14).

Approach B has no FPNet or ranker (only `lv` library match, `sv` ICEBERG, `fr` MetFrag-lite, `ppm`).
So the FPNet split (`_fpnet/pack/splits.npz`) does **not** apply here. The exclusion sets are exactly
the three in `plan.md` §5.3.

---

## 3. User decisions for P1 (2026-09-29)

1. **Scaffold groups are capped.** A molecule can enter `pseudo_c3` only if its Murcko scaffold group
   has ≤ `G_MAX` structures; then the **whole group** is held out of decoder training. An **acyclic**
   molecule (empty Murcko scaffold) is a **group of one**: only the structure itself is held out.
   Without these two rules, benzene or the empty scaffold would wipe out tens of thousands of
   training structures. The resulting bias toward rarer scaffolds is reported.
2. **Sizes:** `pseudo_c3` = 600 molecules, including **every** eligible enveda-np-examples
   molecule. `c2_displace` = 400.

Decisions taken by the planner, to record in the roadmap as P1 amendments:

- **The universe is the P0 corpus.** Every cohort spectrum is a `corpus_meta` row, so it already has
  an embedding in `emb_dbec3a0` and CE in `corpus_ce`.
- **`c2_displace` structures are held out of decoder training (structure only).** A real Class-2
  test molecule is not in our training data. A decoder that had memorised it would generate the
  truth, which gets dropped as `in_pool`, and so understate displacement.
- **Libraries:** the val builder's `NP_LIBS` = riken, mona, msdial, massbank, gnps,
  enveda-np-examples, masaryk. enveda-180 is excluded (synthetic, drug-like). This is a superset of
  roadmap's list and matches the existing val cohorts.
- **`dreams_ft_risk` is recorded, not an eligibility rule** (amended 2026-09-29). The DreaMS head
  was fine-tuned on MoNA `[M+H]+`, so for those spectra the encoder may have seen the molecule. The
  lesson that the exclusion set is the model's training set applies to the encoder too. A molecule
  is flagged when `src_lib == "mona"` and at least one of its picked cohort rows is `[M+H]+`. The
  count is reported in `p1_gate.json`, and P4 reports hit rates split by the flag.
- **`in_scope` is recorded, not an eligibility rule.** It means prep status `ok` and token length
  ≤ `MAX_LEN`. Out-of-scope `pseudo_c3` molecules stay and count as misses in P4, which keeps the
  measure honest.

---

## 4. Design

The template is `enveda-casmi-2026-fast-spectral-validation-creation.ipynb`: cell 5 (`VCFG`), cell 15
(cohort sampling), cell 16 (`_scaf` Murcko), cell 21 (spectrum picking) and cell 23
(`ban_rows`/`ban_keys`).

### 4.1 Per structure (unique ik14 over the corpus; multiprocessing on Colab)

- `nd.prep_structure(normalized_smiles)` gives (status, canonical SMILES, SELFIES, formula counts).
  `in_scope` = status `ok` and `len(split_selfies) ≤ 171`.
- Scaffold key: val `_scaf` (RDKit `MurckoScaffold.GetScaffoldForMol` → canonical SMILES). An empty
  scaffold becomes `"acyclic:" + ik14`.
- `group_size` = the number of corpus ik14 that share the scaffold key.

### 4.2 Label-quality filter per row (val `VCFG`)

- Neutral mass = `simsim_common.neutral_mass1(precursor_mz, adduct)`. Truth mass =
  `simsim_common.formula_mass(molecular_formula)`; no RDKit is needed.
- Keep the row if |error| ≤ max(10 ppm, 0.0075 Da) and the neutral mass is in the test range
  [157, 1159].
- This also removes riken rows whose 2-dp m/z breaks the ppm window (memory:
  query-self-retrieval).

### 4.3 Eligibility (both cohorts)

- ik14 ∉ `msg_ik14`, i.e. not seen by ICEBERG;
- ≥ 1 quality-OK row in an NP library.
- `src_lib` = enveda-np-examples if the molecule has quality-OK rows there (`timstof = True`),
  otherwise its most frequent NP library among quality-OK rows.

### 4.4 `G_MAX` rule (pre-register)

`G_MAX` = the largest value in {200, 100, 50, 25} for which the expected held-out share is ≤ 7 % of
corpus structures:

    600 × (size-biased mean group size among eligible molecules with group ≤ G_MAX) + 400 + 3 % (dec_val)

"Size-biased" means each molecule weights its own group size. This leaves margin under G1's 10 %.
Record the chosen value, and the share of eligible molecules that it excludes.

### 4.5 Cohorts (fixed seed, e.g. 20260929)

1. **`pseudo_c3`:**
   - all eligible timsTOF molecules with group ≤ `G_MAX` (up to 600);
   - then a uniform draw from the other eligible molecules with group ≤ `G_MAX`, to 600 in total;
   - decoder holdout = the whole scaffold group of each;
   - at evaluation (P4): `pool_ban_keys` = `lib_ban_keys` = {truth}. The truth is gone from the
     pool and the library; scaffold relatives stay, as for a real Class-3 molecule.
2. **`c2_displace`:**
   - 400 eligible molecules, drawn uniformly, none in any `pseudo_c3` scaffold group;
   - decoder holdout = the structure only;
   - at evaluation: the truth **stays in the pool**, `lib_ban_keys` = {truth}, `pool_ban_keys` = {}.
3. **Spectra per molecule** (port val cell 21):
   - n is drawn from the test's spectra-per-molecule distribution (`test.parquet`, grouped by
     `molecule_id`), capped at 16 and at what is available;
   - rows come from `src_lib` only, quality-OK, CE-known rows first (`corpus_ce.n_ce > 0`);
   - distinct (adduct, rounded CE) pairs first, then fill.
4. **`dec_split`**: every corpus ik14 gets exactly one label:
   - `heldout_c3`: all members of the `pseudo_c3` scaffold groups;
   - `heldout_c2`: the `c2_displace` structures;
   - `dec_val`: whole random scaffold groups with ≤ `G_MAX` members (acyclic singletons included),
     up to 3 % of structures;
   - `train`: the rest.

### 4.6 Outputs (Drive `casmi_2026/_novel/p1/`)

- `pseudo_c3.parquet`, `c2_displace.parquet`, one row per molecule. Columns: `molecule_id`,
  `inchikey14`, `smiles` (normalized), `can` (stereo-stripped), `selfies`, `formula`, `scaffold`,
  `group_size`, `src_lib`, `timstof`, `in_scope`, `rows` (list of corpus rows = embedding rows),
  `train_rows`, `adducts`, `ces`, `lib_ban_keys`, `pool_ban_keys`, `dreams_ft_risk`.
- `dec_split.parquet`: `inchikey14`, `scaffold_key`, `split`.
- `p1_gate.json`: the G1 items, verdict and every recorded value.

**Recorded, not gating:**

- `G_MAX` and the share of eligible molecules it excludes (the scaffold bias);
- per cohort: counts by library and by adduct, the timsTOF count, the in-scope share, the CE-known
  share, the `dreams_ft_risk` count, and the spectra-per-molecule distribution;
- the held-out share, split into c3 groups, c2 and dec_val;
- the count of mismatches between `molecular_formula` and the RDKit formula;
- overlap with the val_scripts cohorts (`val_scripts/val_key.pkl`), for information only.

---

## 5. Files

| file | role |
|---|---|
| `build/novel_cohort.py` (new) | Pure helpers: numpy/pandas, RDKit imported lazily. `scaffold_key`, `label_ok`, `choose_gmax`, `sample_cohort`, `pick_spectra`, `assign_splits`, `dreams_ft_risk`. Embedded verbatim into the notebook (like `dreams_infer`) and imported directly by the tests. |
| `build/novel_p1_colab.py` (new) → `casmi_novel_p1_colab.ipynb` | Colab CPU High-RAM, ~20–30 min. Drive mount, RDKit 2026.03.3 install cell (copy from `novel_p0_colab.py` lines 104–139), embedded sources, structure pass (fork Pool), cohorts, split, invariants, G1 verdict, outputs of §4.6. |
| `build/mk_novel_p0.py` | Add to `SOURCES`: `NOVEL_COHORT` (`build/novel_cohort.py`) and `SIMSIM_COMMON` (`build/simsim_common.py`, numpy/pandas/pyarrow only). Add the build `(novel_p1_colab.py → casmi_novel_p1_colab.ipynb, ("NOVEL_DECODER", "NOVEL_COHORT", "SIMSIM_COMMON"))`. The placeholder convention is `VAR = None  # @@TAG@@`, and the notebook writes each source to a file so Pool workers can import it. |
| `build/test_novel_p1.py` (new) | Plain-assert script ending in `test_novel_p1: OK`, in the style of `build/test_novel_p0.py`. |
| `build/run_all_tests.sh` | Add `python build/test_novel_p1.py \| tail -1`. The pipe hides exit codes, so read the output. |
| `specs_novel/gate_report.md` | New section **G1 — Cohorts (P1)**: thresholds, the `G_MAX` rule, the holdout/ban definitions, decision rules, and the "recorded, not gating" table. **Committed before the run.** |
| `specs_novel/roadmap.md` | P1 amendments (§3 decisions); set the P1 status as it progresses. |
| `specs_novel/tech_stack.md` | Outputs table rows for `pseudo_c3.parquet`, `c2_displace.parquet`, `dec_split.parquet`. |

**Reuse, and do not reimplement:**

- `nd.prep_structure` and `nd.ik14` (`build/novel_decoder.py`);
- `simsim_common.neutral_mass1`, `formula_mass` and `TEST_ADDUCTS`;
- `save_gate`, `sh`, `pool()` and the Drive paths in `build/novel_p0_colab.py`;
- `load_func` and the notebook-check pattern from `build/test_novel_p0.py` (`test_notebooks`).

### Tests (`build/test_novel_p1.py`, synthetic data)

- `scaffold_key`: benzene derivatives share a key; acyclic → `"acyclic:<ik14>"`; unparsable → a
  singleton key.
- `label_ok`: the ppm floor, the 0.0075 Da rounding tolerance at low mass, and the mass-range edges.
- `choose_gmax`: returns the largest value that meets the 7 % budget, and the smallest one when
  nothing meets it (the notebook then reports it).
- `sample_cohort`:
  - timsTOF first, and all of them when there are ≤ 600;
  - never above `G_MAX`;
  - `c2_displace` never in a `pseudo_c3` group;
  - deterministic for a given seed.
- `pick_spectra`:
  - only `src_lib` rows;
  - CE-known first;
  - distinct (adduct, CE) pairs before repeats;
  - the count equals min(n, available).
- `assign_splits`:
  - each ik14 appears once;
  - scaffold groups are never split across labels;
  - `dec_val` ≈ 3 %;
  - the held-out sets are disjoint.
- `dreams_ft_risk`:
  - true only when `src_lib` is mona and a picked row is `[M+H]+`;
  - false for mona with only other adducts;
  - false for `[M+H]+` from other libraries.
- Notebooks: regenerated, not stale; every cell parses; the embedded sources are verbatim.
  `test_novel_p0.py` still passes, because `mk_novel_p0.BUILDS` grew.

### In-notebook invariants (assert before writing)

- `dec_split` labels are disjoint and cover every corpus ik14;
- no `pseudo_c3` group member is in `train` or `dec_val`;
- every cohort row is a quality-OK corpus row from its `src_lib`;
- no cohort ik14 is in `msg_ik14`, and `pseudo_c3` ∩ `c2_displace` = ∅.

---

## 6. Execution order

1. ~~**Commit the P0 work**~~ **done** (8bdc4dd).
2. **done (2026-09-29)** — **Pre-register G1**: write the `gate_report.md` G1 section plus the roadmap and tech_stack
   amendments, then commit, **before** any P1 run.
3. **done (d31c863)** — Write `build/novel_cohort.py` and `build/test_novel_p1.py`, and make the tests pass locally.
   Miniconda base has RDKit and selfies 2.1.2.
4. **done** — Write `build/novel_p1_colab.py`, extend `mk_novel_p0.py`, and regenerate with
   `python build/mk_novel_p0.py`. Run `test_novel_p0.py`, `test_novel_p1.py` and
   `run_all_tests.sh`, and read each test's output.
5. **Run 1 done (2026-09-29): G1 FAILED on G1.1** (15 timsTOF outside MassSpecGym). Amendment A-G1.1
   (`ts_dec`, `gate_report.md`) is implemented; **run 2 pending.** — **User** runs `casmi_novel_p1_colab.ipynb` on Colab (CPU, High-RAM) and pastes the printout or
   `p1_gate.json`.
6. Fill in the G1 results in `gate_report.md`, update the roadmap status and tech_stack, and commit.
   If G1 fails, follow protocol 2: stop and present the options.

**In parallel, any time before P3:** run G0.1-R2 (`gate_report.md`). Point the re-embed notebook at
`_novel/emb_dbec3a0_lx/` and add the `parity_lx` cross-platform peak leg (Colab CPU, GPU host,
Kaggle). About 35 min on an A100.

After G1: **P3 decoder training**, which needs G0.1-R2. Then **P4-oracle** (G3, G4.0). **P2** comes
only after G4.0 passes (roadmap execution-order amendment, 2026-09-29).

---

## 7. Lessons that apply (from project memory)

- **The exclusion set is the model's training set, not the eval holdout.** Excluding only the
  holdout cost 0.05 LB once. Here: ICEBERG → `msg_ik14`, decoder → `dec_split`.
- **val_scripts is blind to pool misses:** val MRR always reads above the LB. P1/P4 numbers are
  relative guides; the LB is the arbiter.
- **riken reports m/z to 2 dp**, which puts a query outside its own ppm window. The §4.2 filter
  handles it; report the per-library usable share.
- **The DreaMS head** was fine-tuned on MoNA `[M+H]+` (Murcko split). This is not an exclusion
  rule (plan §5.3). The cohorts flag it with `dreams_ft_risk` (§3), and P4 reports results split
  by the flag.
- **The training cache is provisional.** `emb_dbec3a0` was built with the unstable tie-break. P3
  trains on `emb_dbec3a0_lx` once G0.1-R2 passes. Row indices are the same in both caches, so P1
  outputs stay valid.
- **Any future re-embed must write to a new folder.** The re-embed notebook skips chunks that already
  exist. P5 needs a new Kaggle offline dataset and manifest, because the `dreams_infer.py` SHA1
  changed to `57ce1bc1…`.
- **Windows environment:** the Bash tool is Git Bash and PowerShell 5.1 is also available. The
  auto-mode classifier sometimes fails transiently; retry once, or use the Edit/Write tools.
