# specs_novel — Phase 3 handover (decoder training)

*Written 2026-09-29 at the end of the P0/P1 session. **Updated the same day**: the six open decisions
in §4 have been taken with the user and G3 is now pre-registered in `gate_report.md`. Read this file
first. Then read `gate_report.md` → **G3** (the authority on every P3 threshold and parameter),
`roadmap.md` (the execution-order amendment at the top, then P3 and P4), `plan.md` §3.4–§3.5
(embedding, memory) and §4.3 (masks, search), and `search_plan.md` **v1.1** (§0 is the amendment that
corrects the v1.0 search designs).*

---

## 1. Where the project stands

**specs_novel** adds a generator to Approach B (`build/simple_sim_submit.py`, LB 0.289): a SELFIES
decoder conditioned on frozen DreaMS embeddings, searched with a formula-constrained best-first
DFS. Every phase has a gate that is written into `specs_novel/gate_report.md` **before** it runs
(roadmap protocol 1). A failed gate stops the project; the options are then to redesign, use the
single rescue, or end (protocol 2).

**Execution order (amended):** P0 ✅ → P1 ✅ → **P3 (now)** → P4-oracle (G4.0 kill) → P2 (formula
stage) → P4-predicted → P5 → P6. G3 and G4.0 use the **oracle** formula, so P2 is not needed yet.

### P0 — PASS (2026-09-29)

- **G0.2–G0.5** passed in the G0 run.
- **G0.1** passed via **G0.1-R2**. The first cache failed its closure check G0.1-R because
  `np.argsort` let the platform decide which intensity-tied peaks survived the top-100 cut.
  `dreams_infer.dreams_peaks` now uses `np.lexsort`, where ties go to the higher m/z.
- **The corpus was re-embedded** into `_novel/emb_dbec3a0_lx/`. Its peaks are byte-identical on
  three hosts:
  - Colab A100 host: Xeon with AVX-512;
  - Colab CPU: AMD EPYC without AVX-512;
  - Kaggle T4: numpy 2.0.2, Python 3.12.

  Cosine agrees to within about 5e-12 across those hosts. Every row without a tie matches the old
  cache to 1e-12; all the drift was in 31,742 tie rows (2.4 %).

### P1 — PASS (2026-09-29, run 2 under amendment A-G1.1)

Run 1 failed G1.1: only 15 timsTOF molecules are outside MassSpecGym, because 230 of 245 are in
it. Amendment A-G1.1 added `ts_dec`.

| cohort | molecules | what it is for | spectra |
|---|---|---|---|
| `pseudo_c3` | 600 (gnps 483; 15 honest timsTOF) | **G3** and the headline P4 numbers: truth ∉ `msg_ik14`, whole scaffold group held out of decoder training | 1,028; CE known on **15 %**; median **1** per molecule |
| `ts_dec` | 155 (all enveda-np-examples) | **decoder-only** instrument-shift check (G3 / G4.0 reporting). ICEBERG *has* seen these molecules, so never use them for ICEBERG / `sv` / merged-list metrics, and never pool them with `pseudo_c3` | 414; CE known on **100 %**; 2.67 per molecule |
| `c2_displace` | 400 | P4 displacement cost (the truth stays in the pool); structure held out of decoder training | 669; CE known on 16 % |

- **`dec_split`:** train 183,102 (94.06 %), `dec_val` 5,840 (3.00 %, whole scaffold groups),
  `heldout_c3` 4,212, `heldout_ts` 1,114, `heldout_c2` 400.
- **Settings:** `G_MAX` = 100; seed 20260929.

### Fixed values carried into P3

| value | setting |
|---|---|
| `MAX_LEN` | 171 tokens (p99.9) |
| `DBE_PRUNE` | False |
| vocabulary | 81 tokens (`selfies_vocab.json`, SHA1 `3046040b…`, ids identical to MassSpecGym's `SelfiesTokenizer`) |
| selfies | 2.1.2 |
| RDKit | 2026.03.3 |
| `CE_BINS` | [15, 23, 36, 44] eV → 5 known bins + "unknown" (23.9 % of corpus rows have no CE) |
| adducts | the 10 `simsim_common.TEST_ADDUCTS` |
| DreaMS | `dreams_embed.pt` SHA1 `a28488b2…`, commit `dbec3a0`, `dreams_infer.py` SHA1 `57ce1bc1…` |
| `T_embed` | 6.68 ms per spectrum (T4) |
| fast tracker | 19 µs per step (`legal_mask` + `push`) |
| stereo / isotopes | stripped before SELFIES (the round-trip is judged on ik14) |

### Data on Google Drive (`/content/drive/MyDrive/casmi_2026/`)

| path | content |
|---|---|
| `_dreams_out/corpus_meta.parquet` | the corpus: 1,323,878 rows (`ingest_lib`, `inchikey14`, `normalized_smiles`, `molecular_formula`, `ionization_mode`, `adduct`, `precursor_mz`, `oob_mass`). Row `i` is embedding row `i`. |
| **`_novel/emb_dbec3a0_lx/emb_{0000..0006}.npy`** | **THE training cache**: 200,000 rows per chunk, 1024-d float32, plus `manifest.json` (chunk SHA1s) and `cos_lx_vs_dbec3a0.npy` |
| `_dreams_out/emb/`, `_novel/emb_dbec3a0/` | **old caches: never use them for specs_novel** |
| `_novel/p0/corpus_ce.parquet` | per corpus row: `row`, `train_row`, `ce_list`, `n_ce`, `ce_mean`, `instrument_type` |
| `_novel/p0/msg_ik14.npy`, `selfies_vocab.json`, `dreams_embed.pt` | P0 artefacts |
| `_novel/p1/pseudo_c3.parquet`, `ts_dec.parquet`, `c2_displace.parquet` | one row per molecule. Columns: `molecule_id`, `inchikey14`, `smiles`, `can` (stereo-stripped), `selfies`, `formula` (**label** formula), `scaffold`, `group_size`, `src_lib`, `timstof`, `in_scope`, `rows` (corpus rows = embedding rows), `train_rows`, `adducts`, `ces`, `lib_ban_keys`, `pool_ban_keys`, `dreams_ft_risk` |
| `_novel/p1/dec_split.parquet` | `inchikey14`, `scaffold_key`, `split` ∈ {`train`, `dec_val`, `heldout_c3`, `heldout_c2`, `heldout_ts`} |
| `_novel/lx/parity_lx.npz` | 200 parity rows (peaks + cache vectors); the P5 DreaMS fixture |
| `casmi26-offline-rdkit-2026033/` | the RDKit wheel (the install cell pattern is in `build/novel_p1_colab.py`) |

The Kaggle dataset **`casmi26-novel-lx`** holds `dreams_embed.pt`, `parity_lx.npz` and
`manifest_lx.json`. The old `casmi26-novel-offline` dataset is superseded for P5.

### Repository (branch `specs-novel-p0`, everything committed, last commit `d144002`)

| file | role |
|---|---|
| `build/novel_decoder.py` | P0 part only: `SelfiesVocab`, `TokenTable`, `Tracker` (`legal_mask`, `push`, `eos_legal`), `prep_structure`, `ik14`, `parse_formula`, `walk_truth`. **P3 adds the model, the memory builder and the search here** (plan §5.5: one source of truth, imported by Colab training and the Kaggle worker; torch-only core, and any Lightning / massspecgym wrapper only on Colab). |
| `build/dreams_infer.py` | the vendored DreaMS, pinned by SHA1 `57ce1bc1…`. **Do not edit**: the G0.1-R2 notebooks assert the SHA1, and so does `test_novel_p0.py`. |
| `build/novel_cohort.py`, `build/novel_lx.py` | P1 and G0.1-R2 helpers |
| `build/mk_novel_p0.py` | generates **every** specs_novel notebook: the `SOURCES` and `BUILDS` lists, and the `VAR = None  # @@TAG@@` placeholder convention. Add the P3 notebooks here. |
| `build/novel_p1_colab.py` | the most recent notebook template: Drive paths, RDKit / selfies install cell, `save_gate`, `sh`, fork `pool()`, source write-out |
| `build/test_novel_p0.py`, `build/test_novel_p1.py` | plain-assert test scripts ending in `…: OK`. They also check that the notebooks are fresh and embed their sources verbatim. |

---

## 2. P3 question and gate *(pre-registered 2026-09-29 in `gate_report.md` → G3 — that section wins over any number below)*

**Question:** given the true formula, does the decoder's constrained search produce structures
closer to the truth than a formula-constrained random generator?

- **Model.** `DreamsSelfiesDecoder`. The memory is one token per spectrum,
  `W_e·emb + E_adduct + E_ce`, plus a formula token `MLP(formula_counts)`, with no positional
  encoding (plan §3.5). The decoder is a pre-norm `nn.TransformerDecoder` with tied input and output
  embeddings. It is trained with teacher forcing and cross-entropy on SELFIES, given the **true**
  formula.
- **Sample unit.** One molecule, with up to `MAX_SPEC_PER_MOL` of its spectra drawn at random each
  epoch.
- **Evaluation on `dec_val`, then `pseudo_c3`** (G3), with **`ts_dec` reported separately**:
  - teacher-forced loss and token accuracy;
  - unconstrained sampling: the share of samples with the correct heavy-atom formula;
  - constrained search at a fixed `NODE_BUDGET`: top-25 **max Tanimoto** (Morgan r2, 2048 bits,
    MassSpecGym `morgan_fp`), MCES, and exact@25 (ik14);
  - each compared against **`RandomDeNovo`** given the same formula.
- **G3 (provisional):** constrained search beats `RandomDeNovo` on **median top-25 max Tanimoto by
  ≥ 0.10**, and the loss has converged. "Converged" must be defined when G3 is pre-registered.
- **Single rescue if G3 fails:** pretrain the decoder as a SELFIES language model on
  `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv` (`massspecgym.utils.load_unlabeled_mols`),
  with cross-attention off. Then fine-tune and re-run G3 **once**. If it fails again, stop.
- **Then P4-oracle, G4.0 (kill):** oracle-formula exact@25 on `pseudo_c3` < 0.01 → stop.

**`RandomDeNovo` usage** (`MSGYM_novel/massspecgym/models/de_novo/random.py`): call
`generate_random_molecule_graphs_via_traversal(CalcMolFormula(mol))` directly for each truth.
Construct it with `formula_known=True`, `max_top_k=25` and `top_ks=(1, 10, 25)`. The default
`max_top_k` is 10, which is too small for top-25. It generates from the **full** formula, H
included. The massspecgym import on Colab needs `--no-deps` plus its import-chain packages; P0.5
proved the recipe (`build/novel_p0_colab.py`, P0.5 cell).

---

## 3. Findings from P0/P1 that shape P3 (recorded in `gate_report.md`)

1. **Unknown CE dominates the cohorts.** `pseudo_c3` has CE on 15 % of spectra, while every test
   spectrum has CE. The model must work well with CE = "unknown". Use CE dropout in training: set
   known CE to "unknown" with probability p. Report results split by CE known / unknown.
2. **`ts_dec` is the only test-like cohort:** timsTOF, CE known on 100 %, several spectra per
   molecule. `pseudo_c3` measures novelty; `ts_dec` measures the test conditions. Report both;
   G3 is judged on `pseudo_c3` (as registered in the roadmap) unless the G3 pre-registration says
   otherwise.
3. **The oracle formula comes from the structure, not the label.** Use `nd.prep_structure(can)`'s
   formula (RDKit, H included). 165 corpus structures disagree with `molecular_formula`.
4. **Spectra per molecule:** median 1 in `pseudo_c3`. Multi-spectrum pooling is barely exercised
   there; `ts_dec` exercises it.
5. **~80 % GNPS.** Report per `src_lib`.
6. **DreaMS leakage flag:** the cohort column `dreams_ft_risk` only looks at picked rows. P4 also
   reports a **molecule-level** flag: any MoNA `[M+H]+` row for that ik14 in the corpus.
   `ts_dec` molecules are the likely ones.

---

## 4. Decisions taken with the user (2026-09-29)

*These close the six questions this section used to ask. All of them are pre-registered in
`gate_report.md` → G3, which is the authority; this table is the summary.*

| # | question | decision |
|---|---|---|
| 1 | **Training rows** | **All libraries** (including enveda-180, pluskal_ms2, spectraverse, drug_plus), **label-quality-OK rows only** (`novel_cohort.label_ok`, 92.2 %), **in-scope structures only** (prep `ok` and ≤ `MAX_LEN`), `dec_split == "train"` only |
| 2 | **Colab GPU budget** | Colab Pro, ~200 CU total, **50–60 CU for P3** ≈ 5 h A100 or 11 h L4. Note the sample unit is a *molecule*, so an epoch is ≈ 715 steps at batch 256 — minutes, not tens of minutes. Training is cheap; the eval legs are the real cost |
| 3 | **Model size and `MAX_SPEC_PER_MOL`** | `D_MODEL` **512**, `N_DEC_LAYERS` **8**, `N_HEADS` **8**, ff 2048, dropout 0.1, tied embeddings (≈ 28 M params); `MAX_SPEC_PER_MOL` **4**. A calibration cell times 50 steps and prints the projection before the real run |
| 4 | **CE dropout probability** | **p = 0.3** on known CE. 23.9 % of corpus rows are already "unknown", so the effective unknown rate is ≈ 0.47 |
| 5 | **G3 pre-registration** | Judged on **`pseudo_c3` only**; `ts_dec` reported separately. Threshold unchanged at **≥ +0.10** median Δ top-25 max Tanimoto. **Converged** = no `dec_val` token-CE improvement for **5 consecutive epochs** *and* the best epoch is not the last. **MCES** on a fixed random **150-molecule** subset of `pseudo_c3` and on **all 155** `ts_dec`, reported only |
| 6 | **Search implementation** | A **3-method bake-off at an equal candidate budget of `N_CANDIDATES` = 256**: **A** prefix-seeded batched sampling (`L_SEED` 4, `K_ROOTS` 8, `M_ROLLOUT` 32, T 0.7, top-p 0.90), **B** plain masked beam (`BEAM` 256, the **control**), **C** diverse masked beam (`BEAM` 256, `ALPHA_DIVERSE` 0.6). Batched best-first moves to **P4** |
| 6b | **Where the rerank applies** | **P4 only, to both arms.** G3 is decoder-only. G3's metrics are order-invariant set metrics, and reranking only our arm would measure the reranker |
| 6c | **Top-25 selection** | **`maxmin`** (logp-seeded greedy MaxMin, Morgan r2/2048) primary; **`logp`**-only reported beside it from the same candidate pool |

**The bake-off's selection rule is what keeps it honest.** A, B and C run on a fixed random
**300-molecule `dec_val` subset**; the winner (highest median top-25 max Tanimoto; ties A → C → B) is
then run **once** on `pseudo_c3` and `ts_dec`, and G3 is judged on that. The losing arms' `pseudo_c3`
numbers are never produced, so none can be promoted after the fact.

**`search_plan.md` is amended to v1.1.** Its §0 records six corrections to the v1.0 designs: depth-1
root expansion is chemically empty (the mask admits only atom tokens at state 0, and bond-order
prefixes are degenerate there, so seeding must be at depth ~4); ranking the final 25 by `logp`
defeats a max-over-25 metric; peak-guided logit steering (Method 2) is **dropped as unsound** because
SELFIES fixes H only at the leaf; Method 4's in-flight rerank would confound G3 and MetFrag-lite is
**not** in `simsim_common.py`; best-first is not inherently unbatched; and G3's metrics are
order-invariant.

---

## 5. Work breakdown

*Steps 1-8 are **done** (2026-09-29, commits `e3504e4` .. `f855126`). Step 9 is the user's: run the
three notebooks and paste the printouts back.*

| # | step | state |
|---|---|---|
| 1 | specs synced, G3 pre-registered | done `e3504e4` |
| 2 | `novel_decoder.py` P3 part: model, memory, three searches, `select_top25`, funnel | done `e7b49e6`, `ee28df3` |
| 3 | `novel_p3_data.py`: structures, index, EmbeddingStore, sampler | done `ee28df3` |
| 4 | `novel_p3_train_colab.py` -> `casmi_novel_p3_train_colab.ipynb`, per-epoch Drive resume | done `7267f41` |
| 5 | `novel_p3_eval_colab.py` -> `casmi_novel_p3_eval_colab.ipynb` (bake-off, resumable) | done `f855126` |
| 6 | `novel_p3_null_colab.py` -> `casmi_novel_p3_null_colab.ipynb` (RandomDeNovo, MCES, verdict) | done `f855126` |
| 7 | `test_novel_p3.py`, in `run_all_tests.sh` | done `f855126` |
| 8 | `novel_p3_dryrun.py`: 4 legs incl. kill-and-resume | done `7267f41`, `f855126` |
| 9 | **run the notebooks**, record the results in `gate_report.md`, update the roadmap, commit | **user** |

**Run order:** `casmi_novel_p3_train_colab.ipynb` (A100) -> `casmi_novel_p3_eval_colab.ipynb` (GPU)
-> `casmi_novel_p3_null_colab.ipynb` (**CPU High-RAM** -- `RandomDeNovo` and MCES are CPU/ILP bound,
so running them on an A100 would burn ~12x the compute units for nothing). Every notebook is
re-runnable: it resumes rather than restarting.

**Three environment facts worth knowing before the run:**

1. `MassSpecGym`'s own `utils.morgan_fp` calls `Chem.GetMorganFingerprintAsBitVect`, which **does not
   exist on RDKit 2026.03.3** (it moved to `AllChem` and is deprecated in favour of
   `MorganGenerator`). `novel_decoder.morgan_fps` reuses the *definition*, not the call, and is
   asserted bit-identical to the deprecated function.
2. The null notebook asserts the massspecgym install did not move RDKit off the metric's pinned
   version. If it does, the metric changes underneath the gate.
3. `novel_cohort.label_ok` requires the neutral mass in **[157, 1159]**, so the training set is
   implicitly mass-limited. That is inherited from G1 and matches the test's range; it is not a bug,
   but it does mean small molecules are absent from training.

### Original breakdown

1. **Sync the specs and commit first.** `gate_report.md` (new G3 section), `roadmap.md` (P3 fixed
   values, P4 deferrals), `plan.md` (§1.3, §3.5, §4.3, §6), `search_plan.md` (→ v1.1),
   `tech_stack.md` (§4 interfaces, §5 parameters), and this file. **Done 2026-09-29**, one commit,
   before any code.
2. **`build/novel_decoder.py`, P3 part** (appended; the P0 part is not modified):
   - `build_memory(embs, adducts, ces, formula)` → `(k+1, D)`, no positional encoding;
   - `DreamsSelfiesDecoder` (**torch only**, so Kaggle can import it);
   - greedy / sampling decode, for the unconstrained formula-rate diagnostic;
   - `search_prefix_seeded`, `search_masked_beam`, `search_diverse_beam` — one signature returning
     `[(smiles, logp)]`, all driving `Tracker.legal_mask` / `push`; the diverse beam returns the
     **unpenalised** `logp`;
   - `select_top25(cands, mode)` — `"maxmin"` or `"logp"`;
   - a leaf check on the **full** formula including H, then ik14 dedupe.
3. **`build/novel_p3_data.py`:** the training index (ik14 → corpus rows, SELFIES ids, formula vector)
   from `dec_split` + `corpus_meta` + `corpus_ce` + `emb_dbec3a0_lx`; memory-mapped embedding chunks
   (`chunk = row // 200_000`, `offset = row % 200_000`); per-epoch spectrum sampling with CE dropout;
   the oracle formula from `prep_structure(can)`, **not** the label.
4. **`build/novel_p3_train_colab.py` → `casmi_novel_p3_train_colab.ipynb`** (A100): trains, early
   stops on `dec_val` loss with patience 5, saves `decoder.pt` (`state_dict`, hparams, vocab SHA1,
   adduct list, `CE_BINS`) to `_novel/p3/`.
   - **Every epoch is durable on Drive.** `ckpt/epoch_{n:03d}.pt` holds the model, optimizer, LR
     scheduler, epoch, global step, best loss, patience counter **and all four RNG states** (torch CPU
     and CUDA, numpy, python `random`) — without the RNG the resumed run silently trains on a
     different sampling stream. Writes are atomic (local `.tmp` → `os.replace`), the newest two
     checkpoints are kept, and `train_log.jsonl` is append-only so per-epoch numbers survive a failed
     checkpoint write. A resume cell runs first and prints `resumed from epoch N` loudly.
5. **`build/novel_p3_eval_colab.py` → `casmi_novel_p3_eval_colab.ipynb`** (GPU): teacher-forced loss
   and token accuracy, the unconstrained formula rate, the bake-off on the `dec_val` subset, then the
   winner on `pseudo_c3` and `ts_dec`. Candidates are flushed to Drive every **50 molecules** with a
   `done_*.json` skip-list, so generation resumes too — it is the longest leg of P3.
6. **`build/novel_p3_null_colab.py` → `casmi_novel_p3_null_colab.ipynb`** (**CPU High-RAM**):
   `RandomDeNovo` on the same molecules with the same oracle formulas, plus Tanimoto / MCES /
   exact@25 and the G3 verdict → `p3_gate.json`. Both `RandomDeNovo` and myopic-MCES are CPU/ILP
   bound, so this leg costs ~1 CU/h instead of ~12. `RandomDeNovo` needs `max_top_k=25`; the default
   10 is too small for top-25.
7. **`build/test_novel_p3.py`** on synthetic data: memory shapes and the **set-invariance** of the
   spectrum tokens; the tied embeddings; search leaves always satisfy the full formula and are
   ≤ `MAX_LEN`; a tiny overfit model is recovered by all three searches; `select_top25` behaviour;
   the metric helpers against MassSpecGym's definitions; **checkpoint round-trip including RNG**;
   **atomic-write survival**; **generation resume idempotence**; and that the notebooks are fresh and
   embed their sources verbatim. Add it to `build/run_all_tests.sh`.
8. **Dry-run all three notebooks locally** on a fake Drive with a stubbed `google.colab`, a tiny model
   and a small corpus, including a **kill-and-resume rehearsal**, before handing them over.
9. The user runs the notebooks. Record the bake-off table and the G3 verdict in `gate_report.md`,
   update the roadmap status and commit.

---

## 6. Lessons and pitfalls (from this project and its memory)

- **The exclusion set is the model's training set, not the eval holdout.** The decoder's training
  set is `dec_split == "train"`. Never train on `dec_val`, `heldout_*` or anything else from the
  cohorts.
- **Embedding row = `corpus_meta` row.** Chunk = `row // 200_000`, offset = `row % 200_000`. The
  cohort `rows` columns are these indices.
- **Offline numbers are relative guides and the LB is the arbiter** (val MRR always reads above the
  LB).
- **Notebooks are generated.** Edit `build/*.py` and regenerate with `python build/mk_novel_p0.py`;
  never hand-edit an `.ipynb`. `read_text` normalises CRLF, so the embedded SHA1s are of the LF
  text.
- **`build/run_all_tests.sh` has two failures older than specs_novel:** `test_stage_d_colab.py` and
  `test_simple_sim_gate.py`. The second one **regenerates `casmi_simple_sim_gate_kaggle.ipynb`**, so
  restore that file with `git checkout -- casmi_simple_sim_gate_kaggle.ipynb` after a full run. The
  pipe hides exit codes, so read the output.
- **Windows tooling:** the Bash tool is Git Bash. Heredocs that contain both quote kinds can fail to
  parse; write the script to the scratchpad and run it. The auto-mode classifier sometimes fails
  transiently; retry once, or use Read / Grep / Edit instead.
- **Dry-run notebooks locally** before handing them over. The P1 notebook was run end to end on a
  synthetic fake Drive with a stubbed `google.colab` and a `multiprocessing.dummy` pool; the
  pattern is worth reusing for P3 (CPU, a tiny model).
- **The user uploads result JSONs** into `specs_novel/`; one once landed in the wrong folder
  (`specs_simple_ms_simulator/lx_gpu.json`, a stray copy). Check the path.
- **Pre-register before running**, and record an amendment with its date *before* a re-run. Check
  thresholds against numbers you already have: G1.1's 30 timsTOF was unreachable, and
  `P0.5.p1_preview` already showed it.
