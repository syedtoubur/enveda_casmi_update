# specs_novel — gate report

*The thresholds for each gate are written here and committed **before** the gate's measurement runs
(roadmap protocol 1). Results are filled in afterwards, and each result cites the notebook and stage
that produced it. A failed gate stops the project (protocol 2).*

---

## G0 — Feasibility (P0)

*Pre-registered 2026-09-28, spec v3.1. Run 2026-09-29 (Colab, then Kaggle T4 ×2, internet off).*

Sources, once run:

- `casmi_novel_p0_colab.ipynb` writes `_novel/p0/phase0_colab_gate.json` (P0.1a, P0.2–P0.5).
- `casmi_novel_p0_kaggle.ipynb` writes `phase0_kaggle.json` (P0.1b; T4, internet off).

### Verdict: **CLOSED BY DECISION** — not PASS (2026-09-29)

G0.2–G0.5 pass. G0.1 passes its official and Kaggle legs; the cache leg failed and was put on HOLD
(rule 1). Its closure check **G0.1-R failed** as registered (below). The cause is non-deterministic
peak selection, not the network. Read literally, rule 5 would stop the project. **The user decided on
2026-09-29 to proceed to P1 with the `emb_dbec3a0` cache as built** and to make peak selection
deterministic for all future inference. G0 is therefore recorded as closed by that decision, with
the accepted risk stated under G0.1-R. It is **not** a PASS. G0.1-R2 (below, pre-registered) re-embeds
with the fixed code; if it passes, G0 becomes PASS. It must pass before P3.

| gate | pass condition (pre-registered) | result | status |
|---|---|---|---|
| G0.1 | **Kaggle, offline:** `dreams_embed.pt` loads with its SHA1 asserted, and cosine ≥ **0.999** on **all 200** parity rows, Kaggle (b) vs Colab (b). **Colab:** official (a) vs vendored (b) ≥ **0.9999** on all 200 rows, and vendored (b) vs cache (c) ≥ **0.999** on all 200 rows. | Kaggle (b) vs Colab (b): min **0.99999999999**, SHA1 asserted, `cuda:1` (Kaggle P0.1b). (a) vs (b): min **1.000000** (Colab P0.1a). (b) vs (c): min **0.998083** (Colab P0.1a); Kaggle (b) vs (c): min **0.998083** (Kaggle P0.1b). | **HOLD** (rule 1) |
| G0.2 | Replaying `casmi_dreams_colab.ipynb` cell 7 reproduces `corpus_meta.parquet` **row by row**: `ingest_lib`, `inchikey14` and `adduct` equal, `precursor_mz` equal to float64. `corpus_ce.parquet` is written. | 1,323,878 vs 1,323,878 rows; 0 mismatches in every compared column (also `normalized_smiles`, `molecular_formula`, `ionization_mode`, `oob_mass`); `corpus_ce.parquet` written (Colab P0.2). | **PASS** |
| G0.3 | The SELFIES round-trip (stereo and isotopes stripped → SELFIES → SMILES → ik14, RDKit 2026.03.3) is ≥ **0.99** of the unique corpus structures that are single-component and contain only C, H, N, O, P, S, F, Cl, Br and I. The excluded share is reported, and `MAX_LEN` is set at p99.9 of the token lengths. | Round-trip **1.0000** over 194,589; excluded 79 of 194,668 (0.04 %, all disallowed elements); `MAX_LEN` = **171** (Colab P0.3). | **PASS** |
| G0.4 | Fast-tracker vs reference-tracker parity is **100%** on every prefix of 10k random structures (remaining counts, unsaturation, atom placed). Mask soundness over **all** round-trip-OK structures shows **0 violations** of the count, `</s>` and length rules. | Parity 0 / 10,000 structures. Violations: count 0, `</s>` 0, grammar 0, length 182 (exactly the structures above `MAX_LEN`, rule 6), DBE 139 → `DBE_PRUNE = False` (rule 3) (Colab P0.4). | **PASS** |
| G0.5 | `SelfiesTokenizer`, `MolToFormulaVector`, `DeNovoMassSpecGymModel` and `RandomDeNovo` import on Colab Python 3.12, and `msg_ik14.npy` is built. | All four import; `SelfiesTokenizer` ids equal ours (81 = 81 tokens, 0 / 998 mismatches); `msg_ik14.npy` has 28,923 ik14 (Colab P0.5). | **PASS** |

### Pre-registered decision rules

1. **G0.1, cache leg.** If (b) vs (c) fails while (a) vs (b) passes, the cache was built by a
   different DreaMS commit (A6). The gate is then **HOLD, not FAIL**. The drift is diagnosed and a
   re-embed of the corpus with the pinned commit is costed and scheduled. G0.1 passes once the
   re-embed exists and both other legs pass.
2. **G0.2.** A row mismatch is HOLD in the same way: a re-embed with CE stored is costed and
   scheduled (roadmap P0.2).
3. **G0.4, DBE prune (A8).** Any DBE-prune violation sets `DBE_PRUNE = False` and is recorded with
   examples. It **does not** fail G0.4. Only count, `</s>` or length violations fail it.
4. **G0.4, tracker.** A parity failure that cannot be fixed falls back to the reference tracker
   (plan §6). G0.4 then passes only if the reference tracker's mask walk shows 0 violations, and the
   slowdown is recorded for P4.
5. **Anything else that fails** stops the project at P0, as roadmap protocol 2 requires.
6. **G0.4, length rule** *(clarified 2026-09-28, before any run)*. `MAX_LEN` is p99.9 of the token
   lengths (G0.3), so about 0.1 % of structures exceed it by construction. Those structures are out
   of scope and reported under G0.3. The length rule passes when the walk's length violations are
   exactly those structures and no others.
7. **G0.4, grammar violations** *(clarified 2026-09-28, before any run)*. `walk_truth` also records a
   true token masked for a reason other than the count rule. This rule was not pre-registered, so it
   does not change the verdict. Any grammar violation is recorded as a note and reviewed before P1.

### Recorded, not gating

| item | value | source |
|---|---|---|
| SHA1 `embedding_model.ckpt` / `ssl_model.ckpt` | in `phase0_colab_gate.json` → `P0.1a.dreams_info` | P0.1a |
| SHA1 `dreams_embed.pt` | `a28488b277162f42d6361e339ffa54217a9d7381` | P0.1a |
| DreaMS head fine-tuning data (leakage note; provisionally MoNA `[M+H]+`, `fine_tune.sh:45-71`) | the checkpoint does not name its data, so the provisional note stands | P0.1a |
| `T_novel_install` (Kaggle) | 3.7 s (selfies wheel, `--no-index`) | P0.1b |
| `T_embed` per spectrum (T4, `cuda:1`) | 6.68 ms at batch 64; loading `dreams_embed.pt` takes 5.0 s | P0.1b |
| corpus rows / unique ik14 / per-adduct / mode split / share m/z > 1000 | 1,323,878 rows / 194,668 ik14; the splits are in `phase0_colab_gate.json` → `P0.2` | P0.2 |
| CE histogram, missing-CE share, `CE_BINS` | 23.9 % missing; `CE_BINS` [15, 23, 36, 44] plus "unknown"; histogram in `P0.2.ce_hist` | P0.2 |
| round-trip failure classes; excluded share (multi-component, disallowed elements) | no round-trip failures; 79 excluded (0.04 %), all disallowed elements | P0.3 |
| `MAX_LEN`; share of structures above it | 171; 182 of 194,589 (0.094 %) | P0.3 / P0.4 |
| SHA1 `selfies_vocab.json`; vocabulary size | `3046040b23503f2a6dd1fe46419b55dab9774a6a`; 81 | P0.3 |
| `selfies` pinned version and licence | 2.1.2 (MassSpecGym's pin), Apache-2.0 | P0.5 |
| DBE-prune violations (count, examples) and the `DBE_PRUNE` decision | 139 structures; `DBE_PRUNE = False`; examples in `P0.4.violation_examples` | P0.4 |
| fast-tracker cost | 19 µs per step (`legal_mask` + `push`); Kaggle smoke walk 0 / 200 violations | P0.4 / P0.1b |
| P1 preview: `|corpus ik14 \ msg_ik14|`, overall / NP libraries / enveda-np-examples | overall 166,048 / 194,668 (85.3 %); per library in `P0.5.p1_preview` | P0.5 |

### Findings

1. **The vendored DreaMS is exact.** Official (a) and vendored (b) agree to 1e-6 on Colab CPU, and
   the Kaggle T4 run of (b) agrees with Colab's (b) to about 1e-11 on identical inputs. Both are the
   same distance from the old cache: min cosine 0.998083.
   *Corrected by G0.1-R:* the old cache was **not** built by a different DreaMS version. The network
   is identical. The drift comes from peak selection, and so did this 0.998. See G0.1-R.
2. **CE is missing on 23.9 % of corpus rows.** `CE_BINS` = [15, 23, 36, 44], the 20/40/60/80 %
   quantiles of `ce_mean`. That gives 5 known bins **plus an "unknown" bin that is not rare**.
3. **The DBE prune is off.** 139 structures (0.07 %) break the rings + π = DBE identity (A8). This
   only costs search speed; the count and `</s>` masks stay exact.
4. **Leakage flag for P1.** 166,048 of 194,668 corpus ik14 (85.3 %) are outside MassSpecGym, and
   about 99 % of MassSpecGym's 28,923 ik14 are in the corpus. Anything trained on MassSpecGym has
   seen almost all of its structures, and nearly all of them are in our corpus.
5. **Offline costs are small.** `T_novel_install` is 3.7 s; `T_embed` is 6.68 ms per spectrum on a
   T4; the fast tracker takes 19 µs per step.

### G0.1-R — re-embed closure *(pre-registered 2026-09-29, after the G0 run, before the re-embed)*

Rule 1 schedules a re-embed of the corpus with the pinned commit. `casmi_novel_reembed_colab.ipynb`
(GPU) does it:

- all 1,323,878 rows, in the P0.2 replay order (`corpus_ce.parquet` `train_row`);
- the vendored `dreams_infer` with `dreams_embed.pt` (SHA1 `a28488b2…`);
- written to `casmi_2026/_novel/emb_dbec3a0/` in the same 200,000-row chunks. The old cache is not
  touched.

Estimated cost: 1.32 M × 6.68 ms ≈ **2.5 GPU-hours on a T4**, less on an L4 or A100.

| check | pass condition | result | status |
|---|---|---|---|
| G0.1-R | The new cache holds 1,323,878 finite 1024-d float32 rows. For the 200 parity rows, the peaks rebuilt from `train.parquet` equal those in `parity_200.npz` exactly, and the new vectors have cosine ≥ **0.999** with Colab (b) on **every** row. | 1,323,878 finite rows, 7 chunks, row order 0 mismatches. Parity peaks identical: **false**. New vs Colab (b): min **0.998823**, median 1 − 2e-12. (`reembed_gate.json`, A100, 1,892 s embedding + 121 s peaks.) | **FAIL** |

When G0.1-R passes, G0.1 is **PASS**, since its official and Kaggle legs already pass, and the G0
verdict is **PASS**. Recorded, not gating: the cosine distribution of new vs old cache over all rows,
overall and by library, adduct, `oob_mass` and peak-count bin.

#### G0.1-R result and diagnosis *(2026-09-29)*

**New vs old cache, by peaks kept** (`reembed_gate.json` → `new_vs_old`):

| peaks kept | rows | min cosine | share < 0.999 |
|---|---|---|---|
| 2–9 | 354,232 | 1 − 3e-12 | 0 |
| 10–39 | 565,568 | 1 − 1e-12 | 0 |
| 40–100 | 404,078 | 0.546 | 2.6 % |
| all | 1,323,878 | 0.546 (p0.1 % 0.980, p1 % 0.99932, median 1 − 3e-13) | 0.80 % |

Worst libraries by share below 0.999: spectraverse 3.8 %, msdial 2.2 %, riken 1.5 %; pluskal_ms2
0.02 %.

**Diagnosis.** Every spectrum that keeps fewer than 40 peaks is identical in the old and new caches,
so the network is the same and the old cache was built by the pinned DreaMS. Differences appear only
where spectra are cut to the top 100 peaks. `dreams_peaks` (cell 7 `peak_list`, and upstream
`trim_peak_list` too) cut with `np.argsort(it)[-100:]`. That sort is unstable, so when intensities
tie at the cut, the numpy build and CPU decide which tied peaks survive. The P0 CPU runtime, the A100
host and the machine that built the old cache each made different choices; hence
`parity_peaks_identical = false`. The Kaggle leg passed only because it embedded the stored
`parity_200.npz` peaks rather than rebuilding them. Rule 1's premise (a different commit) was wrong.

**Fix (2026-09-29).** `dreams_infer.dreams_peaks` now selects the top peaks with
`np.lexsort((mz, it))`: ties at the cut go to the higher m/z. It orders by m/z, then intensity, so
the output is independent of platform and input order. `to_model_input`'s trim does the same.
Offline tests cover ties at the cut, shuffled input and m/z ties. `dreams_infer.py` SHA1:
`6fd5ebbb…` (G0 run, re-embed) → `57ce1bc1…`.

**Decision (user, 2026-09-29).** Keep `_novel/emb_dbec3a0/` as the training cache without
re-embedding it under the fixed selection, and proceed to P1. **Accepted risk:** for spectra with
intensity ties at the top-100 cut, the training vectors (A100, unstable tie-break) and the Kaggle
test-time vectors (fixed tie-break) can use different peaks. This is bounded by the 40–100-peak rows.
The measured proxy is 0.80 % of rows below 0.999 against the old cache, with a worst case of cosine
0.55. It was not measured against the fixed selection.

**Consequences.**
- `casmi26-novel-offline` and `parity_200.npz` were built with the old `dreams_infer`. The Kaggle and
  re-embed notebooks now embed the fixed source, so their `src_sha1` check against the G0 manifest
  fails by design. P5 builds a new offline dataset and manifest.
- The re-embed notebook would skip every chunk already in `emb_dbec3a0/`. Any future re-embed must
  write to a new folder.
- The accepted risk rests on an argument, not a measurement. G0.1-R2 below measures it. It is a
  **precondition for P3**, the first phase that reads embeddings. It does not block P1 or P2.

### G0.1-R2 — fixed-code re-embed *(pre-registered 2026-09-29, before any run; precondition for P3)*

**Why.** Two things are still unverified after the fix:

- (a) that the lexsort `dreams_peaks` gives the same peaks on every platform. The Kaggle leg passed
  only because it embedded the stored `parity_200.npz` peaks; the offline tests ran on one machine.
- (b) how far the `emb_dbec3a0` training cache is from what the fixed code produces.

The G0.1-R run cost 1,892 s of embedding plus 121 s of peaks on an A100, so this check is cheap.

**Run.**

- Re-embed all 1,323,878 rows in the P0.2 replay order (`corpus_ce.parquet` `train_row`).
- Use `dreams_infer.py` at SHA1 `57ce1bc1…` (lexsort) and `dreams_embed.pt` `a28488b2…`.
- Write to a **new** folder, `casmi_2026/_novel/emb_dbec3a0_lx/`, in 200,000-row chunks. Leave
  `emb_dbec3a0/` untouched.
- Choose a fixed set of **tie-rich parity rows**, `parity_lx`: about 200 corpus rows, at least half
  of them with an intensity tie at the top-100 cut. Rebuild their peaks from `train.parquet` with
  the fixed `dreams_peaks` on three platforms: Colab CPU, the GPU host and Kaggle T4 (offline).

| check | pass condition | result | status |
|---|---|---|---|
| G0.1-R2 | The new cache has 1,323,878 finite 1024-d float32 rows, and its row order (`inchikey14`, `adduct`, `precursor_mz`) has 0 mismatches. The `parity_lx` peaks are **byte-identical** across Colab CPU, the GPU host and Kaggle T4. The new-cache vectors for `parity_lx` have cosine ≥ **0.999** on **every** row with the Kaggle-embedded vectors built from the Kaggle-rebuilt peaks. | — | PENDING |

**Decision rules.**

1. **Pass.** G0.1 becomes **PASS**, since its official and Kaggle legs already pass, and G0 becomes
   **PASS**. P3 trains on `emb_dbec3a0_lx`. `parity_lx` (peaks and vectors) replaces
   `parity_200.npz` as the P5 DreaMS fixture.
2. **The peaks differ across platforms.** The fix is incomplete. Diagnose it, fix `dreams_peaks`,
   and re-run into another new folder. P3 does not start.
3. **The peaks are identical but the cosine fails.** The network is not deterministic across
   hardware at this tolerance. Record the distribution and stop before P3 (protocol 2).

**Recorded, not gating:** the cosine distribution of new vs `emb_dbec3a0` over all rows, overall and
split by library, adduct, `oob_mass` and peak-count bin. This measures the accepted risk above.
Also recorded: the share of corpus rows with a tie at the cut, the GPU, and the timings.
