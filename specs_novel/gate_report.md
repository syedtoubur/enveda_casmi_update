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

### Verdict: **PASS** (2026-09-29, via G0.1-R2). It was first closed by decision, which is kept below as history.

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
| G0.1-R2 | The new cache has 1,323,878 finite 1024-d float32 rows, and its row order (`inchikey14`, `adduct`, `precursor_mz`) has 0 mismatches. The `parity_lx` peaks are **byte-identical** across Colab CPU, the GPU host and Kaggle T4. The new-cache vectors for `parity_lx` have cosine ≥ **0.999** on **every** row with the Kaggle-embedded vectors built from the Kaggle-rebuilt peaks. | All three legs produced the same peaks digest, `cdf94fd6…`, with 0 of 200 rows differing (150 of them tie rows). Kaggle `cuda:1` vs new cache: min cosine **1 − 5.3e-12** (tie rows: 1 − 2.4e-12). CPU vs new cache: min 1 − 5.7e-12. 1,323,878 finite rows, order 0 mismatches. | **PASS** |

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

#### G0.1-R2 GPU leg — result *(2026-09-29, A100, `lx_gpu.json`)*

| value | result |
|---|---|
| rows / chunks / order | 1,323,878 finite 1024-d rows in 7 chunks at `_novel/emb_dbec3a0_lx/`; order mismatches 0 / 0 / 0 |
| host | Intel Xeon @ 2.20 GHz, AVX-512, numpy 2.1.3, torch 2.11.0+cu128; peaks 93 s, embedding 1,876 s |
| rows with a tie at the cut | **31,742 (2.40 %)** |
| `parity_lx` | 200 rows, 150 tie rows (15 per library across 10 libraries); peaks digest `cdf94fd6…` |
| cache vs fresh embed of the parity peaks (batch 64, same host) | min 0.9999999999985 |
| **lx vs `emb_dbec3a0`, non-tie rows** | min **0.99999999999937** (1,292,136 rows): identical |
| lx vs `emb_dbec3a0`, tie rows | median 0.99960; 30.5 % below 0.999; min **0.495** |
| lx vs `emb_dbec3a0`, all rows | 0.73 % below 0.999; 0.13 % below 0.99. Worst libraries: spectraverse 3.6 %, msdial 2.1 %, riken 1.3 % |

**Reading.** Every row that differs is a tie-at-the-cut row, and every non-tie row is identical to
1e-12. The diagnosis is confirmed exactly: the provisional cache differed from the fixed code only
where the unstable tie-break decided which peaks survived. The accepted risk is now measured: 9.7k
rows (0.73 %) differed at a cosine below 0.999, and the worst reached 0.49. It no longer applies
once P3 trains on `emb_dbec3a0_lx`.

#### G0.1-R2 CPU and Kaggle legs — result and verdict *(2026-09-29, `lx_cpu.json`, `lx_kaggle.json`)*

| leg | host | peaks vs GPU host | cosine vs new cache |
|---|---|---|---|
| GPU (Colab) | Intel Xeon 2.20 GHz, **AVX-512**, numpy 2.1.3, Python 3.13, A100 | reference, digest `cdf94fd6…` | fresh re-embed min 1 − 1.5e-12 |
| CPU (Colab) | **AMD EPYC 7B12, no AVX-512**, numpy 2.1.3, Python 3.13 | **identical**, 0 of 200 differ; tie flags match | min 1 − 5.7e-12 (recorded) |
| Kaggle | Intel Xeon 2.00 GHz, AVX-512, **numpy 2.0.2, Python 3.12**, T4 `cuda:1`, internet off | **identical**, 0 of 200 differ; identities match | min **1 − 5.3e-12**; tie rows 1 − 2.4e-12 (gated) |

**Why this closes the question.** numpy's sort kernels dispatch on the CPU's SIMD features: the
AVX-512 path comes from x86-simd-sort. The old unstable `argsort` was therefore expected to
differ between these hosts, and it did in G0.1-R. The fixed `lexsort` selection gives
byte-identical peaks on three hosts with **different CPU vendors, SIMD sets, numpy versions and
Python versions**, and on 150 rows chosen precisely because their cut falls in a tie. The network
agrees across A100, T4 and CPU to about 5e-12.

**G0.1-R2: PASS. G0.1: PASS** (its official and Kaggle legs passed in the G0 run). **G0: PASS.**

**Consequences.**

- **P3 precondition met.** P3 trains on `_novel/emb_dbec3a0_lx/`. The old caches
  (`_dreams_out/emb/`, `_novel/emb_dbec3a0/`) must not be used for any specs_novel training or
  evaluation.
- **P5 fixture.** `parity_lx.npz` from the Kaggle dataset `casmi26-novel-lx` is the DreaMS fixture.
  That dataset already holds `dreams_embed.pt` for `dreams_infer` `57ce1bc1…`, so the P5 offline
  dataset can build on it.
- **Limit of the check.** Byte identity is shown on 200 rows, not on the full corpus. `lexsort` is
  stable and deterministic by construction, and the tie-rich sample tested the only mechanism
  known to differ, so this is judged sufficient.

**Where it runs** *(added 2026-09-29, before any run; the notebooks come from `build/mk_novel_p0.py`,
and the helpers are in `build/novel_lx.py`)*:

- `casmi_novel_lx_reembed_colab.ipynb` (Colab GPU) → `_novel/lx/lx_gpu.json`. It does the
  re-embed, the tie flag, the `parity_lx` pick (150 tie rows + 50 others, seed 20260929), new vs
  `emb_dbec3a0`, and it packages the Kaggle dataset `casmi26-novel-lx`.
- `casmi_novel_lx_cpu_colab.ipynb` (Colab CPU runtime) → `_novel/lx/lx_cpu.json`. It is the CPU
  peak leg.
- `casmi_novel_lx_kaggle.ipynb` (Kaggle T4 ×2, internet off) → `lx_kaggle.json`. It rebuilds the
  peaks from the competition `train.parquet`, embeds them on `cuda:1`, and checks the cosine.

G0.1-R2 passes only if all three legs pass.

---

## G1 — Cohorts and the training split (P1)

*Pre-registered 2026-09-29, before any P1 run. Design: `phase1_plan.md` §3–§4.*

**Source, once run:** `casmi_novel_p1_colab.ipynb` (Colab CPU, High-RAM) writes
`_novel/p1/p1_gate.json`. P1 reads no embeddings. The cohort rows are `corpus_meta` row indices,
and those are valid in both `emb_dbec3a0` and `emb_dbec3a0_lx`.

### Definitions (fixed before the run)

- **Universe.** The P0 corpus: 1,323,878 `corpus_meta.parquet` rows and 194,668 unique ik14.
- **Label-quality filter, per row.** The neutral mass comes from
  `simsim_common.neutral_mass1(precursor_mz, adduct)`, and the truth mass from
  `simsim_common.formula_mass(molecular_formula)`. A row is quality-OK when:
  - |neutral − truth| ≤ max(10 ppm · truth, 0.0075 Da);
  - the neutral mass is in [157, 1159].
- **NP libraries.** riken, mona, msdial, massbank, gnps, enveda-np-examples, masaryk. enveda-180 is
  excluded.
- **Eligible molecule.**
  - Its ik14 ∉ `msg_ik14` (28,923 ik14).
  - It has at least one quality-OK row in an NP library.
  - Its `src_lib` is enveda-np-examples when it has quality-OK rows there (`timstof = True`).
    Otherwise it is the NP library with the most quality-OK rows for that molecule; a tie goes to
    the name that sorts first.
- **Scaffold key.** The canonical SMILES of the RDKit Murcko scaffold, computed from the
  stereo-stripped structure. An empty scaffold (acyclic) gives `"acyclic:" + ik14`, and an
  unparsable structure gives `"unparsable:" + ik14`. `group_size` is the number of corpus ik14 that
  share the key.
- **`G_MAX`.** The largest value in {200, 100, 50, 25} for which
  `600 · m(G) + 400 + 0.03 · U ≤ 0.07 · U`. Here U is the number of corpus ik14. m(G) is the mean
  `group_size` of eligible molecules with `group_size ≤ G`, where each molecule counts once and
  contributes its own group's size. If no value meets the budget, `G_MAX = 25` and the shortfall is
  reported; G1.3 then decides.
- **`pseudo_c3`** (target 600, seed 20260929).
  - First take every eligible timsTOF molecule with `group_size ≤ G_MAX`. If there are more than
    600, draw 600 of them uniformly.
  - Then draw uniformly from the other eligible molecules with `group_size ≤ G_MAX` until there are
    600. If fewer are eligible, take them all.
  - Decoder holdout is the **whole scaffold group**.
  - At evaluation, `pool_ban_keys = lib_ban_keys = {truth ik14}`.
- **`c2_displace`** (target 400).
  - Draw uniformly from eligible molecules that are in no `pseudo_c3` scaffold group. The
    `G_MAX` cap does not apply.
  - Decoder holdout is the **structure only**.
  - At evaluation, `lib_ban_keys = {truth ik14}` and `pool_ban_keys = {}`.
- **Spectra per molecule.**
  - n is drawn from the test's spectra-per-molecule distribution (`test.parquet` grouped by
    `molecule_id`), capped at 16 and at the number available.
  - Only `src_lib` rows that are quality-OK are used, with CE-known rows first.
  - Distinct (adduct, rounded CE) pairs are taken before any repeats.
- **`dec_split`.** Every corpus ik14 gets exactly one label:
  - `heldout_c3`: every member of a `pseudo_c3` scaffold group;
  - `heldout_c2`: the `c2_displace` structures;
  - `dec_val`: whole random scaffold groups with `group_size ≤ G_MAX`, acyclic singletons included,
    drawn up to 3 % of U;
  - `train`: the rest.

### Gate

| item | pass condition | result | status |
|---|---|---|---|
| G1.1 | `pseudo_c3` has ≥ **300** molecules, and ≥ **30** of them are timsTOF (enveda-np-examples) | 600 molecules; **15** timsTOF (run 1, 2026-09-29) | **FAIL** (run 1) → amended, see A-G1.1 |
| G1.2 | `c2_displace` has ≥ **300** molecules | 400 (run 1) | PASS (run 1) |
| G1.3 | the `train` label covers ≥ **90 %** of corpus ik14 (`dec_val` and both held-out labels count as removed) | 0.9463 (run 1) | PASS (run 1) |
| G1.4 | every invariant holds. `dec_split` labels are disjoint and cover every corpus ik14. No member of a `pseudo_c3` group is in `train` or `dec_val`. No `c2_displace` structure is in `train` or `dec_val`. Every cohort row is a quality-OK corpus row from its `src_lib`. No cohort ik14 is in `msg_ik14`. `pseudo_c3` ∩ `c2_displace` = ∅. | all 7 true (run 1) | PASS (run 1) |

### Decision rules

1. **All four pass.** G1 is PASS. Next is P3, once G0.1-R2 has passed (roadmap execution order).
2. **G1.1 fails only on the timsTOF count.** Stop, then report the timsTOF funnel: rows → quality-OK
   → ik14 ∉ `msg_ik14` → `group_size ≤ G_MAX`. Relaxing `G_MAX` for timsTOF molecules is a
   redesign that has to be registered **before** it is re-run. It is not an automatic rescue.
3. **Any other item fails.** Stop (protocol 2) and present the options.
4. **Changing a definition after this section is committed** requires a dated amendment here,
   written before the re-run.

### Recorded, not gating

| item | source |
|---|---|
| `G_MAX`, m(G) at each candidate value, and the share of eligible molecules excluded by `G_MAX` (the rare-scaffold bias) | P1 |
| eligibility funnel, overall and for timsTOF: molecules → quality-OK in an NP library → ∉ `msg_ik14` → `group_size ≤ G_MAX` | P1 |
| quality-OK share of rows per library (riken's 2-dp m/z) | P1 |
| per cohort: counts by `src_lib` and by adduct, the timsTOF count, the `in_scope` share (prep `ok` and ≤ `MAX_LEN` tokens), the CE-known share of picked rows, and the spectra-per-molecule distribution | P1 |
| per cohort: the `dreams_ft_risk` count (`src_lib == mona` with a picked `[M+H]+` row) | P1 |
| held-out share of U, split into `heldout_c3`, `heldout_c2` and `dec_val` | P1 |
| count of `molecular_formula` vs RDKit formula mismatches | P1 |
| overlap with the val_scripts cohorts (`val_scripts/val_key.pkl`), for information | P1 |

### G1 run 1 — result *(2026-09-29, `casmi_novel_p1_colab.ipynb`, `p1_gate.json`)*

**Verdict: FAIL on G1.1 only.** G1.2, G1.3 and G1.4 pass.

| value | result |
|---|---|
| quality-OK rows | 1,220,817 / 1,323,878 (92.2 %). Lowest: massbank 83.5 %, drug_plus 85.3 %, mona 86.1 %, riken 86.1 % |
| structures | 194,668. Prep `ok` 194,589, `element` 79. In scope 99.87 %. Formula mismatches 165. Scaffolds 108,531, of which 12,519 acyclic. Largest group: benzene, 6,577 |
| funnel | 194,668 → 50,223 (quality-OK in an NP library) → 26,966 (∉ `msg_ik14`) → 20,126 (≤ `G_MAX`) |
| **timsTOF funnel** | 250 → 245 → **15** → 15. 230 of 245 (94 %) are in MassSpecGym |
| `G_MAX` | 100, budget met. m(G): 200 → 15.87, 100 → 9.28, 50 → 5.13, 25 → 3.32. It excludes 25.4 % of eligible molecules |
| `src_lib` of eligible molecules | gnps 22,627, massbank 1,804, msdial 938, mona 873, riken 701, enveda-np-examples 15, masaryk 8 |
| `pseudo_c3` | 600 molecules (gnps 483), 1,028 spectra. CE known on 15.4 % of spectra. Spectra per molecule: median 1, mean 1.71. `dreams_ft_risk` 10. Median group size 1 |
| `c2_displace` | 400 molecules (gnps 328), 669 spectra. CE known on 16.3 %. `dreams_ft_risk` 6 |
| split | train 184,216 (94.63 %), dec_val 5,840 (3.00 %), heldout_c3 4,212 (2.16 %), heldout_c2 400 (0.21 %) |
| val_scripts overlap | `pseudo_c3` 29, `c2_displace` 9 (of 1,145 val molecules) |

**Diagnosis.** The binding constraint is the ICEBERG exclusion, not `G_MAX`: `G_MAX` removes 0 of
the 15 timsTOF molecules. No `G_MAX`, seed or sampling choice can exceed 15 honest timsTOF
Class-3 molecules, so decision rule 2's suggestion does not apply. The threshold of 30 was copied
from the roadmap without checking `P0.5.p1_preview`, which already held this count.

**Findings for later phases** (recorded, not gating):

1. About 80 % of both cohorts come from GNPS. P4 reports its numbers per `src_lib`.
2. Only about 15 % of cohort spectra have a known CE, while every test spectrum has one. P3 must
   handle unknown CE well, and P4 reports results split by whether CE is known.
3. Spectra per molecule: median 1. Cohort results are pessimistic about multi-spectrum
   aggregation.
4. The cohort `formula` column is the label formula. The oracle-formula gates (G3, G4.0) use the
   formula of the prepared structure (`can`), because 165 structures disagree with their label.

### A-G1.1 — decoder-only timsTOF cohort `ts_dec` *(amendment, registered 2026-09-29 after run 1, before run 2)*

The user chose to amend G1.1 and not to end P1 or accept 15.

**Why it is honest.** G3 and G4.0 measure the decoder and its constrained search with the oracle
formula. ICEBERG is not involved in either. A timsTOF molecule that ICEBERG has seen is therefore
a valid test of the decoder, **as long as the decoder has not seen its scaffold group**. The DreaMS
head was fine-tuned on MoNA, not on enveda-np-examples.

**Definition of `ts_dec`** (run 2):

- **Candidates.** ik14 ∈ `msg_ik14` with ≥ 1 quality-OK enveda-np-examples row. `src_lib` is
  enveda-np-examples.
- **Excluded** if any of these holds:
  - the molecule is in a `pseudo_c3` scaffold group;
  - it is a `c2_displace` structure;
  - its scaffold group contains a `c2_displace` structure, which keeps every group whole under one
    label;
  - `group_size > G_MAX`. `G_MAX` is still chosen by the unchanged rule and is not re-chosen.
- **Size.** Every remaining candidate is taken; there is no draw.
- **Decoder holdout.** The whole scaffold group, under the new `dec_split` label `heldout_ts`.
  `dec_val` does not draw from `heldout_ts` groups.
- **Spectra.** Picked as for the other cohorts, from enveda-np-examples rows.
- **Use.** Decoder-level metrics only: G3, the G4.0 funnel and exact@k with the oracle formula,
  reported as the instrument-shift check. **It is excluded from every metric that uses ICEBERG
  `sv`, rerank or the merged list** (G4.1–G4.3, P6), and it is never pooled with `pseudo_c3`.

**Amended gate** (run 2; G1.2 and G1.3 are unchanged):

| item | pass condition |
|---|---|
| G1.1′ | `pseudo_c3` ≥ **300** molecules **and** `ts_dec` ≥ **100** molecules. The honest timsTOF molecules stay in `pseudo_c3` and are reported, with no minimum. |
| G1.4′ | G1.4, plus the following. No member of a `ts_dec` group is in `train` or `dec_val`. `ts_dec` is disjoint from `pseudo_c3` and `c2_displace`. Every `ts_dec` ik14 is in `msg_ik14`. Every `ts_dec` row is a quality-OK enveda-np-examples row. |

**Expected before the run:** `pseudo_c3` and `c2_displace` are identical to run 1 (same seed and
draw order), and only `dec_split` changes. The train share drops by roughly
230 × m(100) ≈ 2,100 structures (about 1.1 %). If G1.1′ or G1.3 fails, P1 stops (protocol 2).

**Recorded, not gating:** the `ts_dec` funnel (candidates → after the c3/c2 exclusions → ≤ `G_MAX`),
and the `ts_dec` counts by adduct, CE-known share and spectra per molecule.

### G1 run 2 — result *(2026-09-29, `casmi_novel_p1_colab.ipynb` under A-G1.1, `p1_gate.json`)*

**Verdict: PASS.** P1 is complete.

| item | result | status |
|---|---|---|
| G1.1′ | `pseudo_c3` 600 (≥ 300); `ts_dec` **155** (≥ 100); 15 honest timsTOF molecules in `pseudo_c3` (reported) | PASS |
| G1.2 | `c2_displace` 400 | PASS |
| G1.3 | train 183,102 = **94.06 %** | PASS |
| G1.4′ | all 11 invariants true | PASS |

- **`pseudo_c3` and `c2_displace` are the same as in run 1:** every recorded count is unchanged,
  as expected before the run.
- **`ts_dec` funnel:** 230 candidates → 163 after the c3/c2 group exclusions → 155 within
  `G_MAX`.
  - Spectra: 414, **CE known on 100 %**, 2.67 per molecule on average (median 2).
  - Median group size 4. The adducts are more varied than in the other cohorts: `[M+H]+` 226,
    `[M-H]-` 85, `[M+Na]+` 34, formate 29, `[M+NH4]+` 23.
- **Split:** train 94.06 %, dec_val 3.00 %, heldout_c3 2.16 %, heldout_ts 0.57 % (1,114),
  heldout_c2 0.21 %. `ts_dec` cost 1,114 structures, about half the pre-run estimate of 2,100.
- **val_scripts overlap:** all 155 `ts_dec` molecules are in the val_scripts cohorts. This is
  expected, since those cohorts include every enveda-np-examples molecule. It has no effect on the
  decoder metrics.

**Findings for P3/P4** (recorded, not gating):

1. **`ts_dec` is the only cohort whose conditions look like the test's.** It is timsTOF, has CE
   known on every spectrum, and has several spectra per molecule, whereas `pseudo_c3` has CE on
   15 % of spectra and a median of 1 spectrum. Where the two disagree on decoder metrics, `ts_dec`
   says more about the test conditions and `pseudo_c3` says more about novelty.
2. **`dreams_ft_risk` only looks at the rows picked from `src_lib`.** `ts_dec` shows 0 because its
   `src_lib` is enveda-np-examples. Since `ts_dec` molecules are in MassSpecGym, many probably also
   have MoNA `[M+H]+` spectra that the DreaMS head saw during fine-tuning. P4 therefore also reports
   a molecule-level flag: any MoNA `[M+H]+` row for that ik14 in the corpus. This is reporting only
   and does not change G1.

---

## G3 — Decoder training and constrained search (P3)

*Pre-registered 2026-09-29, **before any training run**. Design: `specs_novel/phase3_plan.md` §4–§5;
search design: `specs_novel/search_plan.md` v1.1.*

**Question (roadmap P3):** given the true formula, does the decoder's constrained search produce
structures closer to the truth than a formula-constrained random generator?

**Source, once run:** `casmi_novel_p3_train_colab.ipynb` (A100) writes `_novel/p3/decoder.pt` and
`train_log.jsonl`; `casmi_novel_p3_eval_colab.ipynb` (GPU) writes the candidate parquets;
`casmi_novel_p3_null_colab.ipynb` (CPU High-RAM) writes `_novel/p3/p3_gate.json` and prints the
verdict.

### Decisions fixed before the run (user, 2026-09-29)

| # | decision | choice |
|---|---|---|
| 1 | training rows | all libraries; **label-quality-OK rows only** (`novel_cohort.label_ok`, 92.2 % of corpus rows); in-scope structures only (prep `ok` and ≤ `MAX_LEN`); ik14 with `dec_split == "train"` only |
| 2 | GPU budget | Colab Pro, ~50–60 compute units for P3 (≈ 5 h A100 or 11 h L4) |
| 3 | model | `D_MODEL` 512, `N_DEC_LAYERS` 8, `N_HEADS` 8, ff 2048, dropout 0.1, tied embeddings; `MAX_SPEC_PER_MOL` 4 (≈ 28 M params) |
| 4 | CE dropout | **p = 0.3** applied to known CE (23.9 % of rows are already "unknown", so the effective unknown rate is ≈ 0.47) |
| 5 | G3 cohort | **`pseudo_c3` only**. `ts_dec` is reported separately as the instrument-shift check and is **not** part of the verdict |
| 6 | top-25 selection | **`maxmin`** (logp-seeded greedy MaxMin on Morgan r2/2048) is primary; **`logp`-only** is reported beside it from the same candidate pool |
| 7 | rerank | G3 is **decoder-only**. No ICEBERG `sv` and no MetFrag-lite `fr` enter any G3 number; the rerank is a P4 ablation applied to **both** arms |

### Search: a 3-method bake-off, selected on `dec_val`

All three methods get the **same candidate budget of N = 256 sequences per molecule**, so GPU compute
is matched by construction. All three drive `Tracker.legal_mask` / `push` from
`build/novel_decoder.py`; none reimplements the grammar.

| id | method | parameters |
|---|---|---|
| **A** | prefix-seeded batched sampling | masked expansion to depth `L_SEED` = 4 → top `K_ROOTS` = 8 distinct prefixes by logp → `M_ROLLOUT` = 32 rollouts each; `T` = 0.7, top-p 0.90 |
| **B** | plain masked beam (**control**) | `BEAM` = 256, no diversity mechanism |
| **C** | diverse (grouped) masked beam | `BEAM` = 256, cross-beam penalty `ALPHA_DIVERSE` = 0.6 on ring and branch tokens; the **unpenalised** logp is kept alongside for P4 ordering |

**Selection rule (pre-registered; this is what keeps the bake-off honest).** A, B and C are run on a
fixed random **300-molecule subset of `dec_val`**. The winner is the method with the highest **median
top-25 max Tanimoto** on that subset; ties go to A, then C, then B. The winner is then run **once** on
`pseudo_c3` and `ts_dec`, and **G3 is judged on that single run**. Selecting on `dec_val` and judging
on `pseudo_c3` is what prevents "best of three on the gate cohort" from inflating the verdict. The
losing methods' `pseudo_c3` numbers are **not** produced, so they cannot be promoted after the fact.

**Null baseline.** `RandomDeNovo` from `MSGYM_novel/massspecgym/models/de_novo/random.py`, constructed
with `formula_known=True`, `max_top_k=25`, `top_ks=(1, 10, 25)`, calling
`generate_random_molecule_graphs_via_traversal(CalcMolFormula(mol))` per truth. It receives the
**same oracle formula** as the decoder arm, on the **same molecules**.

**Oracle formula.** `nd.prep_structure(can)`'s formula (RDKit, H included), **not** the cohort
`formula` column: 165 corpus structures disagree with their label (G1 finding 4).

### Gate

| item | pass condition |
|---|---|
| **G3.1** | on `pseudo_c3`, the winning search beats `RandomDeNovo` on **median top-25 max Tanimoto by ≥ 0.10** (Morgan r2, 2048 bits, MassSpecGym `morgan_fp`), under the `maxmin` selection |
| **G3.2** | **the loss has converged**: `dec_val` token cross-entropy did not improve for **5 consecutive epochs** (early-stop patience 5) **and** the best epoch is **not** the last epoch recorded in `train_log.jsonl` |

Both must hold. G3.2 exists so that a G3.1 failure cannot be explained away as undertraining, and so
that a G3.1 pass cannot come from a run that was still improving when the budget ran out.

### Decision rules

1. **Both pass.** G3 is PASS. Next is P4-oracle, whose G4.0 is the kill gate.
2. **G3.1 fails.** Use the **single** pre-registered rescue, once: pretrain the decoder as a SELFIES
   language model on `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv`
   (`massspecgym.utils.load_unlabeled_mols`) with cross-attention off, then fine-tune as above and
   re-run G3 **once**. If it fails again, specs_novel ends (protocol 2).
3. **G3.2 fails but G3.1 passes.** Not a PASS. Continue training from the last checkpoint until G3.2
   holds, then re-read G3.1 from the resulting model. This is a continuation of the same run, not a
   new arm, and no amendment is needed.
4. **Changing any definition in this section** requires a dated amendment written here **before** the
   re-run.

### Recorded, not gating

| item |
|---|
| the bake-off table: median and mean top-25 max Tanimoto for A, B and C on the `dec_val` subset, with per-method wall-clock per molecule |
| teacher-forced `dec_val` loss and token accuracy per epoch (`train_log.jsonl`), and the measured seconds per epoch and CU burn |
| unconstrained-sampling diagnostic: the share of samples whose heavy-atom formula is correct, with no masks |
| the funnel per cohort: sequences → leaves → valid → single component → full formula incl. H → unique ik14 |
| **median unique formula-valid candidates per molecule** (if this is below 25, the selector is inert and G3.1 is really a funnel measurement) |
| `maxmin` vs `logp`-only top-25 max Tanimoto, side by side, on every cohort |
| exact@1 / @10 / @25 by ik14 on every cohort, for both arms (the G4.0 preview) |
| MCES (myopic, `pulp`) on a fixed random **150-molecule subset** of `pseudo_c3` and on **all 155** `ts_dec` molecules — reported only, never gating, because the ILP is slow |
| every metric split by: `src_lib`, adduct, CE known vs unknown, and neutral mass ≤ 1000 vs > 1000 (the DreaMS range) |
| `dreams_ft_risk`, plus the molecule-level MoNA `[M+H]+` flag (G1 finding 2) |
| `ts_dec` results in full, as the instrument-shift check, never pooled with `pseudo_c3` |
