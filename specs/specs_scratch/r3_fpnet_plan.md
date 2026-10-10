# specs_scratch R3 — FPNet channel in tier-2 ranking (grilling outcome, 2026-10-08)

Status: **built 2026-10-08** (gates pre-registered in `gate_report.md`, incl. the R3.run (a) amendment). Next: the run order below.

## Grilling log (4 rounds, clickable `AskUserQuestion`)
1. "Fingerprint matching" = a **spectrum→FP channel** (not structure-Tanimoto). Build on **both** r1b_glacier and
   r2_gl (R2 audit passed; r2_gl LB 0.311). Reuse existing checkpoints before any training.
2. Checkpoints: own + a forked set, both to LB. Fusion: RRF inside tier 2. The audit runs **in parallel** with the LB,
   not as a gate. Candidate FPs at runtime for now; precompute into the pool later.
3. 4 LB submissions, no same-day control; RRF k = 3 equal weights; stages A + B; audit bar MRR +0.02 and up > down.
4. Keep rule +0.008 (unpaired). The user supplied `specs/Dataset_intro/fp-net-models-intro.md` and the v3/v4 code zips.
   Analysis showed two FPNet families, so the fork probe was **revised** from prvsiyan `fp-models-v2` (same family
   as ours) to Ahmed **`fpnet_full1`** (different family, zenith Engine-1 backbone). Own bank = all 4 ckpts.

## Context
- **The gap.** The best scratch run is `casmi_scratch_r2_gl` at LB 0.311 (r1b_glacier 0.303, plus the class vote). Zenith is at 0.425, and its whole stack sits on FPNet `f·z`. The scratch ranker has only lv (library entropy), fr (MetFrag-lite) and sv (GLACIER/ICEBERG). It has no spectrum→fingerprint channel.
- **Why this lever.** An orthogonal channel fused by rank is the only lever that has ever paid in this project (memory *fusion-beats-tuning*).
- **Process.** Aligned by `/grill-me` (4 rounds, 2026-10-07/08).
- **R3 design.** Add an FPNet channel and fuse it inside tier 2 by RRF. Test two checkpoint families on two bases (4 LB probes). Run an honest offline audit in parallel. It explains the result but does not block the LB.

## Settled decisions
| # | Topic | Decision |
|---|---|---|
| 1 | Channel | Spectrum→FP model; candidate score `f·z` (binary candidate bits · predicted logits) |
| 2 | Bases | `r1b_glacier` (0.303) and `r2_gl` (0.311). Same probe on both |
| 3 | Checkpoints | **own** = `casmi-fpnet` s0,s1,m0,m1 (family P, 6,930 bits, split known). **full1** = Ahmed `casmi26-fpnet-full1/fpnet_full1.pt` (family A, 10,226 bits, trained on all folds) |
| 4 | z aggregation | Average over models, then the zenith way: `z = 0.5·(mean per-spectrum logits + mean per-polarity merged logits)` |
| 5 | Fusion | Tier 1 (lv ≥ 0.75) untouched. Tier 2 key = `1/(3+r_sv) + 1/(3+r_fp)`; r_sv is the rank of (sv + vote bonus), r_fp the rank of `f·z`, both descending with average ranks for ties (so all sv=0 rows tie and FP decides among them). Tie-breaks: fr, then \|ppm\|. Fixed in advance and never tuned |
| 6 | Stages | A and B (as R2). Stage C top-up and non-deep mass-nearest untouched |
| 7 | Candidate FPs | Runtime RDKit, scored-range and stage-B candidates only, cached per key. Precomputing them into pool v3 comes later |
| 8 | LB | 4 submissions with no same-day control: `r3_own_gl`, `r3_full1_gl`, `r3_own_r2gl`, `r3_full1_r2gl` |
| 9 | Keep rule | ≥ its base's old score **+0.008** (r1b_glacier probes ≥ 0.311; r2_gl probes ≥ 0.319). A result in [+0.004, +0.008) gets a paired re-run with its control the next day |
| 10 | Audit | val_key **C2 (400) + C2_hard (200)**, whose structures are banned from own-FPNet training. Own library spectra banned, sv = 0. Pre-registered PASS (own ckpt): MRR@25(with FP) − MRR@25(without FP) ≥ **+0.02** and truth up > down. full1 and C1_npx (slice I, which FPNet trained on) are reported as leaked upper bounds only. The verdict decides whether we keep iterating on FP, not whether we submit |

## Facts the build relies on (verified)
- **Family P**, from `build/fpnet_train.py` and the zenith Engine 2 `pv_fp` (scratch/…zenith.py L125):
  - d512, 6 layers, 8 heads, instrument + adduct embeddings, 128 peaks.
  - Raw bits: Morgan r2 4096 + r3 4096 + RDKitFP(2048, maxPath 6) + MACCS. `fp_bits.npy` (6,930) lives next to `coco_fp.npy`.
  - The own ckpts embed the fp_bits sha1. They were trained with RDKit 2026.03.3, the same RDKit as the scorer, so there is no Morgan drift (Morgan bits are stable from 2021.09 on).
- **Family A**, from `casmi-v4-models.zip` → `casmi/fpnet.py` and `casmi/chem.py`:
  - `FPNet(nbits, d, layers=8)`, `prep_peaks` (160 peaks, sqrt intensity), `chem.ADDUCT_LIST` index, and the `ce_known`, `merged`, `mode` inputs. `load_model` reads nbits/d/layers from the ckpt.
  - Raw bits from `FP_BLOCKS`: ecfp4 4096, ecfp6 4096, fcfp4 2048, rdk6 2048, ap 2048, tt 2048, maccs 167, giving 16,507.
  - It needs the v4b pool's `fp_bits.npy` (10,226), the one zenith finds via `pool_meta.parquet`.
- **Leakage.** Slice I = `C1_npx`, whose ban set is empty, so both families saw it. C2/C2_hard are banned only for the own FPNet.

## Build
### Package `scratch_pkg/casmi_scratch/`
- **`fpnet.py` (new).** Two adapters behind one interface: `load_bank(kind, paths, bits_path, device) → Bank` with `.logits(items)` and `.featurise(smiles) → bits`.
  - `kind="own"`: the family-P model class and featuriser, ported from `build/fpnet_train.py` (`_g['rk']` etc.). Assert that the fp_bits sha1 equals the one in the ckpt.
  - `kind="ahmed"`: vendor `FPNet`, `prep_peaks`, `collate`, `ADDUCT_LIST`/`adduct_index` and `raw_fingerprint` from `casmi-v4-models.zip` (`casmi/fpnet.py`, `casmi/chem.py`), with attribution. fp16 autocast on T4.
  - `molecule_z(bank, spectra)` implements decision 4: per-spectrum items, plus per-polarity merged items (reuse `spectra.merge` if present, else vendor `merge_spectra`).
  - `fz(bank, keys, smiles)` returns f·z using a key-cached bit matrix.
- **`rank.py`.** `rank_two_tier(..., bonus=None, fp=None, rrf_k=3)`. With `fp=None` the order is exactly today's (test).
- **`pipeline.py`.**
  - Pass 1, `plan_molecule`: compute z once per molecule on the GPU, before the simulator.
  - Pass 2, stages A and B: compute `fz` for the stage's candidates and pass it to `rank_two_tier`.
  - Diag fields `fp_ok`, `fp_reordered`; counters `n_fp_mol`, `n_fp_reorders` in `vote_counters` (or a sibling `fp_counters`).
- **`config.py`.** `fp_channel=None|"own"|"ahmed"`, `fp_rrf_k=3`, `fp_ckpt_glob`, `fp_bits_glob`.
- **`audit.py`.**
  - Generalise `discrimination` to take `bonus_fn=None, fp=None`, keeping today's signature as the default.
  - Add `load_cohorts(val_key, train, cohorts=("C2","C2_hard"))`, which gives (truth key, spectra, ban rows/keys) per molecule, with the library ban built from `ban_rows`.

### Drivers and notebooks (regenerate through `scratch_pkg/tools/mk_notebook.py`, never hand-edited)
- **`NOTEBOOKS`:** add 4 Kaggle variants, keeping each base's overrides plus `fp_channel`:
  - `r3_own_gl` = r1b_glacier + `fp_channel="own"`
  - `r3_full1_gl` = r1b_glacier + `"ahmed"`
  - `r3_own_r2gl` = r2_gl + `"own"`
  - `r3_full1_r2gl` = r2_gl + `"ahmed"`
- **`drivers/r0_kaggle.py`:**
  - Load the bank if `fp_channel` is set.
  - Fast-commit asserts:
    - (a) featuriser bit count == nbits, and the sha matches (own);
    - (b) **self-rank smoke**: for 5 train molecules with known structure, the truth's f·z ranks top-5 in its ±8.5 ppm window. This catches a wrong bit order, adduct vocab or peak prep;
    - (c) `n_fp_mol > 0`, `n_fp_reorders > 0`.
  - Print FP timing.
- **Kaggle datasets to attach:**
  - own probes: `casmi-fpnet` plus the dataset holding family-P `fp_bits.npy`;
  - full1 probes: `casmi26-fpnet-full1` plus the v4b pool dataset with family-A `fp_bits.npy`.
  - At build time, check these names against zenith's attached inputs.
- **`drivers/r3_audit_colab.py` (new):** `casmi_scratch_r3_audit_colab.ipynb` on a Colab T4.
  - C2 ∪ C2_hard, sv = 0, with and without FP, for both banks.
  - C1_npx is reported too.
  - Writes `_r3_audit/verdicts.json`, cached and resumable like `r2_audit_colab.py`.

### Tests (`scratch_pkg/tests/test_r3.py`)
- `fp=None` reproduces the R2 order.
- RRF reorders tier 2 only, and sv ties resolve by FP.
- Stage C is untouched.
- Both featurisers return the expected bit counts on 3 SMILES. Family A matches the vendored `raw_fingerprint`.
- `molecule_z` shape and averaging, using a fake bank.
- Audit with `fp` on `toy_analog_lib`.
- `test_dryrun` for one r3 variant with a fake bank.
- The full suite stays green: `python scratch_pkg/tests/run.py`.

### Docs (before any measurement)
- `gate_report.md`: R3 section with R3.run (fast-commit asserts, full run ≤ 7.5 h), R3.LB (decision 9) and R3.audit (decision 10).
- `specs/specs_scratch/r3_fpnet_plan.md`: this plan plus the grilling log.
- `roadmap.md`: S5 = R3.
- `data_contract.md`: the attached FP datasets.

## Run order (user-side)
1. Build and tests (me) → upload nothing new: the pool is unchanged.
2. Kaggle: submit the 4 r3 notebooks and check the fast-commit asserts.
3. In parallel, on a Colab T4: `casmi_scratch_r3_audit_colab` → paste the verdicts.
4. Record the LB scores and the audit in `gate_report.md`, then use `/update-experiments` (also log r2_gl 0.311 if it is missing).

## Risks
- **Adduct and instrument vocab mismatch** between test metadata and each family's vocab. Smoke (b) catches gross errors only.
- **full1 licence** is CC BY-NC, derived from the competition data. Zenith already uses it, so it is treated as allowed.
- **Runtime.** Family-A featurisation is 7 generators per candidate, which is heavier. It is cached per key and run on scored-range candidates only. Measure it in the fast commit.
- **No paired control.** That is why the bar is +0.008 rather than +0.004.

## Verification
- `python scratch_pkg/tests/run.py` is green.
- A local dry run of `r3_own_gl` with a fake bank shows the counters firing.
- On Kaggle, the fast commit passes asserts (a)–(c) and the full run finishes ≤ 7.5 h.
- The audit's verdicts.json is in.
- The LB is judged by decision 9.


## Build log (2026-10-08)
- `casmi_scratch/fpnet.py`: both families (`p_model` / `a_model`, verbatim peak prep, adduct lists, featurisers),
  `FPBank.molecule_z` (family P: single models per spectrum + merged models on the merged pseudo-spectrum, CE 25;
  family A: 0.5 (per-spectrum + per-polarity merged), ≤ 16 spectra, fp16 autocast), `fz` with a per-key cache,
  `release()` after pass 1, `find_bank_files` (fp_bits chosen by sha1 / nbits), `parity_reference`.
- `rank.rank_two_tier(..., fp=, rrf_k=)` with `avg_rank_desc`; `pipeline`: z in pass 1, `fp_scores` in stages A + B,
  counters `n_fp_z`, `n_fp_mol`, `n_fp_reorders`; `config`: `fp_channel`, `fp_rrf_k`.
- `audit`: `load_cohorts`, `ban_mask`, `fp_discrimination`, `summarize_fp`, `fp_self_rank`.
- Driver `r0_kaggle.py`: bank + parity + self-rank before the run; R3.run checks (a)–(c), asserted on the fast commit.
- Notebooks: `casmi_scratch_r3_{own_gl, full1_gl, own_r2gl, full1_r2gl}.ipynb` (diff vs base = `fp_channel` + tag),
  `casmi_scratch_r3_audit_colab.ipynb`.
- Tests: `test_r3.py` (11, incl. dry runs of both families with tiny real checkpoints). Full suite green.

## Run order (user-side, concrete)
1. **Kaggle** — attach to the r3 notebooks, on top of each base's inputs:
   - `*_own_*`: `casmi-fpnet` + the COCONUT arrays dataset (the one with `fp_bits.npy` / `coco_fp.npy`);
   - `*_full1_*`: `casmi26-fpnet-full1` + `casmi26-v2-pool`.
   Fast commit must print `[fp] featuriser parity … | self-rank top-5 ≥4/5` and PASS R3.run (a)–(c).
2. **Colab T4** — copy `casmi-fpnet/`, `casmi26-fpnet-full1/`, `casmi26-v2-pool/` into `MyDrive/casmi_2026/`
   (with `coconut_dataset/`, `val_scripts/` and `casmi26-scratch-pool/` already there), run
   `casmi_scratch_r3_audit_colab.ipynb` and paste the verdict.

## Results (2026-10-08)
LB: `r3_own_gl` 0.322, `r3_own_r2gl` 0.320, `r3_full1_gl` 0.341, `r3_full1_r2gl` **0.346**. All four clear the keep rule.
full1 beats own by +0.019 / +0.026, and the class vote adds nothing measurable once FP is on. Details are in
`gate_report.md` R3 Verdicts. Audit pending.
