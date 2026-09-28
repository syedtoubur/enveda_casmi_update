# CASMI 2026 — runbook

## Where things stand

| submission | FPNet | rank_train.npz | LB |
|---|---|---|---|
| forked | forked | forked | **0.337** |
| yours | new (good) | v1 (leaked) | **0.287** |
| next | new (unchanged) | **v2** | run Stage D first |

**Keep the checkpoints.** On the scaffold-disjoint holdout, against 48 real ±8.5 ppm decoys:

| checkpoint | es_MRR | es_top1 |
|---|---|---|
| `fp_single_s0` | 0.8278 | 0.7425 |
| `fp_single_s1` | 0.8250 | 0.7366 |
| `fp_merged_m0` | 0.8222 | 0.7431 |
| `fp_merged_m1` | 0.8197 | 0.7360 |

Those are unseen scaffolds — the model generalises. The listwise stage earned its keep
(0.556 → 0.828 single, 0.689 → 0.822 merged). The FPNet was never the problem.

## What went wrong, in one paragraph

Stage C v1 excluded the *validation* structures from its query sampling but not the structures
the *FPNet trained on*. The FPNet trained on essentially all of `train.parquet`, so every v1
query was a molecule the model had memorised, and `f·z` behaved as a near-oracle — the truth
was the model's top pick in **90.2%** of windows versus **42.0%** in the forked matrix. A GBDT
handed a feature that alone solves 90% of its training queries will use that feature and
discard the others, and it did: the model channel carried **60.2%** of the ranker's skill while
library fell to 2.7%, analog to 0.2% and MetFrag to 0.0%. On the leaderboard, where the model
has not seen the Class-2/3 structures, there was no fallback left.

---

## The fix — three notebooks, ~6h, zero Colab units

### 1. `casmi_export_fpnet_split.ipynb` — Colab, **CPU-only**, ~2 min

Reads `_fpnet/pack/{uni,splits}.npz` and writes `fpnet_split_keys.npz` (a few MB of InChIKey14
lists — not the 1.9 GB fingerprint matrix). Three disjoint sets:

| set | FPNet saw it? | used for |
|---|---|---|
| `seen_keys` | **yes** | Stage-C cells A and C — the Class-1 regime |
| `es_keys` | no (scaffold-disjoint holdout) | Stage-C cells D and B — the Class-2 regime |
| `banned_keys` | no (val `ban_keys`) | **Stage D only**, never a Stage-C query |

Add the file to the `casmi26-fpnet-v3` Kaggle dataset, next to the checkpoints.

### 2a. `casmi_rank_matrix_colab.ipynb` — **Colab, L4 + High-RAM** (single notebook)

Use this when you cannot run three Kaggle notebooks at once. Same logic; paths resolve under
Drive, one process, checkpointed to Drive every 50 queries.

**Which runtime:** Stage C is CPU-bound, not GPU-bound — the GPU only runs `model_logits` on a
few spectra per query. Wall clock goes to numba entropy search over 2.54M library spectra and
RDKit bond-breaking, both of which scale with vCPU.

| runtime | vCPU | RAM | units/hr | cores per unit |
|---|---|---|---|---|
| T4 + High-RAM | 4 | 51 GB | ~2.0 | 2.0 |
| **L4 + High-RAM** | **8** | **53 GB** | **~4.8** | **1.7** |
| A100 | 12 | 83 GB | ~13 | 0.9 |

**L4 + High-RAM.** ~2× faster than T4 at nearly the same cost-efficiency; A100 costs ~3× an L4
for 1.5× the cores on work the GPU barely touches. High-RAM is mandatory — `load_library` holds
all 2.54M peak lists in memory, and the notebook refuses to start below 40 GB.

Three verification cells print everything needed to judge the result:

* **VERIFY 1 — pre-flight**: self-retrieval and the ban filter over **8** queries (not one),
  printing each query's ppm offset and how many of its window's spectra are the truth;
  `frag_scores_fast == frag_scores`; truth in the pool **and inside the query's own window**;
  then a 6-query smoke test with a per-channel timing breakdown and a **measured ETA**. Nothing
  has started yet at that point — if the ETA is too long, lower `SC.N_SEEN`/`SC.N_UNSEEN` and
  re-run two cells.
* **VERIFY 2 — the gate**: channel-destruction test; writes `rank_train.npz` only if it passes.
* **VERIFY 3 — report**: one paste-back block, including the leak indicator split by regime
  (FPNet-seen vs FPNet-unseen) and a 31-column comparison against the forked matrix.

### 2b. `casmi_rank_matrix.ipynb` (v2) — Kaggle ×3, free, ~5h

Attach: competition data, `casmi26-offline-rdkit-2026033`, `coconut_dataset`, `val_scripts`,
`casmi26-fpnet-v3`. **Detach `casmi26-fp-models-v2` and `casmi26-ranker-features`** —
`load_neural_models()` globs `fp_*.pt` across all of `/kaggle/input`. The sha1 guard stops you
if you forget.

Run `SC.SHARD = 0, 1, 2` concurrently, then the merge cell once.

Three changes from v1:

- **Queries from both model-strength regimes.** Four cells, ~equal size:

  | cell | query structure | library | `M` | mirrors |
  |---|---|---|---|---|
  | A | FPNet-**seen** | visible | 0 | test Class 1 — model strong, `lv` high |
  | C | FPNet-**seen** | banned | 1 | a Class-2 molecule the model happens to nail |
  | D | FPNet-**unseen** | visible | 0 | cross-library Class 1 — model weak, `lv` strong |
  | B | FPNet-**unseen** | banned | 1 | **test Class 2** — absent from v1 entirely |

- **Paired two-regime scoring.** Each query runs the expensive work once (window,
  `model_logits`, `frag_scores`), then re-runs only the library and analog channels under both
  ban settings. This is the forked builder's design — its columns 14–24 are byte-identical
  between the `M=0` and `M=1` halves. Halves the runtime *and* gives a controlled contrast.

- **Queries must be able to retrieve themselves.** The library is indexed on the label's
  exact mass (`CFG.STRUCT_MASS`) and the pool on `ExactMolWt(normalized_smiles)`, but a query's
  window is built from its own reported `precursor_mz`, exactly as at test time. riken reports
  m/z to 2 decimals — ~12 ppm at 400 Da — so a riken query lands outside its own ±8.5 ppm
  window: `lv` is 0 by construction *and* the row group contains no positive row at all. The
  cohort now keeps only spectra whose corrected target is within `SC.QUERY_PPM_MAX`
  (= `CFG.PPM_WIN`) of their label mass, and prints a per-library usable %. The test set is
  high-precision timsTOF and has no such rows, so nothing representative is lost.

- **Full windows and a `G` column.** No hard-negative mining. v1's flat ~37-row cap
  under-weighted crowded windows 2× (row-weighted mean window 106.8 vs the forked 194.7), and
  crowded windows are where MRR is lost.

**Watch these two numbers in the output:**
- `truth is the model's top-1 in …% of windows` — should land well below v1's 90.2%
- the gate's `held-out MRR` — should be **0.65–0.80**, not 0.95

### 3a. `casmi_validate_ab_colab.ipynb` — **Colab, L4 + High-RAM**, ~2.5 h. **Not optional.**

Same four arms, same code, paths under Drive. The GPU only runs `model_logits` twice over
1,145 molecules (~4 min); everything else is numba, RDKit and 16 GBM fits, all CPU.

| runtime | vCPU | units/hr | est. wall | est. cost |
|---|---|---|---|---|
| T4 + High-RAM | 4 | ~2.0 | ~4 h | ~8 units |
| **L4 + High-RAM** | **8** | **~4.8** | **~2.5 h** | **~12 units** |
| A100 | 12 | ~13 | ~1.8 h | ~24 units |

The setup cell copies `_stagec/rank_train.npz` into the staged `casmi26-fpnet-v3/` folder, so
the "new matrix" is identified by co-location with the new checkpoints — the same rule you
must follow when uploading to Kaggle. Pass 1 checkpoints to Drive every 200 molecules and
resumes. A pre-flight cell verifies the `fp_bits` sha1 contract, the `G` column, and that the
new matrix is not v1 (truth-is-model-top1 must read ~72%, not 90.2%).

`build/test_stage_d_colab.py` proves the port differs from the Kaggle build only where
intended: five Stage-D cells byte-identical, the phase1 cells verbatim bar the path.

### 3b. `casmi_validate_ab.ipynb` — Kaggle, ~1.5h, free. **Not optional.**

4-way A/B over all 1,145 val molecules. Ship `both` only if C2 **and** C2_hard each gain more
than 1 paired SE and C1_npx does not regress by more than 1 SE. Expect `model_only` (new FPNet
+ old matrix) to be clearly negative — that arm is the control.

Sanity floor: `both` must beat `base` on **C2_hard**, the closest proxy to where the
leaderboard is decided.

---

## The gate

Stage C's last cell fits the real phase1 ranker on 60% of query groups, then destroys one
channel at a time on the held-out 40% and measures what each was carrying. It **refuses to
write `rank_train.npz`** if:

- the model channel carries > 50% of skill
- library or analog carries < 5%
- **the FPNet is top-1 on > 75% of FPNet-*unseen* queries** — its measured `es_top1` is
  0.7425 against 48 decoys, and a ~130-candidate window is harder, so anything above that
  means the holdout is not a holdout
- **seen−unseen top-1 gap < 0.10** — if the two halves are indistinguishable the split is
  separating nothing
- **MRR > 0.88 on the FPNet-unseen + library-banned cell** — the hardest cell must not be
  near-solved
- (matrices with no unseen half, i.e. v1 and the forked one) held-out MRR > 0.88

Absolute MRR over the whole matrix is deliberately *not* a rule any more. v2 contains an
FPNet-seen half on purpose — a test Class-1 molecule is in `train.parquet` by definition, so
the FPNet memorised it there too, and with a strong FPNet that half is honestly easy. The
0.88 ceiling was calibrated on the forked matrix, whose FPNet was top-1 in 42% of windows;
applied to a model at 74% it fails matrices that are fine. The rules above test the leak
mechanism itself instead of a proxy for it, and they are strictly harder to fool.

It is validated against ground truth — `python build/test_gate.py` runs the shipped cell
against both real matrices:

```
v1 matrix    (LB 0.287)  -> FAIL  (MRR 0.958, model 59%, library 3%, analog 0%)
forked matrix(LB 0.337)  -> PASS  (MRR 0.713, model 34%, library 15%, analog 15%)
```

---

## Known remaining weakness

**FPNet early stopping gated on the wrong metric.** Stage B1 saved on `es_bce`, which bottomed
at ep02, while `es_MRR` kept climbing (0.556 → 0.697 by ep06). B1 shipped ep02 and B2 started
from a worse init. B2 rescued it, so this is not the regression's cause — but it leaves roughly
0.03 es_MRR on the table. Change `train_one` to gate B1 on `es_MRR` if you ever retrain
(~4.5h A100, ~70 units). Not worth doing on its own.

**Validation cannot see pool misses.** Every `val_key` truth comes from `train.parquet`, which
is inside the pool, so truth-in-pool is ~1.0 by construction. Real Class-3 molecules are not in
the pool and score 0 regardless. Val MRR will always read above LB; the gap is the pool miss
rate. Read Stage D as *relative* deltas, never as an LB forecast.

**`CFG.USE_BIO_DB = False` is still an assertion, not a measurement.** You have
`chebi-lipidmaps-casmi26` on Drive. Stage D can settle the decoy-dilution question for zero GPU
units — worth a spare Kaggle slot.

---

## Files

```
casmi_export_fpnet_split.ipynb   split key export      (Colab CPU, 2 min)   <- done
casmi_rank_matrix_colab.ipynb    Stage C v2 + gate     (Colab L4, one run)  <- use this
casmi_rank_matrix.ipynb          Stage C v2 + gate     (Kaggle x3)          <- if you can shard
casmi_validate_ab_colab.ipynb    Stage D 4-way A/B     (Colab L4, ~2.5h)
casmi_validate_ab.ipynb          Stage D 4-way A/B     (Kaggle, ~1.5h, free)
casmi_fpnet_train.ipynb          Stage A+B  -- done, do not re-run
build/
  mk_stage_c_colab.py            generator for the Colab build
  mk_stage_c.py                  generator for the Kaggle build
  mk_stage_d.py                  generator for the A/B (Kaggle)
  mk_stage_d_colab.py            generator for the A/B (Colab; derives from the Kaggle nb)
  test_stage_d_colab.py          proves the Colab port differs only where intended
  export_split.py                cell-marked source for the split export
  py2nb.py                       cell-marked .py -> .ipynb
  run_all_tests.sh               everything below, in order
  selftest.py                    Stage-B machinery on synthetic data (CPU)
  test_process_query.py          the paired two-regime invariant: cols 14-24 identical
  test_cohort.py                 four-cell cohort + the fallback branch (takes a notebook name)
  test_gate.py                   proves the gate rejects v1 and accepts the forked matrix
stage_c_hotfix_cell.py           paste into a LIVE Colab kernel to re-run the cohort +
                                 assembly + VERIFY 1 without reloading the library/pool
stage_c_gate2_cell.py            paste into a LIVE kernel: matrix diagnostics + the
                                 regime-aware gate + the four-cell MRR breakdown
  mk_stage_c_v1_broken.py.bak    the version that caused 0.287, kept for reference
```

Regenerate after editing a generator: `python build/mk_stage_c_colab.py`.
Before shipping any change: `sh build/run_all_tests.sh`.
