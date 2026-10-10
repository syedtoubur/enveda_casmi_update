# Probes P3e and P3/0.415: natural-product pool expansion and NP prior bonus

> **Base:** [`casmi26-v29-fusion-pop-glmh-0-415.ipynb`](file:///c:/Users/h/OneDrive/Desktop/ENVEDA_casmi/casmi26-v29-fusion-pop-glmh-0-415.ipynb) (0.415 public LB; same code as 0.411)
> **Generator:** `build/patch_v29.py` (extended). Tests: `build/test_patch_v29.py`
> **Outputs:** `casmi26-t4v3-npx.ipynb` (P3e), `casmi26-t4v3-np1.ipynb` (P3 rebased)
> **Revised 2026-10-03.** The first version of this plan bundled both changes into one notebook, `casmi26-t4v2-np1-expand`. It was split, and the expansion was moved, for the reasons in §2.

## 1. Motivation

- **Chemistry of the test set.** The test set comes from biological extracts. `natural_products.parquet` (see [`../Dataset_intro/natural_product_knowledge_table.md`](../Dataset_intro/natural_product_knowledge_table.md)) holds 152,717 LOTUS/NPAtlas connectivities, and **2,715 of them are absent from the public 710k pool**.
- **P3e (expansion).** Make those structures retrievable by a ranker. This is the same kind of lever as ChEBI/LIPID MAPS (BIO), which gave **+0.009** (EXPERIMENTS row 13).
- **P3 (NP bonus).** Add `pop += 1·[LOTUS or NPAtlas]` in the pool prior and the PubChem prior. This code already existed in `patch_v29.py --np-bonus`; P3 rebases it onto 0.415. **Risk:** row 30 shows that the popularity voice is already saturated on 0.411/0.415. G = 1 adds only μ·1 = 0.15 z to the pool list, but it is still extra popularity.

## 2. Why the original design was changed

1. **Bundling broke gate rule 1** (one change per probe). With a ±0.004 LB noise floor, a bundled result can't be attributed to either change.
2. **The expansion code in the original BASE cell could never score.**
   - It called `chem.raw_fingerprint`, `bits` and `z`. None of these exist in that scope; `z` lives inside `V.run`.
   - It appended rows with `pid = -1`. The pool-prior block treats `pid < 0` as "generated", so those rows keep their slot after the TOPN = 60 cut, get the `'|gen'` suffix, and receive no ranker score.
   - Adding rows to the v4n pool properly means rebuilding `pool_fp.npy`, the fragment tables and `POOL_POP`/`pool_np` so they stay aligned.
3. **Engine 2's `build_pool` is the natural place for the expansion.** The function sits in the `casmi_engine.py` string inside `ENG_FILES`. It already appends BIO and fingerprints non-COCONUT SMILES with `fp_and_mass`. Rows added there get, natively:
   - packed 6,930-bit fingerprints and exact mass;
   - two-ranker scoring;
   - entry to the engine top 40, then RRF `fuse2`, then the ICE/GLACIER rerank.
4. **`patch_v29.py` could not read 0.415.** The 0.415 notebook already carries the dynamic `KAGGLE_IS_COMPETITION_RERUN` switch, so the run-mode patch is now a no-op on that source.

## 3. What `--np-expand` does

- **Engine cell.** Before `ENG_FILES` is written to disk, a guarded `str.replace` (asserting `count == 1`) inserts the following into `build_pool`, right after the line that builds the training-structure frame `tr`:
  1. Read `natural_products.parquet` with `find()`.
  2. Keep rows with `~in_casmi_pool` and mass in 150–1250 Da.
  3. Drop rows whose `scorer_key` or `ik14` is already among the COCONUT/BIO/train keys, and drop duplicate `scorer_key`s.
  4. Append the rest to `tr`, so they are fingerprinted with the training structures.
  5. Write the added keys to `eng/npx_keys.txt` and log `TRACK4 NP expansion +N structures`.

  If any step raises, the engine logs `failed -> base pool` and carries on (rule R5).
- **Fusion cell (counters only, wrapped in `try`).** One line is printed:

  `TRACK4 NPX keys K | in engine top-40: molecules A | in final top-25 B | at rank 1 C | dMRR bound X | ENG duplicate keys D`

  The bound is Σ 1/(best NPX rank) / n_mol, i.e. the most the expansion could have gained if every NPX hit were a truth.

## 4. Gates (pre-registered in `gate_report.md`)

| probe | must print | keep |
|---|---|---|
| P3e | `NP expansion +N` with N > 0; B > 0; D ≈ 0 | LB ≥ 0.416 **and** B > 0. Read any LB move larger than the `dMRR bound` as noise. VOID if N = 0 |
| P3 / 0.415 | both `np flags ... found` lines | LB ≥ 0.416. VOID if `absent` is printed |

Winners stack as P6.

## 5. Running it

```bash
python build/patch_v29.py --src casmi26-v29-fusion-pop-glmh-0-415.ipynb --np-expand  --out casmi26-t4v3-npx.ipynb
python build/patch_v29.py --src casmi26-v29-fusion-pop-glmh-0-415.ipynb --np-bonus 1 --out casmi26-t4v3-np1.ipynb
python build/test_patch_v29.py      # also wired into build/run_all_tests.sh
```

**Kaggle.** Attach the same datasets as 0.415.
- **P3e** also needs `casmi26-natural-product-knowledge-table`.
- **P3** needs `casmi26-pubchem-popularity-prior` **Version 2**, the one with `pool_np.npy` and `pc_np.npy`.

The commit run (12 placeholder molecules) checks plumbing only: the `+N` line in the engine stdout tail, and that the counters print. Read the counters on the scoring rerun.
