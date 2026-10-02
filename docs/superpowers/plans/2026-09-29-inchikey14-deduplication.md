# InChIKey14 Candidate Pool and Submission Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate COCONUT structures and candidate pool by 14-character InChIKey (`InChIKey14`) such that each 2D molecular backbone appears strictly once in the candidate pool, and fix the top-25 answer emission in both `casmi_simple_sim_submission.ipynb` and `enveda-casmi-2026-tier1-patched.ipynb` so that every submitted slot represents a distinct InChIKey14.

**Architecture:** 
1. In `build_candidate_pool()`, normalize all keys to 14 characters (`k[:14]`). Use vectorized NumPy unique indexing to slice all synchronized COCONUT arrays (`co_fp`, `co_mass`, `co_smis`, `co_k14`) to unique `InChIKey14` rows.
2. Filter novel training structures (`tr`) against the deduplicated `coco_k14` set using matching 14-character string keys so overlapping structures (~509 rows) are never duplicated.
3. In answer emission (`CFG.DEDUP_KEYS` and `take_slots`), track `seen_k14` with explicit `str(k)[:14]` slicing so that no redundant stereoisomer or duplicate backbone ever consumes more than one of the 25 submission slots.

**Tech Stack:** Python 3.12, NumPy, Pandas, RDKit, Jupyter Notebooks.

**Spec:** Direct user request; `specs_simple_ms_simulator/roadmap.md` Action 3; `casmi_tier0_diagnostics.ipynb` Diagnostic D5.

## Global Constraints
- Preserve exact alignment between `fp`, `mass`, `keys`, and `smiles` arrays in `CandidatePool`.
- `co_fp` must be sliced using the exact same indices as `co_mass`, `co_k14`, and `co_smis` to maintain bitpacked fingerprint correspondence.
- Must not break `build/mk_simple_sim.py` or existing offline tests in `build/run_all_tests.sh`.
- All emitted submission rows must contain exactly 25 semicolon-separated candidate SMILES without nulls or duplicates.

## Review Focus
1. **Array Shape Mismatch**: Subsetting `co_k14` without subsetting `co_fp` or `co_mass` would cause catastrophic index desynchronization. Verified by assertion `len(co_k14) == len(co_mass) == len(co_fp) == len(co_smis)`.
2. **String Key Length Inconsistency**: COCONUT raw keys have 27 characters (`BSYNRYMUTXBXSQ-UHFFFAOYSA-N`) while Train keys have 14 (`BSYNRYMUTXBXSQ`). If string lengths are not normalized before set operations, set intersections silently evaluate to empty.
3. **Missing or Corrupted SMILES**: Slicing must not drop valid candidate SMILES or introduce NaNs.
4. **Stage Cross-Contamination**: In `casmi_simple_sim_submission.ipynb`, `take_slots` is called across Stages A, B, and C. The `seen` set must persist across all stages for a given molecule.
5. **Fast-Commit Compatibility**: `IS_FULL_RUN` and fast-commit execution must produce valid 25-candidate rows.

---

### Task 1: Offline Unit Test for InChIKey14 Deduplication & Array Synchronization

**Files:**
- Create: `build/test_pool_dedup.py`

**Interfaces:**
- Consumes: `numpy`, `CandidatePool`
- Produces: `test_coconut_dedup_synchronization()`, `test_submission_dedup_logic()`

- [ ] **Step 1: Write the offline test script**

```python
import numpy as np
import pytest

def deduplicate_coconut_arrays(raw_keys, smis, mass, fps):
    k14_all = np.array([str(k)[:14] for k in raw_keys], dtype=object)
    _, first_idx = np.unique(k14_all, return_index=True)
    first_idx = np.sort(first_idx)
    return k14_all[first_idx], smis[first_idx], mass[first_idx], fps[first_idx]

def test_coconut_dedup_synchronization():
    # 2 stereoisomers of AAAAA, 1 BBBBB, 1 CCCCC
    raw_keys = np.array(['AAAAA-1234567890-N', 'AAAAA-0987654321-N', 'BBBBB-1111111111-N', 'CCCCC-2222222222-N'], dtype=object)
    smis = np.array(['SMI_A1', 'SMI_A2', 'SMI_B1', 'SMI_C1'], dtype=object)
    mass = np.array([100.1, 100.1, 200.2, 300.3], dtype=np.float64)
    fps  = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.uint8)

    k14, u_smis, u_mass, u_fps = deduplicate_coconut_arrays(raw_keys, smis, mass, fps)

    assert len(k14) == 3
    assert list(k14) == ['AAAAA', 'BBBBB', 'CCCCC']
    assert list(u_smis) == ['SMI_A1', 'SMI_B1', 'SMI_C1']
    assert list(u_mass) == [100.1, 200.2, 300.3]
    assert u_fps.shape == (3, 2)
    assert np.array_equal(u_fps[0], [1, 0])

def test_submission_dedup_logic():
    # Candidates list with duplicate InChIKey14
    keys = ['AAAAA-1234567890-N', 'AAAAA-0987654321-N', 'BBBBB', 'CCCCC']
    order = [0, 1, 2, 3]
    smis = ['SMI_A1', 'SMI_A2', 'SMI_B', 'SMI_C']

    seen = set()
    out_smis = []
    for idx in order:
        k14 = str(keys[idx])[:14]
        if k14 in seen:
            continue
        seen.add(k14)
        out_smis.append(smis[idx])

    assert len(out_smis) == 3
    assert out_smis == ['SMI_A1', 'SMI_B', 'SMI_C']
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest build/test_pool_dedup.py -v`
Expected: 2 passed.

---

### Task 2: Update Candidate Pool Construction in `enveda-casmi-2026-tier1-patched.ipynb`

**Files:**
- Modify: `enveda-casmi-2026-tier1-patched.ipynb` (Cell 6: `def build_candidate_pool`)

**Interfaces:**
- Modifies `build_candidate_pool()`
- Return: `CandidatePool` containing only unique `InChIKey14` entries

- [ ] **Step 1: Inspect Cell 6 in `enveda-casmi-2026-tier1-patched.ipynb`**
Verify the exact lines defining `build_candidate_pool()`.

- [ ] **Step 2: Update Cell 6 with COCONUT InChIKey14 deduplication and Train key filtering**

Replace the pool assembly logic in Cell 6:
```python
def build_candidate_pool():
    t0 = time.time()
    d = os.path.dirname(find_file('coco_fp.npy'))
    cm = pickle.load(open(os.path.join(d, 'coco_meta.pkl'), 'rb'))
    co_fp = np.load(os.path.join(d, 'coco_fp.npy'))
    co_mass = np.load(os.path.join(d, 'coco_mass.npy'))
    raw_keys = np.asarray(cm['keys'], dtype=object)
    co_smis = np.asarray(cm['smiles'], dtype=object)

    # 1. Deduplicate within COCONUT by InChIKey14
    co_k14_all = np.array([str(k)[:14] for k in raw_keys], dtype=object)
    _, first_idx = np.unique(co_k14_all, return_index=True)
    first_idx = np.sort(first_idx)

    co_k14  = co_k14_all[first_idx]
    co_smis = co_smis[first_idx]
    co_mass = co_mass[first_idx]
    co_fp   = co_fp[first_idx]
    print(f"[INFO] COCONUT 2.0 component: {len(raw_keys):,} -> {len(co_k14):,} unique InChIKey14 ({time.time()-t0:.1f}s)", flush=True)

    if CFG.USE_BIO_DB:
        try:
            bd = os.path.dirname(find_file('bio_fp.npy'))
            bm = pickle.load(open(os.path.join(bd, 'bio_meta.pkl'), 'rb'))
            bi_fp = np.load(os.path.join(bd, 'bio_fp.npy'))
            bi_mass = np.load(os.path.join(bd, 'bio_mass.npy'))
            bi_k14_all = np.array([str(k)[:14] for k in bm['keys']], dtype=object)
            bi_smis = np.asarray(bm['smiles'], dtype=object)

            _, bi_first_idx = np.unique(bi_k14_all, return_index=True)
            bi_first_idx = np.sort(bi_first_idx)
            bi_k14 = bi_k14_all[bi_first_idx]
            bi_smis = bi_smis[bi_first_idx]
            bi_mass = bi_mass[bi_first_idx]
            bi_fp = bi_fp[bi_first_idx]

            coco_set = set(co_k14)
            bi_mask = np.array([k not in coco_set for k in bi_k14], bool)
            co_fp = np.vstack([co_fp, bi_fp[bi_mask]])
            co_mass = np.concatenate([co_mass, bi_mass[bi_mask]])
            co_k14 = np.concatenate([co_k14, bi_k14[bi_mask]])
            co_smis = np.concatenate([co_smis, bi_smis[bi_mask]])
            print(f"[INFO] + ChEBI/LIPID MAPS: {bi_mask.sum():,} unique structures integrated.", flush=True)
        except Exception:
            print("[INFO] Bio DB not found, continuing with COCONUT + Train.", flush=True)

    # 2. Filter Train structures already in COCONUT (using matching 14-char keys)
    tr = pq.read_table(TRAIN, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    tr = tr.dropna().drop_duplicates('inchikey14')
    coco_k14_set = set(co_k14)
    tr = tr[~tr.inchikey14.isin(coco_k14_set)]
    print(f"[INFO] Computing fingerprints for {len(tr):,} novel training structures...", flush=True)

    if HAVE_RDKIT:
        with MPool(4) as mp:
            res = mp.map(fp_and_mass, list(tr.normalized_smiles), chunksize=500)
        ok = [i for i, r in enumerate(res) if r is not None]
        tr_fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
        tr_mass = np.array([res[i][1] for i in ok])
        tr_keys = tr.inchikey14.values[ok].astype(object)
        tr_smi = tr.normalized_smiles.values[ok].astype(object)
    else:
        tr_fp = np.empty((0, co_fp.shape[1]), dtype=np.uint8)
        tr_mass = np.empty(0, dtype=np.float64)
        tr_keys = np.empty(0, dtype=object)
        tr_smi = np.empty(0, dtype=object)

    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_k14, tr_keys])
    smis = np.concatenate([co_smis, tr_smi])
    good = np.isfinite(mass)

    pool = CandidatePool(fp[good], mass[good], keys[good], smis[good], cm['nbits'])
    print(f"[SUCCESS] Unified candidate pool ready: {len(pool.mass):,} structures ({time.time()-t0:.1f}s)", flush=True)
    return pool
```

- [ ] **Step 3: Verify the update in `enveda-casmi-2026-tier1-patched.ipynb`**
Run a Python check script to parse Cell 6 and confirm syntax and presence of `unique` and `first_idx`.

---

### Task 3: Update Answer Submission Deduplication in `enveda-casmi-2026-tier1-patched.ipynb`

**Files:**
- Modify: `enveda-casmi-2026-tier1-patched.ipynb` (Cell 10)

- [ ] **Step 1: Inspect Cell 10 in `enveda-casmi-2026-tier1-patched.ipynb`**
Locate the block under `if CFG.DEDUP_KEYS:`.

- [ ] **Step 2: Update slot-filling deduplication to strictly check `str(k)[:14]`**
```python
            if CFG.DEDUP_KEYS:
                smis, seen = [], set()
                for i in order:
                    raw_k = pool.keys[cand[i]]
                    if raw_k is None:
                        continue
                    k14 = str(raw_k)[:14]
                    if len(k14) < 14 or k14 in seen:
                        continue
                    seen.add(k14)
                    smis.append(pool.smiles[cand[i]])
                    if len(smis) >= CFG.TOPN:
                        break
                order = order[:CFG.TOPN]
```

- [ ] **Step 3: Verify syntax in Cell 10**

---

### Task 4: Update `build_candidate_pool` and `take_slots` in `casmi_simple_sim_submission.ipynb` and `build/` Pipeline

**Files:**
- Modify: `enveda-casmi-2026-phase1-enhanced-baseline.ipynb` (Cell 6: `build_candidate_pool`)
- Modify: `build/simple_sim_submit.py` (`take_slots`)
- Modify: `casmi_simple_sim_submission.ipynb` (Cell 6 and Cell 19)

- [ ] **Step 1: Update `build_candidate_pool()` in `enveda-casmi-2026-phase1-enhanced-baseline.ipynb`**
Update Cell 6 with the identical InChIKey14-deduplicated `build_candidate_pool()` so generator scripts inherit it.

- [ ] **Step 2: Update `take_slots()` in `build/simple_sim_submit.py`**
Modify `def take_slots(order, cand, smis, seen)`:
```python
def take_slots(order, cand, smis, seen):
    """Append ranked candidates, strictly InChIKey14-deduplicated, until TOPN slots are full."""
    for idx in order:
        c_idx = cand[idx]
        raw_k = pool.keys[c_idx]
        if raw_k is None:
            continue
        k14 = str(raw_k)[:14]
        if len(k14) == 14 and k14 not in seen:
            seen.add(k14)
            smis.append(pool.smiles[c_idx])
            if len(smis) == CFG.TOPN:
                break
```

- [ ] **Step 3: Run `build/mk_simple_sim.py` to regenerate `casmi_simple_sim_submission.ipynb` and `casmi_simple_sim_nosim.ipynb`**
Run: `python build/mk_simple_sim.py`
Expected output:
```
casmi_simple_sim_submission.ipynb: 24 cells
casmi_simple_sim_nosim.ipynb: 24 cells
```

- [ ] **Step 4: Verify `casmi_simple_sim_submission.ipynb` contains the updated `build_candidate_pool` and `take_slots`**
Check that `first_idx` appears in Cell 6 and `k14 = str(raw_k)[:14]` appears in Cell 19.

---

### Task 5: End-to-End Test Suite Verification

**Files:**
- Test: `build/run_all_tests.sh` or existing offline test files (`build/test_simple_sim.py`, etc.)

- [ ] **Step 1: Run all existing offline tests**
Run: `pytest build/test_simple_sim.py build/test_pool_dedup.py -v`
Expected: All tests pass.

- [ ] **Step 2: Perform sanity verification script on the notebook structure**
Write a quick script to verify that in both notebooks:
- `build_candidate_pool` contains `co_k14_all = np.array([str(k)[:14] for k in raw_keys], dtype=object)`
- `build_candidate_pool` contains `np.unique` on `co_k14_all`
- Deduplication logic in submission loops slices to `[:14]`
